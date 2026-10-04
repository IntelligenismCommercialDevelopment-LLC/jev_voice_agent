"""dry_suite.py -- the zero-LLM regression gate. One run takes about 0.1 seconds, costs
nothing, and never touches the real pool/ or workspace/.

Usage:
    cd jev_voice_agent && <venv>/bin/python IAF_AI/dry_suite.py

★Why this file exists:
  An earlier round of this project recorded a "9/9 all green" acceptance -- but that was a
  **one-off run typed at a prompt**, and the script was never kept (grep the whole tree:
  `LaggySink` survives only in the write-up). So the next round's acceptance step, "re-run
  the existing 9/9 dry suite", was, taken literally, impossible to perform. This file pins
  that run down permanently. The order it was built in matters:
      first go green on the **pre-change** in-memory backend (which proves the suite itself
      is faithful)
      -> then switch to the file backend -> still green = a real acceptance
  Done the other way round (writing the suite after the refactor), green proves nothing.

★Which layer it wraps: the driver layer (`reply_list.VoiceDriver`). It does not go through
  a stdin subprocess. The semantics of `/feed` and of "a plain line" are **copied out of
  voice_cli's two branches** into the CASEs, so what runs here is the same path; on top of
  that, C4 drives `voice_cli._drain` directly and covers three member contracts
  (`add` / `len()` / `total_added`) -- which is where the real regression risk lives.
  Not covered: the CLI's stdin pipe and its command parsing (manual smoke only).

★Three timing disciplines (the easiest things to get wrong in a suite like this):
  1. An interruption must be fired by a **deterministic event**, never by racing a sleep --
     a timing suite that goes green intermittently is worse than having no suite at all.
     LaggySink set(reached) once it has played the Nth character, then awaits released.
  2. `dropped > 0` needs a **half-played block**. Interrupt only after a whole block has
     finished and `drop_unspoken()` returns 0 -- that assertion then goes falsely green.
     So the pause point deliberately lands in the middle of a block.
  3. Sandboxing: `assemble_pool`'s four path constants are module-level and read at call
     time -> patching them is enough to isolate. Without that, every run of the suite
     stuffs one more fake exchange into `pool/threads/` -- exactly the kind of residue that
     then has to be cleaned out by hand.
"""

import argparse
import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

def _find_shell_root(start):
    """Walk up to the shell root = the first level that holds both `IAF_AI/` and `config_base/`.

    ★Do not count dirname levels. This script gets moved (it was moved wholesale once
      already), and a level count is wrong the moment it moves -- wrong in the way that
      "the path now points at something that does not exist", so most call sites silently
      get empty content back instead of an error.
    ★Do not match on the name either: the copy-safety law -- code must never depend on the
      root's name or its location.
    ★These five lines of boilerplate have a copy in several files, on purpose (the IAF
      decoupling law: better to repeat "find the root / lock / atomic write" boilerplate
      than to hoist it out into one shared script).
    """
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, "IAF_AI"))
                and os.path.isdir(os.path.join(d, "config_base"))):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("cannot find the shell root (nothing above holds both "
                               "IAF_AI/ and config_base/): " + str(start))
        d = p


IAF_AI_ROOT = os.path.dirname(os.path.abspath(__file__))        # dry_suite lives in IAF_AI/
SHELL_ROOT = _find_shell_root(IAF_AI_ROOT)
LISTENER_DIR = os.path.join(IAF_AI_ROOT, "dispatch", "listener")

# ★All three are needed: the shell root (voice_cli) · IAF_AI (assemble_pool / reply_list /
#   lib.*) · the listener dispatch (listener_loop / jev_bench). The gate straddles two
#   dispatches, which makes it the only file in the shell that needs three sys.path entries.
for _p in (SHELL_ROOT, IAF_AI_ROOT, LISTENER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assemble_pool          # noqa: E402
import listener_loop          # noqa: E402
import reply_list             # noqa: E402


# ── The fake dispatch (these four methods are the entire contact surface with the engine) ──
class FakeDispatch:
    """VoiceDriver only ever touches new_session / append_to_history / run_stream /
    close_session.

    run_stream has to be an **async generator** -- the driver does `await stream.aclose()`,
    and the object a plain async function returns has no aclose.
    """

    def __init__(self, reply, chunk_chars=4, delay_s=0.0):
        self.reply = reply
        self.chunk_chars = chunk_chars
        self.delay_s = delay_s
        self.seen = []          # [(role, text)] <- assert which slice the LLM actually got
        self.opened = 0
        self.closed = 0

    def new_session(self, agent_name=None):
        self.opened += 1

    def append_to_history(self, role, text):
        self.seen.append((role, text))

    def close_session(self):
        self.closed += 1

    async def run_stream(self, agent_name=None):
        yield {"type": "meta", "generation_id": f"fake-{self.opened}"}
        for i in range(0, len(self.reply), self.chunk_chars):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            else:
                await asyncio.sleep(0)
            yield {"type": "delta", "text": self.reply[i:i + self.chunk_chars]}
        yield {"type": "done", "usage": {}}


# ── Two sinks ────────────────────────────────────────────────────────────────
class SilentSink:
    """Plays the whole block through (returns None). A muted console_sink, so the suite's
    own output does not get buried under the body of the reply."""

    def __init__(self):
        self.played = ""

    async def __call__(self, chunk):
        self.played += chunk
        return None


class LaggySink:
    """A sink that plays one character at a time. pause_at: on reaching the Nth character
    it set(reached) and then awaits released.

    ★This is how "the person happens to speak right here" becomes a deterministic event --
      see discipline 1 in the module header.
    ★Since barge-in was removed it no longer honours any abort position and **always plays
      the whole block** (returns None) -- which is exactly the shape of the new behaviour:
      once the machine starts speaking, it finishes the sentence.
    """

    def __init__(self, pause_at=None):
        self.pause_at = pause_at
        self.played = ""
        self.reached = asyncio.Event()
        self.released = asyncio.Event()

    async def __call__(self, chunk):
        for ch in chunk:
            if self.pause_at is not None and len(self.played) == self.pause_at:
                self.pause_at = None                 # pause once and once only
                self.reached.set()
                await self.released.wait()
            self.played += ch
            await asyncio.sleep(0)
        return None


# ── The sandbox ──────────────────────────────────────────────────────────────
_PATCHED = ("POOL_DIR", "THREADS_DIR", "SEMANTIC", "MAILBOX", "CONTROL_CONFIG")


class Sandbox:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="dry_suite_")
        self._saved = {k: getattr(assemble_pool, k) for k in _PATCHED}
        pool = os.path.join(self.tmp, "pool")
        assemble_pool.POOL_DIR = pool
        assemble_pool.THREADS_DIR = os.path.join(pool, "threads")
        assemble_pool.SEMANTIC = os.path.join(pool, "semantic.md")
        assemble_pool.MAILBOX = os.path.join(self.tmp, "mailbox")
        # ★Point at a file inside the sandbox that **does not exist** -> DEFAULT_CONFIG is
        #   used. Without that isolation every assertion would depend on whatever the real
        #   control_config.json at the shell root currently says, and one threshold edit by
        #   a user would turn a whole screen of them red.
        assemble_pool.CONTROL_CONFIG = os.path.join(self.tmp, "control_config.json")
        assemble_pool._warned.clear()
        self.session_dir = os.path.join(self.tmp, "workspace", "session_dry")
        os.makedirs(self.session_dir, exist_ok=True)

    def close(self):
        for k, v in self._saved.items():
            setattr(assemble_pool, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def thread_text(self, peer="user"):
        p = assemble_pool.thread_path(peer)
        if not os.path.exists(p):
            return ""
        with open(p, encoding="utf-8") as f:
            return f.read()


async def _wait_busy(drv, timeout=2.0):
    t0 = time.time()
    while not drv.busy:
        if time.time() - t0 > timeout:
            raise AssertionError("pump never became busy")
        await asyncio.sleep(0)


def _driver(sb, disp, sink):
    return reply_list.VoiceDriver(disp, sb.session_dir, peer="user", sink=sink,
                                  verbose=False)


# ── Assertion bookkeeping ────────────────────────────────────────────────────
RESULTS = []      # [(tag, assertion, result bool|None, evidence)]  None = skipped


def ok(tag, label, cond, evidence=""):
    RESULTS.append((tag, label, bool(cond), evidence))


def skip(tag, label, why):
    RESULTS.append((tag, label, None, why))


FILE_BACKEND = hasattr(reply_list, "UserBuffer")   # only true once step 1 is done


# ── C1: three cuts -> exactly 1 turn, joined with spaces ─────────────────────────
async def c1(sb):
    disp = FakeDispatch("好的，我这边看一下报价。")
    drv = _driver(sb, disp, SilentSink())
    for s in ("我想问一下", "不对是", "报价的事"):
        drv.reservoir.add(s)
    recs = await drv.run_until_quiet()

    ok("1", "/feed x2 + one line -> exactly 1 turn eats all 3 cuts (not 3 turns)",
       len(recs) == 1 and disp.opened == 1,
       f"turns={len(recs)} · new_session={disp.opened} · user_text={recs[0]['user_text']!r}"
       if recs else "no turn ran at all")

    ut = recs[0]["user_text"] if recs else ""
    ok("2", "the three cuts are joined with a SPACE, not a newline (a newline nudges the "
            "LLM into answering each one separately)",
       ut == "我想问一下 不对是 报价的事" and "\n" not in ut,
       f"user_text={ut!r}")

    ok("2b", "what the LLM actually received is that one whole segment "
             "(append_to_history was fed exactly once)",
       disp.seen == [("user", "我想问一下 不对是 报价的事")],
       f"seen={disp.seen!r}")


# ── C2: water arriving while busy -> eaten in one go next turn, never interrupting ──
async def c2(sb):
    disp = FakeDispatch("我在听，你继续说。", delay_s=0.003)
    drv = _driver(sb, disp, SilentSink())
    drv.reservoir.add("第一刀")
    task = asyncio.create_task(drv.pump())
    await _wait_busy(drv)

    # ★Copied from voice_cli's /feed semantics: only add, never set an interrupt flag
    drv.reservoir.add("忙着的时候进的水")
    drv.reservoir.add("再来一刀")
    busy_len = len(drv.reservoir)

    rec1 = await asyncio.wait_for(task, 5)
    rec2 = await asyncio.wait_for(drv.pump(), 5)

    ok("3", "two /feed cuts while the LLM is busy -> eaten in ONE GO on the next turn",
       busy_len == 2 and rec2 is not None
       and rec2["user_text"] == "忙着的时候进的水 再来一刀",
       f"{busy_len} cuts pooled while busy · turn 2 user_text={rec2['user_text']!r}"
       if rec2 else "turn 2 never ran")

    ok("4", "/feed never interrupts (both turns end with reason 'done')",
       rec1["reason"] == "done" and rec2 and rec2["reason"] == "done",
       f"reason=[{rec1['reason']}, {rec2['reason'] if rec2 else '—'}]")

    ok("3b", "back-pressure really happened: total_added counted 3 cuts and the pool is dry",
       drv.reservoir.total_added == 3 and len(drv.reservoir) == 0,
       f"total_added={drv.reservoir.total_added} · {len(drv.reservoir)} cuts left")


# ── C3: water arriving while busy -> NO interruption; the segment finishes and the water
#       waits (the behaviour since barge-in was removed) ────────────────────────────────
async def c3(sb):
    """★This CASE used to verify barge-in (a half-played block / the "N characters were
    never spoken" note / only the prefix being written down). Once barge-in was removed,
    every one of those flipped over -- and the flipped behaviour needs assertions guarding
    it just as much, or the day someone quietly wires barge-in back in, the gate says
    nothing.
    """
    tail = "这句尾巴现在【必须】出现在留痕里"
    reply = "这是一段足够长的回复，用来验证人在中途说话也不会打断它，" + tail + "。"
    disp = FakeDispatch(reply, chunk_chars=4)
    drv = _driver(sb, disp, SilentSink())
    sink = LaggySink(pause_at=10)
    drv.sink = sink

    drv.reservoir.add("第一句")
    task = asyncio.create_task(drv.pump())
    await _wait_busy(drv)
    await asyncio.wait_for(sink.reached.wait(), 5)

    # ★Copied from the "plain line" branch: while busy, only add water (the step that set
    #   an interrupt flag no longer exists)
    was_busy = drv.busy
    drv.reservoir.add("插进来的一句")
    sink.released.set()

    rec = await asyncio.wait_for(task, 5)
    spoken = rec["spoken"]

    ok("5", "★typing a plain line while busy -> NO interruption: this turn still completes, "
            "and that line waits in the pool for the next one",
       was_busy and rec["reason"] == "done" and len(drv.reservoir) == 1,
       f"reason={rec['reason']} · {len(drv.reservoir)} cuts left in the pool")

    ok("6", "★no half-played block: dropped == 0 (a half-played block = barge-in crept back)",
       rec["dropped_chars"] == 0 and sink.played == reply,
       f"played {len(sink.played)}/{len(reply)} chars · dropped {rec['dropped_chars']}")

    ok("8", "what is written down is the FULL text, not a prefix",
       spoken == reply,
       f"spoken {'==' if spoken == reply else '!='} the full text "
       f"({len(spoken)}/{len(reply)} chars)")

    thread = sb.thread_text()
    # ★Cross-file contract: this exact wording comes from `reply_list.VoiceDriver._persist`,
    #   which appends it only when dropped > 0. Change one side alone and this assertion
    #   turns into a no-op -- and because it is a **negative** probe it keeps reporting
    #   green, so barge-in could come back without the gate ever noticing.
    UNSPOKEN_NOTE = "characters were never spoken"
    ok("7", "★the transcript carries NO \"N characters were never spoken\" note "
            "(it should now only ever appear on the exception path)",
       UNSPOKEN_NOTE not in thread,
       "no note" if UNSPOKEN_NOTE not in thread
       else "★the note is still there = something went unspoken")

    ok("8b", "★the tail of the sentence made it into the transcript (this assertion used to "
             "require the opposite; removing barge-in inverted it)",
       tail in thread,
       "tail archived" if tail in thread else "★the tail was lost")


# ── C4: voice_cli's three-member contract + the cut count it displays ────────────
async def c4(sb):
    import voice_cli
    disp = FakeDispatch("知道了。")
    drv = _driver(sb, disp, SilentSink())
    cli_sink = voice_cli.TypingSink()
    drv.sink = cli_sink

    drv.reservoir.add("一")
    drv.reservoir.add("二")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await asyncio.wait_for(voice_cli._drain(drv, cli_sink), 5)
    out = buf.getvalue()

    ok("9a", "voice_cli._drain runs to completion on add/len()/total_added alone "
             "(the three-member contract is intact)",
       len(drv.reservoir) == 0 and drv.reservoir.total_added == 2,
       f"{len(drv.reservoir)} cuts left · total_added={drv.reservoir.total_added}")

    # ★This one FAILED before the refactor -- it caught a genuine defect in `_drain`:
    #   `total_added - before` measures "cuts that arrived during this turn", not "cuts
    #   this turn consumed", and it attributed them to the wrong turn as well (turn 1 ate
    #   1 and displayed 2; turn 2 ate 2 and displayed 0).
    #   Fixed while switching to the file backend: rec["cuts_consumed"] is now the real
    #   line count of reply_list.txt.
    ok("9b", "the observation line \"drew N cuts\" = the cuts this turn really consumed "
             "(the old total_added delta was wrong)",
       "drew 2 cuts" in out,
       f"observation line {'has' if 'drew 2 cuts' in out else 'lacks'} \"drew 2 cuts\"")


# ── C5-C7: the three assertions the file backend added ───────────────────────────
async def c5_c7(sb):
    if not FILE_BACKEND:
        why = ("the file backend is not implemented; before that refactor these three do "
               "not apply")
        skip("10", "concurrent add loses nothing once rename has taken the batch", why)
        skip("11", "a leftover .inflight is recognisable", why)
        skip("12", "the silence counter has a single writer (read -> zero -> write-back "
                   "does not eat the zeroing)", why)
        return

    # ── 10: add and drain interleaved concurrently, not one cut lost ──
    #   This is the entire reason os.rename replaced "read it, then truncate it". Eight
    #   threads write while the main thread takes; taking and writing are bound to overlap.
    res = reply_list.Reservoir(os.path.join(sb.tmp, "s10"))
    n_threads, per = 8, 50
    total = n_threads * per

    def writer(k):
        for i in range(per):
            res.add(f"t{k}-{i}")

    threads = [threading.Thread(target=writer, args=(k,), daemon=True)
               for k in range(n_threads)]
    got = []
    for t in threads:
        t.start()
    spins = 0
    while (any(t.is_alive() for t in threads) or len(res)) and spins < 200000:
        spins += 1
        text, n = res.drain()
        if n:
            got.extend(text.split(" "))
            res.release()
    for t in threads:
        t.join(timeout=5)
    text, n = res.drain()
    if n:
        got.extend(text.split(" "))
        res.release()

    ok("10", "concurrent add loses nothing once rename has taken the batch "
             "(8 threads x 50 cuts, written and taken at the same time)",
       len(got) == total and len(set(got)) == total and res.total_added == total,
       f"put in {total} · got back {len(got)} · {len(set(got))} distinct · "
       f"total_added={res.total_added}")

    # ── 11: a leftover .inflight is recognisable ──
    res2 = reply_list.Reservoir(os.path.join(sb.tmp, "s11"))
    res2.add("说了一半进程就崩了")
    res2.drain()                      # ★simulate a crash: drain, then never reach release()
    residue = res2.inflight_residue()
    ok("11", "a leftover .inflight is recognisable (= the previous turn died halfway)",
       residue == ["说了一半进程就崩了"] and os.path.exists(res2.inflight_path),
       f"residue={residue!r}")

    res2.release()
    ok("11b", ".inflight disappears after a turn ends normally (otherwise the crash signal "
              "is stuck true forever)",
       res2.inflight_residue() is None and not os.path.exists(res2.inflight_path),
       "cleared")

    # ★POSIX os.rename overwrites its target without complaint -> it would silently delete
    #   the crash scene. This assertion guards that.
    res2.add("甲"); res2.drain()       # leave an .inflight that was never released
    res2.add("乙"); res2.drain()       # drain again: the old one must be moved aside, not overwritten
    orphans = [f for f in os.listdir(res2.session_dir) if ".inflight.orphan-" in f]
    ok("11c", "hitting a leftover .inflight does not silently overwrite it "
              "(the crash scene survives)",
       len(orphans) == 1 and _read_orphan(res2.session_dir, orphans[0]) == "甲",
       f"orphan={orphans!r}")

    # ── 12: the silence counter has a single writer ──
    sc = reply_list.SilenceCounter(os.path.join(sb.tmp, "s12"))
    seq = [sc.tick(), sc.tick(), sc.tick()]
    before = open(sc.path, encoding="utf-8").read().strip()
    sc.mark_voice()                                  # <- the only thing the STT callback touches
    after = open(sc.path, encoding="utf-8").read().strip()
    zeroed = sc.tick()
    again = sc.tick()
    ok("12", "single writer: mark_voice does not write the file, so the zeroing cannot be "
             "eaten by a read-then-write-back",
       seq == [1, 2, 3] and before == "3" and after == "3"
       and zeroed == 0 and again == 1,
       f"tick sequence={seq} · file before/after mark_voice={before}/{after} · "
       f"zeroed={zeroed} · next tick={again}")

    # ── 12b ★A reading of N implies at least N whole ticks since the last event (an older
    #    comment said ">= (N-1)", which was off by a full tick) ──
    #    Reconciled tick by tick against 8 real recordings: when the reading is 1, the
    #    smallest gap since the last event measured 1024ms, with no counter-example.
    #    Why: the flag mark_voice sets is consumed by the **next** tick, which turns into a
    #    reading of 0. So for the reading to reach 1 there must have been a whole tick with
    #    no event at all. A standing suggestion to raise a threshold 1 -> 2 had been filed
    #    purely on the strength of that wrong comment.
    clk = [1000.0]
    sc2 = reply_list.SilenceCounter(os.path.join(sb.tmp, "s12b"), clock=lambda: clk[0])
    sc2.mark_voice()                                 # an event at t=1000 (no audio timestamp)
    at_mark = sc2.silence_seconds()
    clk[0] += 0.999                                  # 1ms short of a full tick
    r0 = sc2.tick(); sil0 = sc2.silence_seconds()
    clk[0] += 0.001                                  # exactly one full tick
    r1 = sc2.tick(); sil1 = sc2.silence_seconds()
    ok("12b", "★a reading of 1 implies a whole tick has passed since the last event "
              "(not \"it guarantees nothing\"); the seconds reading is continuous and is "
              "not quantised to the tick",
       at_mark == 0.0 and r0 == 0 and abs(sil0 - 0.999) < 1e-6
       and r1 == 1 and abs(sil1 - 1.0) < 1e-6,
       f"at the event={at_mark}s · at 0.999s reading={r0}/seconds={sil0:.3f} · "
       f"at 1.000s reading={r1}/seconds={sil1:.3f}")

    # ── 12c ★The audio timeline: add the server's pipeline lag back in, clamped to
    #    MAX_PIPELINE_LAG_S ──
    clk = [2000.0]
    sc3 = reply_list.SilenceCounter(os.path.join(sb.tmp, "s12c"), clock=lambda: clk[0])
    sc3.mark_voice(stop_wall=2000.0 - 0.677)         # measured p50: the event lands 677ms
                                                     # after the mouth actually stopped
    with_lag = sc3.silence_seconds()
    # ★A late event carrying an **earlier** audio end must not lengthen the silence
    #   (take the max, not the last one that arrived)
    sc3.mark_voice(stop_wall=2000.0 - 3.0)
    not_lengthened = sc3.silence_seconds()
    # An absurd lag (this is what clock drift looks like) -> clamp it; only under-report
    sc4 = reply_list.SilenceCounter(os.path.join(sb.tmp, "s12d"), clock=lambda: clk[0])
    sc4.mark_voice(stop_wall=2000.0 - 99.0)
    clamped = sc4.silence_seconds()
    sc5 = reply_list.SilenceCounter(os.path.join(sb.tmp, "s12d2"), clock=lambda: clk[0])
    sc5.mark_voice(stop_wall=2000.0 + 5.0)           # describes audio "in the future" -> clamp to now
    future = sc5.silence_seconds()
    ok("12c", "★silence seconds = time since the event arrived + that event's pipeline lag; "
              f"the lag is clamped to {reply_list.MAX_PIPELINE_LAG_S}s, and a late old event "
              "may not lengthen the silence (drift may only under-report, never over-report)",
       abs(with_lag - 0.677) < 1e-9
       and abs(not_lengthened - 0.677) < 1e-9
       and abs(clamped - reply_list.MAX_PIPELINE_LAG_S) < 1e-9
       and future == 0.0,
       f"677ms lag -> {with_lag:.3f}s · another event with an earlier audio end -> still "
       f"{not_lengthened:.3f}s · absurd lag -> clamped to {clamped}s · event claims future "
       f"audio -> {future}s")

    # ── 12d ★With no audio timestamp it degrades to "how long since the event arrived" --
    #    still continuous, and it under-reports ──
    clk = [3000.0]
    sc6 = reply_list.SilenceCounter(os.path.join(sb.tmp, "s12e"), clock=lambda: clk[0])
    sc6.mark_voice()                                 # old recording / server gave none -> None
    clk[0] += 2.5
    ok("12d", "★no audio timestamp is not an error: it degrades to \"how long since the "
              "event arrived\" (continuous, and conservatively low)",
       abs(sc6.silence_seconds() - 2.5) < 1e-9 and sc6.tick() == 0,
       f"degraded reading={sc6.silence_seconds():.3f}s (the true silence is about 0.68s more)")


def _read_orphan(d, name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return f.read().strip()


# ── C8: the shape of U (partial overwrites / is_final appends / clearing is all-or-nothing) ──
async def c8(sb):
    if not FILE_BACKEND:
        skip("13", "U: a partial overwrites the current segment, is_final appends",
             "the file backend is not implemented")
        return
    u = reply_list.UserBuffer(os.path.join(sb.tmp, "s13"))

    u.set_partial("我觉得那")
    u.set_partial("我觉得那个方案")          # ★the interim got rewritten -- overwrite, not append
    ok("13", "an STT interim partial overwrites the current segment (it does not append)",
       u.text() == "我觉得那个方案",
       f"U={u.text()!r}")

    u.commit_final("我觉得那个方案挺好")
    u.set_partial("也不贵")
    ok("13b", "is_final freezes this segment and starts a new one; reading U = the segments "
              "joined with spaces",
       u.text() == "我觉得那个方案挺好 也不贵" and u.char_len() == len("我觉得那个方案挺好 也不贵"),
       f"U={u.text()!r} · {u.char_len()} chars")

    # ★Empty partials really do arrive (measured: 4 of them in one stt_probe run). Not
    #   blocking them **wipes the live segment to empty** = silently throwing away a
    #   half-finished sentence. This is the closest thing here to soundless data loss.
    u.set_partial("我正在说到一半")
    u.set_partial("")
    u.set_partial("   ")
    ok("13e", "an empty partial is ignored and does not wipe the current segment "
              "(measured: the server really does send empty partials)",
       u.text() == "我觉得那个方案挺好 我正在说到一半",
       f"U={u.text()!r}")
    u.set_partial("也不贵")            # reset, then carry on into 13d

    # ★Empty finals really do arrive (measured: 3 in the first 15 seconds of an stt_probe
    #   run). Not blocking them appends a blank line every time -- and text() filters blank
    #   lines out, so **it reads perfectly normal** while the file grows without bound.
    before = len(open(u.path, encoding="utf-8").read())
    for _ in range(5):
        u.commit_final("")
        u.commit_final("   ")
    after = len(open(u.path, encoding="utf-8").read())
    ok("13d", "an empty final is ignored and appends no blank line to user_buffer.txt "
              "(measured: the server really does send empty finals)",
       before == after and u.text() == "我觉得那个方案挺好 也不贵",
       f"file across 10 empty finals {before}B -> {after}B · U={u.text()!r}")

    u.clear()
    ok("13c", "clear has exactly one setting: all of it (a JEV verdict is a boolean, so it "
              "cannot hand back a keep_from offset)",
       u.text() == "" and u.char_len() == 0,
       "cleared")

    # ── Trimming the tail: the tail of a live partial is not trustworthy -- measured, the
    #    server revises it **downwards** ──
    T = reply_list._trim_tail
    ok("13f", "trimming: split on whitespace when there is any, on characters when there is "
              "none; trimming everything away returns an empty string (not the original)",
       T("Do you know what Grok means", 1) == "Do you know what Grok"
       and T("你知道呃Grok这个单词", 1) == "你知道呃Grok这个单"
       and T("嗯", 1) == "" and T("hi there", 2) == ""
       and T("abc", 0) == "abc" and T("", 1) == "",
       "English by word · Chinese by character · trimmed to empty · units=0 is a no-op")

    u.clear()
    u.commit_final("你知道 LanceDB 吗")      # locked; not one character may move
    u.set_partial("一个 embedding 的数据")   # the live segment, which gets trimmed
    # ★This assertion was wrong the first time it was written (on the assumption that a
    #   Chinese tail is trimmed by character): **if the live segment contains any
    #   whitespace at all, the whitespace rule wins**, so the whole of `的数据` goes at
    #   once. Mixing languages therefore trims more than pure Chinese would. Noted here
    #   because it is not a bug -- it is what "split on whitespace when there is any"
    #   necessarily does to mixed text.
    ok("13g", "text_for_jev trims only the LIVE segment; a locked final is untouched "
              "(those can no longer change)",
       u.text() == "你知道 LanceDB 吗 一个 embedding 的数据"
       and u.text_for_jev(1) == "你知道 LanceDB 吗 一个 embedding",
       f"after trimming={u.text_for_jev(1)!r} (the live segment has whitespace, so the "
       f"whole of `的数据` goes at once)")

    u.clear()
    u.set_partial("嗯")
    ok("13h", "when the live segment is trimmed away entirely, text_for_jev returns an empty "
              "string -> should_skip blocks it and this tick sends no request",
       u.text() == "嗯" and u.text_for_jev(1) == "",
       "U='嗯' -> after trimming=''")

    # ── The delivered watermark (the fix for "the same sentence answered twice"; the
    #    mechanism is described in the UserBuffer class comment) ──
    #   ★This group guards a disease that **really happened in a live run**: JEV#1 passes
    #     on a live block's partial -> `_submit` clears U -> that block's final (arriving
    #     250-9764ms later) writes the same sentence back into the now-empty U -> the next
    #     tick passes it again -> answered twice. Two of six turns in one real session.
    v = reply_list.UserBuffer(os.path.join(sb.tmp, "s13w"))
    v.set_partial("And if I want to use them to generate some,")
    v.clear()                                  # <- what `_submit` does
    mark = v._delivered
    v.commit_final("And if I want to use them to generate some")   # the same block's final
                                                                   # (verbatim from a recording)
    ok("13i", "★once a submission has landed on a live block's partial, that block's FINAL "
              "no longer pours the same sentence back into U (unmended, JEV#1 passes it "
              "again on the next tick -> the same sentence answered twice)",
       mark == "And if I want to use them to generate some," and v.text() == "",
       f"watermark={mark!r} -> after the final arrived U={v.text()!r} · {len(v.dedup)} "
       f"traces · {v.dedup_chars} chars cut")

    v2 = reply_list.UserBuffer(os.path.join(sb.tmp, "s13w2"))
    v2.set_partial("suggest me to select Gemini or")
    v2.clear()
    v2.commit_final("suggest me to select Gemini or Grok?")        # the final is longer than the draft
    ok("13j", "★when the final is longer than the draft that was delivered, **only the newly "
              "grown tail** enters U (cut the prefix, do not drop the whole line)",
       v2.text() == "Grok?",
       f"U={v2.text()!r}")

    v3 = reply_list.UserBuffer(os.path.join(sb.tmp, "s13w3"))
    v3.set_partial("Is there is there anybody?")
    v3.clear()
    v3.commit_final("Is any is is anybody here?")   # measured: STT re-heard the whole block
                                                    # as a different sentence
    cov = v3.dedup[-1]["cover"] if v3.dedup else None
    ok("13k", "★coverage below the threshold (STT re-heard this block as a different "
              "sentence) -> **cut nothing at all**: better to repeat a few words than to "
              "bite the head off a sentence (the rule in this repo: never silently drop text)",
       v3.text() == "Is any is is anybody here?" and cov is None,
       f"U={v3.text()!r} · threshold={reply_list.DELIVERED_PREFIX_MIN_COVER}")

    v4 = reply_list.UserBuffer(os.path.join(sb.tmp, "s13w4"))
    v4.commit_final("你知道 LanceDB 吗")      # the final is locked -> the last line is an empty slot
    v4.clear()
    v4.commit_final("你知道 LanceDB 吗")      # on the typing path every line is a final
    ok("13m", "★when the last line is empty at clear time (the final is already locked) -> "
              "**set no watermark**: on the typing path every line is a final, so not one "
              "character may be cut (otherwise asking the same question twice gets eaten)",
       v4._delivered is None and v4.text() == "你知道 LanceDB 吗",
       f"watermark={v4._delivered!r} · U={v4.text()!r}")

    v5 = reply_list.UserBuffer(os.path.join(sb.tmp, "s13w5"))
    v5.set_partial("你好")
    v5.clear()
    v5._delivered_at -= reply_list.DELIVERED_TTL_S + 1        # pretend that block's final never came
    v5.set_partial("你好，能不能做一个自我介绍？")
    ok("13n", f"★the watermark expires after {reply_list.DELIVERED_TTL_S}s: when that block's "
              "final never arrives, the watermark must not survive into the next utterance "
              "and bite the head off it",
       v5.text() == "你好，能不能做一个自我介绍？" and v5._delivered is None,
       f"U={v5.text()!r}")


# ── C9: STT event routing (offline, pure functions, no network and no microphone) ──────
async def c9(sb):
    """★Why this group has to exist (a lesson learned the hard way):

    While editing the latency statistics, the line `await _call(on_speech_final, ...)` got
    **sucked up into the if block above it**, so **every event was dispatched as a
    speech_final**. `ast.parse` passed it happily -- syntactically legal, semantically
    wrong. It only came to light after two live sessions had written a corrupt `kind`
    field throughout.
    **A syntax check is not a behaviour check.** Routing is pure logic; it belongs in an
    offline test, not in a paid live connection.
    """
    from lib import stt_client

    ok("14a", "is_final=F / speech_final=F → partial",
       stt_client.route_event({"is_final": False, "speech_final": False}) == "partial",
       "partial")
    ok("14b", "is_final=T / speech_final=F -> final (the text is locked but the person is "
             "still talking)",
       stt_client.route_event({"is_final": True, "speech_final": False}) == "final",
       "final")

    # ★★This is the only one of the three states with an ordering dependency: measured,
    #   every speech_final=T event **also carries is_final=T** (4 out of 4 in a 13-event
    #   trace). So testing is_final first means the speech_final branch is **never
    #   reached** -> the thing that drives the LLM never happens, and nothing errors.
    ok("14c", "is_final=T and speech_final=T -> speech_final (★is_final must not be tested "
             "first, or the LLM is never driven at all)",
       stt_client.route_event({"is_final": True, "speech_final": True}) == "speech_final",
       "speech_final")

    ok("14d", "the text is read from `text` (the real field name, measured); `transcript` is "
             "only a cushion in case it is ever renamed",
       stt_client.event_text({"text": "甲"}) == "甲"
       and stt_client.event_text({"transcript": "乙"}) == "乙"
       and stt_client.event_text({}) == "",
       "text / transcript / empty -- all three correct")


# ── C10: the JEV#1 client + the question file (offline, pure functions, no network, free) ──
async def c10(sb):
    """★Why this group earns its place:

    The JEV side has two kinds of silent failure that raise nothing, and only an offline
    test can catch either:
      ① If an unreadable noul in the response degrades to `meaningful=False`, the symptom
         is "the person finishes speaking and the machine never opens its mouth" -- with
         **no exception anywhere**. So verdict_from_answer has to raise.
      ② If the examples in the question file overlap the held-out test set, a replay
         measures **recitation**, and the better the number looks the more dangerous it
         is. ★That assertion really did catch one once (the counter-example `嗯` in the
         question file collided with `嗯。` in the corpus).
    """
    from lib import jev_client
    from lib.jev_client import JevError

    # -- 15a Extracting the question: exactly one json block; zero or several both raise
    #    (never guess which one is authoritative) --
    got = jev_client.extract_question_block('preamble\n```json\n{"q": {"type": "noul"}}\n```\npostscript')
    two = "```json\n{\"a\": {\"type\": \"noul\"}}\n```\n```json\n{\"b\": {\"type\": \"noul\"}}\n```"
    raised = []
    for bad, why in (('no block at all', "zero blocks"), (two, "two blocks")):
        try:
            jev_client.extract_question_block(bad)
        except JevError:
            raised.append(why)
    ok("15a", "extracting the question: prose outside the block never reaches the model; "
              "zero blocks and several blocks both raise (with several, do not guess)",
       got == {"q": {"type": "noul"}} and raised == ["zero blocks", "two blocks"],
       f"extracted {list(got)} · raised on {raised}")

    # -- 15b The free pre-gate: an empty U and punctuation-only both send no request
    #    (measured: both kinds really do arrive) --
    ok("15b", "an empty U / punctuation only sends no request (measured: empty partials, "
              "empty finals and lone punctuation really do arrive)",
       jev_client.should_skip("") == "empty_state"
       and jev_client.should_skip("   \n") == "empty_state"
       and jev_client.should_skip(".") == "punctuation_only"
       and jev_client.should_skip("？，") == "punctuation_only"
       and jev_client.should_skip("？一个") is None
       and jev_client.should_skip("嗯。") is None,
       "empty or blank -> empty_state · . and ？， -> punctuation_only · real text -> None")

    # -- 15c The threshold comparison is strictly greater-than; landing on it is not enough --
    mk = lambda v: {"answers": {"complete": {"type": "noul", "noul": v}}}
    ok("15c", "meaningful = noul > threshold (strictly greater: equal to the threshold is "
              "not enough)",
       jev_client.verdict_from_answer(mk(0.7), threshold=0.5) == (True, 0.7)
       and jev_client.verdict_from_answer(mk(0.3), threshold=0.5) == (False, 0.3)
       and jev_client.verdict_from_answer(mk(0.5), threshold=0.5) == (False, 0.5),
       "0.7 -> True · 0.3 -> False · 0.5 -> False")

    # -- 15d ★An unreadable noul always raises and never degrades to False (otherwise the
    #    symptom is "never opens its mouth", with no error) --
    bads = [({}, "no answers"),
            ({"answers": {}}, "no question key"),
            ({"answers": {"complete": {"type": "noul"}}}, "no noul"),
            ({"answers": {"complete": {"type": "noul", "noul": "0.9"}}}, "noul is a string"),
            ({"answers": {"complete": {"type": "noul", "noul": True}}}, "noul is a bool"),
            ({"answers": {"complete": {"type": "choice", "choice": "x"}}}, "a choice answer instead")]
    leaked = []
    for payload, why in bads:
        try:
            jev_client.verdict_from_answer(payload)
            leaked.append(why)          # no raise = silently degraded to False, the banned case
        except JevError:
            pass
    ok("15d", "★an unreadable noul in the response always raises and never silently degrades "
              "to False (the symptom of which is: never opens its mouth, and never errors)",
       not leaked, "all six malformed responses raised" if not leaked
       else f"★swallowed: {leaked}")

    # -- 15e The real question file loads, and its key matches the client's default --
    dd = LISTENER_DIR
    if not os.path.isdir(dd):
        skip("15e", "loading the real question file + the shape of the listener directory",
             f"missing {dd}")
        return
    cfg = jev_client.load_dispatch_config(dd)
    qs = jev_client.load_questions(dd, cfg["agents"]["jev1"])
    ok("15e", f"the real question file loads ({len(qs)} question(s)) and its key matches "
              "jev_client.QUESTION_KEY",
       jev_client.QUESTION_KEY in qs and all("type" in q for q in qs.values()),
       f"question keys {list(qs)}")

    # -- 15f ★The listener must not contain a dispatch.py (it is not a conversational
    #    dispatch) --
    ok("15f", "the listener directory has no dispatch.py (it is not a conversational "
              "dispatch and must never be loaded by chain.load_dispatch)",
       not os.path.exists(os.path.join(dd, "dispatch.py")),
       "no dispatch.py")

    # -- 15g The mirror rule: the authoritative copy and the engine's copy agree field for
    #    field, apart from the translated context_files paths --
    base = os.path.join(SHELL_ROOT, "config_base", "listener", "dispatch_config.json")
    if not os.path.isfile(base):
        skip("15g", "the mirror rule: config_base/listener matches the engine's copy",
             f"missing {base}")
    else:
        with open(base, encoding="utf-8") as f:
            b = json.load(f)
        strip = lambda d: {k: ({kk: {k3: v3 for k3, v3 in vv.items() if k3 != "context_files"}
                                for kk, vv in v.items()} if k == "agents" else v)
                           for k, v in d.items()}
        ok("15g", "the mirror rule: config_base/listener and IAF_AI/dispatch/listener agree "
                  "field for field apart from the translated paths",
           strip(b) == strip(cfg)
           and cfg["agents"]["jev1"]["context_files"] == ["../../../prompt_base/listener/question.md"],
           "the two agree; the paths have been translated")


# ── C11: ListenerLoop -- ticker / submission / three strikes / silence counting
#        (offline, driven deterministically one tick at a time) ──────────────────────
class FakeJudge:
    """A fake verdict. ★Never build timing out of sleeps -- to block, block on an Event
    (discipline 1 in the module header)."""

    def __init__(self, noul=0.9, gate=None, raises=None, gate_only_first=False):
        self.noul = noul
        self.gate = gate            # asyncio.Event: returns only once it is set
        self.gate_only_first = gate_only_first   # block only the 1st call (for "a late verdict")
        self.raises = raises
        self.seen = []
        self.entered = None         # set as soon as control enters the judge

    async def __call__(self, state):
        n = len(self.seen)
        self.seen.append(state)
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None and (n == 0 or not self.gate_only_first):
            await self.gate.wait()
        if self.raises is not None:
            raise self.raises
        th = 0.65
        return self.noul, self.noul > th


_LOOP_N = [0]


def _loop(sb, judge, **knobs):
    """★Open a SEPARATE session_dir every time. All three state files live in session_dir,
    so sharing one lets `reply_list.txt` bleed between sub-tests -- the first version of
    16f/16g was broken by two cuts left behind by 16e, and the symptom was "the evidence
    string looks entirely correct but the result is FAIL", which is the worst kind to
    track down."""
    disp = FakeDispatch("好。")
    _LOOP_N[0] += 1
    sd = os.path.join(sb.tmp, "workspace", f"session_loop{_LOOP_N[0]}")
    os.makedirs(sd, exist_ok=True)
    drv = reply_list.VoiceDriver(disp, sd, peer="user", sink=SilentSink(), verbose=False)
    k = {"meaningless_strikes": 3, "jev_trim_units": 1}
    k.update(knobs)
    return listener_loop.ListenerLoop(drv, judge, knobs=k, verbose=False), drv


async def c11(sb):
    # ── 16a ★The tail is trimmed only for judging; what gets submitted is always the full
    #    text (get this wrong and every cut silently swallows words) ──
    j = FakeJudge(noul=0.9)
    lp, drv = _loop(sb, j)
    lp.on_partial("你知道 LanceDB 吗")
    await lp.tick_once(); await lp.settle()
    cut, n = drv.reservoir.drain(); drv.reservoir.release()
    ok("16a", "★what JEV is fed is trimmed; what enters the reply list is the full U "
              "(submitting the trimmed version would swallow words on every cut)",
       j.seen == ["你知道 LanceDB"] and cut == "你知道 LanceDB 吗" and n == 1,
       f"the judge saw {j.seen[0]!r} · submitted {cut!r}")

    # ── 16b A tick with U empty: no call is made, and no strike is recorded ──
    j = FakeJudge()
    lp, drv = _loop(sb, j)
    for _ in range(5):
        await lp.tick_once(); await lp.settle()
    lp.on_partial("。")                      # punctuation only must be blocked too
                                             # (lone punctuation really does arrive)
    await lp.tick_once(); await lp.settle()
    ok("16b", "a tick where U is empty or punctuation-only makes no call and records no "
              "strike (nothing to judge is not the same as judged insufficient)",
       j.seen == [] and lp.strikes == 0 and lp.stats["skipped"] == 6
       and lp.stats["asked"] == 0,
       f"asked=0 skipped={lp.stats['skipped']} strikes=0")

    # ── 16c Three strikes: 3 consecutive ticks judged insufficient with U not growing
    #    -> clear U ──
    j = FakeJudge(noul=0.1)
    lp, drv = _loop(sb, j, meaningless_strikes=3)
    lp.on_final("还有那个")
    seq = []
    for _ in range(3):
        await lp.tick_once(); await lp.settle()
        seq.append((lp.strikes, drv.user_buffer.char_len()))
    ok("16c", "three strikes: 3 consecutive ticks judged insufficient with U not growing "
              "-> U is cleared and strikes reset to zero",
       seq[0][0] == 1 and seq[1][0] == 2 and seq[2] == (0, 0)
       and lp.stats["strike_cleared"] == 1 and len(drv.reservoir) == 0,
       f"strike trace {[x[0] for x in seq]} · U ended at {seq[2][1]} chars · nothing "
       f"entered the reply list")

    # ── 16d ★Three strikes must not kill a live speaker: U is growing -> never clear
    #    ("not growing" is the entire safeguard) ──
    j = FakeJudge(noul=0.1)
    lp, drv = _loop(sb, j, meaningless_strikes=3)
    for i in range(6):
        lp.on_partial("我想问一下" + "啊" * i)      # longer on every tick
        await lp.tick_once(); await lp.settle()
    ok("16d", "★no false kill: U grows on every tick -> not cleared once in 6 ticks (the "
              "person is still talking; clearing would be a false kill)",
       lp.stats["strike_cleared"] == 0 and drv.user_buffer.char_len() > 0
       and lp.strikes == 1,
       f"cleared 0 times · U still holds {drv.user_buffer.char_len()} chars · "
       f"strikes={lp.strikes}")

    # ── 16e ★A late verdict is DISCARDED (the request is not cancelled -- cancelling
    #    poisons the connection pool) ──
    #   The timeline: A is sent first, B second, B comes back first. By the time A returns,
    #   the U it looked at is stale -> acting on it decouples the verdict from the text,
    #   and a fragment gets submitted as if it were complete.
    gate = asyncio.Event()
    j = FakeJudge(noul=0.9, gate=gate, gate_only_first=True)
    j.entered = asyncio.Event()
    lp, drv = _loop(sb, j)
    lp.on_partial("我觉得那个方案挺好")
    await lp.tick_once()                       # A: blocked on the gate
    await j.entered.wait()
    lp.on_partial("我觉得那个方案挺好也不贵")
    await lp.tick_once()                       # B: returns at once, judged sufficient -> submits
    await asyncio.sleep(0)
    submitted_before_a = lp.stats["submit_jev"]
    gate.set()                                 # only now does A return -- already stale
    await lp.settle()
    ok("16e", "★a late verdict is discarded rather than acted on (out-of-order responses): "
              "B submitted, A is thrown away when it returns, nothing is submitted twice",
       submitted_before_a == 1 and lp.stats["submit_jev"] == 1
       and lp.stats["stale_discarded"] == 1 and len(drv.reservoir) == 1
       and j.seen == ["我觉得那个方案挺", "我觉得那个方案挺好也不"],   # ★both trimmed
       f"submit_jev=1 · 1 discarded · 1 cut in the reply list (both A's and B's U really "
       f"were sent, and both were trimmed)")

    # ── 16j ★An in-flight request must not be cancelled (cancelling poisons the connection
    #    pool -- measured: the next call is a second slower) ──
    gate = asyncio.Event()
    j = FakeJudge(noul=0.9, gate=gate, gate_only_first=True)
    j.entered = asyncio.Event()
    lp, drv = _loop(sb, j)
    lp.on_partial("我觉得那个方案挺好")
    await lp.tick_once()
    await j.entered.wait()
    first = next(iter(lp._inflight))
    lp.on_partial("我觉得那个方案挺好也不贵")
    await lp.tick_once()
    still_running = not first.cancelled() and not first.done()
    gate.set(); await lp.settle()
    ok("16j", "★a new tick does NOT cancel the previous tick's request (cancelling poisons "
              "the connection pool: 352-421ms when clean, 1119-1425ms after a cancel)",
       still_running and not first.cancelled() and lp.stats.get("aborted", 0) == 0,
       "the previous tick runs to completion; only its verdict is ignored")

    # ── 16k The in-flight cap: at max_inflight, skip this tick (when the endpoint degrades,
    #    piling on more only makes it slower) ──
    gate = asyncio.Event()
    j = FakeJudge(noul=0.9, gate=gate)          # block every one of them
    lp, drv = _loop(sb, j, max_inflight=2)
    for i in range(4):
        lp.on_partial("我想问一下报价" + "啊" * i)
        await lp.tick_once()
        await asyncio.sleep(0)
    capped = lp.stats["inflight_capped"]
    gate.set(); await lp.settle()
    ok("16k", "once in-flight requests reach max_inflight=2 the tick is skipped; they do not "
              "pile up without bound",
       capped == 2 and lp.stats["asked"] == 2 and len(j.seen) == 2,
       f"of 4 ticks, 2 were sent and 2 were held back by the cap")

    # ── 16f ★"could not get a verdict" is not the same as "judged insufficient" ──
    j = FakeJudge(raises=RuntimeError("JEV HTTP 429"))
    lp, drv = _loop(sb, j, meaningless_strikes=2)
    lp.on_final("你知道 LanceDB 吗")
    for _ in range(4):
        await lp.tick_once(); await lp.settle()
    ok("16f", "★a tick where the judge raises does nothing at all: it does not submit and "
              "**does not record a strike** (treating it as insufficient would wipe out what "
              "the person said)",
       lp.stats["judge_errors"] == 4 and lp.strikes == 0
       and lp.stats["strike_cleared"] == 0 and drv.user_buffer.text() == "你知道 LanceDB 吗"
       and len(drv.reservoir) == 0,
       "4 errors · strikes=0 · U left exactly as it was")

    # ── 16g ★★speech_final is a pure observation point: it does not submit, does not touch
    #    U, and does not reset strikes ──
    #   (The fallback that used to live here was deliberately deleted. This assertion exists
    #    to stop anyone **wiring it back in**.)
    #   ★Why even strikes is asserted on: before that fallback was removed, three strikes
    #     never got past 2/6 on real recordings, and the root cause was exactly this
    #     zeroing, which happened every 400ms. If someone casually adds `strikes = 0` back
    #     into _apply_speech_final, three strikes becomes a **dead path all over again**,
    #     and not one assertion turns red.
    j = FakeJudge(noul=0.1)                 # always insufficient -> strikes should climb
    lp, drv = _loop(sb, j, meaningless_strikes=99)
    lp.on_final("对于这次对话之")
    await lp.tick_once(); await lp.settle()          # strikes 1
    lp.on_speech_final("对于这次对话之")             # <- stir things up
    await lp.tick_once(); await lp.settle()          # strikes 2 (if it were zeroed, only 1)
    u_after = drv.user_buffer.text()
    n_res = len(drv.reservoir)
    ok("16g", "★★speech_final is only an observation point: it submits no cut, puts not one "
              "character into U, and **never resets strikes** (resetting it reinstates, in "
              "place, the \"three strikes never accumulate\" bug that was fixed -- and "
              "nothing turns red)",
       n_res == 0 and u_after == "对于这次对话之" and lp.strikes == 2
       and lp.stats["speech_finals"] == 1
       and not any(k.startswith("submit_") for k in lp.stats),
       f"{n_res} cuts in the reply list · U={u_after!r} · strikes={lp.strikes} · "
       f"speech_finals={lp.stats['speech_finals']} · submit counters="
       f"{[k for k in lp.stats if k.startswith('submit_')]}")

    # ── 16h ★The reset trigger for three strikes is a config field, and it is measured
    #    CUMULATIVELY ──
    #   (It is controlled by knobs.strike_reset_growth_chars rather than being hard-coded.)
    #   ★Why "cumulative" has to be asserted: measured per-tick, someone squeezing out one
    #     character a second never reaches the threshold -> strikes climb and wipe out the
    #     speech of a person who is still talking. The two readings are **exactly
    #     equivalent** at the default of 1, so only a value > 1 can tell them apart -- at
    #     the default this assertion is blind.
    #   ★Grow the LIVE partial one character at a time; do not use commit_final. `u.text()`
    #     joins the segments with spaces, so each commit_final conjures an extra space and
    #     the length is wrong (that is how the first version got its expected values wrong:
    #     it assumed +1 character where the real figure was +2).
    j = FakeJudge(noul=0.1)
    lp, drv = _loop(sb, j, meaningless_strikes=99, strike_reset_growth_chars=3)
    traj = []
    for live in ("一二", "一二三", "一二三四", "一二三四五"):   # 2->3->4->5: one more char per tick
        lp.u.set_partial(live)
        await lp.tick_once(); await lp.settle()
        traj.append(lp.strikes)
    #   Cumulative: baseline 2 -> 3 (diff 1 < 3) -> 2 -> 4 (diff 2 < 3) -> 3 -> 5
    #               (diff 3 >= 3) -> reset, then +1 = 1
    #   Per-tick:   every tick differs by only 1 -> the threshold is never reached ->
    #               [1,2,3,4]: climbing while the person is still talking
    ok("16h", "★the reset trigger for three strikes comes from knobs.strike_reset_growth_chars, "
              "and it counts growth ACCUMULATED SINCE THE LAST RESET (measured per tick, "
              "someone squeezing out one character a second never reaches the threshold -> "
              "strikes climb and wipe out a person who is still talking; per-tick would give "
              "[1,2,3,4] here)",
       traj == [1, 2, 3, 1],
       f"strike trace at 3: {traj} (cumulative expects [1,2,3,1]; per-tick would be [1,2,3,4])")

    # ── 16h2 The default of 1 reproduces the behaviour that used to be hard-coded (moving
    #    it into a field changed nothing on the day) ──
    j = FakeJudge(noul=0.1)
    lp, drv = _loop(sb, j, meaningless_strikes=99)      # not passed -> use the default
    traj1 = []
    for live in ("一二", "一二三", "一二三四"):
        lp.u.set_partial(live)
        await lp.tick_once(); await lp.settle()
        traj1.append(lp.strikes)
    ok("16h2", "at the default strike_reset_growth_chars=1, \"one more character resets it\" "
               "= the behaviour from before it became a field",
       traj1 == [1, 1, 1] and lp.knobs["strike_reset_growth_chars"] == 1,
       f"trace {traj1} · default={lp.knobs['strike_reset_growth_chars']}")

    # ── 16i The ticker is the only writer of silence_ticks + the gate tests > 1 ──
    j = FakeJudge()
    lp, drv = _loop(sb, j, silence_ticks_to_speak=2)
    lp.on_partial("嗯")                     # the inbound side only mark_voice; it writes no file
    before = drv.silence.value()
    await lp.tick_once(); await lp.settle()  # speech present -> zeroed
    zero = drv.silence.value()
    await lp.tick_once(); await lp.settle()  # no speech -> 1
    one = drv.silence.value()
    can_at_1 = drv.silence.can_speak(2)
    await lp.tick_once(); await lp.settle()  # -> 2
    ok("16i", "silence_ticks is written only by the ticker (the inbound side only calls "
              "mark_voice); the gate tests > 1, not > 0",
       before == 0 and zero == 0 and one == 1 and can_at_1 is False
       and drv.silence.can_speak(2) is True,
       "mark_voice writes no file · speech zeroes it · 1 tick is not yet enough to speak · "
       "2 ticks is")


# ── C12: the whole loop running (a long-lived consumer coroutine + water pooling while
#        busy) ────────────────────────────────────────────────────────────────────────
async def c12(sb):
    """★This verifies that "one utterance cut three ways = one LLM call" also holds in the
    REAL loop -- C2 verifies it at the driver layer, this verifies it once the ticker, the
    submission and the consumer are wired together."""
    release = asyncio.Event()
    disp = FakeDispatch("好的，我这边看一下。", chunk_chars=4)

    orig = disp.run_stream

    async def gated(agent_name=None):
        async for ev in orig(agent_name=agent_name):
            if ev.get("type") == "delta":
                await release.wait()      # block after the first chunk, letting new water in
            yield ev
    disp.run_stream = gated

    drv = _driver(sb, disp, SilentSink())
    # ★★A fake clock: the LLM trigger gained a second condition, "the person has been quiet
    #   for at least llm_min_silence_s seconds" (discipline 6), which makes this loop
    #   time-driven -- so a test either controls time or is reduced to racing sleeps.
    clk = [1_000_000.0]
    drv.silence.set_clock(lambda: clk[0])

    def quiet(n):
        """The person stays quiet for n seconds."""
        clk[0] += float(n)

    j = FakeJudge(noul=0.9)
    lp = listener_loop.ListenerLoop(drv, j, knobs={"tick_interval_s": 0.01},
                                    verbose=False)
    runner = asyncio.create_task(lp.run())
    try:
        lp.on_final("我想问一下")
        quiet(1)                                     # ★without this the LLM never fires
                                                     #   (discipline 6)
        await lp.tick_once(); await lp.settle()      # cut 1 -> wake the consumer
        await _wait_busy(drv)
        lp.on_final("报价的事")
        quiet(1)
        await lp.tick_once(); await lp.settle()      # cut 2: the LLM is busy, so only pool it
        lp.on_final("还有交期")
        quiet(1)
        await lp.tick_once(); await lp.settle()      # cut 3
        cuts_waiting = len(drv.reservoir)
        release.set()
        for _ in range(400):                          # wait for the consumer to finish turn 2
            if len(drv.turns) >= 2 and not drv.busy and len(drv.reservoir) == 0:
                break
            await asyncio.sleep(0)
        ok("17a", "the whole loop: the two cuts that arrived while busy pool up and are "
                  "eaten in ONE GO on the next turn (the long-lived consumer coroutine)",
           cuts_waiting == 2 and len(drv.turns) == 2
           and drv.turns[0]["cuts_consumed"] == 1 and drv.turns[1]["cuts_consumed"] == 2,
           f"{cuts_waiting} cuts pooled while busy · cuts eaten per turn "
           f"{[t['cuts_consumed'] for t in drv.turns]}")
        ok("17b", "afterwards U is empty, the reply list is empty, and the consumer did not "
                  "start a second pump (busy has fallen back)",
           drv.user_buffer.char_len() == 0 and len(drv.reservoir) == 0
           and not drv.busy and lp.stats["llm_turns"] == 2,
           f"llm_turns={lp.stats['llm_turns']}")

        # ── 17c ★The LLM's second trigger condition: the person is still talking -> cuts
        #    go into the pool, but the LLM does not run once ──
        #    JEV#1 answers only "is there enough here to answer", **not "has he finished"**.
        #    Without this condition, a complete thought formed mid-sentence gets answered
        #    over the top of the speaker -- and with barge-in gone, the machine then says
        #    the whole segment.
        turns_before = len(drv.turns)
        lp.on_final("那个报价")                      # <- mark_voice: silence back to 0
        await lp.tick_once(); await lp.settle()      # JEV#1 passes (submission is not bound
                                                     # by this condition)
        for _ in range(200):                          # give the consumer every chance to jump in
            await asyncio.sleep(0)
        ok("17c", "★the LLM's second trigger (discipline 6): the person is still talking "
                  "(true silence 0s < 1.0s) -> JEV#1 still passes and the cut enters the "
                  "reply list, but **the LLM does not run once** (the water is not dropped, "
                  "it waits)",
           len(drv.reservoir) == 1 and len(drv.turns) == turns_before
           and lp.stats["llm_held"] > 0,
           f"{len(drv.reservoir)} cuts pooled · LLM turns still {len(drv.turns)} · "
           f"held back {lp.stats['llm_held']} times")

        # ── 17d ★Once the person has been quiet long enough, the ticker calls the consumer
        #    back ──
        #    This also guards "those two lines must sit at the very TOP of tick_once": by
        #    now U has been submitted and cleared, so control takes should_skip's earliest
        #    return. Put the wake-up after that and the water lies in the pool forever --
        #    **no error, the machine simply never speaks again**.
        quiet(1)
        await lp.tick_once(); await lp.settle()
        for _ in range(400):
            if len(drv.reservoir) == 0 and not drv.busy and len(drv.turns) > turns_before:
                break
            await asyncio.sleep(0)
        ok("17d", "★a full 1.0s of quiet -> the ticker calls the consumer back (U is empty "
                  "by now and control takes skip's earliest return, which is why the "
                  "wake-up has to be written at the very top of tick_once) -> the LLM "
                  "really runs and the pool is emptied",
           len(drv.turns) == turns_before + 1 and len(drv.reservoir) == 0
           and drv.turns[-1]["user_text"] == "那个报价",
           f"LLM turns {turns_before} -> {len(drv.turns)} · {len(drv.reservoir)} cuts "
           f"pooled · this turn ate {drv.turns[-1]['user_text']!r}")
    finally:
        await lp.aclose()
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner



# ── C13: voice_cli's wiring (no audio, no network, no microphone) ────────────────────
async def c13(sb):
    """★Why the wiring step needs offline assertions too:

    "Swap the water source from the keyboard to the microphone" sounds like plumbing, but
    plumbing is what fails most quietly. It happened once already: the `on_speech_final`
    line got sucked into the if above it by indentation, so **every event was dispatched
    as a speech_final**; `ast.parse` passed it, and two sessions' worth of recordings had
    a corrupt kind field. One live run that "sounds right" cannot catch this -- when a
    person finishes a sentence all three events have occurred, and nothing on screen says
    which port the water came in through.
    So voice_cli's three callbacks are packaged as `VoiceWiring`, and here **synthetic
    events** drive it one at a time.
    """
    import voice_cli
    from lib import stt_client
    src = io.open(os.path.join(SHELL_ROOT, "voice_cli.py"), encoding="utf-8").read()

    # -- 18a The CLI lives at the shell root, and importing it must not drag sounddevice in
    #    (it has to be importable with no microphone and no PortAudio) --
    ok("18a", "voice_cli.py lives at the shell root, and importing it does not drag in "
              "sounddevice (opening the mic is a runtime concern)",
       os.path.isfile(os.path.join(SHELL_ROOT, "voice_cli.py"))
       and "sounddevice" not in sys.modules,
       f"no sounddevice in sys.modules · shell root {os.path.basename(SHELL_ROOT)}")

    # -- 18b ★Never open an InputStream directly: the channels=1 precondition is pinned
    #    inside mic_stream --
    ok("18b", "★voice_cli does not open sd.InputStream itself (bypassing mic_stream bypasses "
              "channels=1, and reading a 1-channel device as 2 channels yields noise at the "
              "wrong speed)",
       "InputStream(" not in src and "stt_client.mic_stream(" in src,
       "goes through stt_client.mic_stream only")

    # -- 18c ★A dump filename must never fall inside jev_bench's corpus glob --
    names = re.findall(r'f"([^"]*\.jsonl?)"', src)
    ok("18c", "★dump filenames do not match stt_*.jsonl (that glob is jev_bench's corpus; "
              "dropping an unlabelled file into it silently corrupts the bench, which "
              "assumes every case has a label)",
       names and all(n.startswith("voice_") for n in names),
       f"dump names {names}")

    # -- 18d ★The echo gate: anything but y refuses (a feedback loop costs real money on
    #    every lap, so this gate is mandatory by default) --
    class _A:
        yes_headphones = False
    import builtins
    saved = builtins.input
    try:
        builtins.input = lambda *a: "n"
        no = voice_cli.headphone_gate(_A())
        builtins.input = lambda *a: "y"
        yes = voice_cli.headphone_gate(_A())
        _A.yes_headphones = True
        builtins.input = lambda *a: (_ for _ in ()).throw(AssertionError("should not ask"))
        skipped = voice_cli.headphone_gate(_A())
    finally:
        builtins.input = saved
    ok("18d", "★the echo gate: only y passes, anything else refuses; --yes-headphones passes "
              "straight through without asking",
       no is False and yes is True and skipped is True,
       f"n -> {no} · y -> {yes} · --yes-headphones -> {skipped} (never asked)")

    # -- 18e ★Each of the three events goes into its own port + conf is passed through +
    #    the audio timestamp really reaches the criterion --
    lp, drv = _loop(sb, FakeJudge(noul=0.1))
    # ★Install a fake clock: this asserts that "the moment the person actually stopped"
    #   travels all the way through to the gate's criterion (see below).
    drv.silence.set_clock(lambda: 2000.0)
    seen = []

    def _wrap(name):
        orig = getattr(lp, name)

        def f(*a, **kw):
            # ★Accept **kw: the wiring layer passes stop_wall to on_speech_final as a
            #   keyword. Accepting only *a raises TypeError right here (which really did
            #   happen once).
            seen.append((name,) + a + tuple(v for _, v in sorted(kw.items())))
            return orig(*a, **kw)
        setattr(lp, name, f)

    for _n in ("on_partial", "on_final", "on_speech_final"):
        _wrap(_n)

    w = voice_cli.VoiceWiring(lp, display=False)
    w.mark_mic_open()
    # ★`_stop_wall` is the key `stt_client` computes from start/duration and tucks into
    #   the event (= the wall-clock moment this person actually stopped). 1999.2 here is
    #   0.8 seconds before the fake clock's 2000.0.
    evs = [
        {"type": "transcript.partial", "text": "你知道", "is_final": False,
         "start": 0.0, "duration": 0.6, "language": "zh", "_stop_wall": 1998.6},
        {"type": "transcript.partial", "text": "你知道 LanceDB 吗", "is_final": True,
         "start": 0.0, "duration": 1.2, "language": "zh", "_stop_wall": 1999.2},
        # ★The speech_final event **also** carries is_final=true -- get the routing order
        #   backwards and this branch is never reached
        {"type": "transcript.partial", "text": "你知道 LanceDB 吗", "is_final": True,
         "speech_final": True, "end_of_turn_confidence": 0.97,
         "start": 0.0, "duration": 1.2, "language": "zh", "_stop_wall": 1999.2},
    ]
    for ev in evs:
        getattr(w, "on_" + stt_client.route_event(ev))(stt_client.event_text(ev), ev)
    cut, n = drv.reservoir.drain()
    drv.reservoir.release()
    ok("18e", "★each of the three events goes into its own port (partial -> U's live "
              "segment / final -> locked / speech_final -> **observation only**). The text "
              "on a speech_final is passed into the loop **purely to decide whether this "
              "event carries text at all** (discipline 8), but it puts **not one character "
              "into U and submits no cut** (the fallback was deleted; forwarding it to "
              "on_final repeats either the whole turn or the last segment into U, and both "
              "shapes show up in real recordings)",
       seen == [("on_partial", "你知道", 1998.6),
                ("on_final", "你知道 LanceDB 吗", 1999.2),
                ("on_speech_final", "你知道 LanceDB 吗", 1999.2)]
       and n == 0 and cut == "" and lp.u.text() == "你知道 LanceDB 吗",
       f"ports {[x[0] for x in seen]} · speech_final args={seen[-1][1:]} · "
       f"{n} cuts submitted · U={lp.u.text()!r} (the whole-turn text did not enter U)")

    # -- 18e3 ★An empty-text event must not count as "currently speaking" (discipline 8) --
    #    In a real 180s session, 203 of 247 events were empty: the server opens a window
    #    over **every stretch of silence** and sends one empty event per second (`start`
    #    pinned to the window's beginning, `duration` growing). An unconditional
    #    `mark_voice()` therefore announced "he is still speaking" once a second -> the
    #    silence reading was pinned at about 1.5s and could not rise. **Nothing errors**
    #    (a threshold of 1.0 still passes); the reading is simply meaningless above 1.5s,
    #    and tightening the threshold to 2.0 would mean the gate **never opens at all**.
    _CK = [5000.0]
    lp3, drv3 = _loop(sb, FakeJudge(noul=0.1))
    drv3.silence.set_clock(lambda: _CK[0])
    w3 = voice_cli.VoiceWiring(lp3, display=False)
    w3.mark_mic_open()
    # one real utterance (audio ends at 4999.3), then 6 seconds of silence heartbeat:
    # one empty partial + one empty final per second
    w3.on_final("你好", {"is_final": True, "start": 0.0, "duration": 1.0,
                        "_stop_wall": 4999.3})
    for i in range(1, 7):
        _CK[0] = 5000.0 + i
        # ★Empty events carry **no** `_stop_wall` (`stt_client` only adds that key to
        #   events that have text)
        w3.on_partial("", {"is_final": False, "start": 1.0, "duration": float(i)})
        w3.on_final("", {"is_final": True, "start": 1.0, "duration": float(i)})
    _CK[0] = 5006.0
    sil = drv3.silence.silence_seconds()
    ok("18e3", "★empty-text events (the silence-window heartbeat, measured at 82% of all "
               "events in one session) must not call mark_voice: 6 seconds of silence has "
               "to read as about 6 seconds, not be pinned at MAX_PIPELINE_LAG_S of about "
               "1.5s (being pinned raises no error; it just means the gate never opens "
               "once the threshold is tightened to 2.0)",
       6.0 <= sil <= 6.8 and lp3.u.text() == "你好",
       f"true silence {sil:.3f}s (clamp {reply_list.MAX_PIPELINE_LAG_S}s · before the fix "
       f"this read about 1.5s) · U={lp3.u.text()!r} (empty events are still blocked by "
       f"UserBuffer itself -- 13d and 13e are not made redundant)")

    # -- 18e2 ★The audio timestamp really travels from the STT event through to the
    #    gate's criterion --
    #    This guards a **silent gap** that existed for a while: the wiring layer received
    #    `(text, ev)` but passed only `text` onward, so the audio timeline never entered
    #    the control loop at all. Nothing errored, no feature was lost -- it just added
    #    about 1.4 seconds to every reply. Exactly the "it runs, it is green, it is simply
    #    slow" category.
    ok("18e2", "★the wiring layer passes ev['_stop_wall'] into the silence counter: true "
               "silence = 0.8s (not 0 -- dropped on the way it reads as 0, the gate then "
               "waits a whole extra tick, and nothing reports an error)",
       abs(drv.silence.silence_seconds() - 0.8) < 1e-9,
       f"fake clock 2000.0 · the last event says audio ended at 1999.2 -> true silence "
       f"{drv.silence.silence_seconds():.3f}s")

    # -- 18f The event log records conf/language (the calibration data for the free
    #    pre-gate) --
    kinds = [r["kind"] for r in w.rows]
    ok("18f", "the word-by-word log has all three states and records conf/language "
              "(calibrating a threshold needs paired conf-vs-completeness data, and not "
              "capturing it at the time means going back and re-recording)",
       kinds == ["partial", "final", "speech_final"]
       and w.rows[-1]["conf"] == 0.97 and w.rows[-1]["language"] == "zh",
       f"kinds={kinds} · conf={w.rows[-1]['conf']} · language={w.rows[-1]['language']!r}")

    # -- 18g ★After JEV#1 has passed, speech_final has **no side effect whatsoever** on the
    #    live path --
    #    (This used to measure `w.leads`, the lead time from "submitted early" to
    #     speech_final. With the fallback gone there is no "early" to measure against, so
    #     that figure went too. What is asserted now is that **it does nothing** --
    #     including producing no empty-submission counter such as
    #     `submit_empty_speech_final`.)
    lp, drv = _loop(sb, FakeJudge(noul=0.9))
    w = voice_cli.VoiceWiring(lp, display=False)
    w.mark_mic_open()
    w.on_partial("你知道 LanceDB 吗", evs[0])
    await lp.tick_once()
    await lp.settle()
    before = dict(lp.stats)
    w.on_speech_final("你知道 LanceDB 吗", evs[2])       # <- speech_finals is the only thing that may change
    delta = {k: v for k, v in lp.stats.items() if before.get(k) != v}
    cut, n = drv.reservoir.drain()
    drv.reservoir.release()
    ok("18g", "★after JEV#1 has passed, speech_final has zero side effects on the live "
              "path: it only increments speech_finals, adds no cut, and produces no "
              "empty-submission counter (a `submit_empty_*` appearing means _submit is "
              "being called from it again)",
       n == 1 and cut == "你知道 LanceDB 吗" and delta == {"speech_finals": 1}
       and not any(k.startswith("submit_empty_") for k in lp.stats),
       f"{n} cuts · counters this changed={delta}")



# ── C14: the real TTS sink (voice_out/speak_sink) ────────────────────────────────────
class FakeTts:
    """A fake TTS: `speak_delta` records the text, and `end_turn` **immediately** emits an
    equivalent length of PCM according to a duration model.

    ★Why no sleeps: a timing suite that races sleeps goes falsely green now and then
      (discipline 1 at the top of this file). Here the audio is generated
      **deterministically** and playback progress is advanced by the test itself
      (`player._played`), so every assertion is definite.
    """

    #: "Seconds per character", for tests. ★This is the **test fixture's own** model -- the
    #  production `SpeechClock` went when barge-in did (its only purpose was working out
    #  "how many characters of the block currently playing were spoken when it got cut").
    #  All that is needed here is "text -> a stretch of fake audio of predictable length",
    #  for which a single constant is enough; no per-character classification required.
    SECONDS_PER_CHAR = 0.2

    def __init__(self, player):
        self.player = player
        self.sent = []
        self.ops = []                 # ★the only evidence of event order (the two-level clear relies on it)
        self.cleared = 0

    async def speak_delta(self, text):
        self.sent.append(text)
        self.ops.append(("text.delta", text))
        return len(text)

    async def end_turn(self, timeout=60.0):
        text = self.sent[-1]
        n = int(len(text) * self.SECONDS_PER_CHAR * self.player.bps)
        self.player.feed(b"\x00" * n)
        self.ops.append(("audio.done", len(text)))
        return {"ttfa_ms": 500, "audio_seconds": n / self.player.bps,
                "chars": len(text), "reason": "done", "total_ms": 900}

    async def clear(self, timeout=5.0):
        self.cleared += 1
        self.ops.append(("text.clear", None))
        return {"ok": True, "clear_ms": 12, "dropped_after_clear": 2}


def _advance(pl, nbytes):
    """Simulate the PortAudio callback: pop from the queue and count it into _played.

    ★Do not just change `_played` -- the real callback does two things, **pop and count**.
      Doing only the second leaves `queued_seconds` never draining, and the prefetch
      assertion hangs waiting forever (which is exactly how the first version timed out).
    """
    with pl._lock:
        take = min(int(nbytes), len(pl._buf))
        del pl._buf[:take]
        pl._played += take
    return take


async def c14(sb):
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import speak_sink
    from lib import tts_client

    # ── 19a Importable with no sound card (sounddevice is only loaded inside start()) ──
    #   ★Do not assert `"sounddevice" not in sys.modules`: C13's echo gate has already
    #     pulled it in, so such an assertion would measure **global state left behind by an
    #     earlier case**, not this module's behaviour.
    #     -> The real behaviour is only observable in a clean interpreter, hence a subprocess.
    src = open(speak_sink.__file__, encoding="utf-8").read()
    top_imports = {n.names[0].name.split(".")[0]
                   for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))
                   for _ in [0] if isinstance(n, ast.Import)}
    sub = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import speak_sink; "
         "print('sounddevice' in sys.modules)" % os.path.dirname(speak_sink.__file__)],
        capture_output=True, text=True, timeout=60)
    clean = sub.stdout.strip() == "False"
    ok("19a", "speak_sink imports with no sound card: in a clean interpreter, importing it "
              "does not pull in sounddevice (the gate has to run in an audio-free environment)",
       clean and "sounddevice" not in top_imports,
       f"subprocess import -> sounddevice in sys.modules = {sub.stdout.strip()!r} · "
       f"module-level imports {sorted(top_imports)}")

    # ★There used to be a 19b (refuse to start without a calibrated speech_cost) and a 19c
    #   (spoken characters computed as "handed to the sound card minus the card's own
    #   backlog"). Both guarded the **duration model**, and the duration model went in one
    #   piece when barge-in did (`SpeechClock` / `spoken_chars()` /
    #   `LocalPlayer.spoken_bytes` / the whole of `tts_calibrate.py` / `knobs.speech_cost`).
    #   19c2 remains, guarding the one part that survived: closing out has to wait for the
    #   card's own backlog, and Bluetooth under-reports it, which extra_latency_ms makes up.
    pl = speak_sink.LocalPlayer(24000, extra_latency_ms=150)
    ok("19c2", "★the \"card's own backlog\" that closing out must wait for = the latency "
               "PortAudio reports + extra_latency_ms (Bluetooth routinely under-reports; "
               "this is the only latency compensation point left)",
       pl.latency_s == 0.15
       and speak_sink.LocalPlayer(24000, extra_latency_ms=0).latency_s == 0.0,
       f"extra 150ms -> latency_s={pl.latency_s}s · extra 0 -> 0.0s")

    # ── 19d/e/f: three blocks in one turn ──
    pl = speak_sink.LocalPlayer(24000)            # no start(): the test advances playback itself
    tts = FakeTts(pl)
    sink = speak_sink.SpeakSink(tts, pl, playback_lead_s=999)          # prefetch not throttled yet
    chunks = ["你知道 LanceDB", "吗？它是个", "向量数据库。"]
    for c in chunks:
        await sink(c)
    ok("19d", "route B: one utterance per block (each block sends one text.delta and then "
              "waits for audio.done)",
       [o[0] for o in tts.ops] == ["text.delta", "audio.done"] * 3
       and tts.sent == chunks,
       f"event order {[o[0] for o in tts.ops]}")

    # ★19e (exact accounting: only the block currently playing uses the duration model)
    #   went with the duration model. The per-block byte ranges themselves remain
    #   (`sink._chunks` start/end); they are what locates a dropout.
    ok("19e2", "★the per-block byte ranges are still exact (locating a dropout depends on "
               "them): blocks join end to end and cover every byte that was fed",
       [c["start"] for c in sink._chunks] == [0] + [c["end"] for c in sink._chunks[:-1]]
       and sink._chunks[-1]["end"] == pl.fed_bytes,
       f"ranges={[(c['start'], c['end']) for c in sink._chunks]} · {pl.fed_bytes}B fed")

    # ── 19f The ORDER of the two-level clear now lives only in tts_client (source level) ──
    #    With barge-in gone, `speak_sink` no longer calls `clear()`; the only live caller
    #    is tts_probe.py (a manual probe that costs money). So this assertion guards the
    #    statement order inside `tts_client.clear()` instead.
    src_tc = io.open(tts_client.__file__, encoding="utf-8").read()
    body = src_tc[src_tc.index("    async def clear(self"):]
    body = body[:body.index("\n    async def ", 10)] if "\n    async def " in body[10:] else body
    i_drop = body.find("self._dropping = True")
    i_cb = body.find("_call(self.on_clear)")
    i_send = body.find('"text.clear"')
    ok("19f", "★the order of the two-level clear: raising the in-flight discard window and "
              "firing on_clear (which empties the local queue) both happen **before** "
              "`text.clear` is sent (reversed, audio still in flight pours back into the "
              "queue the moment it is cleared -- the terminal looks right and the sound "
              "keeps coming out anyway)",
       0 <= i_drop < i_cb < i_send,
       f"_dropping@{i_drop} < on_clear@{i_cb} < text.clear@{i_send}")

    # ── 19f2 A turn that finished normally needs no estimation ──
    # ★A non-zero card latency is required: at zero the two routes compute the same value,
    #   so the assertion looks green while measuring nothing at all
    pl3 = speak_sink.LocalPlayer(24000, extra_latency_ms=150)
    tts3 = FakeTts(pl3)
    sink3 = speak_sink.SpeakSink(tts3, pl3, playback_lead_s=999)
    full = ["你知道 LanceDB", "吗？就是那个向量库。"]
    for c in full:
        await sink3(c)
    _advance(pl3, pl3._fed)                      # everything has played
    n_full = await sink3.settle()
    ok("19f2", "★settle records it as FULLY SPOKEN, and `settle` has **only this one route** "
               "(no `interrupted` parameter any more) -- with barge-in gone there is no "
               "situation in which the sink should stop early",
       n_full == sum(len(c) for c in full)
       and "interrupted" not in inspect.signature(sink3.settle).parameters,
       f"settle reports {n_full} chars = all {sum(len(c) for c in full)} · signature "
       f"{inspect.signature(sink3.settle)}")

    # ── 19g Prefetch: do not return while the queue holds more than lead seconds ──
    pl2 = speak_sink.LocalPlayer(24000)
    tts2 = FakeTts(pl2)
    sink2 = speak_sink.SpeakSink(tts2, pl2, playback_lead_s=1.0)
    t = asyncio.create_task(sink2("这是一段足够长的话会产生超过一秒的音频所以要等"))
    for _ in range(5):
        await asyncio.sleep(0)
    blocked = not t.done()
    _advance(pl2, pl2._fed)                      # ★pop + count (as the real callback does)
                                                 #   -> the queue drains
    await asyncio.wait_for(t, timeout=2)
    ok("19g", "★prefetch: the sink does not return while more than playback_lead_s seconds "
              "remain queued; it returns once the queue drains (without prefetch, a "
              "0.5-1.0s silence is inserted between blocks -- measured at 18% of the time)",
       blocked and t.done(),
       f"queue {len('这是一段足够长的话会产生超过一秒的音频所以要等') * FakeTts.SECONDS_PER_CHAR:.2f}s"
       f" > lead 1.0s -> blocked; once played -> released")

    # ── 19h reconcile only truncates; it never extends ──
    led = reply_list.SpokenLedger()
    for c in ("一二三", "四五六", "七八九"):
        led.add_chunk(c)
        led.mark_spoken(None)                    # optimistic: mark everything as played
    led.reconcile(4)
    ok("19h", "★reconcile truncates to the real character count (over-reporting records "
              "words that never left the speaker as having been said -- a lying transcript)",
       led.spoken_text() == "一二三四" and led.drop_unspoken() == 5,
       f"optimistic 9 chars -> closed out at 4 -> spoken={led.spoken_text()!r} · 5 dropped")

    # ── 19i pump calls settle; a sink without settle behaves exactly as before ──
    class SettleSink:
        def __init__(self):
            self.calls = []
        async def __call__(self, chunk):
            return None
        async def settle(self):
            self.calls.append(True)
            return 2                              # insist that only 2 characters were spoken
    ss = SettleSink()
    drv = _driver(sb, FakeDispatch("一二三四五六七八九十"), ss)
    drv.reservoir.add("问一句")
    rec = await drv.pump()
    ok("19i", "★pump calls sink.settle and reconciles against it (with a prefetch buffer, "
              "the block has not been spoken yet when the sink returns)",
       len(ss.calls) == 1 and rec["spoken"] == "一二" and rec["dropped_chars"] == 8,
       f"settle called {len(ss.calls)} time(s) · recorded spoken={rec['spoken']!r} · "
       f"dropped={rec['dropped_chars']}")

    drv2 = _driver(sb, FakeDispatch("一二三四五"), SilentSink())
    drv2.reservoir.add("再问一句")
    rec2 = await drv2.pump()
    ok("19j", "a sink without settle (TypingSink/console_sink) behaves byte for byte as "
              "before (the protocol is optional)",
       rec2["spoken"] == "一二三四五" and rec2["dropped_chars"] == 0,
       f"spoken={rec2['spoken']!r} · dropped={rec2['dropped_chars']}")


# ── C15: a turn has exactly one ending (after barge-in and invalidation were removed) ──
#    ★★This CASE used to be called "the invalidation path", with nine assertions 20a-20i
#      verifying "water arrives while blocks are accumulating -> this turn is invalidated,
#      not one character is spoken, nothing is written down, user_text goes back to the
#      head of the queue, the next turn re-runs it". Once **the invalidation test and the
#      invalidated re-run were deleted**, every one of those nine that was bound to
#      invalidation went with them. What remains are two **regression gates**, whose job
#      is to make sure these mechanisms cannot be quietly wired back in.
def _code_refs(path, names):
    """**Real code references** in a source file (parsed with `ast`, not grepped).

    ★Why not grep: the prose explaining these regression gates has to name the very things
      that were deleted ("`cancel()` went with the invalidation mechanism"), so grep picks
      the docstrings and comments up too -> the assertion is permanently red. The first
      version of 21f2 went red in exactly that way.
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read(), filename=path)
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            hits.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in names:
            hits.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name in names:
            hits.add(node.name)
    return sorted(hits)


class HoldSink:
    """A stand-in for the pending-speech queue: it accepts blocks and speaks **not one
    character** (returns 0). This is the shape while the gate has not opened.

    `on_first_chunk` fires once when the first block arrives -- which turns "the reply
    list takes on more water while blocks are accumulating" into a deterministic event
    (discipline 1 in the module header: never race a sleep for timing).
    """

    def __init__(self, on_first_chunk=None):
        self.held = ""
        self.played = ""
        self._hook = on_first_chunk

    async def __call__(self, chunk):
        self.held += chunk
        if self._hook is not None:
            hook, self._hook = self._hook, None
            hook()
        return 0


class HoldSettleSink(HoldSink):
    """The pending-speech queue plus the close-out protocol: settle honestly reports "not
    one character was spoken", and records how many times it was called."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    async def settle(self):
        self.calls.append(True)
        return 0


_OLD = "我想问三个问题，第一个是关于索引的"
_NEW = "第二个是关于向量维度的"


async def c15(sb):
    # ── 20e Water arriving does not abort: this turn says the whole segment anyway, and
    #    the new water waits in the pool for the next one ──
    #    This is the **positive behaviour** left after invalidation was removed (this used
    #    to produce reason=invalidated, speak nothing, and push the text back to the head
    #    of the queue).
    disp = FakeDispatch("这是一段会完整说完的回复，中间人又说了一句也不管。", chunk_chars=6)
    drv = _driver(sb, disp, SilentSink())

    def water():
        drv.reservoir.add(_NEW)      # <- JEV#1 submitted another cut while blocks accumulated

    drv.sink = HoldSink(on_first_chunk=water)
    drv.reservoir.add(_OLD)
    threads_before = len(sb.thread_text())
    rec = await asyncio.wait_for(drv.pump(), 5)
    ok("20e", "★the reply list takes on water while blocks accumulate -> **this turn still "
              "says the whole segment and still writes it down**, and the new water waits "
              "in the pool for the next turn (this used to be reason=invalidated, nothing "
              "spoken, user_text pushed back to the head of the queue)",
       rec["reason"] == "done" and rec["spoken"] == drv.sink.played
       and rec["user_text"] == _OLD and len(sb.thread_text()) > threads_before
       and drv.reservoir.peek() == _NEW and len(drv.reservoir) == 1,
       f"reason={rec['reason']} · spoke {len(rec['spoken'])} chars · this turn ate "
       f"{rec['user_text']!r} · left in the pool {drv.reservoir.peek()!r}")

    # ── 20g Close-out runs on every turn, and the protocol has no `interrupted` parameter ──
    spy = HoldSettleSink()
    drv3 = _driver(sb, FakeDispatch("Yep"), spy)        # 3 chars, no punctuation -> only flush emits
    drv3.reservoir.add(_OLD)
    rec3 = await asyncio.wait_for(drv3.pump(), 5)
    ok("20g", "★close-out (sink.settle) runs on every turn, which is what makes the books "
              "balance; and the protocol has **no** `interrupted` parameter -- with "
              "barge-in and invalidation gone there is no such thing as \"make the sink "
              "stop early\" (passing that parameter from `reply_list.pump` raises TypeError)",
       len(spy.calls) == 1 and rec3["reason"] == "done"
       and "interrupted" not in inspect.signature(spy.settle).parameters,
       f"settle called {len(spy.calls)} time(s) · reason={rec3['reason']} · "
       f"signature {inspect.signature(spy.settle)}")

    # ── 20h ★The regression gate for removing barge-in (source level): the flag is gone
    #    and so is the reason vocabulary ──
    #    Why a source-level assertion is needed: barge-in is the kind of thing that can be
    #    wired back in without anything erroring. If someone restores
    #    `self.interrupt = asyncio.Event()`, not one behavioural assertion turns red
    #    (nobody sets it) -- but the next person will see it and reconnect the trigger.
    #    So this guards the file directly.
    src_rl = io.open(reply_list.__file__, encoding="utf-8").read()
    # ★★Use `_code_refs` (AST), not a regex over lines. The old version grepped
    #   `\binterrupt\b` and only excluded whole-line `#` comments, so the moment any
    #   docstring in reply_list.py mentions interrupting -- which the English prose
    #   does, because it has to explain what was removed -- this assertion goes red on
    #   a file that is perfectly correct. That is the exact trap `_code_refs` was
    #   written for; 20j already used it and 20h did not.
    live_int = _code_refs(reply_list.__file__, {"interrupt"})
    ok("20h", "★barge-in removed: `reply_list.py` has no `interrupt` code reference, no "
              "\"interrupted\" reason, and no such attribute on a `VoiceDriver` instance "
              "(wire it back in and this goes red at once)",
       not live_int and '"interrupted"' not in src_rl and not hasattr(drv3, "interrupt"),
       f"leftover code references={live_int} · vocabulary has interrupted="
       f"{'\"interrupted\"' in src_rl} · drv has the attribute="
       f"{hasattr(drv3, 'interrupt')}")

    # ── 20j ★The regression gate for removing invalidation (source level + instance level) ──
    #    Same reasoning as 20h, and invalidation is even easier to reinstate by accident:
    #    its trigger was a single line (`len(drv.reservoir) > 0`), so anyone who adds "take
    #    a look at the pool" to the mouth loop brings it back.
    #    ★Three things are guarded together: the `stale` flag / `put_back` / the "aborted"
    #    reason.
    live_stale = _code_refs(reply_list.__file__, {"stale", "put_back", "_Aborted"})
    ok("20j", "★invalidation removed: `reply_list.py` has no code reference to `stale` / "
              "`put_back` / `_Aborted`, neither the \"invalidated\" nor the \"aborted\" "
              "reason, and no `stale` attribute on a `VoiceDriver` instance -- a turn has "
              "exactly one ending, and that ending is \"finished\"",
       not live_stale and '"invalidated"' not in src_rl and '"aborted"' not in src_rl
       and not hasattr(drv3, "stale") and not hasattr(drv3.reservoir, "put_back"),
       f"leftover code references={live_stale} · vocabulary has invalidated="
       f"{'\"invalidated\"' in src_rl} · has aborted={'\"aborted\"' in src_rl}"
       f" · drv has stale={hasattr(drv3, 'stale')}"
       f" · Reservoir has put_back={hasattr(drv3.reservoir, 'put_back')}")


# ── C16: the speaking gate (voice_out/voice_out_loop) ────────────────────────────────
_GATE_N = [0]
_GATE_CLOCK = [1_000_000.0]     # ★the fake clock this group uses (_gate_env installs it,
                                #   _silent_for advances it)


def _gate_env(sb, judge=None, reply="这是一段要出声的回复，够长到能切成好几块。",
              inner=None, **knobs):
    """Build "one driver + one gate + one mouth loop". ★A separate session_dir every time,
    for the same reason as _loop."""
    import voice_out_loop
    _GATE_CLOCK[0] = 1_000_000.0          # a clean starting point per env
    _GATE_N[0] += 1
    sd = os.path.join(sb.tmp, "workspace", f"session_gate{_GATE_N[0]}")
    os.makedirs(sd, exist_ok=True)
    disp = FakeDispatch(reply, chunk_chars=6)
    inner = inner if inner is not None else SilentSink()
    gate = voice_out_loop.GateSink(inner)
    drv = reply_list.VoiceDriver(disp, sd, peer="user", sink=gate, verbose=False)
    # ★★Install a fake clock. Once the gate's criterion changed from "tick count" to "real
    #   seconds of silence", the mechanism itself became time-driven -- so a test either
    #   controls time or is reduced to racing sleeps (which discipline 1 in the module
    #   header forbids). `_silent_for` is what advances this clock. In a live run the clock
    #   defaults to time.time and not one of these lines is reached.
    drv.silence.set_clock(lambda: _GATE_CLOCK[0])
    k = {"tick_interval_s": 0.01, "jev2_min_interval_s": 1.0,
         "speak_min_silence_s": 1.0, "max_hold_s": 6.0,
         "turn_over_threshold": 0.6, "max_inflight": 2}
    k.update(knobs)
    lp = voice_out_loop.VoiceOutLoop(drv, gate, judge, knobs=k, verbose=False)
    return lp, gate, drv, inner


async def _wait_gate(gate, timeout=2.0):
    """Wait until a block is genuinely stuck at the gate. ★A deterministic wait, not a
    sleep racing the timing."""
    t0 = time.time()
    while not gate.waiting:
        if time.time() - t0 > timeout:
            raise AssertionError("no block ever got stuck at the gate")
        await asyncio.sleep(0)


def _silent_for(drv, n):
    """Stay quiet for another n seconds -- advance the fake clock by n seconds, and strike
    n empty ticks the way the ear does (the mouth itself must NEVER tick).

    ★Both are needed because both readings have to move: `silence_seconds()` (the gate's
      real criterion) reads the clock, while `value()` (the figure in the record) counts
      ticks. Advance the clock without ticking and 21k, "the mouth must not touch
      silence", becomes untestable.
    """
    _GATE_CLOCK[0] += float(n)
    for _ in range(n):
        drv.silence.tick()


async def c16(sb):
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import voice_out_loop

    # ── 21a The lower bound: the person has just spoken (real silence < the lower bound)
    #    -> hold, and do not ask JEV#2 (this blocks the commonest case for free) ──
    j = FakeJudge(noul=0.9)
    lp, gate, drv, inner = _gate_env(sb, j)
    drv.reservoir.add("你知道 LanceDB 吗")
    task = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    ok("21a2", "the moment a block sticks at the gate, the ticker is woken (gate.wants) -- "
               "at a plain 1 Hz the worst case is waiting a whole tick before judging even "
               "begins, by which time the person finished long ago and the machine is "
               "still waiting for the beat",
       gate.wants.is_set(),
       "wants is raised (_tick_loop waits on it rather than idling for a tick)")

    drv.silence.mark_voice(); drv.silence.tick()          # speech this tick -> reading to 0
    rec_a = await lp.tick_once(); await lp.settle()
    ok("21a", "★the lower bound: the person has just spoken (real silence 0s < 1.0s) -> "
              "hold, do NOT ask JEV#2, and the block stays stuck at the gate",
       rec_a["seg"] == "hold" and j.seen == [] and gate.waiting
       and inner.played == "" and not gate.opened.is_set(),
       f"seg={rec_a['seg']} · asked {len(j.seen)} times · the inner sink received "
       f"{len(inner.played)} chars")

    # ── 21b The middle band: ask JEV#2, verdict true -> open -> the block really reaches
    #    the inner sink ──
    _silent_for(drv, 2)                                   # real silence 2s (between the bounds)
    rec_b = await lp.tick_once(); await lp.settle()
    rec = await asyncio.wait_for(task, 5)
    ok("21b", "★the middle band: ask JEV#2, verdict true -> open, and the whole segment "
              "really reaches the inner sink and is written down",
       rec_b["seg"] == "ask" and len(j.seen) == 1 and gate.opened.is_set()
       and inner.played == drv.turns[0]["spoken"] and rec["reason"] == "done"
       and rec["dropped_chars"] == 0,
       f"asked once · opened because="
       f"{gate.turns[-1]['open_reason'] if gate.turns else gate.turn['open_reason']}"
       f" · the inner sink played {len(inner.played)} chars")

    # ── 21b2 An open gate = transparent forwarding (only the first block ever waited) ──
    t = gate.turns[-1] if gate.turns else gate.turn
    ok("21b2", "once the gate opens it forwards transparently: in the whole turn only the "
               "FIRST block ever waited at the gate",
       t["queued"] == 1 and rec["spoken"] == inner.played and len(inner.played) > 6,
       f"blocks that waited at the gate={t['queued']} · {len(inner.played)} chars over the "
       f"turn (blocks are 6-24 chars)")

    # ── 21c Middle band, verdict false -> do not open; the block keeps waiting (the inner
    #    sink receives nothing at all) ──
    j = FakeJudge(noul=0.1)
    lp, gate, drv, inner = _gate_env(sb, j)
    drv.reservoir.add("我想问三个问题，第一个是关于索引的")
    task = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    _silent_for(drv, 2)
    for i in range(3):
        if i:
            _silent_for(drv, 1)             # the person stays quiet: another ask is only
                                            # allowed per extra second (rate limiting)
        await lp.tick_once(); await lp.settle()
    ok("21c", "★middle band, verdict false -> do not open; the block stays stuck at the "
              "gate (not one character is spoken)",
       len(j.seen) == 3 and not gate.opened.is_set() and gate.waiting
       and inner.played == "" and lp.stats["jev2_not_over"] == 3,
       f"3 consecutive ticks all judged false · the inner sink received "
       f"{len(inner.played)} chars")

    # ── 21c2 ★Rate limiting (discipline 8): checking the gate is free, asking JEV#2 costs
    #    money -- while the silence has not grown, not one ask is allowed ──
    #    This guards the bill for dropping tick_interval_s from 1.0 to 0.25: without rate
    #    limiting, JEV#2 call volume quadruples outright -- and it still runs, still raises
    #    nothing, the bill is simply larger.
    asks_before = len(j.seen)
    v_before = drv.silence.value()
    for _ in range(3):
        await lp.tick_once(); await lp.settle()
    ok("21c2", "★rate limiting: the silence has not grown -> 3 consecutive ticks ask JEV#2 "
               "not once (checking the gate is free, asking JEV costs money)",
       len(j.seen) == asks_before and lp.stats["rate_limited"] == 3,
       f"all 3 ticks held back by the rate limiter (still {len(j.seen)} asks)")

    # ── 21k The mouth's ticks must NOT touch silence (the ear's ticker is its only writer) ──
    ok("21k", "★the mouth struck 3 ticks and the silence reading did not move (the ear's "
              "ticker is the only writer)",
       drv.silence.value() == v_before == 4,
       f"{drv.silence.value()} both before and after the 3 ticks (had it ticked, it would "
       f"now read 7)")

    # ── 21d The upper bound: real silence reaches max_hold_s -> open unconditionally, and
    #    do NOT ask JEV#2 ──
    _silent_for(drv, 2)                                   # 4s -> 6s = max_hold_s
    rec_d = await lp.tick_once(); await lp.settle()
    rec = await asyncio.wait_for(task, 5)
    ok("21d", "★the upper bound: real silence reaching max_hold_s opens the gate "
              "unconditionally and asks JEV#2 nothing (with no upper bound it would wait "
              "forever)",
       rec_d["seg"] == "max_hold" and len(j.seen) == asks_before
       and gate.opened.is_set() and rec["reason"] == "done" and inner.played,
       f"seg={rec_d['seg']} · this tick asked nothing extra (still {len(j.seen)}) · "
       f"really spoke {len(inner.played)} chars")

    # ── 21e The shape of state: the finished segment comes from .inflight, not from U ──
    j = FakeJudge(noul=0.9)
    # ★★This used to require a real 1.0s tick, because the seconds in the note were
    #   computed as "silence reading x tick_interval_s", so a 0.01s tick wrote 2 ticks as
    #   "0 seconds". Once that was changed to read silence_seconds() directly, **the
    #   question became independent of the tick length** -- so the special case was
    #   dropped and this env uses the same fast tick as the others.
    lp, gate, drv, inner = _gate_env(sb, j)
    said_text = "I have three questions, the first one is about indexes"
    drv.reservoir.add(said_text)
    task = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    _silent_for(drv, 2)
    drv.user_buffer.set_partial("and that other thing")    # he started talking again while
                                                           # blocks accumulated (U non-empty)
    state, said = lp.build_state()
    drv.user_buffer.clear()                                # the note's other half, when he did not
    state_quiet, _ = lp.build_state()
    ok("21e", "★JEV#2's state = the whole segment just submitted (.inflight) plus a "
              "deterministic note -- not U",
       said == said_text and state.startswith(said_text + "\n\n(system note:")
       and "quiet for 2 seconds" in state
       and 'started speaking again: "and that other thing"' in state
       and "They have not spoken since." in state_quiet,
       f"state={state!r}")

    # ── 21f2 ★The regression gate for removing invalidation (the mouth side, source level) ──
    #    21f used to live here: water arrives while blocks accumulate -> raise stale, abort
    #    at the gate, speak nothing, write nothing down. With invalidation deleted, that
    #    behaviour is gone, and this assertion now guards "do not wire it back in" --
    #    ★on the mouth side invalidation costs almost nothing to revive: its test was one
    #    line, `len(drv.reservoir) > 0`.
    src_vo = io.open(voice_out_loop.__file__, encoding="utf-8").read()
    # ★`cancel` cannot be looked up directly: `task.cancel()` / `t.cancel()` belong to
    #   asyncio, and both are legitimate and must stay. What is checked is whether the
    #   **method definition** `GateSink.cancel` has come back (see the hasattr below).
    live_vo = _code_refs(voice_out_loop.__file__,
                         {"aborted", "_first_of", "stale_turns", "abort_reason"})
    g_probe = voice_out_loop.GateSink(SilentSink())
    ok("21f2", "★invalidation removed (mouth side): `voice_out_loop.py` has no code "
               "reference to `aborted`/`cancel`/`_first_of`/`stale_turns`, and `GateSink` "
               "has no `cancel`/`aborted`/`bind`/`drv` -- the gate has exactly one exit, "
               "and that is opening (discipline 3)",
       not live_vo and not hasattr(g_probe, "cancel") and not hasattr(g_probe, "aborted")
       and not hasattr(g_probe, "bind") and not hasattr(g_probe, "drv")
       and "abort_reason" not in src_vo,
       f"leftover code references={live_vo} · GateSink has cancel="
       f"{hasattr(g_probe, 'cancel')} · aborted={hasattr(g_probe, 'aborted')} · "
       f"bind={hasattr(g_probe, 'bind')} · drv={hasattr(g_probe, 'drv')} · "
       f"abort_reason in the source={'abort_reason' in src_vo}")

    # ── 21g ★A verdict arriving a turn late: acting on it means "this turn opened without
    #    ever being asked" ──
    hold = asyncio.Event()
    j = FakeJudge(noul=0.95, gate=hold, gate_only_first=True)
    lp, gate, drv, inner = _gate_env(sb, j)
    drv.reservoir.add("第一句")
    task = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    _silent_for(drv, 2)
    await lp.tick_once()                                   # turn 1 asks, then blocks on hold
    seq1 = gate.turn_seq
    _silent_for(drv, 4)                                    # 6 ticks -> upper bound opens,
                                                           # turn 1 runs to completion
    await lp.tick_once()
    await asyncio.wait_for(task, 5)
    drv.reservoir.add("第二句")                            # turn 2
    task2 = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    seq2 = gate.turn_seq
    hold.set()                                             # turn 1's verdict only returns now
    await lp.settle()
    crossed = lp.stats["cross_turn_discarded"]
    opened_now = gate.opened.is_set()
    _silent_for(drv, 6)
    await lp.tick_once(); await lp.settle()
    await asyncio.wait_for(task2, 5)
    ok("21g", "★a JEV#2 verdict from the previous turn comes back across the turn boundary "
              "-> discard it (acting on it means this turn OPENED WITHOUT BEING ASKED)",
       seq2 == seq1 + 1 and crossed == 1 and not opened_now,
       f"turn number {seq1} -> {seq2} · {crossed} cross-turn discard(s) · the gate was "
       f"still shut when it was discarded={not opened_now}")

    # ── 21h Close-out: use the inner sink's settle figure when it has one, otherwise the
    #    forwarded character count -- never invent a number ──
    class SettleInner:
        def __init__(self):
            self.played = ""
            self.calls = []

        async def __call__(self, chunk):
            self.played += chunk
            return None

        async def settle(self):
            self.calls.append(True)
            return 2                                       # insist only 2 characters were spoken

    si = SettleInner()
    lp, gate, drv, inner = _gate_env(sb, FakeJudge(noul=0.9), inner=si)
    drv.reservoir.add("问一句")
    task = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    _silent_for(drv, 6)
    await lp.tick_once(); await lp.settle()
    rec_h = await asyncio.wait_for(task, 5)
    ok("21h", "★the gate passes the inner sink's settle straight through to pump "
              "(inventing a number of its own would make the transcript lie)",
       len(si.calls) == 1 and rec_h["spoken"] == si.played[:2]
       and rec_h["dropped_chars"] == len(si.played) - 2,
       f"the inner sink reported 2 chars -> recorded spoken={rec_h['spoken']!r} · "
       f"{rec_h['dropped_chars']} dropped")

    # ── 21i Config validation: the middle band must not be empty ──
    try:
        _gate_env(sb, FakeJudge(), speak_min_silence_s=6.0, max_hold_s=6.0)
        raised = False
    except ValueError:
        raised = True
    ok("21i", "config validation: speak_min_silence_s >= max_hold_s raises at construction "
              "time (otherwise the gate silently degrades into a plain timer and the whole "
              "JEV#2 path was built for nothing)",
       raised, "min=6.0s max=6.0s -> ValueError")

    # ── 21j Both arguments to the real verdict must be passed explicitly (omitting either
    #    is a silent error) ──
    src = open(os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out",
                            "voice_out_loop.py"), encoding="utf-8").read()
    ok("21j", "★open_jev2 passes question_key='turn_over' and "
              "threshold=turn_over_threshold explicitly (omitting the first raises; "
              "omitting the second SILENTLY falls back to 0.5)",
       'question_key="turn_over"' in src and 'threshold=float(k["turn_over_threshold"])' in src,
       "both arguments are pinned inside open_jev2")

    # ── 21l ★Water arriving after the gate opened -> that tick is idle, and the turn still
    #    finishes (the invariant itself) ──
    #    Before invalidation was removed this guarded "invalidation only happens before the
    #    first sound". With it gone, this guards something stronger: **water arriving at
    #    any moment does not affect the turn in progress**. It stays because it is a
    #    BEHAVIOURAL assertion -- the source gates 21f2 and 20j can stop invalidation being
    #    reinstated verbatim, but not someone reimplementing it under another name.
    j = FakeJudge(noul=0.95)
    lp, gate, drv, inner = _gate_env(sb, j)
    drv.reservoir.add("放行之后再进水的那一轮")
    task = asyncio.create_task(drv.pump())
    await _wait_gate(gate)
    _silent_for(drv, 2)
    rec_open = await lp.tick_once()                       # -> jev2 opens the gate
    t0 = time.time()
    while gate.waiting:
        if time.time() - t0 > 2:
            raise AssertionError("the gate opened, but the block stuck at it was never woken")
        await asyncio.sleep(0)
    drv.reservoir.add("机器正在说的时候我又说了一句")     # <- water enters the pool
    rec_after = await lp.tick_once()                      # -> must be idle
    rec = await asyncio.wait_for(task, 5)
    ok("21l", "★water arriving after the gate opened -> that tick is idle: this turn says "
              "the whole segment, records it honestly, and the new water waits in the pool "
              "for the next turn (with barge-in and invalidation gone, this is the ONLY "
              "possible behaviour)",
       rec_open["opened"] and rec_after["seg"] == "idle"
       and rec["reason"] == "done" and rec["dropped_chars"] == 0
       and len(drv.reservoir) == 1,
       f"opening tick seg={rec_open['seg']} noul={rec_open['noul']} · the tick after="
       f"{rec_after['seg']} · reason={rec['reason']} · {rec['dropped_chars']} dropped · "
       f"{len(drv.reservoir)} cuts in the pool")

    # ── 21m ★The production threshold has to sit above "the peak reading while the person
    #    is still talking" ──
    #    This guards a mistake made right here: assuming `now - stop_wall` means "how long
    #    the person has been quiet", and therefore setting the lower bound to 1.0s. What it
    #    really means is "how far the audio we have a transcript for extends" -- STT sends
    #    a partial every ~958ms describing audio from ~677ms ago, so while the person is
    #    **still talking** this reading naturally climbs to about 1.635s. A threshold below
    #    that peak means the gate starts asking JEV#2 mid-sentence: more money spent, and
    #    the whole decision resting on turn_over_threshold, which **has never been
    #    calibrated**. With barge-in gone, every false positive is the machine talking over
    #    the person for a whole segment. And it still runs, and still goes green.
    clk = [500.0]
    sc_m = reply_list.SilenceCounter(os.path.join(sb.tmp, "s21m"),
                                     clock=lambda: clk[0])
    sc_m.mark_voice(stop_wall=clk[0] - 0.677)      # measured p50 pipeline lag
    clk[0] += 0.958 - 1e-9                         # the instant before the next partial arrives
    peak = sc_m.silence_seconds()
    # ★Both silence thresholds must **really be read from dispatch_config** (both were
    #   deliberately made config fields). Why that is worth an assertion: when `load_knobs()`
    #   does not find a key in the config it **silently falls back to DEFAULT_KNOBS** -- so
    #   renaming a key turns that config line into a dead field, while the system keeps
    #   running, keeps going green, and simply stops listening to the config from then on.
    cfg_vo = json.load(open(os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out",
                                         "dispatch_config.json"), encoding="utf-8"))["knobs"]
    cfg_li = json.load(open(os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "listener",
                                         "dispatch_config.json"), encoding="utf-8"))["knobs"]
    k_prod = voice_out_loop.load_knobs()
    k_li = listener_loop.load_knobs()
    in_cfg = ("speak_min_silence_s" in cfg_vo and "max_hold_s" in cfg_vo
              and "llm_min_silence_s" in cfg_li)
    matches = (k_prod["speak_min_silence_s"] == cfg_vo["speak_min_silence_s"]
               and k_li["llm_min_silence_s"] == cfg_li["llm_min_silence_s"])
    side = ("★below" if k_prod["speak_min_silence_s"] <= peak else "above")
    ok("21m", "★both silence thresholds (the mouth's speak_min_silence_s and the ear's "
              "llm_min_silence_s) really are read from dispatch_config rather than falling "
              "back to DEFAULT_KNOBS (renaming a key silently reverts to the code default: "
              "still runs, still green, and stops listening to the config); and the "
              "mid-sentence peak reading of about 1.635s is recorded here",
       in_cfg and matches and abs(peak - 1.635) < 1e-6,
       f"read from the config: speak={k_prod['speak_min_silence_s']}s · "
       f"LLM={k_li['llm_min_silence_s']}s · upper bound={k_prod['max_hold_s']}s; "
       f"mid-sentence peak reading {peak:.3f}s (measured p50 1.606 / max 2.185) -> the "
       f"current speaking threshold is {side} the peak"
       f"{': the gate will often ask JEV#2 while the person is mid-sentence. This is a '
         'knowing trade-off; tighten towards 2.0 to be safer'
         if k_prod['speak_min_silence_s'] <= peak else ''}")


# ── C17: the whole chain wired up (voice_cli --tts + voice_cli._report) ──────────────
async def _one_tts_turn(sb, *, gaps=0, wrap_gate=True):
    """Run one turn through a real `SpeakSink` (fake TTS + a player that never opens the
    sound card) and return `(sink_for_report, rec)`.

    ★This reproduces a whole `--tts` turn on the offline bench: `GateSink(SpeakSink)` ->
    pump -> settle.
    """
    import speak_sink
    import voice_out_loop
    from lib import tts_client
    pl = speak_sink.LocalPlayer(24000)
    inner = speak_sink.SpeakSink(FakeTts(pl), pl, playback_lead_s=999)
    sink = voice_out_loop.GateSink(inner) if wrap_gate else inner
    if wrap_gate:
        sink.open("jev2 noul=0.930")
        sink.turn["asks"], sink.turn["wait_ms"] = 2, 1234
    await sink("一二三四五")
    pl.t_first_play = inner.turn["t0"] + 0.865        # the moment the ear hears it
                                                      # (stamped by the callback in a live run)
    pl.gaps, pl.gap_log = gaps, [(1.46, 121)] * gaps
    _advance(pl, pl._fed)
    n = await sink.settle()
    rec = {"reason": "done", "dropped_chars": 0,
           "first_chunk_ms": 120, "total_ms": 900, "usage": {}, "spoken": "一二三四五"}
    return sink, rec, n


def _report_text(rec, sink, cuts=1):
    import voice_cli
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        voice_cli._report(rec, sink, cuts=cuts)
    return buf.getvalue()


async def c17(sb):
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import voice_cli
    import voice_out_loop
    src = io.open(os.path.join(SHELL_ROOT, "voice_cli.py"), encoding="utf-8").read()

    # ── 22a ★_report must TAKE THE FIGURES BEFORE calling new_turn (this really was
    #    broken once) ──
    sink, rec, n = await _one_tts_turn(sb, wrap_gate=False)
    line = _report_text(rec, sink)
    ok("22a", "★_report takes this turn's figures before calling new_turn: first sound and "
              "spoken really are printed (reversed, it reads the next turn's empty shell -> "
              "neither is printed, while [first text · total] still shows up as normal)",
       "first sound 865ms" in line and "spoken 5/5 chars" in line,
       f"output={line.strip()!r}")

    # ── 22b ★_report has to see through the wrapping sink (the live chain is
    #    GateSink(SpeakSink)) ──
    sink, rec, n = await _one_tts_turn(sb, wrap_gate=True)
    line = _report_text(rec, sink)
    ok("22b", "★_report follows .inner through the gate: the TTS figures still print, plus "
              "a section of the gate's own (without that, sink.turn is the gate's "
              "bookkeeping and the TTS keys are simply not in it)",
       "first sound 865ms" in line and "spoken 5/5 chars" in line and "gate " in line
       and "waited 1234ms" in line and "opened jev2 noul=0.930" in line
       and "asked JEV#2 2" in line,
       f"输出={line.strip()!r}")

    # ── 22b2 ★EchoSink wraps the outermost layer: it prints the reply text, and it must
    #    NOT eat the gate's figures ──
    #    This is the one place EchoSink can break **silently**: `_report` reads the gate's
    #    bookkeeping out of `sink.turn` (how long it waited / why it opened / how many
    #    times JEV#2 was asked). If EchoSink does not pass `turn` through to the inner
    #    sink, that whole section disappears from the closing line -- no error, the text
    #    still prints, and everything looks completely normal.
    import speak_sink
    import voice_cli as _tc
    pl2 = speak_sink.LocalPlayer(24000)
    inner2 = speak_sink.SpeakSink(FakeTts(pl2), pl2, playback_lead_s=999)
    gate2 = voice_out_loop.GateSink(inner2)
    gate2.open("jev2 noul=0.930")
    gate2.turn["asks"], gate2.turn["wait_ms"] = 2, 1234
    echo = _tc.EchoSink(gate2, gap_ms=0)      # gap_ms=0 -> both timings get printed,
                                              # which is what makes this assertable
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        await echo("第一块。")
        await echo("第二块。")
    shown = buf2.getvalue()
    pl2.t_first_play = inner2.turn["t0"] + 0.865
    _advance(pl2, pl2._fed)
    n2 = await echo.settle()
    rec2 = {"reason": "done", "dropped_chars": 0, "first_chunk_ms": 120,
            "total_ms": 900, "usage": {}, "spoken": "第一块。第二块。"}
    line2 = _report_text(rec2, echo)
    ok("22b2", "★EchoSink (the outermost layer under --tts) prints each block's **text** to "
               "the terminal and passes `turn` through to the inner sink -- otherwise the "
               "gate's own figures silently vanish from the closing line while the text "
               "still prints and everything looks completely normal",
       "bot>" in shown and "第一块。" in shown and "第二块。" in shown
       and "gen " in shown and n2 == 8
       and "gate waited 1234ms" in line2 and "asked JEV#2 2" in line2
       and "spoken 8/8 chars" in line2,
       f"screen={shown.strip().replace(chr(10), ' ⏎ ')!r} · settle={n2} chars · "
       f"closing line={line2.strip()!r}")

    # ── 22c Dropouts are reported from the `gaps` key (`underruns` was retired: it counted
    #    the idle spin before playback began) ──
    sink, rec, n = await _one_tts_turn(sb, gaps=2)
    line = _report_text(rec, sink)
    ok("22c", "dropouts are reported from `gaps` (the `underruns` key went long ago, and "
              "keeping it would mean never raising the alarm), and the timestamps from "
              "gap_log are printed too (those are what make a dropout locatable)",
       "dropouts 2" in line and "(1.46, 121)" in line,
       f"that part of the output={[b for b in line.split(' · ') if 'dropouts' in b]}")

    # ── 22d ★Barge-in removed: the person speaks while the machine is speaking -> the
    #    words only enter the pool, and this turn is unaffected ──
    lp, drv = _loop(sb, FakeJudge(noul=0.1))
    gate = voice_out_loop.GateSink(SilentSink())
    w = voice_cli.VoiceWiring(lp, display=False)
    w.mark_mic_open()
    ev = {"type": "transcript.partial", "text": "等一下", "is_final": False}
    drv.busy = True                                    # a turn is running
    gate.open("测试")                                  # the gate is open = currently speaking
    w.on_partial("等一下", ev)                         # <- the person starts talking
    ok("22d", "★barge-in removed: the person speaks while the machine is speaking -> the "
              "words only enter U (there is no interrupt flag, and this turn finishes)",
       "等一下" in drv.user_buffer.text() and not hasattr(drv, "interrupt"),
       f"U={drv.user_buffer.text()!r} · drv has an interrupt attribute="
       f"{hasattr(drv, 'interrupt')}")

    # ── 22e ★Source level: every wiring point for barge-in is gone (so nobody can quietly
    #    reconnect it) ──
    live = _code_refs(voice_cli.__file__,
                      {"interrupt", "interrupted", "barge", "barge_in"})
    # ★★Check only that `drv` / `gate` are absent from the parameters; do **not** pin the
    #   whole signature verbatim. This used to read
    #   `"def __init__(self, lp, *, display=True, plain=False):" in src`, which turned the
    #   assertion red whenever **any** legitimate keyword was added to the constructor (it
    #   really did go red when `hallucination_gate=` was added) -- and what it exists to
    #   guard was never "the signature may not change", it is "no relic of barge-in has
    #   been quietly reconnected". An assertion pinned tighter than its own intent only
    #   trains people to edit the assertion, at which point it has stopped earning its keep.
    _sig = re.search(r"def __init__\(self, lp,[^)]*\)", src)
    sig_ok = bool(_sig) and not re.search(r"\b(drv|gate)\b", _sig.group(0))
    ok("22e", "★source level: `voice_cli.py` has no code reference to `interrupt`/`barge`, "
              "and `VoiceWiring` no longer takes drv/gate (they existed purely for barge-in)",
       not live and sig_ok,
       f"leftovers={live} · constructor signature narrowed={sig_ok}")

    # ── 22f The closing-report path (print_gate) really does print something ──
    sink, rec, n = await _one_tts_turn(sb, wrap_gate=True)
    sink.new_turn()                                    # archive this turn, as _report does
    lp3, drv3 = _loop(sb, FakeJudge(noul=0.9))
    vo = voice_out_loop.VoiceOutLoop(drv3, sink, FakeJudge(noul=0.9),
                                     knobs={"tick_interval_s": 0.01}, verbose=False)
    w3 = voice_cli.VoiceWiring(lp3, display=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        voice_cli.print_gate(vo, sink, w3, None)
    rep = buf.getvalue()
    ok("22f", "the closing report print_gate prints first-sound p50, why the gate opened, "
              "and the wait at the gate (this path broke silently once, which is why it "
              "needs an assertion of its own)",
       "first sound p50 865ms" in rep and "jev2 noul=0.930×1" in rep
       and "waited at the gate p50 1234ms" in rep,
       f"report={rep.strip().splitlines()}")

    # ── 22g Source-level wiring checks (each one fails without raising anything) ──
    i_tts = src.find("stack.enter_async_context(conn)")
    i_mic = src.find("stt_client.mic_stream(")
    ok("22g", "★two wiring rules (there were three: the `gate.bind(drv)` one went with the "
              "invalidation mechanism, and **having no bind is now the correct state** -- "
              "assertion 21f2 keeps it from coming back): ① JEV#2 is warmed up too "
              "(without that the first tick is bound to be late, and a late verdict is "
              "always discarded) ② the TTS handshake happens BEFORE the mic opens "
              "(1.4-2.9s, otherwise the first sentence waits out a handshake)",
       '("JEV#2", jev2)' in src and 0 < i_tts < i_mic
       and not _code_refs(voice_cli.__file__, {"bind"}),
       f"TTS handshake@{i_tts} < mic open@{i_mic} · code references to bind in voice_cli="
       f"{_code_refs(voice_cli.__file__, {'bind'})} (a mention in a comment does not count)")

    # ── 22h Shutdown order: the mouth closes first (while it is alive it keeps opening the
    #    gate for the turn that is draining) ──
    ok("22h", "on shutdown the mouth closes FIRST and the ear second (the gate is on the "
              "mouth side; the other way round it keeps judging and keeps opening)",
       0 < src.find("await vo.aclose()") < src.find("await lp.aclose()"),
       f"vo.aclose@{src.find('await vo.aclose()')} < lp.aclose@{src.find('await lp.aclose()')}")

    # ── 22i ★Walk the whole --tts path (every external IO stubbed) ──
    #    Without this, any misspelled name inside the `--tts` branch would first surface
    #    during a live run -- wearing headphones, spending real money.
    got = await _tts_wiring_smoke(sb)
    ok("22i", "★the whole `--tts` path runs offline: connect TTS -> warm up both JEVs -> "
              "open the mic -> one turn really spoken -> both loops shut down -> the report "
              "is written (only four external IOs are stubbed: mic/STT/TTS/LLM)",
       got["turns"] == 1 and got["spoken"] and got["gate_opened"]
       and got["report_has_gate"] and got["dumped"] and got["warmups"] == ["JEV#2"]
       and got["player_stopped"] and got["conn_closed"],
       f"turns={got['turns']} · spoken={got['spoken']!r} · opened={got['open_reason']!r}"
       f" · warm-ups={got['warmups']} · shutdown: player stopped={got['player_stopped']}"
       f" connection closed={got['conn_closed']} · "
       f"written={os.path.basename(got['dumped'] or '')}")

class InstantPlayer:
    """A stub player: anything fed to it counts as played, and it never opens the sound
    card. ★This verifies wiring only, never timing (timing lives in C14)."""

    def __init__(self, sr=24000):
        self.sr, self.bps = sr, sr * 2
        self._fed = self._played = 0
        self.t_first_play = None
        self.gaps, self.gap_log, self.underruns = 0, [], 0
        self.started = self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    @property
    def latency_s(self):
        return 0.0

    def feed(self, pcm):
        self._fed += len(pcm)
        self._played = self._fed
        if self.t_first_play is None:
            self.t_first_play = time.time()

    def flush(self):
        n = self._fed - self._played
        self._played = self._fed
        return n

    @property
    def fed_bytes(self):
        return self._fed

    @property
    def played_bytes(self):
        return self._played

    @property
    def queued_seconds(self):
        return 0.0


async def _tts_wiring_smoke(sb):
    """Run the whole of `voice_cli._run`, stubbing four external IOs: mic / STT / TTS / LLM.

    ★What is stubbed is the **boundary**, not the logic: the gate, the three-band decision,
    pump, the transcript, close-out, the report and the dump are all real.
    """
    import speak_sink
    import voice_cli
    import voice_out_loop
    from lib import tts_client

    tmp = os.path.join(sb.tmp, "listener_stub")
    os.makedirs(os.path.join(tmp, "fixtures"), exist_ok=True)
    real_listener = os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "listener")
    shutil.copy(os.path.join(real_listener, "dispatch_config.json"), tmp)

    player = InstantPlayer()
    fake_tts = FakeTts(player)
    fake_tts.m = {"connect_ms": 1234, "chars_sent": 0, "errors": []}
    fake_tts.cost_usd = 0.0
    state = {"conn_closed": False, "warmups": [], "reports": []}

    class _Conn:
        async def __aenter__(self):
            return fake_tts

        async def __aexit__(self, *e):
            state["conn_closed"] = True
            return False

    def _build(device=None, verbose=False):
        return _Conn(), player, (lambda tts: speak_sink.SpeakSink(
            tts, player, playback_lead_s=999))

    class _SttStub:
        @staticmethod
        def resolve_key(shell_root):
            return "sk-fake"

        @staticmethod
        def mic_stream(device=None, on_frame=None, on_status=None):
            class _M:
                def stop(self): pass
                def close(self): pass
            return _M()

        @staticmethod
        async def stream(cfg, key, q, on_partial=None, on_final=None,
                         on_speech_final=None, on_ready=None, on_event=None):
            on_ready()                                  # ★= the mic only opens after
                                                        #   transcript.created
            ev = {"type": "transcript.partial", "text": "你知道 LanceDB 吗",
                  "is_final": True, "speech_final": True,
                  "end_of_turn_confidence": 0.97, "start": 0.0, "duration": 1.2}
            on_partial("你知道", dict(ev, is_final=False, speech_final=False))
            # ★★Three events, shaped the way real recordings are (13 of 17 across 8
            #   sessions have exactly this shape): the `is_final` one carries text and
            #   enters U; the `speech_final` one carries **a repeat of that text**, and it
            #   neither submits nor enters U (listener_loop discipline 2).
            #   ★So this turn can only run if **JEV#1 genuinely passes** -- and that is
            #     precisely why this smoke now exercises the one real production
            #     submission path (it used to ride the fallback).
            on_final("你知道 LanceDB 吗", dict(ev, speech_final=False))
            on_speech_final("你知道 LanceDB 吗", ev)
            await q.get()                               # wait for the stopper's sentinel
            return {"connect_ms": 9, "created_ms": 9, "events": 3, "frames_sent": 10,
                    "audio_seconds": 1.0, "event_latency_ms": [10], "errors": []}

    def _new_session(peer="user", sink=None, flush_kwargs=None):
        sd = os.path.join(sb.tmp, "workspace", "session_smoke")
        os.makedirs(sd, exist_ok=True)
        return reply_list.VoiceDriver(FakeDispatch("知道，它是向量库。", chunk_chars=6),
                                      sd, peer=peer, sink=sink, verbose=False), sd

    class _FakeJev2:
        threshold, calls, total_cost_usd, model = 0.6, 0, 0.0, "fake/jev2"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def warmup(self):
            state["warmups"].append("JEV#2")
            return 12.0

        async def ask(self, s):
            self.calls += 1
            return {"noul": 0.95, "meaningful": True}

    # ★★Take the defaults from the **real parser**; never hand-copy a Namespace. A
    #   hand-copied one drifts every time a flag is added, and the symptom is this CASE
    #   raising `AttributeError` outright while every other assertion stays green (which
    #   really happened when --no-echo was added). Only override what this smoke needs.
    args = voice_cli.build_parser().parse_args([])
    # ★seconds went from 0 to 0.3: 0 means shut down immediately, so the ticker never
    #   strikes a single tick. While the fallback existed this smoke passed anyway, because
    #   speech_final submitted **synchronously** inside the STT stub; with the fallback
    #   gone the only submission path hangs off the ticker, which therefore has to turn a
    #   few times (0.3s / tick 0.02 is about 15 ticks).
    # ★fake_len went from 999 to 5: 999 means "JEV#1 never passes", which was only a
    #   workable setting while the fallback existed. With it gone, JEV#1 is the only
    #   submission path and it has to genuinely pass.
    for _k, _v in dict(peer="smoke", seconds=0.3, drain=8.0, fake_jev="len", fake_len=5,
                       tick=0.02, plain=True, yes_headphones=True, tts=True,
                       vo_tick=0.02).items():
        setattr(args, _k, _v)

    saved = {k: getattr(voice_cli, k) for k in
             ("DISPATCH_DIR", "stt_client", "chain", "_spawn_reader")}
    real_build, real_open = speak_sink.build, voice_out_loop.open_jev2
    buf = io.StringIO()
    try:
        voice_cli.DISPATCH_DIR = tmp
        voice_cli.stt_client = _SttStub
        voice_cli.chain = type("C", (), {"new_voice_session": staticmethod(_new_session)})
        voice_cli._spawn_reader = lambda loop, q: None      # never touch stdin
        speak_sink.build = _build
        voice_out_loop.open_jev2 = lambda d, k=None, **kw: _FakeJev2()
        with contextlib.redirect_stdout(buf):
            async with contextlib.AsyncExitStack() as stack:
                await asyncio.wait_for(voice_cli._run(args, stack), timeout=20)
    finally:
        for k, v in saved.items():
            setattr(voice_cli, k, v)
        speak_sink.build, voice_out_loop.open_jev2 = real_build, real_open

    out = buf.getvalue()
    dumped = None
    for name in os.listdir(os.path.join(tmp, "fixtures")):
        if name.startswith("voice_") and name.endswith(".json"):
            dumped = os.path.join(tmp, "fixtures", name)
    rep = json.load(open(dumped, encoding="utf-8")) if dumped else {}
    vo = rep.get("voice_out") or {}
    gt = [t for t in (vo.get("gate_turns") or []) if t.get("opened")]
    return {"turns": len(rep.get("turns") or []),
            "spoken": (rep.get("turns") or [{}])[0].get("spoken"),
            "gate_opened": bool(gt),
            "open_reason": gt[0]["open_reason"] if gt else None,
            "report_has_gate": "· gate " in out and "first sound" in out,
            "warmups": state["warmups"], "dumped": dumped,
            "player_stopped": player.stopped, "conn_closed": state["conn_closed"],
            "stdout": out}


# ── C18: going idle -> cutting STT -> pressing enter -> reconnecting ─────────────────
async def _silence_cycle_smoke(sb, fail_at=None, tag="silence", control=None, dwell=0.0,
                               keys=None):
    """Run the whole circuit -- active -> idle -> woken -> active again -> /q -- offline.

    ★★Why this circuit **has** to have offline assertions: its three failure modes look
      identical on screen ("it is ignoring me"), and not one of them raises an error --
      ① the process does not exit after `/q` (the shutdown path forgets the sentinel, so
         the sending side of `stream()` is stuck in queue.get() forever)
      ② pressing enter does nothing (`wake_evt` was already set during the active phase,
         and nothing cleared it before the wait)
      ③ it reconnects and immediately goes back to sleep (at the moment of waking,
         `silence_ticks` still holds the previous stretch's stale reading, so it decides
         all over again that it should sleep)
      ★① and ③ were both **really written wrong** during the build; neither is
        hypothetical. Catching them in a live run means headphones, 60 seconds of waiting,
        and real money.
    Five things are stubbed: mic / STT / TTS / LLM / stdin. The gate, both loops, the
    transcript, the report and the dump are all real.
    """
    import speak_sink
    import voice_cli
    import voice_out_loop

    tmp = os.path.join(sb.tmp, "listener_" + tag)
    os.makedirs(os.path.join(tmp, "fixtures"), exist_ok=True)
    real_listener = os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "listener")
    shutil.copy(os.path.join(real_listener, "dispatch_config.json"), tmp)

    player = InstantPlayer()
    fake_tts = FakeTts(player)
    fake_tts.m = {"connect_ms": 1234, "chars_sent": 0, "errors": []}
    fake_tts.cost_usd = 0.0
    st = {"cycles": 0, "closed": 0, "mics": 0, "mics_closed": 0, "raised": 0,
          "conn_enter": 0, "conn_closed_at": None, "stale_at_wake": None}
    hold = {}

    class _Conn:
        async def __aenter__(self):
            st["conn_enter"] += 1
            return fake_tts

        async def __aexit__(self, *e):
            st["conn_closed_at"] = st["cycles"]      # ★the TTS connection has to outlive
                                                     #   the final stretch
            return False

    def _build(device=None, verbose=False):
        return _Conn(), player, (lambda t: speak_sink.SpeakSink(
            t, player, playback_lead_s=999))

    class _SttStub:
        @staticmethod
        def resolve_key(shell_root):
            return "sk-fake"

        @staticmethod
        def mic_stream(device=None, on_frame=None, on_status=None):
            st["mics"] += 1
            hold["on_frame"] = on_frame

            class _M:
                def stop(self):
                    pass

                def close(self):
                    st["mics_closed"] += 1
            return _M()

        @staticmethod
        async def stream(cfg, key, q, on_partial=None, on_final=None,
                         on_speech_final=None, on_ready=None, on_event=None):
            st["cycles"] += 1
            n = st["cycles"]
            on_ready()                                  # = the mic only opens after transcript.created
            if n == fail_at:
                # ★Simulate a **real disconnect**: this is how `websockets`
                #   ConnectionClosed surfaces. Before the fix, the consequence here was
                #   "stream() returns silently -> the program idles on with a dead ear
                #   until the time limit"; now it has to become one warning line on screen
                #   plus a wait for enter.
                st["raised"] += 1
                raise ConnectionResetError("simulated: the server closed the connection")
            ev = {"type": "transcript.partial", "is_final": False,
                  "speech_final": False, "start": 0.0, "duration": 0.2}
            talker = None
            if n == 1:
                on_partial("你好", dict(ev, text="你好"))   # one is enough: the log needs a cycle=1
            else:
                # ★Read silence_ticks once, right after reconnecting. It **should still be
                #   the large number left over from the previous stretch** (mark_voice only
                #   sets an in-memory flag; the file is not written until the next tick).
                #   This is not "expect it to be 0" -- it **pins the existence of that
                #   stale reading as a fact**, which is what makes the assertion below,
                #   "it did not go straight back to sleep because of it", mean anything.
                st["stale_at_wake"] = hold["drv"].silence.value()

                async def _keep_talking():
                    """★Somebody **keeps talking** throughout the second stretch, so it
                    should not go to sleep again.

                    Written this way so the assertion **does not depend on timing**: if
                    the second stretch were quiet too, it would rightly sleep again after
                    12 ticks, and `cycles == 2` would become a bet on whether `/q` or the
                    next silence arrives first -- the kind of assertion that eventually
                    goes red for no reason, which is worse than no assertion at all.
                    """
                    while True:
                        await asyncio.sleep(0.05)
                        on_partial("嗯", dict(ev, text="嗯"))
                talker = asyncio.create_task(_keep_talking())
            try:
                await q.get()                           # the sentinel: shared by idle /
                                                        # time limit / /q
            finally:
                if talker is not None:
                    talker.cancel()
            st["closed"] += 1
            return {"connect_ms": 10 * n, "created_ms": 9,
                    "events": {"transcript.partial": n}, "frames_sent": 5,
                    "audio_seconds": 0.5, "event_latency_ms": [10 + n], "errors": []}

    def _new_session(peer="user", sink=None, flush_kwargs=None):
        sd = os.path.join(sb.tmp, "workspace", "session_" + tag)
        os.makedirs(sd, exist_ok=True)
        drv = reply_list.VoiceDriver(FakeDispatch("在。", chunk_chars=6),
                                     sd, peer=peer, sink=sink, verbose=False)
        hold["drv"] = drv
        return drv, sd

    class _FakeJev2:
        threshold, calls, total_cost_usd, model = 0.6, 0, 0.0, "fake/jev2"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def warmup(self):
            return 12.0

        async def ask(self, s):
            self.calls += 1
            return {"noul": 0.95, "meaningful": True}

    def _fake_reader(loop, q):
        """Replaces `_spawn_reader`: never touches real stdin, and feeds keys based on
        **state** rather than on time (which would be flaky)."""
        if keys is not None:
            async def _typed():
                while st["cycles"] < 1:
                    await asyncio.sleep(0.01)
                for k in keys:
                    await asyncio.sleep(0.05)
                    q.put_nowait(k)
            asyncio.get_event_loop().create_task(_typed())
            return None
        async def _drive():
            while st["closed"] < 1:                     # ① wait for the first stretch to be cut by idleness
                await asyncio.sleep(0.01)
            # ★**Linger** inside the idle period before pressing enter. The default of 0
            #   presses immediately (which exercises the wake-up path); verifying the
            #   heartbeat during idleness needs a real pause, or the idle window lasts a
            #   few milliseconds and not one line gets printed.
            # ★After the cut, push one more frame into the mic callback: leftover frames
            #   during idleness must not pile up in a dead queue (the gate in _on_frame)
            await asyncio.sleep(dwell)
            with contextlib.suppress(Exception):
                hold["on_frame"](b"\x00" * 3200)
            q.put_nowait("")                            # ② enter = wake up
            while st["cycles"] < 2:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.75)                   # ③ let stretch 2 really survive 3 polls
            q.put_nowait("/q")                          # ④ normal shutdown (when disconnected,
                                                        #    this exits the wait for enter)
        asyncio.get_event_loop().create_task(_drive())
        return None

    args = voice_cli.build_parser().parse_args([])
    # ★tick 0.05 with a threshold of 12: the polling interval is max(0.2, tick)=0.2s, so
    #   the count has to **climb more slowly than the poll**, or the very first poll is
    #   already past the threshold before `armed` ever got to show itself -> nothing is
    #   ever cut and the test hangs on itself. 12 ticks x 0.05s is about 0.6s = the third
    #   poll hits. In production, tick=1s with a threshold of 60 satisfies this naturally.
    for _k, _v in dict(peer="silence", seconds=0, drain=8.0, fake_jev="len", fake_len=5,
                       tick=0.05, vo_tick=0.05, sleep_ticks=12, plain=True,
                       yes_headphones=True, tts=True).items():
        setattr(args, _k, _v)

    saved = {k: getattr(voice_cli, k) for k in
             ("DISPATCH_DIR", "stt_client", "chain", "_spawn_reader", "CONTROL_CONFIG")}
    if control is not None:
        # ★This substitution works because `load_control(path=None)` only consults the
        #   module global **inside the function body**. Written as `path=CONTROL_CONFIG`,
        #   the default would be bound at def time and changing it here would do nothing.
        ctl_tmp = os.path.join(tmp, "control_config.json")
        json.dump({"silence": control}, io.open(ctl_tmp, "w", encoding="utf-8"))
        voice_cli.CONTROL_CONFIG = ctl_tmp
    real_build, real_open = speak_sink.build, voice_out_loop.open_jev2
    buf = io.StringIO()
    try:
        voice_cli.DISPATCH_DIR = tmp
        voice_cli.stt_client = _SttStub
        # ★The stubbed chain has to **forward** maybe_compress rather than omit it:
        #   `/compact` goes through `chain.maybe_compress`, and the symptom of the stub
        #   class missing one attribute is the whole CASE raising AttributeError.
        #   Forwarding via a lambda (rather than binding the current function object) is
        #   what lets an outer stub of it take effect.
        import chain as _chain
        voice_cli.chain = type("C", (), {
            "new_voice_session": staticmethod(_new_session),
            "maybe_compress": staticmethod(
                lambda *a, **k: _chain.maybe_compress(*a, **k))})
        voice_cli._spawn_reader = _fake_reader
        speak_sink.build = _build
        voice_out_loop.open_jev2 = lambda d, k=None, **kw: _FakeJev2()
        with contextlib.redirect_stdout(buf):
            async with contextlib.AsyncExitStack() as stack:
                await asyncio.wait_for(voice_cli._run(args, stack), timeout=25)
    finally:
        for k, v in saved.items():
            setattr(voice_cli, k, v)
        speak_sink.build, voice_out_loop.open_jev2 = real_build, real_open

    out = buf.getvalue()
    dumped = None
    for name in sorted(os.listdir(os.path.join(tmp, "fixtures"))):
        if name.startswith("voice_") and name.endswith(".json"):
            dumped = os.path.join(tmp, "fixtures", name)
    rep = json.load(open(dumped, encoding="utf-8")) if dumped else {}
    return {"raised": st["raised"], "errors": (rep.get("metrics") or {}).get("errors"),
            "cycles": st["cycles"], "closed": st["closed"], "mics": st["mics"],
            "mics_closed": st["mics_closed"], "conn_enter": st["conn_enter"],
            "conn_closed_at": st["conn_closed_at"], "stale_at_wake": st["stale_at_wake"],
            "said_sleep": "gone idle" in out, "said_enter": "press enter to reconnect STT" in out,
            "said_sleep_only": ("gone idle" in out) and ("we did not cut it" not in out),
            "said_woke": "woken" in out, "said_dropped": "we did not cut it" in out,
            "sessions": (rep.get("metrics") or {}).get("stt_sessions"),
            "connect_all": (rep.get("metrics") or {}).get("connect_ms_all"),
            "audio_s": (rep.get("metrics") or {}).get("audio_seconds"),
            "cycles_in_rows": sorted({r.get("cycle") for r in (rep.get("rows") or [])}),
            "dumped": dumped, "out": out}


async def c18(sb):
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import voice_cli
    src = io.open(os.path.join(SHELL_ROOT, "voice_cli.py"), encoding="utf-8").read()

    # ── 23a The idle thresholds really are read from control_config.json (not supplied by
    #    a code default) ──
    ctl_path = os.path.join(SHELL_ROOT, "control_config.json")
    on_disk = json.load(open(ctl_path, encoding="utf-8"))
    got = voice_cli.load_control(ctl_path)
    ok("23a", "★the idle thresholds are read from the silence block of `control_config.json` "
              "at the shell root (the designated place to manage them), and annotation keys "
              "beginning with `_` are not treated as parameters",
       got["sleep_after_ticks"] == on_disk["silence"]["sleep_after_ticks"]
       and got["announce_every_ticks"] == on_disk["silence"]["announce_every_ticks"]
       and not any(k.startswith("_") for k in got),
       f"read={got} · on disk={on_disk['silence']['sleep_after_ticks']}")

    # ── 23b A bad or missing config must **say so**; it may not silently revert to the
    #    defaults ("the config did not take effect" is this module's oldest trap) ──
    bad = os.path.join(sb.tmp, "bad_control.json")
    io.open(bad, "w", encoding="utf-8").write("{ this is not json")
    b1 = io.StringIO()
    with contextlib.redirect_stdout(b1):
        r_bad = voice_cli.load_control(bad)
        r_none = voice_cli.load_control(os.path.join(sb.tmp, "no_such_file.json"))
    noise = b1.getvalue()
    ok("23b", "★bad JSON / a missing file -> fall back to the built-in defaults, but "
              "**print a line saying so** (a silent fallback means \"I changed the config "
              "and nothing happened\" with nothing to find, which this module has already "
              "been caught by)",
       r_bad == voice_cli.DEFAULT_CONTROL and r_none == voice_cli.DEFAULT_CONTROL
       and noise.count("built-in defaults") == 2,
       f"bad JSON={r_bad} · missing file={r_none} · notice lines="
       f"{noise.count('built-in defaults')}")

    # ── 23c All three shutdown paths share one sentinel exit (miss one and the process
    #    never exits on that path) ──
    n_cut = len([ln for ln in src.splitlines()
                 if "_cut_stt()" in ln and not ln.strip().startswith("#")])
    ok("23c", "★cutting STT has exactly one exit, `_cut_stt()`, and all three shutdown "
              "paths (idle / `/q` / the `--seconds` limit) call it once each -- on any path "
              "that forgets, `stream()` stays stuck in queue.get() forever, which looks "
              "like \"I typed /q and the process will not exit\"",
       n_cut == 4 and "await q.put(None)" in src,   # 1 definition + 3 call sites
       f"_cut_stt appears on {n_cut} lines (1 definition + 3 calls)")

    # ── 23d ★The whole circuit: active -> idle cuts STT -> enter -> reconnect -> /q ──
    g = await _silence_cycle_smoke(sb)
    ok("23d", "★★the whole circuit runs offline: the first stretch is cut by idleness -> "
              "the screen prints \"gone idle\" and \"press enter to reconnect STT\" -> "
              "enter really does reconnect into a second stretch -> `/q` really does shut "
              "it down; the mic is closed in both active stretches, and the TTS connection "
              "**survives across the idle period** (it is the one with the 1.4-2.9s "
              "handshake)",
       g["cycles"] == 2 and g["closed"] == 2 and g["mics"] == 2
       and g["mics_closed"] == 2 and g["conn_enter"] == 1
       and g["said_sleep"] and g["said_enter"] and g["said_woke"]
       and not g["said_dropped"] and g["dumped"],
       f"{g['cycles']} active stretches · {g['closed']} shutdowns · mic opened/closed="
       f"{g['mics']}/{g['mics_closed']} · TTS connection opened {g['conn_enter']} time(s) "
       f"(must be 1) · idle notice={g['said_sleep']} enter prompt={g['said_enter']} "
       f"woken={g['said_woke']} · false disconnect report={g['said_dropped']} · "
       f"written={os.path.basename(g['dumped'] or '')}")

    # ── 23e ★The stale reading at the moment of waking **exists**, and it did not cause
    #    "reconnect and immediately sleep" ──
    ok("23e", "★★waking goes through `mark_voice()` (which writes no file, discipline 4), so "
              "at the moment of reconnecting `silence_ticks` **is still the previous "
              "stretch's large number** -- this assertion pins that stale reading down as "
              "a fact and proves the `armed` gate held it back: the second stretch was not "
              "cut again on the spot (which would have produced a third stretch)",
       g["stale_at_wake"] is not None and g["stale_at_wake"] >= 12
       and g["cycles"] == 2 and 'if not ear["armed"]:' in src,
       f"stale reading at reconnect={g['stale_at_wake']} (threshold 12) · total stretches="
       f"{g['cycles']} (cut again on the spot would give >= 3)")

    # ── 23f Metrics aggregate across stretches, and each stretch stays identifiable (a run
    #    now has N stretches, not 1) ──
    ok("23f", "★the closing report aggregates N active stretches into one set of books: "
              "stretch count / per-stretch handshake / audio seconds summed, and the "
              "word-by-word log carries a `cycle` key (without it, \"was the second "
              "stretch really listening\" cannot be checked at all)",
       g["sessions"] == 2 and g["connect_all"] == [10, 20]
       and abs((g["audio_s"] or 0) - 1.0) < 1e-6
       and g["cycles_in_rows"] == [1, 2],
       f"stretches={g['sessions']} · per-stretch handshake={g['connect_all']} · "
       f"audio={g['audio_s']}s (0.5 each) · cycle numbers in the log={g['cycles_in_rows']}")

    # ── 23h ★A real disconnect: no silent idling; print a warning and wait for enter
    #    (this is what replaced the old trap) ──
    d2 = await _silence_cycle_smoke(sb, fail_at=2, tag="dropped")
    ok("23h", "★★the second stretch's STT **raises and disconnects** (not a cut of ours) -> "
              "the process survives, the screen prints \"STT dropped -- we did not cut "
              "it\" with the errors laid out, and then waits for enter; before the fix the "
              "consequence here was \"stream() returns silently -> idle on with a dead ear "
              "until the time limit\", with the queue still growing and nobody consuming it",
       d2["raised"] == 1 and d2["said_dropped"] and not d2["said_sleep_only"]
       and d2["dumped"] and any(e.get("type") == "_stream_raised"
                                for e in (d2["errors"] or [])),
       f"raised {d2['raised']} time(s) · printed the disconnect warning={d2['said_dropped']} "
       f"· errors written={d2['errors']} · stretches={d2['cycles']}")

    # ── 23i ★The heartbeat during idleness really prints (its entire purpose is to prove
    #    the process has not hung) ──
    d3 = await _silence_cycle_smoke(sb, tag="beat", dwell=0.9,
                                    control={"sleep_after_ticks": 12,
                                             "announce_every_ticks": 3})
    beats = d3["out"].count("· idle · silence_ticks=")
    ok("23i", "★while idle, print a \"· idle · silence_ticks=N\" line every "
              "`announce_every_ticks` ticks, with the reading **still climbing** (counting "
              "does not stop during idleness, by design). This is not decoration: an idle "
              "period can last hours, and with nothing on the terminal \"asleep\" and "
              "\"the process has hung\" look identical",
       beats >= 2 and d3["said_sleep"] and d3["cycles"] == 2,
       f"{beats} heartbeat lines · stretches={d3['cycles']} · sample="
       f"{[l.strip() for l in d3['out'].splitlines() if '· idle ·' in l][:3]}")

    # ── 23g No time limit by default (the terminal stays up indefinitely) ──
    d = voice_cli.build_parser().parse_args([])
    ok("23g", "★`--seconds` defaults to 0 = no time limit (the terminal stays up, and normal "
              "shutdown goes through /q). The limit exists as a **spend fuse**, not as a "
              "shutdown path",
       d.seconds == 0 and d.sleep_ticks is None,
       f"--seconds default={d.seconds} · --sleep-ticks default={d.sleep_ticks}")


# ── C19: the retention window is a token budget + compression lands without losing blocks ──
def _mk_blocks(peer, n, body):
    """Make n verbatim blocks (identical body each, so boundaries can be worked back out
    from est_tokens)."""
    for i in range(1, n + 1):
        assemble_pool.append_thread(peer, "in" if i % 2 else "out", i, f"{body}{i:03d}")


async def c19(sb):
    import chain as chain_mod

    # ── 24a The retention window is computed from a token budget, not a block count ──
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 300, "min_keep_blocks": 2})
    _mk_blocks("u1", 20, "今天天气不错我们出去走走吧" * 2)      # about 30 tokens per block
    blks = assemble_pool._split_blocks(assemble_pool._read(assemble_pool.thread_path("u1")))[1]
    keep = assemble_pool.keep_blocks_for("u1")
    # ★What is asserted is an **invariant**, not a guessed block count: the headers differ
    #   in length (digits in the step number, in/out), so blocks are not exactly equal in
    #   tokens, and an assertion that guesses an exact value goes red for unrelated reasons.
    got = sum(assemble_pool.est_tokens(b) for b in blks[-keep:])
    one_more = got + assemble_pool.est_tokens(blks[-keep - 1])
    ok("24a", "★the retention window (the part that is never compressed and is injected "
              "every turn) takes the most recent blocks against a **token budget**, not a "
              "block count -- because one turn is either 1 or 2 blocks (`_persist` has two "
              "independent ifs), so \"a turn\" is not well defined at the storage layer. "
              "The invariant: fit one more in whenever it fits, and one beyond that busts "
              "the budget",
       0 < keep < len(blks) and got <= 300 and one_more > 300,
       f"{keep} of 20 blocks retained = {got} tok <= 300; one more = {one_more} tok > 300")

    # ── 24b min_keep_blocks is a genuine floor: hold on even when a single block busts
    #    the budget ──
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 50, "min_keep_blocks": 6})
    _mk_blocks("u2", 4, "很长的一段独白" * 60)                  # one block alone is ~300 tok >> 50
    keep2 = assemble_pool.keep_blocks_for("u2")
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 50, "min_keep_blocks": 2})
    keep2b = assemble_pool.keep_blocks_for("u2")
    ok("24b", "★`min_keep_blocks` is a floor, not decoration: when a single block far "
              "exceeds the budget (one long monologue), a pure token budget computes 0 "
              "blocks -- not even the sentence just spoken gets injected, which in the "
              "room looks like \"it forgot what I just said\"",
       keep2 == 4 and keep2b == 2,
       f"all 4 blocks bust the budget: floor 6 -> {keep2} retained (only 4 exist) · "
       f"floor 2 -> {keep2b} retained")

    # ── 24c ★★New blocks arrive during compression and not one may be lost (stage and
    #    place have to line up on n_older) ──
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 130, "min_keep_blocks": 2})
    _mk_blocks("u3", 10, "一二三四五六七八九十" * 2)             # ~27 tok per block -> 4 retained
    before = assemble_pool._split_blocks(
        assemble_pool._read(assemble_pool.thread_path("u3")))[1]
    n_older = assemble_pool.stage_thread_input(sb.session_dir, "u3")
    # ★The compressor is running (a few seconds)... and meanwhile the main thread has two
    #   more turns:
    assemble_pool.append_thread("u3", "in", 11, "压缩跑着的时候我又说了一句")
    assemble_pool.append_thread("u3", "out", 12, "这是机器的回答")
    assemble_pool.place_thread_compression("u3", "【摘要】前面聊了十段。", n_older)
    after = assemble_pool._read(assemble_pool.thread_path("u3"))
    # the blocks that were inside the retention window when it was staged (and so are not
    # in the summary) must still be on disk
    staged = assemble_pool._read(os.path.join(sb.session_dir, "compress_input.md"))
    in_summary = [b for b in before if b.split("\n")[0] in staged]
    kept_verbatim = [b for b in before if b.split("\n")[0] in after]
    lost = [b.split("\n")[0] for b in before
            if b.split("\n")[0] not in staged and b.split("\n")[0] not in after]
    ok("24c", "★★compression runs on a worker thread and takes seconds, during which the "
              "main thread may append new blocks. If landing it **recomputes** the boundary "
              "as \"keep the last keep blocks\", the blocks that were inside the window at "
              "staging time end up **neither in the summary nor in the retained verbatim "
              "text** -- history silently lost. So place locates by block count via "
              "`n_older` and recomputes nothing.",
       not lost and len(in_summary) == n_older
       and "step 11" in after and "step 12" in after,
       f"compressed the first {n_older} blocks · {len(kept_verbatim)} kept verbatim · "
       f"lost={lost or 'none'} · steps 11/12 added during compression both present="
       f"{('step 11' in after) and ('step 12' in after)}")

    # ── 24d Archive the original before overwriting (the thread file is the only store of
    #    the verbatim words) ──
    arch = sorted(os.listdir(assemble_pool.archive_dir()))
    arch_text = assemble_pool._read(os.path.join(assemble_pool.archive_dir(), arch[-1])) if arch else ""
    ok("24d", "★compression is a **lossy overwrite**, and the thread file is the only store "
              "of the verbatim words (`fixtures/voice_*.jsonl` holds only the STT-side "
              "recognition log, nothing the machine said). So the whole pre-overwrite file "
              "is copied into `threads/_archive/` before landing",
       len(arch) == 1 and all(b.split("\n")[0] in arch_text for b in before)
       and "_archive/" in after,
       f"archived {arch} · all 10 pre-overwrite blocks are in the archive="
       f"{all(b.split(chr(10))[0] in arch_text for b in before)} · the summary records the "
       f"archive path={'_archive/' in after}")

    # ── 24e _archive/ must not be mistaken for a peer (it lives under threads/) ──
    ok("24e", "★`_archive/` lives under `threads/`, and `active_peers()` must not treat it "
              "as a peer (otherwise compression goes off to compress a conversation that "
              "does not exist, and the pool's token accounting double-counts)",
       "_archive" not in assemble_pool.active_peers(),
       f"active_peers={assemble_pool.active_peers()}")

    # ── 24f The config is read from the memory block of control_config.json at the shell
    #    root, and an unrecognised key has to be shouted about ──
    io.open(assemble_pool.CONTROL_CONFIG, "w", encoding="utf-8").write(json.dumps(
        {"silence": {"sleep_after_ticks": 60},
         "memory": {"keep_recent_tokens": 777, "min_keep_blocks": 3, "mistyped_key": 1}}))
    assemble_pool._warned.clear()
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        cfg = assemble_pool.load_config()
        assemble_pool.load_config()          # the second call must not shout again
    noise = b.getvalue()
    ok("24f", "★compression parameters are read from the `memory` block of "
              "`control_config.json` at the shell root (the same file as the idle "
              "thresholds, the designated place to manage them); the `silence` block does "
              "not interfere; an unrecognised key **is shouted about once, and only once** "
              "(ignoring it silently means \"I changed the config and nothing happened\" "
              "with nothing to find)",
       cfg["keep_recent_tokens"] == 777 and cfg["min_keep_blocks"] == 3
       and noise.count("mistyped_key") == 1,
       f"read keep={cfg['keep_recent_tokens']} / floor={cfg['min_keep_blocks']} · "
       f"warned {noise.count('mistyped_key')} time(s) (must be 1)")

    # ── 24g ★Compression must not block the event loop: the hook is a coroutine and pump
    #    awaits it ──
    src = io.open(os.path.join(SHELL_ROOT, "IAF_AI", "chain.py"), encoding="utf-8").read()
    marks = []

    async def _hook(rec):
        await asyncio.sleep(0)
        marks.append("ran")
        return {"compressed": True}

    drv = _driver(sb, FakeDispatch("好。"), SilentSink())
    drv.on_turn_end = _hook
    drv.reservoir.add("你好")
    rec = await drv.pump()
    ok("24g", "★★`maybe_compress` -> `run_role` is a **blocking** LLM call, and running it "
              "directly inside pump jams the event loop (for those seconds nobody consumes "
              "mic frames and the STT receive side stops turning). So `chain._after_turn` "
              "became a coroutine plus `asyncio.to_thread`, and `pump` must **await the "
              "awaitable it returns** -- without the await, `rec['post']` holds a coroutine "
              "that never ran, compression simply does not happen, and the only trace is "
              "one RuntimeWarning at process exit",
       marks == ["ran"] and rec["post"] == {"compressed": True}
       and "async def _after_turn" in src and "asyncio.to_thread(maybe_compress" in src,
       f"the hook really ran={marks} · rec['post']={rec['post']!r} (not a coroutine object) "
       f"· it is a coroutine in chain={'async def _after_turn' in src} · "
       f"goes through to_thread={'asyncio.to_thread(maybe_compress' in src}")


# ── C20: the /compact command (a peer of /q) ────────────────────────────────────────
async def c20(sb):
    import chain as chain_mod
    import voice_cli

    # ── 25a forced_plan ignores the threshold but **still** skips a peer with no older
    #    blocks ──
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 130, "min_keep_blocks": 2,
                                         "thread_trigger_tokens": 999999})
    _mk_blocks("p_old", 10, "一二三四五六七八九十" * 2)     # 10 blocks, window fits 4 -> older blocks exist
    _mk_blocks("p_new", 2, "短")                            # 2 blocks, all inside the window -> none older
    auto = assemble_pool.compression_plan()
    forced = assemble_pool.forced_plan()
    ok("25a", "★`/compact`'s plan ignores the threshold (it must still compress with the "
              "threshold set to 999999), but it **must** skip a peer with no older blocks "
              "-- asked to summarise an empty input the LLM invents something, which then "
              "lands as a real summary and replaces the original words (`maybe_compress`'s "
              "\"an empty output may not be written\" rule only catches an empty return, "
              "not an input that was empty to begin with)",
       auto["threads"] == [] and forced["threads"] == ["p_old"] and forced["pool"] is False,
       f"threshold 999999 -> automatic={auto['threads']} · forced={forced['threads']} "
       f"(p_new has only 2 blocks, all inside the window, and must not be included) · "
       f"force-compress pool={forced['pool']} (left alone by default)")

    # ── 25b Manual and automatic share one implementation (only the plan's source differs) ──
    csrc = io.open(os.path.join(SHELL_ROOT, "IAF_AI", "chain.py"), encoding="utf-8").read()
    vsrc = io.open(os.path.join(SHELL_ROOT, "voice_cli.py"), encoding="utf-8").read()
    ok("25b", "★manual compression goes through `chain.maybe_compress(session_dir, plan)` "
              "-- the **same implementation** as automatic compression, with only the "
              "plan's source swapped. Write a second one and its archiving and landing "
              "behaviour eventually drifts away from the automatic path",
       "def maybe_compress(session_dir, plan=None):" in csrc
       and "chain.maybe_compress, session_dir, plan" in vsrc
       and "asyncio.to_thread(chain.maybe_compress" in vsrc,
       f"chain accepts a plan={'def maybe_compress(session_dir, plan=None):' in csrc} · "
       f"voice_cli uses to_thread and the same function="
       f"{'asyncio.to_thread(chain.maybe_compress' in vsrc}")

    # ── 25c ★Walk the whole /compact path (a real keypress -> a real landing -> a real
    #    archive), with the LLM stubbed ──
    calls = []

    def _fake_compress(session_dir, plan=None):
        """Only the **single LLM call** is stubbed: landing, archiving and the n_older
        alignment are all real."""
        calls.append(dict(plan or {}))
        rec = {"plan": plan, "ran": []}
        for peer in (plan or {}).get("threads", []):
            n_older = assemble_pool.stage_thread_input(session_dir, peer)
            assemble_pool.place_thread_compression(
                peer, f"[summary] the first {n_older} blocks of {peer} are compressed.", n_older)
            rec["ran"].append({"thread": peer, "blocks": n_older})
        return rec

    real_mc = chain_mod.maybe_compress
    saved_vc = voice_cli.chain
    try:
        chain_mod.maybe_compress = _fake_compress
        g = await _silence_cycle_smoke(
            sb, tag="compact", dwell=0.0,
            control={"sleep_after_ticks": 999999, "announce_every_ticks": 0},
            keys=["/m", "/compact", "/q"])
    finally:
        chain_mod.maybe_compress = real_mc
        voice_cli.chain = saved_vc
    out = g["out"]
    ok("25c", "★★the whole `/compact` path works: the command is recognised -> a before/after "
              "reconciliation is printed (total length / block count / ★what is injected "
              "per turn) -> it really lands -> it really archives; `/m` prints the current "
              "state of memory; and a `/q` immediately afterwards **does not cut the "
              "compression short** (the commander processes in order and /compact is "
              "awaited)",
       "/compact starting" in out and "injected per turn" in out
       and "memory · retention window" in out
       and "unknown command" not in out and len(calls) == 1
       and calls[0].get("_debug", {}).get("forced") is True,
       f"recognised={'/compact starting' in out} · reconciliation line="
       f"{'injected per turn' in out} · /m={'memory · retention window' in out}"
       f" · compression called {len(calls)} time(s) · the plan is forced="
       f"{calls[0].get('_debug', {}).get('forced') if calls else None}")

    # ── 25e ★★When the only thing outside the window is last time's summary, compress it
    #    anyway -- this is the specified design; do not "fix" it as a bug ──
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 130, "min_keep_blocks": 2,
                                         "thread_trigger_tokens": 999999})
    _mk_blocks("p_conv", 10, "一二三四五六七八九十" * 2)
    n1 = assemble_pool.stage_thread_input(sb.session_dir, "p_conv")
    assemble_pool.place_thread_compression("p_conv", "[first summary] ten earlier exchanges.", n1)
    k2 = assemble_pool.keep_blocks_for("p_conv")
    _h2, b2 = assemble_pool._split_blocks(
        assemble_pool._read(assemble_pool.thread_path("p_conv")))
    older2 = b2[:max(0, len(b2) - k2)]
    again = assemble_pool.forced_plan()["threads"]
    ok("25e", "★★after one round of compression, the only thing outside the window is LAST "
              "TIME'S SUMMARY -- and `/compact` **must still be able to compress it**. "
              "Repeatedly re-compressing a summary is the specified **convergence "
              "mechanism** (the older something is the more abstract it becomes, with the "
              "length held by the prompt's character cap). It is **not a leak**. A gate "
              "saying \"skip when only a summary is left\" was added here once and "
              "rejected outright -- this assertion is what stops it coming back",
       len(older2) == 1
       and assemble_pool._COMPRESSED_HEAD_RE.match(older2[0])
       and "p_conv" in again,
       f"{len(older2)} block(s) outside the window · that block is a compression header="
       f"{bool(assemble_pool._COMPRESSED_HEAD_RE.match(older2[0])) if older2 else None}"
       f" · the forced plan still includes it={('p_conv' in again)} (must be True)")

    # ── 25d With nothing older to compress, /compact must say so rather than silently
    #    doing nothing ──
    assemble_pool.DEFAULT_CONFIG.update({"keep_recent_tokens": 100000, "min_keep_blocks": 2})
    empty_forced = assemble_pool.forced_plan()
    ok("25d", "★when the retention window is large enough to hold the entire conversation, "
              "`/compact` has to **say** \"nothing old to compress\" and print the current "
              "state -- doing nothing silently just reads as \"the command had no effect\"",
       empty_forced["threads"] == []
       and "nothing old to compress" in vsrc and "print_memory(session_dir)" in vsrc,
       f"window 100000 tok -> forced plan={empty_forced['threads']} · the notice exists in "
       f"voice_cli={'nothing old to compress' in vsrc}")


# ── C21: the second baseline for going idle (time spent with the machine talking must not
#        be judged as "nobody is speaking") ────────────────────────────────────────────
class HeldPlayer(InstantPlayer):
    """A stub player whose **playback progress is released by the test** (`release()`).

    ★★Why holding it is essential: `SpeakSink.settle()` exits when "every byte fed in has
      finished playing", so on an offline bench the state "the machine is still talking"
      can **only** be produced by holding playback progress back. With `InstantPlayer`
      anything fed counts as played -> a turn ends instantly -> the assertion "do not cut
      the mic while speaking" becomes a bet on timing, and a timing bet eventually goes
      red for no reason, which is worse than having no assertion (discipline 1 of this
      file).
    """

    def feed(self, pcm):
        self._fed += len(pcm)
        if self.t_first_play is None:
            self.t_first_play = time.time()

    def release(self):
        """Release -> `settle()`'s while loop exits -> this turn ends -> `drv.busy` falls
        back to False."""
        self._played = self._fed


async def _speak_grace_smoke(sb, *, sleep_ticks=20, tick=0.05):
    """Run offline: the machine talks past the threshold -> no cut; it stops -> no cut
    while the grace period is unfinished; only once it is finished is there a cut.

    ★★The value of this offline assertion is that the scene it tests **costs 60 seconds and
      real money to reproduce live**, and its failure mode looks like "it is ignoring me"
      on screen: the machine talks for 59 seconds, stops, and the mic is cut before the
      person can get a word in.
    Five external IOs are stubbed: mic / STT / TTS / LLM / stdin. The gate, both loops, the
    ticker and the criteria are all real.
    """
    import speak_sink
    import voice_cli
    import voice_out_loop

    tmp = os.path.join(sb.tmp, "listener_grace")
    os.makedirs(os.path.join(tmp, "fixtures"), exist_ok=True)
    real_listener = os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "listener")
    cfg = json.load(io.open(os.path.join(real_listener, "dispatch_config.json"),
                            encoding="utf-8"))
    # ★Two knobs are turned down. **Neither has anything to do with the criterion under
    #   test**; they exist purely so the assertion does not bet on timing:
    #   ① `llm_min_silence_s` 1.0 -> 0.1: the threshold/grace here is only 1.0s
    #      (20 ticks x 0.05s). Left alone, "the machine starts speaking" and "the tick
    #      count crosses the threshold" collide at the same instant, leaving about 0.45s
    #      of margin; turned down it is about 0.85s.
    #   ② `jev_trim_units` 1 -> 0: U holds a single partial unit, so trimming one leaves
    #      an empty string, the fake verdict (which counts characters) can never judge it
    #      sufficient, this turn never starts, and the test spins until it times out.
    cfg["knobs"]["llm_min_silence_s"] = 0.1
    cfg["knobs"]["jev_trim_units"] = 0
    json.dump(cfg, io.open(os.path.join(tmp, "dispatch_config.json"), "w",
                           encoding="utf-8"))

    grace_s = sleep_ticks * tick
    player = HeldPlayer()
    fake_tts = FakeTts(player)
    fake_tts.m = {"connect_ms": 1234, "chars_sent": 0, "errors": []}
    fake_tts.cost_usd = 0.0
    st = {"cycles": 0, "closed": 0}
    hold = {}
    obs = {"busy_seen": False, "max_ticks_busy": 0, "closed_while_busy": 0,
           "t_release": None, "t_cut": None, "closed_at_half_grace": None,
           "ticks_at_cut": None}

    class _Conn:
        async def __aenter__(self):
            return fake_tts

        async def __aexit__(self, *e):
            return False

    def _build(device=None, verbose=False):
        return _Conn(), player, (lambda t: speak_sink.SpeakSink(
            t, player, playback_lead_s=999))

    class _SttStub:
        @staticmethod
        def resolve_key(shell_root):
            return "sk-fake"

        @staticmethod
        def mic_stream(device=None, on_frame=None, on_status=None):
            class _M:
                def stop(self):
                    pass

                def close(self):
                    pass
            return _M()

        @staticmethod
        async def stream(cfg_, key, q, on_partial=None, on_final=None,
                         on_speech_final=None, on_ready=None, on_event=None):
            st["cycles"] += 1
            on_ready()
            # ★The person says this one line and then **never makes another sound**: every
            #   subsequent rise in the tick count can therefore only come from "the machine
            #   itself is talking" -- which is precisely the cause under test.
            # ★★`duration` has to **match the text length**: 6 characters over 0.2s is 30
            #   characters a second, which is an event real STT cannot produce -- the
            #   hallucination gate would take it for a fabrication and block it on the
            #   spot, so `mark_voice` never happens, the idle tick count never resets, and
            #   this smoke goes red as "the mic was cut while speaking". 2.2s is about 2.7
            #   characters a second, right on the measured median of 2.78.
            #   ★The lesson: stub data has to be **physically plausible** too, or what it
            #   exercises is a path that does not exist in reality.
            on_partial("你好你好你好", {"type": "transcript.partial", "text": "你好你好你好",
                                        "is_final": False, "speech_final": False,
                                        "start": 0.0, "duration": 2.2})
            await q.get()                       # the sentinel: idle and /q share one
                                                # shutdown path
            st["closed"] += 1
            return {"connect_ms": 10, "created_ms": 9, "events": {}, "frames_sent": 1,
                    "audio_seconds": 0.1, "event_latency_ms": [10], "errors": []}

    def _new_session(peer="user", sink=None, flush_kwargs=None):
        sd = os.path.join(sb.tmp, "workspace", "session_grace")
        os.makedirs(sd, exist_ok=True)
        drv = reply_list.VoiceDriver(FakeDispatch("在。", chunk_chars=6),
                                     sd, peer=peer, sink=sink, verbose=False)
        hold["drv"] = drv
        return drv, sd

    class _FakeJev2:
        threshold, calls, total_cost_usd, model = 0.6, 0, 0.0, "fake/jev2"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def warmup(self):
            return 12.0

        async def ask(self, s):
            self.calls += 1
            return {"noul": 0.95, "meaningful": True}

    def _fake_reader(loop, q):
        async def _observe():
            while hold.get("drv") is None:
                await asyncio.sleep(0.01)
            drv = hold["drv"]
            t_start = time.monotonic()
            # ── ① While the machine is talking: the tick count climbs past the
            #    threshold, and there must not be a single cut ──
            while obs["t_release"] is None and time.monotonic() - t_start < 15:
                await asyncio.sleep(0.01)
                if not drv.busy:
                    continue
                obs["busy_seen"] = True
                obs["max_ticks_busy"] = max(obs["max_ticks_busy"], drv.silence.value())
                obs["closed_while_busy"] = max(obs["closed_while_busy"], st["closed"])
                if obs["max_ticks_busy"] >= sleep_ticks + 3:
                    obs["t_release"] = time.monotonic()
                    player.release()            # ★release -> settle returns -> the machine stops
            if obs["t_release"] is None:         # could not produce "while talking" -> let
                                                 # the assertion go red rather than hang
                player.release()
                q.put_nowait("/q")
                return
            # ── ② After it stops, while the grace period is unfinished: still no cut ──
            await asyncio.sleep(grace_s * 0.5)
            obs["closed_at_half_grace"] = st["closed"]
            # ── ③ Only once the grace period is finished is there a cut ──
            while st["closed"] < 1 and time.monotonic() - obs["t_release"] < 15:
                await asyncio.sleep(0.01)
            obs["t_cut"] = time.monotonic()
            obs["ticks_at_cut"] = drv.silence.value()
            q.put_nowait("/q")                   # leave the "wait for enter" state and shut down

        asyncio.get_event_loop().create_task(_observe())
        return None

    args = voice_cli.build_parser().parse_args([])
    for _k, _v in dict(peer="grace", seconds=0, drain=8.0, fake_jev="len", fake_len=5,
                       tick=tick, vo_tick=tick, sleep_ticks=sleep_ticks, plain=True,
                       yes_headphones=True, tts=True).items():
        setattr(args, _k, _v)

    saved = {k: getattr(voice_cli, k) for k in
             ("DISPATCH_DIR", "stt_client", "chain", "_spawn_reader")}
    real_build, real_open = speak_sink.build, voice_out_loop.open_jev2
    buf = io.StringIO()
    try:
        voice_cli.DISPATCH_DIR = tmp
        voice_cli.stt_client = _SttStub
        import chain as _chain
        voice_cli.chain = type("C", (), {
            "new_voice_session": staticmethod(_new_session),
            "maybe_compress": staticmethod(
                lambda *a, **k: _chain.maybe_compress(*a, **k))})
        voice_cli._spawn_reader = _fake_reader
        speak_sink.build = _build
        voice_out_loop.open_jev2 = lambda d, k=None, **kw: _FakeJev2()
        with contextlib.redirect_stdout(buf):
            async with contextlib.AsyncExitStack() as stack:
                await asyncio.wait_for(voice_cli._run(args, stack), timeout=40)
    finally:
        for k, v in saved.items():
            setattr(voice_cli, k, v)
        speak_sink.build, voice_out_loop.open_jev2 = real_build, real_open

    out = buf.getvalue()
    obs["grace_s"] = grace_s
    obs["closed"] = st["closed"]
    obs["held_s"] = (None if obs["t_cut"] is None or obs["t_release"] is None
                     else obs["t_cut"] - obs["t_release"])
    obs["said_sleep"] = "gone idle" in out
    obs["out"] = out
    return obs


async def c21(sb):
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import voice_cli
    src = io.open(os.path.join(SHELL_ROOT, "voice_cli.py"), encoding="utf-8").read()

    got = await _speak_grace_smoke(sb)

    # ── 28a The tick count climbs past the threshold while the machine is talking, and
    #    there must not be a single cut ──
    ok("28a", "★★while the machine is talking, `silence_ticks` climbs past the threshold as "
              "usual (the ticker and pump are two legs of one gather and neither misses a "
              "beat) -- but during that time **STT must not be cut even once**. The scene "
              "if this is not held: the machine talks for 59 seconds, stops, and the mic "
              "is cut before the person can get a word in -- and it takes about 1.6s from "
              "someone starting to speak to the first partial landing, so they cannot "
              "possibly squeeze into a window of one second or less",
       got["busy_seen"] and got["max_ticks_busy"] >= 20 and got["closed_while_busy"] == 0,
       f"highest tick count seen while talking={got['max_ticks_busy']} (threshold 20, must "
       f"be >=) · cuts while talking={got['closed_while_busy']} (must be 0)")

    # ── 28b After it stops, while the grace period is unfinished, still no cut ──
    ok("28b", "★after it stops, allow a full grace period (= `sleep_after_ticks x "
              "tick_interval_s`) before even considering sleep: halfway through the grace "
              "period there is **still no cut**. That stretch is \"the person has finished "
              "listening and it is their turn to speak\" -- the clock on that starts here",
       got["closed_at_half_grace"] == 0,
       f"grace={got['grace_s']}s · cuts by the halfway point="
       f"{got['closed_at_half_grace']} (must be 0)")

    # ── 28c Only once the grace period is finished does the cut really happen ──
    ok("28c", "★the cut happens only after the grace period is finished, and the screen "
              "still prints \"gone idle\" (this path gained no extra exit and lost none -- "
              "it is still the one `_cut_stt()`)",
       got["closed"] == 1 and got["said_sleep"]
       and got["held_s"] is not None and got["held_s"] >= got["grace_s"] * 0.9,
       f"{got['held_s'] and round(got['held_s'], 2)}s between stopping and the cut >= "
       f"grace {got['grace_s']}s · {got['closed']} cut(s) · idle notice printed="
       f"{got['said_sleep']} · tick count at the cut={got['ticks_at_cut']} (far past the "
       f"threshold -- the count itself was never touched)")

    # ── 28d Do not add a fifth knob ──
    ctl = json.load(io.open(os.path.join(SHELL_ROOT, "control_config.json"),
                            encoding="utf-8"))
    sil_keys = sorted(k for k in ctl["silence"] if not k.startswith("_"))
    ok("28d", "★the length of the grace period is **not a new knob**; it is "
              "`sleep_after_ticks x tick_interval_s`. Two knobs for one concept drift "
              "apart eventually, and the symptom of the drift is \"it sleeps earlier/later "
              "than I set\", which starts with having to doubt two numbers at once. So the "
              "set of keys in `control_config.json`'s `silence` block **may not change**",
       sil_keys == ["announce_every_ticks", "sleep_after_ticks"]
       and 'grace_s = sleep_ticks * max(0.0, float(knobs["tick_interval_s"]))' in src,
       f"keys in the silence block={sil_keys} · grace is derived="
       f"{'grace_s = sleep_ticks *' in src}")

    # ── 28e The inputs to the other three criteria are untouched ──
    rl_path = os.path.join(SHELL_ROOT, "IAF_AI", "reply_list.py")
    lp_path = os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "listener",
                           "listener_loop.py")
    rl = io.open(rl_path, encoding="utf-8").read()

    def _n_refs(path, name):
        """★Counts **code references** (via ast), not greps -- the comments in these files
        mention `mark_voice` constantly (the fact that it is SilenceCounter's only input
        is itself something the comments have to say), and grep would count those too,
        leaving the assertion permanently red. Same reasoning as `_code_refs`."""
        n = 0
        for node in ast.walk(ast.parse(io.open(path, encoding="utf-8").read(),
                                       filename=path)):
            if isinstance(node, ast.Attribute) and node.attr == name:
                n += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                n += 1
        return n

    marks = (_n_refs(rl_path, "mark_voice") + _n_refs(lp_path, "mark_voice")
             + _n_refs(voice_cli.__file__, "mark_voice"))
    ok("28e", "★★`SilenceCounter` and `silence_seconds()` are untouched: `mark_voice()` is "
              "still the only input, and the number of call sites has not changed (3 STT "
              "events + 1 wake-up + 1 definition). **Those two readings continuing to "
              "climb while the machine talks is an asset, not a bug** -- the LLM's second "
              "firing condition uses it to save a second of waiting, and the mouth's upper "
              "bound uses it to say a held sentence in one go. Only one consumer, \"going "
              "idle\", is a liability, which is why the extra baseline is added there and "
              "nowhere else",
       marks == 5 and "def mark_voice" in rl
       and "cur = 0 if had_voice else self._raw() + 1" in rl,
       f"mark_voice appears {marks} times (1 definition + 3 in the ear + 1 wake-up) · "
       f"tick()'s formula is unchanged="
       f"{'cur = 0 if had_voice else self._raw() + 1' in rl}")

    # ── 28f Five pieces of wording may no longer say "nobody spoke for N ticks" ──
    lies = [ln for ln in src.splitlines()
            if ("nobody spoke for" in ln or "nobody speaks for" in ln)
            and "machine" not in ln]
    ok("28f", "★★all five pieces of wording have to say \"**after the machine finished**\": "
              "the criterion is now two conditions side by side, and \"nobody spoke for 60 "
              "consecutive ticks\" is **false** in the scene \"the machine talked for 75 "
              "seconds and the person was quiet for 60\" (it sounds like 60 seconds when "
              "135 have passed). The harm from that line is very concrete: it points the "
              "investigation at \"why is the user not speaking\". The heartbeat line has "
              "the same problem, and now prints both numbers together",
       not lies and "since anything was spoken" in src
       and "after the machine finished" in src
       and "after the machine finishes" in src
       and "when the machine stops talking" in src,
       f"lines still saying \"nobody spoke for N ticks\"={lies or 'none'} · the heartbeat "
       f"carries \"since anything was spoken\"={'since anything was spoken' in src} · the "
       f"notice carries \"after the machine finished\"={'after the machine finished' in src}")


# ── C22: the half-duplex gate (mute the mic while the machine speaks -> loudspeakers
#        become usable) ──────────────────────────────────────────────────────────────
async def _mic_gate_smoke(sb, *, tag, enabled=True, feed_silence=True, hangover_ms=400):
    """Run the mic gate's whole timeline offline: before speaking -> while speaking ->
    released but hangover unfinished -> hangover elapsed.

    ★★Why it has to be offline: a live run needs loudspeakers and real money, and **the
      symptom of a broken mic gate is "occasional feedback"** -- irregular, raising
      nothing, and impossible to pin down by ear as either "it never muted" or "the
      hangover was too short".
    ★The probe is **four frames**: each is pushed into `_on_frame` in a known state, and
      what shows up on STT's queue is inspected (unchanged / an equal-length frame of
      zeros / never arrived at all). States advance on events, never by racing a sleep.
    """
    import speak_sink
    import voice_cli
    import voice_out_loop

    tmp = os.path.join(sb.tmp, "micgate_" + tag)
    os.makedirs(os.path.join(tmp, "fixtures"), exist_ok=True)
    real_listener = os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "listener")
    cfg = json.load(io.open(os.path.join(real_listener, "dispatch_config.json"),
                            encoding="utf-8"))
    # ★As in C21: two knobs unrelated to the criterion under test are turned down, purely
    #   so this turn starts at once (the reasoning is in _speak_grace_smoke)
    cfg["knobs"]["llm_min_silence_s"] = 0.1
    cfg["knobs"]["jev_trim_units"] = 0
    json.dump(cfg, io.open(os.path.join(tmp, "dispatch_config.json"), "w",
                           encoding="utf-8"))
    ctl_tmp = os.path.join(tmp, "control_config.json")
    json.dump({"mic_gate": {"enabled": enabled, "hangover_ms": hangover_ms,
                            "feed_silence": feed_silence}},
              io.open(ctl_tmp, "w", encoding="utf-8"))

    LOUD = b"\x01" * 320
    player = HeldPlayer()
    fake_tts = FakeTts(player)
    fake_tts.m = {"connect_ms": 1234, "chars_sent": 0, "errors": []}
    fake_tts.cost_usd = 0.0
    st = {"frames": [], "closed": 0}
    hold = {}

    class _Conn:
        async def __aenter__(self):
            return fake_tts

        async def __aexit__(self, *e):
            return False

    def _build(device=None, verbose=False):
        return _Conn(), player, (lambda t: speak_sink.SpeakSink(
            t, player, playback_lead_s=999))

    class _SttStub:
        @staticmethod
        def resolve_key(shell_root):
            return "sk-fake"

        @staticmethod
        def mic_stream(device=None, on_frame=None, on_status=None):
            hold["on_frame"] = on_frame         # ★the probe goes in through this real callback

            class _M:
                def stop(self):
                    pass

                def close(self):
                    pass
            return _M()

        @staticmethod
        async def stream(cfg_, key, q, on_partial=None, on_final=None,
                         on_speech_final=None, on_ready=None, on_event=None):
            on_ready()
            # ★★`duration` has to **match the text length**: 6 characters over 0.2s is 30
            #   characters a second, which is an event real STT cannot produce -- the
            #   hallucination gate would take it for a fabrication and block it on the
            #   spot, so `mark_voice` never happens, the idle tick count never resets, and
            #   this smoke goes red as "the mic was cut while speaking". 2.2s is about 2.7
            #   characters a second, right on the measured median of 2.78.
            #   ★The lesson: stub data has to be **physically plausible** too, or what it
            #   exercises is a path that does not exist in reality.
            on_partial("你好你好你好", {"type": "transcript.partial", "text": "你好你好你好",
                                        "is_final": False, "speech_final": False,
                                        "start": 0.0, "duration": 2.2})
            while True:                         # ★really drain the queue -- this sequence
                                                #   is exactly what the assertions read
                buf = await q.get()
                if buf is None:
                    break
                st["frames"].append(bytes(buf))
            st["closed"] += 1
            return {"connect_ms": 10, "created_ms": 9, "events": {}, "frames_sent": 1,
                    "audio_seconds": 0.1, "event_latency_ms": [10], "errors": []}

    def _new_session(peer="user", sink=None, flush_kwargs=None):
        sd = os.path.join(sb.tmp, "workspace", "session_" + tag)
        os.makedirs(sd, exist_ok=True)
        drv = reply_list.VoiceDriver(FakeDispatch("在。", chunk_chars=6),
                                     sd, peer=peer, sink=sink, verbose=False)
        hold["drv"] = drv
        return drv, sd

    class _FakeJev2:
        threshold, calls, total_cost_usd, model = 0.6, 0, 0.0, "fake/jev2"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def warmup(self):
            return 12.0

        async def ask(self, s):
            self.calls += 1
            return {"noul": 0.95, "meaningful": True}

    def _fake_reader(loop, q):
        async def _drive():
            t0 = time.monotonic()
            while hold.get("on_frame") is None and time.monotonic() - t0 < 10:
                await asyncio.sleep(0.01)
            drv = hold["drv"]
            # ── ① Not speaking yet: this frame must arrive unchanged (which first proves
            #    the probe itself works) ──
            hold["on_frame"](LOUD)
            await asyncio.sleep(0.05)
            hold["n_before"] = len(st["frames"])
            # ── ② Currently speaking: `fake_tts.sent` non-empty = the first block has
            #    already passed through MicGate (so it has latched) ──
            while not fake_tts.sent and time.monotonic() - t0 < 10:
                await asyncio.sleep(0.005)
            hold["on_frame"](LOUD)
            await asyncio.sleep(0.05)
            # ── ③ Release playback -> settle returns -> `busy` falls back to False, but
            #    the hangover has not elapsed ──
            player.release()
            while drv.busy and time.monotonic() - t0 < 10:
                await asyncio.sleep(0.005)
            hold["on_frame"](LOUD)
            await asyncio.sleep(0.05)
            # ── ④ The hangover elapses: the mic must come back ──
            await asyncio.sleep(hangover_ms / 1000.0 + 0.25)
            hold["on_frame"](LOUD)
            await asyncio.sleep(0.05)
            q.put_nowait("/q")
        asyncio.get_event_loop().create_task(_drive())
        return None

    args = voice_cli.build_parser().parse_args([])
    for _k, _v in dict(peer="micgate", seconds=0, drain=8.0, fake_jev="len", fake_len=5,
                       tick=0.05, vo_tick=0.05, sleep_ticks=0, plain=True,
                       yes_headphones=True, tts=True).items():
        setattr(args, _k, _v)

    saved = {k: getattr(voice_cli, k) for k in
             ("DISPATCH_DIR", "stt_client", "chain", "_spawn_reader", "CONTROL_CONFIG")}
    real_build, real_open = speak_sink.build, voice_out_loop.open_jev2
    buf = io.StringIO()
    try:
        voice_cli.DISPATCH_DIR = tmp
        voice_cli.CONTROL_CONFIG = ctl_tmp
        voice_cli.stt_client = _SttStub
        import chain as _chain
        voice_cli.chain = type("C", (), {
            "new_voice_session": staticmethod(_new_session),
            "maybe_compress": staticmethod(
                lambda *a, **k: _chain.maybe_compress(*a, **k))})
        voice_cli._spawn_reader = _fake_reader
        speak_sink.build = _build
        voice_out_loop.open_jev2 = lambda d, k=None, **kw: _FakeJev2()
        with contextlib.redirect_stdout(buf):
            async with contextlib.AsyncExitStack() as stack:
                await asyncio.wait_for(voice_cli._run(args, stack), timeout=40)
    finally:
        for k, v in saved.items():
            setattr(voice_cli, k, v)
        speak_sink.build, voice_out_loop.open_jev2 = real_build, real_open

    dumped = None
    for name in sorted(os.listdir(os.path.join(tmp, "fixtures"))):
        if name.startswith("voice_") and name.endswith(".json"):
            dumped = os.path.join(tmp, "fixtures", name)
    rep = json.load(io.open(dumped, encoding="utf-8")) if dumped else {}
    zeros = LOUD == b"\x01" * 320 and bytes(320)
    vo_blk = rep.get("voice_out") or {}
    return {"frames": st["frames"],
            "shape": ["unchanged" if f == LOUD else ("zeros" if f == zeros else "?")
                      for f in st["frames"]],
            # ★This one lands inside the `voice_out` block (not at the top level), and what
            #   it reads is `gate.inner` -- once MicGate is inserted, `gate.inner` IS the
            #   MicGate, which makes this live evidence that this turn passed through.
            "tts_turns": [t for t in (vo_blk.get("tts_turns") or []) if t.get("chunks")],
            "gate_turns": [t for t in (vo_blk.get("gate_turns") or []) if t.get("queued")],
            "out": buf.getvalue()}


def _VO(name):
    """The path of a file under `IAF_AI/dispatch/voice_out/` (there are assertions that
    those two files are untouched, so they have to be pointed at by name)."""
    return os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out", name)


async def c22(sb):
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import voice_cli
    src = io.open(os.path.join(SHELL_ROOT, "voice_cli.py"), encoding="utf-8").read()

    # ── 29a/29b The default (on + feeding zeros): the shape of the four frames is the
    #    entire timeline ──
    g = await _mic_gate_smoke(sb, tag="on")
    ok("29a", "★★while the machine is speaking, **the mic really is muted**: real audio "
              "pushed into `_on_frame` arrives on STT's queue as **an equal-length frame "
              "of zeros**. This is \"my own mouth does not reach my own ears\" -- without "
              "it the shape is speech into the mic -> STT turns it into text -> it drives "
              "another turn, a full LLM plus a full TTS every lap, **spending real "
              "money**; and with barge-in gone it sounds like a normal conversation, so "
              "it is not easy to notice in the moment",
       g["shape"][:2] == ["unchanged", "zeros"],
       f"four-frame shape={g['shape']} (before speaking unchanged / while speaking zeros / "
       f"within hangover zeros / after it elapses unchanged)")

    ok("29b", "★★after `settle()` returns, **stay muted for another `hangover_ms`** before "
              "releasing the mic. `settle()` waits for the sound card to finish the bytes "
              "and adds `latency_s` on top -- it has already accounted for the card's own "
              "backlog; what the hangover covers is the **room's reverberation**, which it "
              "cannot account for. The symptom of omitting it is \"it still feeds back "
              "occasionally\" (those few hundred milliseconds of reverb leaking through) "
              "-- irregular, and raising nothing",
       len(g["shape"]) == 4 and g["shape"][2] == "zeros" and g["shape"][3] == "unchanged",
       f"four-frame shape={g['shape']}")

    # ── 29c With feed_silence off -> not a single frame is sent while muted ──
    g2 = await _mic_gate_smoke(sb, tag="nofeed", feed_silence=False)
    ok("29c", "★`feed_silence: false` -> **not one frame is sent** while muted (saving the "
              "STT cost of however much of the time is TTS). The default is `true`, and "
              "for a reason: sending nothing **freezes** the server's audio timeline -> "
              "`endpointing`/VAD never fires while muted -> the half-sentence left hanging "
              "before the mute never closes out, and once audio resumes the server treats "
              "the two stretches as one continuous utterance. Both options **cost the "
              "same** (billing is by duration sent, and silence counts), so the default is "
              "the one with well-defined semantics",
       g2["shape"] == ["unchanged", "unchanged"],
       f"frames that arrived={g2['shape']} (must be only the one before speaking and the "
       f"one after the hangover)")

    # ── 29d With the switch off -> back to the old behaviour, but the layer stays inserted ──
    g3 = await _mic_gate_smoke(sb, tag="off", enabled=False)
    ok("29d", "★`enabled: false` -> back to the behaviour from before the mic gate existed "
              "(the mic runs as usual and **headphones are mandatory**). ★And the `MicGate` "
              "layer **stays inserted** (its two callbacks simply no-op): wiring it two "
              "different ways would mean the \"mic gate off\" path is never exercised by "
              "anyone -- and that is precisely the path to fall back to when the mic gate "
              "misbehaves",
       g3["shape"] == ["unchanged"] * 4
       and "MicGate(make_sink(tts), _mute_mic, _unmute_mic_later)" in src
       and "if not mic_gate[\"on\"]:\n            return" in src,
       f"four-frame shape={g3['shape']} (all unchanged) · inserted unconditionally="
       f"{'MicGate(make_sink(tts)' in src}")

    # ── 29e Transparent to `_report` and to the closing report ──
    class _Inner:
        def __init__(self):
            self.turn = {"chunks": 3, "spoken_chars": 7}
            self.turns = [{"chunks": 1}]
            self.newed = 0
            self.settled = 0

        def new_turn(self):
            self.newed += 1

        async def __call__(self, chunk):
            return len(chunk)

        async def settle(self):
            self.settled += 1
            return 7

    inner = _Inner()
    events = []
    mgate = voice_cli.MicGate(inner, lambda: events.append("mute"),
                             lambda: events.append("unmute"))
    n1 = await mgate("甲甲")
    n2 = await mgate("乙乙")
    spoken = await mgate.settle()
    mgate.new_turn()
    ok("29e", "★★`MicGate` must be **completely transparent** to the record: `_report` can "
              "follow `.inner` down to the innermost layer, and `turn` / `turns` pass "
              "through as before (the two closing sites read `gate.inner`, which once this "
              "layer is inserted is the MicGate). The symptom of not passing them through "
              "is that the TTS metrics section and `tts_turns` in the report **silently "
              "vanish** -- the same trap as `EchoSink` needing to pass `turn` through, "
              "which really was fallen into once. Also: it latches **only on this turn's "
              "first block** (the second does not call back), and it releases inside "
              "`settle`",
       mgate.inner is inner and mgate.turn == inner.turn and mgate.turns == inner.turns
       and (n1, n2, spoken) == (2, 2, 7) and events == ["mute", "unmute"]
       and inner.newed == 1 and inner.settled == 1
       and len(g["tts_turns"]) == 1 and g["tts_turns"][0].get("chunks"),
       f"callbacks={events} (must be one mute and one unmute) · forwarded="
       f"{(n1, n2, spoken)} · turn passed through={mgate.turn == inner.turn} · "
       f"tts_turns in the report={g['tts_turns']}")

    # ── 29f Where it sits: inside the gate, and released in a finally ──
    i_micgate = src.find("MicGate(make_sink(tts)")
    i_gatesink = src.find("gate = voice_out_loop.GateSink(")
    cls = src[src.find("class MicGate:"):src.find("def headphone_gate(args):")]
    ok("29f", "★★`MicGate` has to sit **inside `GateSink` and outside `SpeakSink`**: the "
              "gate may hold for several seconds before opening (the `speak_min_silence_s` "
              "band is exactly \"waiting for the person to finish\"), so putting MicGate "
              "outside it mutes the mic while **nothing has been spoken yet** -- which "
              "destroys the very criterion \"wait until they finish before speaking\", "
              "since the `silence_seconds()` it reads depends on STT events. ★Releasing is "
              "written in `settle`'s `finally`: the mic has to come back even when the "
              "inner layer raises, or one exception leaves **the ears permanently deaf** "
              "without a word on screen to say so",
       0 < i_gatesink < i_micgate and "finally:" in cls
       and cls.find("finally:") < cls.find("self._on_end()")
       # ★`MicGate` lives in voice_cli (the layer that holds both the ear and the mouth),
       #   and neither file under `voice_out/` mentions it at all -- the mouth does not
       #   need to know about the ear (the IAF decoupling law).
       and not _code_refs(_VO("voice_out_loop.py"), {"MicGate"})
       and not _code_refs(_VO("speak_sink.py"), {"MicGate"}),
       f"GateSink@{i_gatesink} < MicGate@{i_micgate} · released in a finally="
       f"{cls.find('finally:') < cls.find('self._on_end()')} · MicGate mentioned under "
       f"voice_out/={_code_refs(_VO('voice_out_loop.py'), {'MicGate'})}")

    # ── 29g The human gate and how it is worded ──
    on_disk = json.load(io.open(os.path.join(SHELL_ROOT, "control_config.json"),
                                encoding="utf-8"))
    live = voice_cli.load_mic_gate()
    b1 = io.StringIO()

    class _A:
        yes_headphones = False
    import builtins
    saved_in = builtins.input
    try:
        builtins.input = lambda *a: "n"
        with contextlib.redirect_stdout(b1):
            refused = voice_cli.headphone_gate(_A())
    finally:
        builtins.input = saved_in
    noise = b1.getvalue()
    ok("29g", "★the human gate is **still mandatory** (muting the mic stops the machine's "
              "own TTS; it does not stop another person talking or a second device playing "
              "sound, and it is not AEC -- only a human can see what the room is actually "
              "like). All that changed is the question: with the mic gate on, it asks "
              "whether the loudspeaker setup has been checked and spells out the two costs",
       refused is False and "speakers are usable" in noise and "not AEC" in noise
       and live == {k: v for k, v in on_disk["mic_gate"].items()
                    if not k.startswith("_")},
       f"answering n still refuses={refused is False} · the question mentions speakers="
       f"{'speakers are usable' in noise} · read={live} · on disk="
       f"{ {k: v for k, v in on_disk['mic_gate'].items() if not k.startswith('_')} }")


# ── C23: the hallucination gate (speech rate) + the lifetime of the delivered watermark ──
async def c23(sb):
    """★Why this group has to exist (it is the regression for two real bugs):

    Both of them **raise nothing and sound like a normal conversation**, so assertions are
    the only thing that can guard them:

    ① **Hallucination**: in one live session, three seconds after the mic opened -- the
       machine had not spoken and neither had the person -- STT produced
       「我现在在使用LanceDB。」 with `conf=0.983` over an audio window of **0.62
       seconds**. It really did drive a full LLM turn plus a full TTS turn, and it was
       written permanently into `pool/threads/user.md`. The criterion is **speech rate**
       (11.31 characters a second versus a real person's fastest 4.31), not content -- so
       this gate does not drift with the topic.
    ② **The same sentence entering twice**: the delivered watermark's TTL used to run from
       **delivery**, and a single sentence was measured at 27.56 seconds (10 of 49 turns
       exceeded the old 15s TTL) -> the watermark expired mid-sentence -> not one
       character of the full final that arrived afterwards was cut -> the first half was
       repeated. The fix is **refresh on a hit**.

    ★Both bugs were originally regressed against **real recordings**, not hand-made
      synthetic events: the thresholds were derived from that corpus, so testing against
      synthetic data would be marking one's own homework. The corpus was recordings of a
      specific person's speech and is not shipped, so those two replays are not in this
      file; what remains below is everything that can be established without it.
    """
    sys.path.insert(0, os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", "voice_out"))
    import voice_cli
    from lib import stt_client

    # ── 31a/31b How the pure functions count ──
    ok("31a", "★counting rule: one Chinese character counts 1, and a run of letters counts 1 "
              "(`LanceDB` is **one** unit, not seven). Counting by `len(text)` overstates "
              "English several times over in mixed text, so the speech-rate threshold "
              "drifts with the **language** -- the same threshold is loose for Chinese and "
              "tight for English, and nothing reports it",
       stt_client.text_units("我现在在使用LanceDB。") == 7
       and stt_client.text_units("LanceDB") == 1
       and stt_client.text_units("你好") == 2,
       f"'我现在在使用LanceDB。'={stt_client.text_units('我现在在使用LanceDB。')} · "
       f"'LanceDB'={stt_client.text_units('LanceDB')}")

    ok("31b", "★★with no `duration` available, `implausible` returns **False (let it "
              "through)**. This layer's default has to be \"let it through\": better to "
              "let one hallucination past than to swallow **whole batches** of what a "
              "person said because the server stopped sending `duration` one day -- that "
              "is soundless data loss, and far worse than one extra answer",
       stt_client.implausible("我现在在使用LanceDB。", None) is False
       and stt_client.implausible("我现在在使用LanceDB。", 0) is False
       and stt_client.speech_rate("x", None) is None,
       "duration missing / duration 0 / speech_rate returning None -- all three pass")

    ok("31c", f"★anything shorter than `min_units` ({stt_client.IMPLAUSIBLE_MIN_UNITS} units) "
              "is not judged: a one- or two-character acknowledgement (\"mm\", \"right\", "
              "\"the code?\") has a short audio window to begin with, and dividing by a "
              "small denominator produces a large ratio. The shortest real final measured "
              "was 2 units over 1.58s",
       stt_client.implausible("嗯", 0.05) is False
       and stt_client.implausible("代码？", 0.1) is False,
       f"cap={stt_client.IMPLAUSIBLE_CPS} units/sec · minimum judged="
       f"{stt_client.IMPLAUSIBLE_MIN_UNITS} units")
    # ── 31e The wiring layer really blocks it: it does not enter the loop, does not enter
    #    U, and leaves a trace ──
    lp, drv = _loop(sb, FakeJudge(noul=0.1))
    w = voice_cli.VoiceWiring(lp, display=False,
                              hallucination_gate={"enabled": True, "max_cps": 8.0,
                                                  "min_units": 4})
    w.mark_mic_open()
    # this is the hallucination from a real session, copied verbatim
    # (including start/duration/conf)
    bad = {"type": "transcript.partial", "text": "我现在在使用LanceDB。", "is_final": True,
           "start": 0.001, "duration": 0.619, "end_of_turn_confidence": 0.983}
    good = {"type": "transcript.partial", "text": "呃，是的，就是这个意思。", "is_final": True,
            "start": 864.42, "duration": 2.66, "end_of_turn_confidence": 0.974}
    w.on_final(bad["text"], bad)
    u_after_bad = drv.user_buffer.text()
    w.on_final(good["text"], good)
    u_after_good = drv.user_buffer.text()
    ok("31e", "★★a hallucination puts **not one character into U**, and real speech still "
              "gets in. It is blocked at the wiring layer rather than in `stt_client` -- "
              "that layer is a faithful mirror of the wire protocol, and `stt_probe` "
              "diagnoses by seeing every event, so blocking there would blind it in one eye",
       u_after_bad == "" and "这个意思" in u_after_good
       and "LanceDB" not in u_after_good,
       f"after the block U={u_after_bad!r} · after real speech U={u_after_good!r}")

    ok("31f", "★★a blocked transcript **leaves a trace** (`w.dropped` + the `cps`/`dropped` "
              "keys in the log + a yellow line in the closing report). This is the third "
              "place in this repo that deliberately throws away something a person said "
              "(the other two are three strikes and the delivered watermark), and all "
              "three are required to leave a trace -- so that later, investigating \"why "
              "did the machine not hear that\", there is something to hold on to",
       len(w.dropped) == 1 and w.dropped[0]["text"] == bad["text"]
       and round(w.dropped[0]["cps"], 2) == 11.31
       and any(r.get("dropped") and r.get("cps") for r in w.rows),
       f"{len(w.dropped)} trace(s) · cps={w.dropped[0]['cps']:.2f} · "
       f"{sum(1 for r in w.rows if r.get('dropped'))} log row(s) carrying dropped")

    # ── 31g The config really is read (a missing block silently falls back to the
    #    defaults, which testing behaviour alone cannot detect) ──
    live = voice_cli.load_hallucination_gate(
        os.path.join(SHELL_ROOT, "control_config.json"))
    on_disk = json.load(io.open(os.path.join(SHELL_ROOT, "control_config.json"),
                                encoding="utf-8")).get("hallucination_gate") or {}
    missing = voice_cli.load_hallucination_gate(os.path.join(sb.tmp, "nope.json"))
    ok("31g", "★the `hallucination_gate` block **really is read from "
              "`control_config.json`**, and a missing file falls back to the default (on). "
              "This gate throws away things people said, so it has to be switchable off or "
              "loosenable without editing code -- and \"did the config actually take "
              "effect\" cannot be seen from behaviour alone (the same shape as the old "
              "`load_knobs` trap)",
       live == {k: v for k, v in on_disk.items() if not k.startswith("_")}
       and bool(on_disk) and missing == voice_cli.DEFAULT_HALLUCINATION_GATE,
       f"read={live} · on disk={ {k: v for k, v in on_disk.items() if not k.startswith('_')} } "
       f"· missing file={missing}")

    # ── 31j Each of the watermark's three ways of dying has its own counter (the entire
    #    reason this observation was added) ──
    v = reply_list.UserBuffer(os.path.join(sb.tmp, "s31j"))
    v.set_partial("你好")
    v.clear()                                          # armed
    v.set_partial("你好，能不能做一个自我介绍")          # a hit -> refreshed
    armed_refreshed = dict(v.delivered_stats)
    v.commit_final("你好，能不能做一个自我介绍？")       # disarmed normally
    normal = dict(v.delivered_stats)

    v2 = reply_list.UserBuffer(os.path.join(sb.tmp, "s31j2"))
    v2.set_partial("你好")
    v2.clear()
    v2._delivered_at -= reply_list.DELIVERED_TTL_S + 1  # pretend that block's final never came
    v2.set_partial("你好，能不能做一个自我介绍？")
    ok("31j", "★★each of the watermark's three ways of dying **has its own counter**, and "
              "expiring by timeout records the original text alongside it. Only \"how many "
              "characters were cut\" used to be recorded, while \"it died when it should "
              "have been alive\" was completely silent -- and that one is precisely the "
              "cause of \"the same sentence answered twice\". `delivered_max_idle_s` "
              f"against `DELIVERED_TTL_S` ({reply_list.DELIVERED_TTL_S}s) is the margin "
              "this safety net has left; without that number, a hundred live turns still "
              "would not tell you whether the margin is enough (the worst case measured on "
              "real recordings was 12.75s, leaving 2.25s)",
       normal == {"armed": 1, "refreshed": 1, "disarmed": 1}
       and v.delivered_max_idle_s >= 0
       and v2.delivered_stats.get("expired") == 1
       and v2.delivered_stats.get("disarmed", 0) == 0
       and len(v2.delivered_expired) == 1
       and v2.delivered_expired[0]["delivered"] == "你好"
       and v2.delivered_expired[0]["idle_s"] > reply_list.DELIVERED_TTL_S
       and v.delivered_ttl_s == reply_list.DELIVERED_TTL_S,
       f"normal path={normal} (at the refresh={armed_refreshed}) · timeout path="
       f"{dict(v2.delivered_stats)} · expiry trace={list(v2.delivered_expired)}")


CASES = [("C1  three cuts, one turn", c1),
         ("C2  water arriving while busy", c2),
         ("C3  no interruption, and the record", c3),
         ("C4  the voice_cli contract", c4),
         ("C5-7 the file backend", c5_c7),
         ("C8  the shape of U", c8),
         ("C9  STT event routing", c9),
         ("C10 the JEV#1 client and its question", c10),
         ("C11 ListenerLoop, seam by seam", c11),
         ("C12 the whole loop running", c12),
         ("C13 voice_cli wiring", c13),
         ("C14 the real TTS sink", c14),
         ("C15 a turn has exactly one ending", c15),
         ("C16 the speaking gate", c16),
         ("C17 the whole chain wired up", c17),
         ("C18 going idle and waking", c18),
         ("C19 the retention window and where compression lands", c19),
         ("C20 /compact, compressing on demand", c20),
         ("C21 the second baseline for going idle", c21),
         ("C22 the half-duplex gate (muting the mic)", c22),
         ("C23 the hallucination gate and the watermark's lifetime", c23)]


async def main():
    print(f"dry_suite · file backend = "
          f"{'implemented' if FILE_BACKEND else 'not implemented (pre-change baseline)'}")
    broke = []
    for name, fn in CASES:
        sb = Sandbox()
        try:
            await fn(sb)
        except Exception as e:
            broke.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        finally:
            sb.close()

    passed = [r for r in RESULTS if r[2] is True]
    failed = [r for r in RESULTS if r[2] is False]
    skipped = [r for r in RESULTS if r[2] is None]

    print()
    for tag, label, res, ev in RESULTS:
        mark = "OK  " if res is True else ("FAIL" if res is False else "skip")
        print(f"  [{mark}] {tag:<3} {label}")
        if ev:
            print(f"         {ev}")
    print()
    print(f"{len(passed)}/{len(passed) + len(failed)} passed"
          f"{f' · {len(skipped)} skipped' if skipped else ''}"
          f"{f' · {len(broke)} CASE(s) raised' if broke else ''}")
    for name, err in broke:
        print(f"  ! {name}: {err}")
    return 0 if not failed and not broke else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
