"""
listener_loop.py -- the ear: a ticker + the submission rule + three strikes + the silence counter

This wires the finished parts into a loop. **This layer knows nothing about the microphone
and nothing about JEV's HTTP** -- input comes through `on_partial` / `on_final`
(`on_speech_final` is an observation port only, see discipline 2), and the verdict comes
through a `judge` callback.
So typed input, real STT (`voice_cli.py`) and a fake verdict (`dry_suite`) all
drive the same loop: **there is no "the one that is tested" and "the one that runs"**.

★★Eight implementation disciplines (each one corresponds to a way of failing silently)

1. **Trimming is for JUDGEMENT only; submission always submits the whole text.**
   What is fed to JEV is `U.text_for_jev(trim)` (the tail of an active partial gets
   **revised downward**), but what enters the reply list once it passes is `U.text()` --
   **the complete one**. Submitting the trimmed version silently eats a word or two from
   every cut, and **reads perfectly naturally** while doing it.

2. **★★`speech_final` is an observation port only: it does not submit, does not touch U,
   and throws away even its own text.**
   ★**Throwing away its text loses nothing**, and this was measured over 8 real
   recordings: all 17 of the 17 speech_final events carried text that had **already** been
   delivered by the `is_final` before it (13 equalled the last final, 4 were the whole-turn
   accumulation). So all that happens here is `mark_voice()` plus a counter -- **not one
   character of the text is needed**.
   ★**Why it must not be forwarded to `on_final`**: the 4 "whole-turn accumulation" cases
   would put the entire turn into U a second time, and the 13 "equals the last final" cases
   would repeat the last segment -- neither raises an error; the LLM just receives a
   stuttering draft. `route_event` checks `speech_final` first, so `on_final` could never
   receive it anyway: "forward it to on_final" is the one way to write this wrong.
   ★**Why the STT layer still splits three ways**: `route_event` is a faithful mirror of
   the wire protocol and `stt_probe` relies on it to diagnose STT. What was removed is **the
   mechanism inside the ear**, not the protocol.

3. **The ticker is the only writer of the `silence_ticks` file.** The input side only calls
   `mark_voice(stop_wall)`, which touches memory -- letting the callbacks write the file
   would make "read 3 -> reset -> write back 4" swallow the reset, with no error.
   ★★This **tick count** no longer feeds any control criterion; it is kept only as a
   record. Control goes through `silence.silence_seconds()` (continuous seconds on the
   audio timeline), whose only consumer is the mouth's gate.
   So the ear's `tick_interval_s` now **only affects JEV#1's cost** and no longer also
   determines the silence resolution -- those two constraints pull in opposite directions,
   and while they shared one knob neither could be moved.

4. **Not getting a verdict != a verdict of "not enough".** When `judge` raises (rate limit
   / endpoint changed / fields renamed) this tick **does nothing**: it neither submits nor
   records a strike. Treating it as "not enough" means three strikes eventually discards
   what the person said.

5. **The consumer is one long-lived coroutine plus an Event, not a task per input.**
   Spawning a task per arrival races with itself at 1 Hz.

6. **★★The LLM has TWO firing conditions, not one**:
   (1) JEV#1 judges the content sufficient to answer (-> it enters the reply list) and
   (2) **the person has stopped speaking for at least `llm_min_silence_s` seconds**
   (`llm_gate_ok()`). JEV#1 only answers "is there enough to answer", it **does not answer
   "have they finished"** -- without (2), a complete thought formed mid-sentence gets
   answered over the top of the speaker, and with barge-in removed the machine says the
   whole thing.
   ★(2) **does not hold back submission**: the cut still enters the pool; only the LLM
   fires later. Keeping the two separate is what stops the pool losing water.
   ★The way to wait for the condition is **to let the ticker call you** (the two lines at
   the top of `tick_once`); the consumer **never sleeps** on its own: silence is measured
   with a fake clock in tests, and a second real clock means two clocks to keep aligned --
   a mismatch shows up as a **false green**.
   ★Those two lines must be at the **very top** of `tick_once` -- every branch below
   returns early, and the path taken once a person stops speaking is precisely the earliest
   return, `should_skip` (U has been cleared). Put them lower and the water sits in the
   pool forever: **no error, the machine simply never speaks again**.

7. **★★After a submission, the STT chunk that produced that text has not been finalised --
   its final will come back on its own.**
   One utterance from this STT is a sequence of **chunks**: `partial` is the current
   chunk's draft and `final` is **that chunk's** final (not the whole sentence); they
   append one after another to form the sentence. When JEV#1 passes on an active chunk's
   partial, `_submit` takes the whole text and calls `u.clear()` -- but that chunk's final
   arrives 250-9764ms later, so the same sentence lands back in an empty U -> the next tick
   passes it again -> **the same sentence is answered twice**.
   Two of six turns in one real run came about this way. The fix lives on `UserBuffer`'s
   **delivered watermark** (see its class comment) --
   ★**not here**: the ear should not know about STT's chunking protocol; all it can see is
   "how many characters are in U".
   ★The characters the watermark trimmed go into `summary()`'s `dedup_*` -- that is the
   only place you can see it doing its job.

8. **★★An empty-text event is evidence that NOBODY IS SPEAKING, not that someone is.**
   The server opens a window over **every stretch of silence** and sends one **empty-text**
   event per second, with `start` pinned to the window's beginning and `duration` growing.
   Measured: in one 180s run, 203 of 247 events were empty (4 windows), and each window's
   `start` **exactly equalled the previous utterance's end point on the audio timeline**
   (all three cases).
   Originally all three ports called `mark_voice()` unconditionally -> every second
   announced "they are still speaking", which pinned `silence_seconds()` near the pipeline
   lag of ~1.5s (measured peak 4.0s, while the longest real silence was 53.7s). **No error**:
   a threshold of 1.0 still passes; the reading is simply meaningless above 1.5s -- and the
   recommended direction to tighten it is 2.0, at which the gate would **never open**.
   ★So the criterion is "does this event carry text": yes -> `mark_voice`; no -> do nothing
   and let `silence_seconds()` grow. Empty events are still passed down to `UserBuffer`
   (which has its own two guards against empty input; those guards prevent "erasing a
   half-finished sentence into nothing" and must not be dropped).
   ★**Mid-sentence readings did not change at all** (while a person is speaking every event
   carries text; measured p50 1.35s / max 2.60s before and after) -- this change only fixes
   the "genuine silence" stretch and does not alter the risk of talking over someone.

★★"A new tick cancels the previous tick" is implemented as DISCARDING, not cancelling --
  cancelling poisons the connection pool (measured: the next call is a second slower, which
  self-sustains into an avalanche at 1 Hz) and saves nothing (JEV output is free and the
  input was already sent). The evidence is in the comment inside `tick_once`.

★Why `tick_once()` is public: `dry_suite` must be able to drive this loop **deterministically,
  tick by tick**. A test that uses `sleep` to produce timing goes intermittently green for
  the wrong reasons. `_tick_loop()` is only "sleep one tick -> call tick_once", and the real
  clock lives on that one line.
"""

import asyncio
import collections
import os
import sys
import time

def _find_shell_root(start):
    """Walk up to the shell root = the level that contains both `IAF_AI/` and
    `config_base/`.

    ★Do not count dirname levels: scripts get moved, and counting levels breaks the moment
      they do -- in a way that points the path at somewhere that does not exist, which most
      call sites then silently read as empty content.
    ★Do not match on names either: a cluster copy may be renamed, so code must never depend
      on the root's name or location.
    ★These five lines of boilerplate are duplicated in several files on purpose
      (decoupling beats DRY: better to repeat "find root / lock / atomic write" than to
      hoist it into a shared script).
    """
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, "IAF_AI"))
                and os.path.isdir(os.path.join(d, "config_base"))):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("cannot find the shell root (no ancestor contains both "
                               "IAF_AI/ and config_base/) from: " + str(start))
        d = p


# This file lives in IAF_AI/dispatch/listener/ (= the dispatch directory it serves).
# The clients live in IAF_AI/lib/ (alongside llm_client.py).
HERE = os.path.dirname(os.path.abspath(__file__))       # = dispatch_dir
SHELL_ROOT = _find_shell_root(HERE)
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
if IAF_AI_ROOT not in sys.path:
    sys.path.insert(0, IAF_AI_ROOT)

from lib import jev_client          # noqa: E402  only its pure function should_skip is used

DEFAULT_KNOBS = {
    "tick_interval_s": 1.0,
    "llm_min_silence_s": 1.0,       # ★the LLM's second firing condition: the person must
                                    #   have stopped speaking for at least this long
    "meaningless_strikes": 5,
    "strike_reset_growth_chars": 1,
    "meaningful_threshold": 0.65,
    "silence_ticks_to_speak": 2,
    "jev_trim_units": 1,
    "max_inflight": 2,
    "min_end_of_turn_confidence": None,
}


def load_knobs(dispatch_dir=None):
    """Read knobs from the listener's dispatch_config.json, filling any missing field from
    DEFAULT_KNOBS."""
    cfg = jev_client.load_dispatch_config(dispatch_dir or HERE)
    knobs = dict(DEFAULT_KNOBS)
    for k, v in (cfg.get("knobs") or {}).items():
        if not k.startswith("_"):
            knobs[k] = v
    return knobs


class ListenerLoop:
    """The ear-side control loop.

        loop = ListenerLoop(driver, judge, knobs)
        task = asyncio.create_task(loop.run())
        loop.on_partial("do you know")               # <- safe to call from another thread
        loop.on_final("do you know LanceDB")         # <- enters U, waits for JEV#1 on the ticker
        # on_speech_final exists too, but it only counts (discipline 2); it does not submit

    `judge(state) -> (noul: float|None, meaningful: bool|None)`
      -- the real path passes a wrapper around `JevClient.ask`; offline tests pass a fake.
    """

    def __init__(self, driver, judge, knobs=None, verbose=True, log_cap=2000,
                 on_turn=None):
        self.drv = driver
        self.judge = judge
        self.knobs = dict(DEFAULT_KNOBS)
        self.knobs.update(knobs or {})
        self.verbose = verbose
        # Hook fired after each pump completes (signature on_turn(rec)). ★It cannot use
        # driver.on_turn_end -- chain.new_voice_session already took that one for
        # maybe_compress.
        self.on_turn = on_turn

        self.u = driver.user_buffer
        self.reservoir = driver.reservoir
        self.silence = driver.silence

        self.strikes = 0
        self._last_verdict_len = None      # strike false-kill guard: how long U was at the
                                           # last "not enough" verdict
        self._inflight = set()             # in-flight _ask tasks
        self._seq = 0                      # monotonically increasing tick number
        self._latest_seq = 0               # the last tick sent -- only its verdict counts
        self._wake = asyncio.Event()       # wakes the consumer
        self._stop = asyncio.Event()
        self._loop = None                  # bound in run(), for call_soon_threadsafe

        # ★Bounded: at 1 Hz that is 3600 entries an hour, and an unbounded deque really does
        #   grow over a long session. The most recent ones are enough to reconstruct events.
        self.log = collections.deque(maxlen=log_cap)
        self.stats = collections.Counter()

    # -- input side (safe to call from another thread) ------------------------

    def _thread_safe(self, fn, *a):
        """★Everything enters the event loop through here. Called before run() (so with no
        loop) it simply executes -- which is exactly how `dry_suite` and the unit tests use
        it, and they should not be forced to start a loop first."""
        if self._loop is None:
            fn(*a)                      # run() has not been called yet: execute directly
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None              # we are on another thread (a daemon thread, or the
                                        # PortAudio callback)
        if running is self._loop:
            fn(*a)
        else:
            self._loop.call_soon_threadsafe(fn, *a)

    # ★All three ports take an optional `stop_wall` (= the end of the audio this event
    #   covers, converted to wall clock; `stt_client` computes it into `ev["_stop_wall"]`
    #   and the wiring layer `.get()`s it out and passes it in).
    #   **Only the float is passed, never the whole `ev`** -- what the wire protocol looks
    #   like is the STT layer's business and this layer should not know.
    #   When it is unavailable (old recordings / the server did not send it) None is passed
    #   and `SilenceCounter` degrades on its own without raising.
    #   ★★And: **only events that carry text call `mark_voice()`** (discipline 8). An
    #     empty-text event is the server's once-a-second heartbeat over a silence window --
    #     treating it as "speaking" pins the silence reading at 1.5s.
    def on_partial(self, text, stop_wall=None):
        if (text or "").strip():
            self.silence.mark_voice(stop_wall)   # memory marker, no file write (discipline 3)
        self._thread_safe(self._apply_partial, text)

    def on_final(self, text, stop_wall=None):
        if (text or "").strip():
            self.silence.mark_voice(stop_wall)
        self._thread_safe(self._apply_final, text)

    def on_speech_final(self, text=None, conf=None, stop_wall=None):
        """★A pure observation port (discipline 2). **No submission, no touching U, text
        discarded.**

        The port is kept rather than unwired for two things: `mark_voice()` (in case some
        speech_final one day arrives without a final in the same millisecond, silence
        should not be over-reported because of it), and the `stats["speech_finals"]`
        number -- how many times "STT thinks the person finished" happened, to read
        alongside "how many times JEV#1 passed".

        ★★`text` is used here **only to decide whether this event carries text at all**
          (discipline 8: an empty event does not count as speaking). It **never touches U
          and never submits** -- assertion 18e guards that. The wiring layer still passes
          it in; without it this layer would have to infer "was there text" from "is
          `stop_wall` present", which is a cross-file implicit convention (`stt_client`
          only stamps `_stop_wall` on events that carry text), and changing an implicit
          convention does not raise.
        """
        if (text or "").strip() or stop_wall is not None:
            self.silence.mark_voice(stop_wall)
        self._thread_safe(self._apply_speech_final)

    # -- what the input side actually does (all on the event-loop thread) -----

    def _apply_partial(self, text):
        self.u.set_partial(text)           # empty partials are blocked by UserBuffer itself
        self.stats["partials"] += 1

    def _apply_final(self, text):
        self.u.commit_final(text)          # empty finals likewise
        self.stats["finals"] += 1

    def _apply_speech_final(self):
        """Just a counter. ★**Never reset strikes here** -- that is precisely why three
        strikes could never accumulate before: with `endpointing: 400` it fired a reset
        every 400ms."""
        self.stats["speech_finals"] += 1

    # -- submission -----------------------------------------------------------

    def _submit(self, reason, noul=None, conf=None):
        text = self.u.text()               # ★discipline 1: the full text, not the trimmed one
        if not text.strip():
            self.stats[f"submit_empty_{reason}"] += 1
            return None
        self.u.clear()                     # there is only one setting: clear it all
        self.reservoir.add(text)
        self.strikes = 0
        self._last_verdict_len = None
        self.stats[f"submit_{reason}"] += 1
        self._wake.set()                   # wake the long-lived consumer
        self._note("submit", f"[{reason}] {len(text)} chars -> reply list"
                             + (f"  noul={noul:.3f}" if noul is not None else "")
                             + (f"  conf={conf}" if conf is not None else ""))
        return text

    # -- three strikes --------------------------------------------------------

    def _strike(self, u_len):
        """Consecutive "not enough" verdicts **with U not having grown enough in between**
        -> clear U. "Not having grown enough" is the entire false-kill guard.

        ★★The growth baseline lives in config: `knobs.strike_reset_growth_chars`,
          deliberately a field rather than hard-coded. Meaning = how many characters U must
          have grown by **since the last reset** to count as "they are still talking". 1 =
          any single extra character resets it.

        ★The baseline only moves on a reset, which makes the length a **cumulative**
          measure rather than a per-tick one: for someone producing one character per
          second, a setting of 3 resets on the third tick; if the baseline moved every
          tick, a slow trickle would never reach the threshold and strikes would climb
          until they wiped out what the person said. At 1 the two algorithms are exactly
          equivalent -- so this choice changes no behaviour today and merely leaves room to
          tune later.
        """
        grow = int(self.knobs["strike_reset_growth_chars"])
        if self._last_verdict_len is None or u_len - self._last_verdict_len >= grow:
            self.strikes = 0                # U grew enough = they are still talking -> the
                                            # preceding ticks do not count
            self._last_verdict_len = u_len   # ★the baseline only moves on a reset (cumulative)
        self.strikes += 1
        if self.strikes >= int(self.knobs["meaningless_strikes"]):
            # ★The discarded text must leave a trace. Three strikes is the **only place in
            #   this system that throws away what a person said** outright, and everywhere
            #   else leaves a trace. Record only "16 characters" and there is nothing in
            #   hand later when someone asks "why did the machine ignore me".
            discarded = self.u.text()
            self.u.clear()
            self.stats["strike_cleared"] += 1
            self._note("strike", f"struck out: {self.strikes} consecutive 'not enough' "
                                 f"verdicts with no growth in U -> discarding "
                                 f"{u_len} chars: {discarded!r}")
            self.strikes = 0
            self._last_verdict_len = None

    # -- the ticker -----------------------------------------------------------

    async def tick_once(self):
        """Run one tick. ★Public so tests can drive it deterministically, tick by tick,
        rather than using sleep to produce timing.

        Returns this tick's record dict (which also goes into self.log).
        """
        n_silent = self.silence.tick()      # ★the only writer (discipline 3)
        self.stats["ticks"] += 1
        # ★★The LLM's second firing condition (`llm_gate_ok`): there is water in the pool
        #   AND the person has been quiet long enough -> wake the consumer.
        #   This goes at the **very top** of tick_once, because every branch below returns
        #   early -- put it lower and it only takes effect on the "we asked JEV" path,
        #   whereas what happens after a person stops speaking is exactly the skip path
        #   (U has been cleared -> should_skip -> early return). Then the water would sit
        #   in the pool forever: **no error, the machine simply never speaks again**.
        if len(self.reservoir) and self.llm_gate_ok():
            self._wake.set()
        trim = int(self.knobs["jev_trim_units"])
        u_full = self.u.text()
        state = self.u.text_for_jev(trim)
        skip = jev_client.should_skip(state)
        rec = {"t": time.time(), "silence_ticks": n_silent,
               "silence_s": round(self.silence.silence_seconds(), 3),
               "u_len": len(u_full),
               "state": state, "skipped": skip, "noul": None, "submitted": False}

        if skip:
            # U is empty / punctuation only -> make no call (zero cost while silent), and
            # **record no strike** either: nothing to judge != judged insufficient. U was
            # empty to begin with, so clearing it would mean nothing.
            self.stats["skipped"] += 1
            self.log.append(rec)
            return rec

        # ★★The goal of "a late verdict must not be used" is implemented as DISCARDING
        #   rather than CANCELLING. Two measured reasons:
        #   (1) **Cancelling poisons the connection pool**: back-to-back clean calls are
        #      352-421ms, whereas the one **immediately following a cancellation** is
        #      1119/1403/1425ms -- cancelling makes httpx tear down that HTTP/1.1
        #      connection and the next call has to redo the TLS handshake. At a 1 Hz tick
        #      that is a **self-sustaining avalanche**: cancel once -> the next one takes
        #      over a second -> it gets cancelled too. A real run hit exactly this: 17
        #      ticks sent, 10 cancelled, only 7 verdicts received.
        #   (2) **Cancelling saves nothing at all**: JEV is billed at input-only rates and
        #      the input was already sent when the request went out. "Cancellation only
        #      bills what was generated" applies to GENERATION, and JEV generates no tokens.
        #   -> So: send it anyway, and on return use the tick number to tell old from new,
        #     **discarding** the old. Out-of-order results remain structurally impossible
        #     (a verdict is always paired with its text), no tick is skipped, the connection
        #     is not poisoned, and the cost is the same.
        if len(self._inflight) >= int(self.knobs["max_inflight"]):
            # Only a genuinely degraded endpoint gets here. Skipping a tick is the right
            # thing then: piling on more in-flight requests only makes it slower.
            self.stats["inflight_capped"] += 1
            rec["skipped"] = "inflight_capped"
            self.log.append(rec)
            return rec

        self._seq += 1
        self._latest_seq = self._seq
        self.stats["asked"] += 1
        task = asyncio.create_task(self._ask(state, len(u_full), rec, self._seq))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        self.log.append(rec)
        return rec

    async def _ask(self, state, u_len, rec, seq):
        try:
            noul, meaningful = await self.judge(state)
        except asyncio.CancelledError:
            raise                          # only reached while shutting down; no side effects
        except Exception as e:
            # Discipline 5: no verdict != judged insufficient. Do nothing.
            self.stats["judge_errors"] += 1
            self._note("error", f"no JEV verdict this tick (not treated as insufficient): "
                                f"{type(e).__name__}: {e}")
            return None
        rec["noul"] = noul
        if seq != self._latest_seq:
            # ★A late verdict: it was looking at a stale U. Using it is exactly the
            #   "verdict decoupled from text" failure -- a fragment submitted as complete.
            #   Discard it, and record no strike.
            self.stats["stale_discarded"] += 1
            rec["stale"] = True
            return None
        if meaningful:
            rec["submitted"] = True
            self._submit("jev", noul=noul)
        else:
            self._strike(u_len)
            rec["strikes"] = self.strikes
        return noul

    async def settle(self):
        """Wait for every in-flight tick to land (for tests and for shutdown)."""
        while self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)

    # -- consumer side --------------------------------------------------------

    def llm_gate_ok(self):
        """★★The LLM's **second** firing condition: the person has stopped speaking for at
        least `llm_min_silence_s` seconds. ★Public so tests can assert this one directly
        without going around the whole loop.

        Why it is needed: JEV#1 passing only answers "is there enough content to answer",
        it **does not answer "have they finished"**. Originally passing meant starting
        immediately, so a complete thought formed mid-sentence ("you remember LanceDB,
        right" -- still to be followed by "…what happened with that in the end") got
        answered over the top of the speaker, and with barge-in removed the machine says
        the whole thing.
        ★This **does not hold back submission**: JEV#1 still passes and the cut still
          enters the reply list (its job is done); only the LLM fires a little later.
          Keeping the two separate is what stops the pool losing water.

        ★The unit is the SAME QUANTITY as the mouth's `speak_min_silence_s`
          (`silence_seconds()`, seconds, on the audio timeline), but they are **two knobs
          with two values**: this one governs "when to start thinking", that one governs
          "when to start speaking".
        """
        return self.silence.silence_seconds() >= float(self.knobs["llm_min_silence_s"])

    async def _consume(self):
        """One long-lived coroutine. ★Do not spawn a _drain task per input -- at 1 Hz that
        races with itself."""
        while not self._stop.is_set():
            waiter = asyncio.create_task(self._wake.wait())
            stopper = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait({waiter, stopper},
                                              return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if self._stop.is_set():
                return
            self._wake.clear()
            while True:
                if not self.llm_gate_ok():
                    # ★The person is still talking (as far as we know) -> this cut stays in
                    #   the pool and we go back to waiting. **It is not discarded**: the
                    #   ticker glances once per tick and calls us back with set(_wake) the
                    #   moment the condition holds (see the top of tick_once). So this must
                    #   never sleep on its own -- that would be a second clock, and since
                    #   silence is measured with a fake one, a mismatch shows up as a false
                    #   green.
                    self.stats["llm_held"] += 1
                    break
                rec = await self.drv.pump()
                if rec is None:
                    break
                self.stats["llm_turns"] += 1
                if self.on_turn is not None:
                    try:
                        self.on_turn(rec)
                    except Exception as e:
                        # The words are already out; an observation failure must not
                        # interrupt the conversation (same as driver.on_turn_end).
                        self._note("error", f"on_turn hook failed (this turn is unaffected): "
                                            f"{type(e).__name__}: {e}")

    async def _tick_loop(self):
        interval = float(self.knobs["tick_interval_s"])
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return                     # stop arrived first
            except asyncio.TimeoutError:
                pass
            await self.tick_once()

    async def run(self):
        self._loop = asyncio.get_running_loop()
        await asyncio.gather(self._tick_loop(), self._consume())

    async def aclose(self):
        self._stop.set()
        self._wake.set()
        for t in list(self._inflight):
            t.cancel()
        await self.settle()

    # -- observation ----------------------------------------------------------

    def _note(self, kind, msg):
        if self.verbose:
            print(f"[listener·{kind}] {msg}", flush=True)

    def summary(self):
        s = dict(self.stats)
        s["silence_s"] = round(self.silence.silence_seconds(), 2)   # ★the number the mouth really uses
        # ★A recorded reading. Its only criterion-consumer is `voice_cli.silence_watch`
        #   (the sleep decision), which is **not in this file**. The complete list of the
        #   four consumers of these two readings, so nobody has to count them again:
        #     silence_seconds() -- 3 control criteria:
        #       1. listener_loop.llm_gate_ok()   >= llm_min_silence_s  -> fire the LLM
        #       2. voice_out_loop.tick_once()    <  speak_min_silence_s -> hold, do not ask JEV#2
        #       3. voice_out_loop.tick_once()    >= max_hold_s          -> pass unconditionally
        #     value() / the silence_ticks file -- exactly 1 control criterion:
        #       4. voice_cli.silence_watch()     >= sleep_after_ticks   -> cut STT, go idle
        #   Everything else that touches them is record-keeping or display.
        s["silence_ticks"] = self.silence.value()
        # ★`can_speak` / `silence_ticks_to_speak` are misnamed: the real gate is the
        #   mouth's `speak_min_silence_s`, in seconds rather than ticks. Known and pending.
        s["can_speak"] = self.silence.can_speak(int(self.knobs["silence_ticks_to_speak"]))
        # ★How much work the delivered watermark (discipline 7) is doing: `dedup_events` =
        #   the number of events discarded as entirely old content, `dedup_chars` = how many
        #   characters were trimmed in total. Both permanently 0 means either that every
        #   pass landed after the final (good), or that **the watermark is broken** (whether
        #   `u.dedup` holds any trace tells the two apart).
        s["dedup_events"] = self.u.dedup_events
        s["dedup_chars"] = self.u.dedup_chars
        # ★★**How the watermark died**. The two numbers above only say how many characters
        #   it trimmed, not whether it died while it should still have been alive -- and
        #   that death (expiry) is precisely the cause of "the same sentence answered twice".
        #   `delivered_expired > 0` = the `DELIVERED_TTL_S` backstop was eaten through:
        #     that chunk broke off entirely, and nothing from it arriving afterwards gets
        #     trimmed.
        #   `delivered_max_idle_s` = the largest value the watermark's "time since the last
        #     content from the same chunk" ever reached. How far that is from
        #     `DELIVERED_TTL_S` is the margin left.
        #     The worst case measured on the corpus was 12.75s, leaving 2.25s -- but that
        #     was a 4-session sample, so measure it in your own setting.
        st = self.u.delivered_stats
        s["delivered_armed"] = st.get("armed", 0)
        s["delivered_disarmed"] = st.get("disarmed", 0)       # normal: the final arrived
        s["delivered_refreshed"] = st.get("refreshed", 0)     # hit, so its life was extended
        s["delivered_expired"] = st.get("expired", 0)         # ★the backstop really fired
        s["delivered_max_idle_s"] = round(self.u.delivered_max_idle_s, 2)
        s["delivered_ttl_s"] = self.u.delivered_ttl_s          # reported side by side, so the
                                                               # margin is easy to read
        s["reply_list_cuts"] = len(self.reservoir)
        s["u_len"] = self.u.char_len()
        s["strikes"] = self.strikes
        return s
