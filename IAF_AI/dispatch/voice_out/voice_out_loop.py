"""
voice_out_loop.py -- the speaking gate (three bands + a pending-speech queue + JEV#2)

The gate sits on the **sink**: the LLM runs as normal and chunks form as normal, they are
**held at the exit**. The time spent waiting for the person to finish overlaps with the
LLM's TTFT (measured 1264ms) -> the shortest possible time to first sound.

```text
FlushBuffer --> GateSink (pending-speech queue) --> SpeakSink --> TTS --> speakers
                    ^
                    | pass (**pass is the only verdict** -- invalidation was deleted)
              VoiceOutLoop (three-band gate + JEV#2)
```

This layer **knows nothing about TTS and nothing about JEV's HTTP**: the real sink arrives
as a constructor argument and the verdict comes through a `judge` callback. So the offline
path (`dry_suite`'s `FakeJudge` + `SilentSink`) and the real path (`SpeakSink` +
`JevClient`) drive **the same loop** -- there is no "the one that is tested" and "the one
that runs".

★★Eight implementation disciplines (each one corresponds to a way of failing silently)

1. **This tick must not call `SilenceCounter.tick()`.** The criterion reads
   `silence_seconds()` (for control) and the record reads `value()` (for observation);
   both are read-only. The file's only writer is the ear's `_tick_loop` (see the
   `reply_list.SilenceCounter` class comment). A second writer makes "read 3 -> reset ->
   write back 4" swallow the reset, **with no error**.

2. **The gate BLOCKS; it does not "return 0 and accumulate".** `GateSink.__call__`
   `await`s while the gate is shut rather than returning. Return 0 instead and pump runs
   the whole turn through and then `settle`s -- **not one character was spoken, yet the
   turn is over**, and every figure in the books agrees, so nothing can be found.

3. **★★The gate has exactly one exit: pass.** "Invalidate and re-run" was deleted, so
   `GateSink` no longer has `cancel()` / `aborted` / `bind()`, and the "woke up but was not
   passed -> return 0" line in `__call__` is gone too -- **a chunk waiting at the gate can
   only ever be passed**.
   ★This was the last of three related removals (barge-in -> the speech_final fallback ->
   invalidation): **there is now nothing in this system that lets words already committed
   to change their mind halfway.**
   Before adding a second exit here, work out whether you are reconnecting one of them
   (assertions 20j/21f2 guard it).

4. **The upper bound (`max_hold_s`) is a free, deterministic gate; do not drop it.**
   Without it, a JEV#2 that keeps answering false waits forever -- the symptom being "the
   machine listened to you for half a minute and then said nothing", with no error.

5. **Not getting a verdict != a verdict of "not finished".** When `judge` raises, this tick
   does nothing -- it keeps holding and `max_hold_s` catches it. Treating it as "not
   finished" means permanent silence whenever the endpoint is sick; treating it as
   "finished" means talking over the person whenever the endpoint is sick (the false
   positive costs more; see the threshold note in the config).

6. **A late verdict is always DISCARDED; never `task.cancel()`.**
   ★★A note on wording: the "discard" here has nothing to do with the deleted
   TURN invalidation -- what is discarded here is **one JEV verdict** (one HTTP reply),
   whereas that discarded **an entire generated response**. The two remaining keys
   `stale_discarded` / `stale_verdict` both belong to this discipline; do not mistake them
   for invalidation residue. Cancelling poisons the httpx connection pool (352-421ms when
   clean -> 1119-1425ms immediately after a cancel), which self-sustains into an avalanche
   at 1 Hz; and since the JEV input was already sent and the output is free, cancelling
   saves nothing.
   ★There are **two kinds** of "late" here, and the second is unique to the mouth:
     (1) a newer tick in the same turn has already been sent -> the old verdict was looking
         at a stale silence reading;
     (2) **the verdict comes back across turns** (last turn's JEV#2 answer, while this turn
         is already waiting at the gate) -> accepting it means "passing without asking". So
         `_ask` carries `turn_seq` along, and anything that does not match is dropped.

7. **`tick_once()` is public**: `dry_suite` must be able to drive it deterministically,
   tick by tick. A test that uses `sleep` to produce timing goes intermittently green for
   the wrong reasons. The real clock lives on one line, inside `_tick_loop`.

8. **★★"Checking the gate" and "asking JEV#2" are different things, and their rates must
   be separate.** Checking the gate reads two floats and is free -> 4 Hz
   (`tick_interval_s: 0.25`). Asking JEV#2 costs money -> rate-limited by
   `jev2_min_interval_s`, **and the criterion is "how much the silence counter has grown
   since the last question"**, not wall clock. Why not wall clock: the only time reference
   in this loop is `silence_seconds()`, and introducing a second clock means two clocks to
   keep aligned -- while the offline harness drives a fake version of the first, so a
   mismatch would show up as a **false green**.
   ★Drop `tick_interval_s` from 1.0 to 0.25 **without** adding this rate limit and JEV#2's
     bill quadruples, with no error (just a fatter invoice). Assertion 21c2 guards it.

★Why the ticker also gets woken by "a chunk is waiting at the gate" (`gate.wants`): at a
  flat 1 Hz, the gap between a chunk arriving and the next tick averages half a tick and is
  at worst a full one -- the person finished long ago and the machine is still waiting for
  a tick. The cost is one extra JEV#2 question per turn (~$0.00002) in exchange for up to a
  second of time-to-first-sound.

★JEV#2's `state` is **the whole chunk just submitted**, not U. What is taken here is
  `reservoir.inflight_residue()` -- after `drain()` and before `release()` inside pump,
  those cuts are sitting in `.inflight`, and they are **exactly "the chunk they just
  finished saying"**, with no need to add a hook to pump.
"""

import asyncio
import collections
import os
import sys
import time


def _find_shell_root(start):
    """Walk up to the shell root = the level that contains both `IAF_AI/` and
    `config_base/`.

    ★Do not count dirname levels and do not match on names (a cluster copy may be renamed
      or moved to a different depth). These five lines of boilerplate are duplicated in
      several files on purpose (decoupling beats DRY: better to repeat "find root / lock /
      atomic write" than to hoist it into a shared script).
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


HERE = os.path.dirname(os.path.abspath(__file__))       # = dispatch_dir
SHELL_ROOT = _find_shell_root(HERE)
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
if IAF_AI_ROOT not in sys.path:
    sys.path.insert(0, IAF_AI_ROOT)

from lib import jev_client          # noqa: E402

DEFAULT_KNOBS = {
    "tick_interval_s": 0.25,        # rate for checking the gate (free). Asking JEV#2 is
                                    # rate-limited separately; see discipline 8
    "jev2_min_interval_s": 1.0,     # silence must grow by this much before JEV#2 may be
                                    # asked again
    "speak_min_silence_s": 1.0,     # lower bound: genuine silence shorter than this ->
                                    # hold without asking JEV#2
                                    # ★Same default as the ear's llm_min_silence_s (one
                                    #   governs when to start thinking, the other when to
                                    #   start speaking).
                                    # ★★1.0 is **below** the peak mid-sentence reading
                                    #   (measured 1606ms p50 / 2185ms max) -> the gate will
                                    #   often ask JEV#2 while the person is mid-sentence.
                                    #   That is a knowing choice; the full scan table and
                                    #   its cost are in _min_silence_note in
                                    #   dispatch_config. To tighten it, go to 2.0 (the
                                    #   equal-safety point).
    "max_hold_s": 6.0,              # upper bound: genuine silence this long -> pass
                                    # unconditionally (discipline 4)
    "turn_over_threshold": 0.6,
    "max_inflight": 2,
}


def load_knobs(dispatch_dir=None):
    """Read knobs from the mouth's `dispatch_config.json`, filling any missing field from
    `DEFAULT_KNOBS`."""
    cfg = jev_client.load_dispatch_config(dispatch_dir or HERE)
    knobs = dict(DEFAULT_KNOBS)
    for k, v in (cfg.get("knobs") or {}).items():
        if not k.startswith("_"):
            knobs[k] = v
    return knobs


# -- pending-speech queue ----------------------------------------------------
class GateSink:
    """A gate wrapped around the real sink. The protocol is identical to
    `reply_list.console_sink`, so they nest.

        gate = GateSink(SpeakSink(...))
        drv  = chain.new_voice_session(sink=gate)[0]

    Gate open = transparent forwarding (not one extra `await`). Gate shut = the chunk waits
    here for `open()`.
    ★**There is only this one exit** (discipline 3): `cancel()` / `aborted` / `bind()` were
    deleted along with the invalidation mechanism.
    """

    def __init__(self, inner, verbose=False):
        self.inner = inner
        self.verbose = verbose
        self.opened = asyncio.Event()      # pass (★the gate's only exit, discipline 3)
        self.wants = asyncio.Event()       # "a chunk is waiting" -> wake the ticker
        self.waiting = False               # right now a chunk really is stuck at the gate
        self.turn_seq = 0                  # ★how a cross-turn late verdict is caught (6.2)
        self.forwarded_chars = 0           # characters really forwarded inward (settle fallback)
        self.turns = []
        self._settled = False
        self.turn = self._fresh_turn()

    # -- lifecycle --
    def _fresh_turn(self):
        return {"seq": self.turn_seq, "queued": 0, "held_ticks": 0, "asks": 0,
                "opened": False, "open_reason": None, "wait_ms": None}

    def new_turn(self):
        """Shut the gate and start a new turn. ★`__call__` calls this automatically after a
        settle -- pump does not know this method exists."""
        if self.turn["queued"] or self.turn["opened"]:
            self.turns.append(self.turn)
        self.turn_seq += 1
        self.turn = self._fresh_turn()
        self.opened.clear()
        self.wants.clear()
        self.forwarded_chars = 0
        self._settled = False
        inner_new = getattr(self.inner, "new_turn", None)
        if inner_new is not None:
            inner_new()
        return self

    # -- the gate --
    def open(self, why):
        """Pass. Idempotent -- a second call within the same turn only records a note."""
        if self.opened.is_set():
            return False
        self.turn["opened"] = True
        self.turn["open_reason"] = why
        self.opened.set()
        self._note("open", f"passed ({why})")
        return True

    # -- protocol part one: async sink(chunk) --
    async def __call__(self, chunk):
        if self._settled:
            self.new_turn()                 # previous turn settled -> this chunk belongs to
                                            # a new turn, so shut the gate again
        if not self.opened.is_set():
            self.turn["queued"] += 1
            t0 = time.time()
            self.waiting = True
            self.wants.set()                # wake the ticker rather than waiting for the
                                            # next tick (see the module header)
            try:
                # ★Wait on the pass flag only (discipline 3). This used to also wait on its
                #   own `aborted` and on `drv.stale`, both of which went with the
                #   invalidation mechanism -- so waking up necessarily means being passed,
                #   and there is no longer a "woke up but was not passed" branch below.
                await self.opened.wait()
            finally:
                self.waiting = False
            self.turn["wait_ms"] = int((time.time() - t0) * 1000)
        n = await self.inner(chunk)
        self.forwarded_chars += len(chunk) if n is None else int(n)
        return n

    # -- protocol part two (optional): async settle() -> characters really spoken --
    async def settle(self):
        """★If the inner sink has a `settle`, use its number (only it knows how much is
        still inside the sound card); otherwise use the forwarded character count.

        Do not invent a number here: `ledger.reconcile` **truncates the record** by it, so
        over-reporting by one character means recording words that never left the speaker
        as spoken = the record lying (`reply_list.SpokenLedger.reconcile`).
        """
        self._settled = True
        inner_settle = getattr(self.inner, "settle", None)
        if inner_settle is not None:
            return await inner_settle()
        # The inner sink is a typing one (TypingSink / console_sink): it has really been
        # spoken the moment it was handed over, so the true total = what was forwarded
        # (= the optimistic number pump recorded, which makes reconcile a no-op).
        return self.forwarded_chars

    def _note(self, kind, msg):
        if self.verbose:
            print(f"[gate·{kind}] {msg}", flush=True)


# -- the speaking gate's control loop ----------------------------------------
class VoiceOutLoop:
    """The mouth-side control loop: read the silence counter each tick -> three-band gate.

        gate = GateSink(real_sink)
        loop = VoiceOutLoop(drv, gate, judge, knobs)
        task = asyncio.create_task(loop.run())

    `judge(state) -> (noul: float|None, turn_over: bool|None)`
      -- the real path passes `make_judge(JevClient(...))`; offline tests pass a fake.

    ★This loop **does not drive turns**: `drv.pump()` is called by the ear's `_consume`
      (`listener_loop`). The mouth only manages the gate. The two loops share one `drv` and
      touch different fields (the ear writes silence, the mouth reads it).
    """

    def __init__(self, driver, gate, judge, knobs=None, verbose=True, log_cap=2000):
        self.drv = driver
        self.gate = gate
        self.judge = judge
        self.knobs = dict(DEFAULT_KNOBS)
        self.knobs.update(knobs or {})
        self.verbose = verbose

        self.min_silence_s = float(self.knobs["speak_min_silence_s"])
        self.max_hold_s = float(self.knobs["max_hold_s"])
        self.jev2_min_interval_s = float(self.knobs["jev2_min_interval_s"])
        # ★The lower bound must not be pushed up onto the upper bound: that empties the
        #   middle band, which is the only place JEV#2 has any say, and the gate degenerates
        #   into "speak once silence reaches max_hold_s". It would still run and still not
        #   complain -- every utterance would simply be 6 seconds late, and the entire JEV#2
        #   path would have been built for nothing.
        if self.min_silence_s >= self.max_hold_s:
            raise ValueError(
                f"speak_min_silence_s({self.min_silence_s}) must be < max_hold_s"
                f"({self.max_hold_s}), otherwise JEV#2's band is empty and the gate "
                f"degenerates into a plain timer")

        self._inflight = set()
        self._seq = 0
        self._latest_seq = 0
        self._stop = asyncio.Event()
        # The rate limit's reference is **silence seconds**, not wall clock (discipline 8).
        # Reset each turn, so a new turn does not inherit the previous turn's debt.
        self._ask_turn = None
        self._ask_at_sil = None
        self.log = collections.deque(maxlen=log_cap)   # ★bounded: at 4 Hz a long session
                                                       #  grows this faster
        self.stats = collections.Counter()

    # -- JEV#2's state (the authoritative template is in prompt_base/voice_out/question.md) --
    def build_state(self):
        """`(state, said)`; empty `said` = no `.inflight`, so no question can be built
        (see `tick_once`).

        ★`said` comes from `.inflight`, NOT from U: the person finishes a sentence ->
          JEV#1 calls it complete -> `_submit()` runs `u.clear()` -> **U is now empty**.
          Using U as the state while chunks are still accumulating means judging an
          empty string.
        ★★The seconds come straight from `silence_seconds()`. It used to be
          `value() x tick_interval_s`, and that formula pinned the question's content to
          the tick length: the same "quiet for 2 ticks" rendered as "2 seconds" at a
          1.0s tick and "0 seconds" at a 0.01s tick. The tick length no longer touches
          the question.
        ★Still **rounded to whole seconds**: the question text says outright that the
          number of seconds is not the deciding factor, and `question.md` section 2
          requires bench fixtures to have the **same shape** as a real state -- "1.37
          seconds" would not. The `or 1` keeps "quiet for 0 seconds" from ever appearing
          (while the person is still talking this loop does not ask JEV#2 at all).
        ★The note is English because the whole prompt surface is English; if you
          localise it, localise the examples in `prompt_base/voice_out/question.md` and
          the assertion in `dry_suite` 21e in the same commit -- they are one shape in
          three places.
        """
        cuts = self.drv.reservoir.inflight_residue() or []
        said = " ".join(cuts)
        secs = int(round(self.drv.silence.silence_seconds())) or 1
        u = self.drv.user_buffer.text().strip()
        tail = (f'Since then they started speaking again: "{u}" -- not finished.)'
                if u else "They have not spoken since.)")
        note = (f"(system note: after finishing the above, this person has been quiet "
                f"for {secs} seconds. {tail}")
        return f"{said}\n\n{note}", said

    # -- one tick (the three-band table) --------------------------------------
    async def tick_once(self):
        """Run one tick. ★Public so that `dry_suite` can drive it deterministically tick by
        tick, instead of using sleep to produce timing."""
        sil_s = self.drv.silence.silence_seconds()   # ★the control criterion (discipline 1: read-only)
        n_silent = self.drv.silence.value()          # ★record only; never a criterion
        self.stats["ticks"] += 1
        rec = {"t": time.time(), "silence_s": round(sil_s, 3),
               "silence_ticks": n_silent, "seg": None,
               "noul": None, "opened": False, "turn_seq": self.gate.turn_seq}

        if not self.gate.waiting:
            # No chunk is waiting at the gate (the LLM has not produced its first chunk, or
            # the gate has already been passed) -> do nothing.
            rec["seg"] = "idle"
            self.stats["idle"] += 1
            self.log.append(rec)
            return rec

        # ★★This used to be the "invalidate before passing" branch (more water entering the
        #   reply list while chunks accumulate -> raise stale -> abort at the gate -> throw
        #   the turn away and re-run). It was deleted outright.
        #   -> Now, water arriving while chunks accumulate does **nothing**: this turn still
        #     speaks the whole passage, and the new water stays in the pool for the next
        #     turn (whose firing still has to pass "stopped speaking >= llm_min_silence_s").
        #   ★Do not add any "take a look at the pool" branch here -- that is reconnecting
        #     invalidation (assertion 21f2 guards it).
        self.gate.turn["held_ticks"] += 1

        # -- lower bound: still speech this tick = the person is talking -> hold, do not ask
        #    JEV#2 (blocks the most common case for free) --
        if sil_s < self.min_silence_s:
            rec["seg"] = "hold"
            self.stats["hold"] += 1
            self.log.append(rec)
            return rec

        # -- upper bound: silent long enough -> pass unconditionally, do not ask JEV#2
        #    (discipline 4) --
        if sil_s >= self.max_hold_s:
            rec["seg"] = "max_hold"
            self.stats["open_max_hold"] += 1
            self.gate.open(f"max_hold {sil_s:.2f}s>={self.max_hold_s}s")
            rec["opened"] = True
            self.log.append(rec)
            return rec

        # -- rate limit (discipline 8): checking at 4 Hz is free; asking JEV#2 requires the
        #    silence counter to have grown enough --
        #    Reset on a new turn -- having asked last turn should not put this turn in debt.
        if self._ask_turn != self.gate.turn_seq:
            self._ask_turn, self._ask_at_sil = self.gate.turn_seq, None
        if self._ask_at_sil is not None \
                and sil_s - self._ask_at_sil < self.jev2_min_interval_s:
            rec["seg"] = "rate_limited"
            self.stats["rate_limited"] += 1
            self.log.append(rec)
            return rec

        # -- the middle ticks: ask JEV#2 --
        state, said = self.build_state()
        if not said.strip():
            # `.inflight` is absent = "the chunk they just finished saying" cannot be
            # assembled. This should not happen (inside pump it is necessarily present
            # between drain and release); if it does, the bug is elsewhere.
            # ★Judging a state that is only the system note is paying for noise -> keep
            #   holding and let max_hold catch it.
            rec["seg"] = "no_state"
            self.stats["no_state"] += 1
            self._note("error", "cannot assemble JEV#2's state (reply_list's .inflight is "
                                "absent) -> not asking this tick; max_hold will catch it")
            self.log.append(rec)
            return rec

        if len(self._inflight) >= int(self.knobs["max_inflight"]):
            # Only a genuinely degraded endpoint gets here. Piling on more in-flight
            # requests only makes it slower.
            rec["seg"] = "inflight_capped"
            self.stats["inflight_capped"] += 1
            self.log.append(rec)
            return rec

        self._seq += 1
        self._latest_seq = self._seq
        self._ask_at_sil = sil_s            # <- the rate-limit reference only moves when a
                                            #    question is really sent
        self.stats["asked"] += 1
        self.gate.turn["asks"] += 1
        rec["seg"] = "ask"
        rec["state"] = state
        task = asyncio.create_task(self._ask(state, rec, self._seq, self.gate.turn_seq))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        self.log.append(rec)
        return rec

    async def _ask(self, state, rec, seq, turn_seq):
        try:
            noul, turn_over = await self.judge(state)
        except asyncio.CancelledError:
            raise                               # only reached while shutting down; no side effects
        except Exception as e:
            # Discipline 5: no verdict != a verdict of "not finished". Do nothing and let
            # max_hold catch it.
            self.stats["judge_errors"] += 1
            self._note("error", f"no JEV#2 verdict this tick (keep holding, max_hold will "
                                f"catch it): {type(e).__name__}: {e}")
            return None
        rec["noul"] = noul
        # Discipline 6.1: a newer tick in this turn has already gone out -> this verdict was
        # looking at a stale silence reading.
        if seq != self._latest_seq:
            self.stats["stale_discarded"] += 1
            rec["stale_verdict"] = "older_tick"
            return None
        # Discipline 6.2: ★late across turns -- accepting it means passing this turn's gate
        # without having asked.
        if turn_seq != self.gate.turn_seq:
            self.stats["cross_turn_discarded"] += 1
            rec["stale_verdict"] = "other_turn"
            self._note("stale", f"JEV#2 verdict came back across turns (answer for turn "
                                f"{turn_seq}, we are now on turn {self.gate.turn_seq}) -> dropped")
            return None
        if turn_over:
            self.stats["open_jev2"] += 1
            rec["opened"] = self.gate.open(f"jev2 noul={noul:.3f}"
                                           if noul is not None else "jev2")
        else:
            self.stats["jev2_not_over"] += 1
        return noul

    async def settle(self):
        """Wait for every in-flight verdict to land (for tests and for shutdown)."""
        while self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)

    # -- the real clock (lives in this one function only) ---------------------
    async def _tick_loop(self):
        interval = float(self.knobs["tick_interval_s"])
        while not self._stop.is_set():
            # ★Clear before waiting: clearing after the wait would let the next clear eat
            #   the "chunk arrived" edge; and clearing after the tick but before the wait
            #   leaves wants still set while the gate is shut -> it spins into asking JEV#2
            #   continuously (which costs real money).
            self.gate.wants.clear()
            waiter = asyncio.create_task(self.gate.wants.wait())
            stopper = asyncio.create_task(self._stop.wait())
            _, pending = await asyncio.wait({waiter, stopper}, timeout=interval,
                                            return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if self._stop.is_set():
                return
            await self.tick_once()

    async def run(self):
        await self._tick_loop()

    async def aclose(self):
        self._stop.set()
        for t in list(self._inflight):
            t.cancel()
        await self.settle()

    # -- observation ----------------------------------------------------------
    def _note(self, kind, msg):
        if self.verbose:
            print(f"[voice_out·{kind}] {msg}", flush=True)

    def summary(self):
        s = dict(self.stats)
        # ★The real criterion first, the recorded reading second (they are in different
        #   units, and reversed the reader will try to match ticks against the gate)
        s["silence_s"] = round(self.drv.silence.silence_seconds(), 2)
        s["silence_ticks"] = self.drv.silence.value()
        s["gate_waiting"] = self.gate.waiting
        s["gate_open"] = self.gate.opened.is_set()
        s["turn_seq"] = self.gate.turn_seq
        s["reply_list_cuts"] = len(self.drv.reservoir)
        return s


# -- the real verdict (wiring only; offline tests pass FakeJudge, so neither of these two
#    functions appears anywhere on the test path) ------------------------------
def open_jev2(dispatch_dir=HERE, knobs=None, **kw):
    """Build the JEV#2 client from the mouth's config (**not** `await`ed; the caller
    manages its lifetime with `async with`).

    ★Both arguments must be passed explicitly; omitting either is a silent error:
      `question_key="turn_over"` -- the module default of `jev_client.QUESTION_KEY` is
        jev1's `"complete"`, and using the default raises (`verdict_from_answer` reports
        rather than silently answering false).
      `threshold=turn_over_threshold` -- `JevClient` looks for `meaningful_threshold`
        (jev1's key) on its own, and the mouth's config has no such key -> it would
        **silently** fall back to the module default of 0.5 when we want 0.6. That one does
        not raise; the threshold is just quietly wrong.
    """
    k = knobs if knobs is not None else load_knobs(dispatch_dir)
    return jev_client.JevClient(dispatch_dir, agent_name="jev2",
                                question_key="turn_over",
                                threshold=float(k["turn_over_threshold"]), **kw)


def make_judge(jev):
    """Wrap a `JevClient` into `judge(state) -> (noul, turn_over)`."""

    async def judge(state):
        r = await jev.ask(state)
        return r["noul"], r["meaningful"]

    return judge
