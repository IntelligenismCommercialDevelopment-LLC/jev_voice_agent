"""speak_sink.py -- the real TTS sink: feed chunks into TTS, wait for the player to
actually finish, and report how many characters were spoken this turn.

Lives in `dispatch/voice_out/` rather than `lib/`: the connection protocol is generic
(`lib/tts_client.py`), but **how to feed it, when to return, and how to account for it
are the mouth's own implementation**.

──────────────────────────────────────────────────────────────────────────────
★Design choice: **one chunk = one utterance**
  (`text.delta` + `text.done` -> wait for `audio.done`), not one utterance per turn.
  What that buys is that **the audio byte range of every chunk is known exactly** (it
  originally also supported "when interrupted, only convert the chunk currently playing";
  that reason went away with barge-in, but the ranges themselves are still the basis for
  per-chunk accounting and for locating dropouts).

★★**One utterance per chunk MUST prefetch, or 0.5-1.0s of silence is inserted between
  every pair of chunks** (measured: a chunk's time-to-first-sound is 546-1035ms while a
  chunk only plays for 1.9-3.4s -> silence is **18%** of total duration, and it happens
  between every chunk).
  The same measurements give the fix: **generation takes only 1.0-1.9 seconds while the
  previous chunk plays for 2.5-3.4 seconds** -- send the next chunk while the previous one
  is still playing and the silence is entirely masked.
  -> So `__call__` returns as soon as "the queue has <= `playback_lead_s` seconds left",
  **without waiting for this chunk to finish playing**.

★That adds one **optional** closing action to the sink protocol (called by
  `reply_list.VoiceDriver.pump`):

      async def settle() -> int          # TOTAL characters actually spoken this turn

  A sink without `settle` (`TypingSink` / `console_sink`) behaves byte-for-byte as before
  -- a character printed to the terminal has been "spoken" the instant it is printed, so
  there is no "sitting in a queue" state for it.
──────────────────────────────────────────────────────────────────────────────

★★Since barge-in was removed, **this sink no longer clears anything**: once a turn has
  started speaking it plays to the end. The three-level clear protocol itself still lives
  in `lib/tts_client.py` (`clear()` + the in-flight discard window) and its live user is
  `tts_probe.py` (which is what verifies that protocol) -- this file is no longer a caller.

★**The sound card's own tail cannot be recalled**: `_played` counts bytes handed to
  PortAudio, and they actually become sound `stream.latency` seconds later. So closing out
  has to wait for that tail as well (`extra_latency_ms` exists for Bluetooth devices that
  under-report it).
"""

import asyncio
import os
import sys
import threading
import time


def _find_shell_root(start):
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, "IAF_AI"))
                and os.path.isdir(os.path.join(d, "config_base"))):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("cannot find the shell root from: " + str(start))
        d = p


HERE = os.path.dirname(os.path.abspath(__file__))
SHELL_ROOT = _find_shell_root(HERE)
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
if IAF_AI_ROOT not in sys.path:
    sys.path.insert(0, IAF_AI_ROOT)

from lib import tts_client        # noqa: E402


class LocalPlayer:
    """PCM playback queue. **`sounddevice` is imported inside the method** -- this file
    must be importable on a machine with no sound card.

    The only writer is `feed()` (the asyncio thread) and the only reader is PortAudio's
    callback thread, with one lock between them. `_played` is only ever incremented by the
    callback.
    """

    def __init__(self, sample_rate, device=None, extra_latency_ms=0, jitter_ms=120):
        self.sr = int(sample_rate)
        self.bps = self.sr * 2                 # PCM16 mono
        self.device = device
        self.extra_latency_s = extra_latency_ms / 1000.0
        # ★Jitter buffer (80-150ms is the usual recommendation): build up this much before
        #   starting playback. Without it the player starts on the first audio.delta and
        #   immediately outruns the audio still arriving -- the symptom is **one dropout at
        #   the start of every turn**, and it only appears on the real path (the offline
        #   harness feeds everything at once, so it cannot reproduce it).
        self.jitter_bytes = int(self.sr * 2 * jitter_ms / 1000.0)
        self._primed = False
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._played = 0                       # bytes handed to PortAudio (not bytes heard)
        self._fed = 0                          # bytes fed into the queue
        self._stream = None
        self.t_first_play = None               # wall clock when the callback first really
                                               # took data (= the moment the ear hears it)
        self.underruns = 0                     # callback ran dry this many times
                                               # (★includes the silence before playback
                                               #  starts -- do not read it as stuttering)
        self.gaps = 0                          # ★real dropouts: the queue emptied AND more
                                               #  audio arrived afterwards
        self.gap_log = []                      # [(seconds into playback, silence ms)] -- for locating them
        self._starved_at = None
        self._starved_t = None

    # -- lifecycle --
    def start(self):
        import sounddevice as sd
        self._stream = sd.RawOutputStream(
            samplerate=self.sr, channels=1, dtype="int16",
            device=self.device, callback=self._cb)
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def latency_s(self):
        """The sound card's own tail (which cannot be recalled). Bluetooth frequently
        under-reports it -> add extra_latency_ms."""
        base = float(getattr(self._stream, "latency", 0.0) or 0.0)
        return base + self.extra_latency_s

    # -- PortAudio callback thread --
    def _cb(self, outdata, frames, time_info, status):
        need = frames * 2
        with self._lock:
            if not self._primed:
                if len(self._buf) < self.jitter_bytes:
                    outdata[:need] = b"\x00" * need     # not enough buffered yet -> emit
                    return                              # silence and **leave the buffer alone**
                                                        # ★not counted as an underrun: this
                                                        #  is priming, not starvation
                self._primed = True
            take = bytes(self._buf[:need])
            del self._buf[:need]
            self._played += len(take)
            if take and self.t_first_play is None:
                self.t_first_play = time.time()
            if len(take) < need:
                self.underruns += 1
                # ★Only "ran dry and then more audio arrived" is a dropout the ear can
                #   hear. The vast majority of raw underruns are the idle callbacks
                #   **before playback starts** and **after the turn finishes** -- the first
                #   version used that number as the stutter metric and measured 1374 in one
                #   turn, all of it those two stretches.
                if self._fed > 0 and self._starved_at is None:
                    self._starved_at = self._fed
                    self._starved_t = time.time()
                    self._primed = False       # ★once starved, build up again rather than
                                               #  stringing micro-dropouts together
        outdata[:len(take)] = take
        if len(take) < need:
            outdata[len(take):] = b"\x00" * (need - len(take))

    # -- asyncio thread --
    def feed(self, pcm):
        with self._lock:
            if self._starved_at is not None and self._fed >= self._starved_at:
                self.gaps += 1                 # audio after a dry spell = that silence
                                               # really was heard
                now = time.time()
                self.gap_log.append((
                    round(now - (self.t_first_play or now), 2),      # seconds into playback
                    int((now - (self._starved_t or now)) * 1000)))   # how long it was silent
                self._starved_at = None
                self._starved_t = None
            self._buf.extend(pcm)
            self._fed += len(pcm)
        return self._fed

    @property
    def fed_bytes(self):
        return self._fed

    @property
    def played_bytes(self):
        """Bytes handed to the sound card (**without subtracting its latency**). This is
        the one to use when deciding "has it finished playing"."""
        with self._lock:
            return self._played

    @property
    def queued_seconds(self):
        with self._lock:
            return len(self._buf) / self.bps


class SpeakSink:
    """The real implementation of `async sink(chunk)` (one utterance per chunk + prefetch).

        player = LocalPlayer(tts.sample_rate).start()
        async with tts_client.TtsConnection(cfg, key, on_audio=...) as tts:
            sink = SpeakSink(tts, player)
            drv = chain.new_voice_session(sink=sink)[0]

    ★This used to need `sink.bind(drv)` to reach `drv.interrupt`; once barge-in was
      removed the sink no longer needs drv at all and `bind` was deleted outright.
      ★`GateSink.bind` went the same way along with the invalidation mechanism (its only
      purpose had been letting the gate reach `drv.stale`) -- **this layer is now entirely
      independent of drv**.
    """

    def __init__(self, tts, player, *, playback_lead_s=2.5, verbose=False):
        self.tts = tts
        self.player = player
        self.playback_lead_s = float(playback_lead_s)
        self.verbose = verbose
        self._chunks = []                # [{text, start, end}] byte ranges
        self.turn = None
        self.turns = []
        self._new_turn()

    # -- turns --
    def _new_turn(self):
        self._chunks = []
        self.turn = {"chunks": 0, "chars": 0, "ttfa_ms": None,
                     "gaps_at_start": self.player.gaps,
                     "t0": None, "settled": False, "per_chunk": []}

    def new_turn(self):
        """Called by `voice_cli._report` (same name and meaning as on TypingSink). Starts a
        new turn once this one has been settled."""
        if self.turn["settled"] or self.turn["chunks"]:
            self.turns.append(self.turn)
            self._new_turn()

    # -- protocol part one: async sink(chunk) --
    async def __call__(self, chunk):
        if self.turn["t0"] is None:
            self.turn["t0"] = time.time()
        start = self.player.fed_bytes
        t_send = time.time()
        await self.tts.speak_delta(chunk)
        rec = await self.tts.end_turn()          # ★one utterance per chunk
        end = self.player.fed_bytes
        self._chunks.append({"text": chunk, "start": start, "end": end})
        self.turn["chunks"] += 1
        self.turn["chars"] += len(chunk)
        self.turn["per_chunk"].append({
            "chars": len(chunk),
            "ttfa_ms": rec and rec.get("ttfa_ms"),          # server's first audio block
            "gen_ms": int((time.time() - t_send) * 1000),   # ★sent -> audio.done (whole chunk generated)
            "audio_s": rec and rec.get("audio_seconds"),
            "queued_s": round(self.player.queued_seconds, 2)})
        # ★Time-to-first-sound = first chunk handed to the sink -> **the callback first
        #   really taking data** (the moment the ear hears it), NOT audio.done. Measuring to
        #   audio.done folds in "the whole chunk finished generating" and inflates it badly.
        if self.turn["ttfa_ms"] is None and self.player.t_first_play is not None:
            self.turn["ttfa_ms"] = int((self.player.t_first_play - self.turn["t0"]) * 1000)

        # ★Prefetch: return as soon as the queue has <= playback_lead_s seconds left,
        #   **without waiting for this chunk to finish playing**. Generating the next chunk
        #   (measured 1.0-1.9s) happens inside those remaining seconds.
        while self.player.queued_seconds > self.playback_lead_s:
            await asyncio.sleep(0.02)

        # The return value here is **necessarily optimistic** (this chunk is probably still
        # queued) -- the real accounting is closed out by settle(), and pump calls
        # ledger.reconcile().
        return None

    # -- protocol part two (optional): async settle() -> characters really spoken --
    async def settle(self):
        """Wait until the player has genuinely finished this turn, then report the number
        of characters spoken.

        ★★The exit condition **must** use `played_bytes` (bytes handed to the sound card)
          and must not use the "minus sound-card latency" figure -- the latter is
          **permanently short by exactly that tail**, so as an exit condition it can never
          be satisfied and the loop spins until the timeout. The measured symptom was a
          130-second turn with 25808 empty callbacks -- and every acceptance check was
          still green (the words really had been spoken); only the end-to-end duration gave
          it away.
        ★Since barge-in was removed there is only one path here: once it starts, it plays
          to the end. There used to be an `interrupted=True` branch (clear the local queue
          + `text.clear`) and a jump that checked the interrupt flag while draining -- that
          jump was the cause of the self-contradictory record "this segment completed, and
          N characters were not spoken" (`reason` was fixed as "done" **before** settle was
          called, and the jump then cut the accounting short).
        """
        t0 = time.time()
        while self.player.played_bytes < self.player.fed_bytes:
            if time.time() - t0 > 120:
                self.turn["drain_timeout"] = True
                break
            await asyncio.sleep(0.02)
        # Handed to the sound card != heard: wait out the card's own tail before this turn
        # has genuinely finished.
        await asyncio.sleep(self.player.latency_s)
        # ★Everything fed has been played = everything was spoken, **no estimation needed**.
        #   Converting would miss the card's last few dozen milliseconds -> every turn would
        #   under-report the final character and the record would permanently carry
        #   "1 character was not spoken".
        n = sum(len(c["text"]) for c in self._chunks)
        self.turn["settled"] = True
        self.turn["spoken_chars"] = n
        self.turn["gaps"] = self.player.gaps - self.turn["gaps_at_start"]
        self.turn["gap_log"] = self.player.gap_log[self.turn["gaps_at_start"]:]
        if self.turn["ttfa_ms"] is None and self.player.t_first_play is not None and self.turn["t0"]:
            self.turn["ttfa_ms"] = int((self.player.t_first_play - self.turn["t0"]) * 1000)
        if self.verbose:
            print(f"[tts] settle spoke {n}/"
                  f"{self.turn['chars']} chars · {self.turn['gaps']} dropouts", flush=True)
        return n

def build(dispatch_dir=HERE, *, device=None, verbose=False, player=None):
    """Build `(TtsConnection context, LocalPlayer, a function that makes the sink)` from
    this dispatch's config.

    ★The connection is deliberately not `await`ed here -- the caller manages its lifetime
    with `async with` (it is reused across turns).
    """
    import json
    with open(os.path.join(dispatch_dir, "dispatch_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tts_cfg, knobs = cfg["tts"], cfg["knobs"]
    key = tts_client.resolve_key(SHELL_ROOT, provider="xai")
    sr = int(tts_cfg["sample_rate"])
    player = player or LocalPlayer(
        sr, device=device,
        extra_latency_ms=int(knobs.get("extra_latency_ms") or 0),
        jitter_ms=int(tts_cfg.get("jitter_buffer_ms") or 120))
    # ★No `on_clear` passed: since barge-in was removed this sink never calls `clear()`
    #   (and `on_clear` is only ever fired by it).
    conn = tts_client.TtsConnection(tts_cfg, key, on_audio=player.feed)
    lead = float(knobs.get("playback_lead_s") or 2.5)

    def make_sink(tts):
        return SpeakSink(tts, player, playback_lead_s=lead, verbose=verbose)

    return conn, player, make_sink
