"""tts_client.py -- streaming TTS client (text in, PCM out), **zero LLM logic**.

Lives in `IAF_AI/lib/` alongside `stt_client.py` / `llm_client.py`: same model type, same
shared lib script, and each dispatch supplies only its own config.

★Five implementation disciplines (the pits you fall into by copying the vendor
  pseudo-code literally):

  1. **Sending and receiving must be concurrent** (same as `stt_client` discipline 1). The
     vendor's `tts_speak` / `tts_recv` are two functions, and calling them in the order
     written sends all the text before receiving anything -- at which point the
     time-to-first-sound is not streaming at all.

  2. ★**`audio.done` ends a TURN, not the CONNECTION** (a single WS supports multiple
     turns: after audio.done you may send the next turn's text.delta). `stt_client`'s
     receiver `break`s on `transcript.done`, and **copying that here makes the second
     utterance wait forever for audio, with no error**. So the receiver here is a
     long-lived coroutine that does not exit between turns, and signals each turn through
     an Event.

  3. ★**`text.clear` has to clear two levels** (the one acceptance item that is easy to
     fudge). This file owns two of them, "server side" and "in flight": between sending
     `text.clear` and receiving `audio.clear`, **every in-flight `audio.delta` is
     discarded** and never handed to `on_audio`. The local playback queue (the stretch of
     `jitter_buffer_ms` that has been received but not yet spoken) is the caller's third
     level -> `clear()` fires `on_clear` synchronously **first** and only then sends
     `text.clear`. **The order must not be reversed**: send clear first and clear the
     local queue second, and the instant the queue is cleared more in-flight audio pours
     into it.

  4. **No key in `dispatch_config.json`** (same as `stt_client` discipline 3): that file
     is pushed down through the mirror rule into `config_base/`, and a key would travel
     with it. `resolve_key()` has its own copy in this file, **deliberately duplicated**
     from the ones in `stt_client` / `jev_client` (decoupling beats DRY here).

  5. ★**The sample rate in and out is not the same number**: STT takes 16000, TTS emits
     **24000**. Open the playback stream at 16k and speech comes out slow and deep,
     **with no error** -> the playback side must read `metrics["sample_rate"]` and must
     never hard-code it.

★Billing: TTS is charged per **character**, $15/M. `metrics["chars_sent"]` is the billing
  figure.
★`chars_per_second` (used by the sink to convert "how many characters have been spoken")
  is recorded per turn by this file, but **calibration is `tts_probe.py`'s job** -- a
  single turn's figure includes leading/trailing silence and punctuation pauses, so short
  utterances come out noticeably low.
"""

import asyncio
import base64
import contextlib
import inspect
import json
import os
import time
import urllib.parse

_TTS_URL = "wss://api.x.ai/v1/tts"

# Recommended parameters for live conversation. The caller's tts_config overrides these.
DEFAULT_TTS = {
    "language": "zh",
    "voice": "eve",
    "codec": "pcm",
    "sample_rate": 24000,
    "speed": 1.0,
    "optimize_streaming_latency": 1,
    "text_normalization": False,
}

# These keys in the dispatch_config tts block are read by other code, not put in the URL
_NON_URL_KEYS = ("jitter_buffer_ms",)
_ALLOWED_SR = (8000, 16000, 22050, 24000, 44100, 48000)
MAX_DELTA_CHARS = 60000          # protocol cap on a single text.delta


class TtsError(Exception):
    pass


# -- config and key ----------------------------------------------------------
def build_url(tts_config=None):
    """Assemble the `tts` block of `dispatch_config.json` into a connection URL.
    Returns `(url, effective config)`.

    Keys starting with `_` are annotations (the config is full of them) and are
    **filtered out here** -- leave each caller to filter them and the one that forgets
    sends a line of documentation as a URL parameter.
    """
    cfg = dict(DEFAULT_TTS)
    cfg.update({k: v for k, v in (tts_config or {}).items() if not k.startswith("_")})
    endpoint = cfg.pop("endpoint", None) or _TTS_URL
    for k in _NON_URL_KEYS:
        cfg.pop(k, None)
    sr = int(cfg.get("sample_rate") or 0)
    if sr not in _ALLOWED_SR:
        raise TtsError(f"sample_rate={sr} is not in the allowed set {_ALLOWED_SR}")
    if not cfg.get("language"):
        raise TtsError("language is required; use a language code such as en, or auto if unsure")
    pairs = []
    for k, v in cfg.items():
        if v is None:
            continue
        pairs.append((k, "true" if v is True else "false" if v is False else str(v)))
    return f"{endpoint}?{urllib.parse.urlencode(pairs)}", cfg


def resolve_key(project_root, provider="xai"):
    """Fetch `providers.{provider}.api_key` from `IAF_AI/config.json`.
    Raises if empty -- never silently returns an empty string."""
    p = os.path.join(project_root, "IAF_AI", "config.json")
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise TtsError(f"cannot read {p}")
    key = ((cfg.get("providers") or {}).get(provider) or {}).get("api_key") or ""
    if not key:
        raise TtsError(f"providers.{provider}.api_key is empty in {p}")
    return key


def split_delta(text, limit=MAX_DELTA_CHARS):
    """A single text.delta is capped at 60,000 characters. Split when over; never
    truncate."""
    return [text[i:i + limit] for i in range(0, len(text), limit)] or [""]


async def _call(cb, *args):
    if cb is None:
        return
    r = cb(*args)
    if inspect.isawaitable(r):
        await r


# ★A "speech clock" used to live here (`SpeechClock` + `features`/`CLASSES`/`char_class`
#   + a regex character-class table + `spoken_prefix_len`) -- a duration model answering
#   "N seconds played = how many characters spoken", calibrated into `knobs.speech_cost`
#   by `dispatch/voice_out/tts_calibrate.py`.
#   **The whole block was deleted when barge-in was removed**: its only purpose was to
#   work out how many characters of the currently playing chunk had been spoken when it
#   got cut off, and nothing can cut off a turn that has already started -> the thing it
#   calibrated no longer exists.
#   (The `queued_seconds` prefetch logic works in bytes/bps and never went through this
#   model, so prefetching is unaffected.)

# -- connection --------------------------------------------------------------
class TtsConnection:
    """One TTS WS connection, **reused across turns**.

        async with TtsConnection(tts_cfg, key, on_audio=play, on_clear=flush) as tts:
            await tts.speak_delta("do you know LanceDB?")
            rec = await tts.end_turn()            # text.done -> wait for audio.done
            await tts.speak_delta("it is a vector store.")
            rec = await tts.end_turn()            # ★second turn on the same connection

    `on_audio(pcm_bytes)` receives decoded mono PCM16 LE at `self.sample_rate`.
    `on_clear()` is called **before** `text.clear` on an interrupt; the caller clears its
    local playback queue there.
    """

    def __init__(self, tts_config, api_key, *, on_audio=None, on_clear=None,
                 on_event=None, open_timeout=10.0, connect_retries=3):
        self.url, self.cfg = build_url(tts_config)
        self.sample_rate = int(self.cfg["sample_rate"])
        self.bytes_per_s = self.sample_rate * 2      # PCM16 mono
        self.api_key = api_key
        self.on_audio = on_audio
        self.on_clear = on_clear
        self.on_event = on_event
        self.open_timeout = open_timeout
        self.connect_retries = int(connect_retries)

        self._ws = None
        self._recv_task = None
        self._done = asyncio.Event()        # this turn's audio.done (or error / clear)
        self._cleared = asyncio.Event()     # audio.clear
        self._dropping = False              # in-flight audio discard window (discipline 3)
        self._turn = None

        self.m = {
            "connect_ms": None,
            "connect_attempts": 0,          # ★>1 is normal; see the note in __aenter__
            "sample_rate": self.sample_rate,
            "url_config": self.cfg,
            "chars_sent": 0,                # ★the billing figure ($15/M characters)
            "audio_bytes": 0,
            "turns": [],
            "events": {},
            "first_of_type": {},            # ★first occurrence of each type kept verbatim
            "dropped_after_clear": 0,
            "errors": [],
        }

    # -- lifecycle -------------------------------------------------
    async def __aenter__(self):
        import websockets      # ★>=14 uses additional_headers, not extra_headers
        # ★★**The first connection in a process almost always fails.** Measured across 12
        #   consecutive connects: the 1st raised `ConnectionResetError` after 3412ms; the
        #   2nd-12th all succeeded with handshakes of 1355-2127ms. Three independent
        #   observations all had the failure on the cold first attempt. A bare TLS
        #   connection to the host works, and an unauthenticated WSS gets a clean 401 ->
        #   it is not a parameter problem and not a key problem.
        #   -> So retrying here is **mandatory**. Without the retry the symptom is "the
        #     script randomly fails to start", and the traceback points into websockets
        #     internals every time, which looks like a library bug.
        t0 = time.time()
        last = None
        for attempt in range(1, self.connect_retries + 1):
            self.m["connect_attempts"] = attempt
            try:
                self._ws = await websockets.connect(
                    self.url, additional_headers={"Authorization": f"Bearer {self.api_key}"},
                    max_queue=None, open_timeout=self.open_timeout)
                break
            except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                last = e
                self.m["errors"].append({"type": "_connect_retry", "attempt": attempt,
                                         "detail": f"{type(e).__name__}: {e}"})
                if attempt == self.connect_retries:
                    raise TtsError(
                        f"cannot connect to TTS (tried {attempt} times): {type(e).__name__}: {e}") from e
                await asyncio.sleep(0.4 * attempt)
        self.m["connect_ms"] = int((time.time() - t0) * 1000)
        self._recv_task = asyncio.create_task(self._receiver(), name="tts-recv")
        return self

    async def __aexit__(self, *exc):
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._recv_task
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        return False

    # -- receive side (long-lived, does not exit between turns: discipline 2) --
    async def _receiver(self):
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray)):
                    # The spec says audio arrives as base64 inside audio.delta. If a real
                    # binary frame does turn up, accept it as audio -- but **leave
                    # evidence**: it means the real protocol and the documentation
                    # disagree, and next time this should be written to match.
                    self.m["errors"].append({"type": "_binary_frame", "detail": len(raw)})
                    await self._audio(bytes(raw))
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    self.m["errors"].append({"type": "_bad_json", "detail": str(raw)[:200]})
                    continue
                typ = ev.get("type") or "?"
                self.m["events"][typ] = self.m["events"].get(typ, 0) + 1
                # ★Strip the base64 audio out of the kept sample: keeping it would grow
                #   the report file to several MB, and the value of a first occurrence is
                #   in the **field names**, not in that block of audio.
                self.m["first_of_type"].setdefault(
                    typ, {k: v for k, v in ev.items() if k not in ("delta", "audio")})
                await _call(self.on_event, ev)

                if typ == "audio.delta":
                    if self._dropping:
                        self.m["dropped_after_clear"] += 1
                        continue
                    b64 = ev.get("delta") or ev.get("audio") or ""
                    try:
                        pcm = base64.b64decode(b64)
                    except Exception as e:
                        self.m["errors"].append({"type": "_bad_base64",
                                                 "detail": f"{type(e).__name__}: {e}"})
                        continue
                    await self._audio(pcm)
                elif typ == "audio.done":
                    self._finish_turn("done")
                    self._done.set()          # ★do not break (discipline 2)
                elif typ == "audio.clear":
                    self._dropping = False
                    self._cleared.set()
                elif typ == "error":
                    self.m["errors"].append(ev)
                    self._finish_turn("error")
                    self._done.set()          # do not leave end_turn stuck until timeout

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # If the receiver dies while the caller is awaiting end_turn(), it would only
            # be noticed at the timeout -- release it immediately.
            self.m["errors"].append({"type": "_recv_raised",
                                     "detail": f"{type(e).__name__}: {e}"})
            self._done.set()
            self._cleared.set()

    async def _audio(self, pcm):
        if not pcm:
            return
        self.m["audio_bytes"] += len(pcm)
        t = self._turn
        if t is not None:
            if t["ttfa_ms"] is None:
                t["ttfa_ms"] = int((time.time() - t["_t0"]) * 1000)
            t["audio_bytes"] += len(pcm)
        await _call(self.on_audio, pcm)

    def _finish_turn(self, reason):
        t = self._turn
        if t is None:
            return
        self._turn = None
        secs = t["audio_bytes"] / self.bytes_per_s
        t["reason"] = reason
        t["audio_seconds"] = round(secs, 3)
        t["total_ms"] = int((time.time() - t.pop("_t0")) * 1000)
        # ★This figure comes out noticeably low for short utterances (leading and
        #   trailing silence dominate). Calibrate from tts_probe's aggregate; do not put a
        #   single turn's number into knobs.chars_per_second.
        t["chars_per_second"] = round(t["chars"] / secs, 3) if secs > 0 else None
        self.m["turns"].append(t)

    # -- send side -------------------------------------------------
    async def speak_delta(self, text):
        """Push along with the LLM's chunks; do not wait for the whole passage. Returns
        the number of characters actually sent."""
        if not text:
            return 0
        if self._turn is None:
            self._turn = {"chars": 0, "audio_bytes": 0, "ttfa_ms": None,
                          "_t0": time.time()}
            self._done.clear()
        n = 0
        for piece in split_delta(text):
            await self._ws.send(json.dumps({"type": "text.delta", "delta": piece}))
            n += len(piece)
        self._turn["chars"] += n
        self.m["chars_sent"] += n
        return n

    async def end_turn(self, timeout=60.0):
        """Send `text.done` and wait for `audio.done`. Returns this turn's record (it
        returns on timeout too, carrying a reason)."""
        if self._turn is None:
            return None
        await self._ws.send(json.dumps({"type": "text.done"}))
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
        except asyncio.TimeoutError:
            self.m["errors"].append({"type": "_no_audio_done",
                                     "detail": f"no audio.done within {timeout}s of text.done"})
            self._finish_turn("timeout")
        return self.m["turns"][-1] if self.m["turns"] else None

    async def clear(self, timeout=5.0):
        """Interrupt. **This is the part of "clear two levels" that this file owns.**

        The order is fixed:
          (1) raise the in-flight discard window -> (2) fire `on_clear` synchronously
          (the caller clears its local playback queue) -> (3) send `text.clear`
          -> (4) wait for `audio.clear`
        ★(1) and (2) must come before (3). Send `text.clear` first and clear the local
          queue second, and the instant it is cleared the in-flight `audio.delta` pours
          back into the queue -- the terminal looks right and the ear still hears sound.
        """
        self._dropping = True
        await _call(self.on_clear)
        self._cleared.clear()
        t0 = time.time()
        await self._ws.send(json.dumps({"type": "text.clear"}))
        ok = True
        try:
            await asyncio.wait_for(self._cleared.wait(), timeout)
        except asyncio.TimeoutError:
            ok = False
            self.m["errors"].append({"type": "_no_audio_clear",
                                     "detail": f"no audio.clear within {timeout}s of text.clear"})
        finally:
            # ★Close the discard window even on timeout: otherwise every subsequent turn
            #   on this connection is silently mute. The cost is that a late audio.clear
            #   lets a few in-flight blocks leak through -- which is recorded in errors.
            self._dropping = False
        # ★★This method **has no caller in the production path** (verified by grep):
        #   removing barge-in deleted the speak_sink route, and removing invalidation cut
        #   the last route that was ever contemplated. The only live caller is
        #   `tts_probe.py` (a manual probe that costs money). It stays because it is a
        #   real capability of this TTS connection and the probe uses it; **do not delete
        #   it as dead residue**. The statement order is guarded by assertion 19f.
        self._finish_turn("cleared")
        self._done.set()
        return {"ok": ok, "clear_ms": int((time.time() - t0) * 1000),
                "dropped_after_clear": self.m["dropped_after_clear"]}

    # -- convenience -----------------------------------------------
    async def say(self, text, timeout=60.0):
        """Say one whole passage at once (for probes and self-tests). The real path uses
        speak_delta + end_turn."""
        await self.speak_delta(text)
        return await self.end_turn(timeout=timeout)

    @property
    def cost_usd(self):
        """$15 per million characters."""
        return self.m["chars_sent"] * 15.0 / 1_000_000
