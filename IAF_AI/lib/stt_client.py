"""stt_client.py -- streaming STT client (websocket + microphone frames), **zero LLM logic**.

Lives in `IAF_AI/lib/` alongside `llm_client.py`: same model type, same shared lib script,
and each dispatch supplies only its own config.

★Three implementation disciplines (every one of them is a pit you fall into by copying the
  vendor pseudo-code literally, and every one has been verified):

  1. **Sending and receiving must be concurrent.** The vendor pseudo-code carries the line
     "audio sending should be concurrent with receiving" and then shows code that
     deadlocks if written literally: `async for raw in ws` never exits, so the sending code
     below it never runs. Here they are two tasks, and the sender first awaits
     `transcript.created` (push before that and the opening words are dropped).

  2. **Field names come from real events, not the docs.** The vendor docs say outright to
     print the first real event and check whether it is `transcript` or `text`. So the text
     accessor is `transcript or text or ""`, and **the first occurrence of each event type
     is kept verbatim in `metrics["first_of_type"]`** -- without that, a field rename shows
     up as "it recognised an empty string", with no error.

  3. **No key in `dispatch_config.json`.** That file goes through the mirror rule down into
     `config_base/`, and a key would travel with it. `api_key` is a separate argument,
     fetched by `resolve_key()` from `providers.xai.api_key` in `IAF_AI/config.json` (same
     convention as `llm_client.call_llm(url, key, model, …)`: the client does not go
     hunting for config on its own).

★Billing note: charged by **audio hour**, and **silent blocks are billed by duration sent**
  just the same. So a real deployment needs silence gating (if VAD says there is no speech,
  do not send empty PCM). This file does **not** gate -- it only exposes
  `metrics["audio_seconds"]` so the cost is visible; gating is the caller's job.
"""

import asyncio
import contextlib
import inspect
import json
import os
import re
import time
import urllib.parse

_STT_URL = "wss://api.x.ai/v1/stt"

# Recommended parameters for a voice assistant. The caller's stt_config overrides these.
DEFAULT_STT = {
    "model": "grok-voice-transcribe-2.0",
    "encoding": "pcm",
    "sample_rate": 16000,
    "interim_results": True,
    "endpointing": 400,
    "language": "zh",
    "smart_turn": 0.55,
    "smart_turn_timeout": 1800,
    "vad_threshold": 0.08,
    "filler_words": False,
}

# Hard protocol requirement: PCM16 LE / 16 kHz / mono, one frame per 100ms
FRAME_MS = 100
FRAME_SAMPLES = 1600
FRAME_BYTES = 3200
_ALLOWED_SR = (8000, 16000, 22050, 24000, 44100, 48000)


class SttError(Exception):
    pass


# -- config and key ----------------------------------------------------------
def build_url(stt_config=None):
    """Assemble the `stt` block of `dispatch_config.json` into a connection URL.
    Returns `(url, effective config)`.

    `keyterm` becomes **repeated same-name parameters** (at most 100, each <=50
    characters), not a comma-joined string.
    """
    cfg = dict(DEFAULT_STT)
    cfg.update(stt_config or {})
    keyterms = cfg.pop("keyterm", None) or []
    sr = int(cfg.get("sample_rate") or 0)
    if sr not in _ALLOWED_SR:
        raise SttError(f"sample_rate={sr} is not in the allowed set {_ALLOWED_SR}")
    pairs = []
    for k, v in cfg.items():
        if v is None:
            continue
        pairs.append((k, "true" if v is True else "false" if v is False else str(v)))
    for t in list(keyterms)[:100]:
        pairs.append(("keyterm", str(t)[:50]))
    return f"{_STT_URL}?{urllib.parse.urlencode(pairs)}", cfg


def resolve_key(project_root, provider="xai"):
    """Fetch `providers.{provider}.api_key` from `IAF_AI/config.json`.
    Raises if empty -- never silently returns an empty string."""
    p = os.path.join(project_root, "IAF_AI", "config.json")
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise SttError(f"cannot read {p}")
    key = ((cfg.get("providers") or {}).get(provider) or {}).get("api_key") or ""
    if not key:
        raise SttError(f"providers.{provider}.api_key is empty in {p}")
    return key


# -- microphone --------------------------------------------------------------
def mic_stream(device=None, on_frame=None, on_status=None,
               samplerate=16000, blocksize=FRAME_SAMPLES):
    """Open the mic and call `on_frame(bytes)` every 100ms. Returns a **started**
    `sd.InputStream`.

    The caller is responsible for `.stop()` / `.close()` (or stopping it before using it as
    a context manager).

    ★`on_frame` runs on **PortAudio's own thread** -- inside it you may only do
      `loop.call_soon_threadsafe(...)`. You must not touch asyncio objects directly, and you
      must not do anything slow (you will drop frames).
    ★`channels=1` **must be explicit**: some USB mics report `Input Channels: 2` while having
      a single diaphragm, and reading them as 2 channels yields noise at the wrong rate.
    ★`samplerate=16000` directly -- measured to work, so there is **no 48k->16k decimation**.
    ★`on_status` is not optional decoration: a PortAudio overflow IS a dropped frame, and
      without this callback the drop is completely silent -- it presents as "STT is missing
      words" with no way to find out why.
    """
    import sounddevice as sd

    def _cb(indata, frames, time_info, status):
        if status and on_status is not None:
            on_status(str(status))
        if on_frame is not None:
            on_frame(indata.tobytes())

    st = sd.InputStream(samplerate=samplerate, channels=1, dtype="int16",
                        blocksize=blocksize, device=device, callback=_cb)
    st.start()
    return st


# -- main body ---------------------------------------------------------------
def route_event(ev):
    """Event -> one of three states: `"speech_final"` / `"final"` / `"partial"`.
    A pure function, so it is easy to test offline.

    ★Routes on **flags**, not on the `type` name -- measured, **only `transcript.partial`
      ever carries a transcript**; there is no `transcript.final`.
    ★The order cannot be reversed: an event with `speech_final=true` **also carries
      `is_final=true`**, so testing is_final first would make the speech_final branch
      unreachable.
    """
    if ev.get("speech_final"):
        return "speech_final"
    if ev.get("is_final"):
        return "final"
    return "partial"


def event_text(ev):
    """Get the transcript text. ★Measured, the real name is `text`; the `transcript`
    fallback stays as a buffer against a rename."""
    return ev.get("transcript") or ev.get("text") or ""


# -- speech-rate check (the criterion behind the hallucination gate) ----------
#: "How many units per second can a person physically produce." A transcript above this
#: **cannot physically have been spoken by a human**.
#: Measured calibration (all 60 non-empty finals in the original corpus):
#:   p50 2.78 · p75 3.40 · p90 3.83 · p95 4.01 · **fastest real human 4.31** units/second.
#: One line in that same corpus sat at **11.31** units/second: a 0.62s audio window
#: containing a complete seven-unit sentence, with `conf=0.983`, three seconds after the
#: mic opened, with the machine not yet having made a sound and nobody having spoken.
#: That was the STT model, biased by a `keyterm` entry on a **very short burst of audio**,
#: having its language prior fill in a whole plausible sentence -- and it really did drive
#: one LLM call plus one TTS synthesis, and was written permanently into the thread file.
#: 8.0 is 1.9x the fastest human and 0.71x that hallucination, landing in the empty space
#: between 4.31 and 11.31.
#: ★This gate blocks on **speech rate**, not content -- it never needs to know "what a
#: hallucination sounds like", so it does not drift as the subject matter changes.
#: ★★Calibrated on Chinese speech. Under language=en the unit is effectively "words per
#: second" (see `text_units`), which the same rule handles correctly, but 8.0 itself has not
#: been revalidated on English -- rescan on your own corpus.
IMPLAUSIBLE_CPS = 8.0

#: Shorter than this many units and no judgement is made. A one- or two-unit acknowledgement
#: ("mm", "right", "code?") has a short audio window by nature, and dividing by a small
#: denominator produces a very large ratio. The shortest real human final measured was
#: 2 units / 1.58s.
#: ★That hallucination was 7 units, so it was inside the judging range.
IMPLAUSIBLE_MIN_UNITS = 4


def text_units(text):
    """Unit counting: one CJK character counts as 1, one run of consecutive letters counts
    as 1 (`LanceDB` is one word, not seven).

    ★Not `len(text)`: in mixed script the Latin part gets overstated several times over, and
    the speech-rate threshold then drifts with the language.
    """
    t = text or ""
    return len(re.findall(r"[\u4e00-\u9fff]", t)) + len(re.findall(r"[A-Za-z]+", t))


def speech_rate(text, duration):
    """Units per second. Returns `None` when `duration` is missing or <=0 -- **it does not
    guess**. A pure function, so it is easy to test offline."""
    try:
        du = float(duration)
    except (TypeError, ValueError):
        return None
    if du <= 0:
        return None
    return text_units(text) / du


def implausible(text, duration, max_cps=IMPLAUSIBLE_CPS, min_units=IMPLAUSIBLE_MIN_UNITS):
    """Is this transcript's speech rate **physically impossible** for a human? A pure
    function (tested offline; see dry_suite).

    Returns `False` when `duration` is unavailable -- **if it cannot judge, it lets it
    through**. The default at this layer has to be "let it through": better to miss one
    hallucination than to swallow everything a person says because the server stopped
    sending `duration` one day.
    """
    n = text_units(text)
    if n < int(min_units):
        return False
    r = speech_rate(text, duration)
    return r is not None and r > float(max_cps)


async def _call(cb, *args):
    if cb is None:
        return
    r = cb(*args)
    if inspect.isawaitable(r):
        await r


async def stream(stt_config, api_key, audio_queue, *,
                 on_partial=None, on_final=None, on_speech_final=None,
                 on_event=None, on_ready=None, ready_timeout=10.0, done_grace=8.0):
    """Run one STT connection. Put bytes frames into `audio_queue`; put `None` to signal
    end of audio.

    The three callbacks correspond to the protocol's three states (all receive
    `(text, ev)`, and may be sync or async):
      `on_partial`      `is_final=false`                      display only, **not sent to the LLM**
      `on_final`        `is_final=true` and not speech_final  text locked -> enters U
      `on_speech_final` `speech_final=true`                   ★**observation only**, drives nothing
    ★★`on_speech_final` used to be documented as "this is the one that drives the LLM".
      **That is no longer true** -- that fallback was deliberately removed, and the **only**
      thing that drives the LLM now is JEV#1 on the ear-side ticker. This layer still splits
      three ways (`route_event` is a faithful mirror of the wire protocol, and `stt_probe`
      relies on it for diagnosis), but the third state's text is discarded on the ear side.
    ★All three callbacks receive an `ev` with **one extra field injected by this layer**:
      `ev["_stop_wall"]` = the end of the audio this event covers, converted to wall clock
      (= the moment the person actually stopped speaking). The leading underscore means
      "not a server field, added here". The silence gate depends on it (`SilenceCounter`).
      When `start`/`duration` are unavailable the key is **absent** -- downstream must use
      `.get()` and must not assume it is there.
    `on_event` receives **every** raw event (for debugging and instrumentation).

    Returns metrics, whose **names honestly reflect what they measure** (see the comments).
    """
    url, eff = build_url(stt_config)
    sr = int(eff["sample_rate"])
    bytes_per_s = sr * 2

    import websockets      # ★>=14 uses additional_headers, not extra_headers

    m = {
        "connect_ms": None,          # TLS + WS handshake (first one is ~700ms, so pre-warm)
        "created_ms": None,          # handshake done -> transcript.created
        # ★**This is the algorithm stt_ttfp_ms / stt_final_ms should have had all along**
        #   (settled by measurement): every server event carries its own `start` +
        #   `duration` (on the audio timeline), so
        #       event latency = arrival wall clock - (start + duration) x 1000
        #   -- which subtracts both "the silence before the person spoke" and "how long this
        #   utterance took", leaving pure pipeline latency.
        #   Measured 13/13 between 651-723ms (mean ~683ms), and **partial / final /
        #   speech_final all have the same latency**: there is no speech_final penalty. That
        #   matches the vendor's own budget (RTT 180-250 + interim results 200-500), at the
        #   high end of it.
        "event_latency_ms": [],
        # ttfp measured from the first frame is kept for comparison, but **do not use it as a
        # latency budget** -- it includes "when the person decided to start talking".
        "ttfp_after_first_frame_ms": None,
        "audio_seconds": 0.0,        # duration of audio sent = the billing unit
        "frames_sent": 0,
        "events": {},                # count per event type
        "first_of_type": {},         # ★first occurrence of each type kept verbatim (discipline 2)
        "audio_status": [],          # PortAudio overflow and friends (fed in by the caller)
        "errors": [],
        "url_config": eff,
    }
    ready = asyncio.Event()
    stop = asyncio.Event()
    t_first_frame = None
    t_last_text = None
    # Origin of the audio timeline. The server's start/duration are relative to when IT
    # began receiving audio, and the first frame it receives is the first one recorded after
    # on_ready -> so use the moment of on_ready as the origin.
    t_mic_epoch = [None]

    t0 = time.time()
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"},
                                  max_queue=None) as ws:
        m["connect_ms"] = int((time.time() - t0) * 1000)
        t_conn = time.time()

        async def sender():
            nonlocal t_first_frame
            # ★Wait for transcript.created before pushing, or the opening is dropped -- and
            #   nothing reports it.
            try:
                await asyncio.wait_for(ready.wait(), timeout=ready_timeout)
            except asyncio.TimeoutError:
                m["errors"].append({"type": "_no_created",
                                    "detail": f"no transcript.created within {ready_timeout}s"})
                return
            while not stop.is_set():
                item = await audio_queue.get()
                if item is None:
                    break
                if t_first_frame is None:
                    t_first_frame = time.time()
                await ws.send(item)
                m["frames_sent"] += 1
                m["audio_seconds"] += len(item) / bytes_per_s
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "audio.done"}))

        async def receiver():
            nonlocal t_last_text
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue            # the server should not send binary; ignore it if it does
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    m["errors"].append({"type": "_bad_json", "detail": raw[:200]})
                    continue
                typ = ev.get("type") or "?"
                m["events"][typ] = m["events"].get(typ, 0) + 1
                m["first_of_type"].setdefault(typ, ev)
                await _call(on_event, ev)

                if typ == "transcript.created":
                    if m["created_ms"] is None:
                        m["created_ms"] = int((time.time() - t_conn) * 1000)
                    # ★on_ready exists so the caller can "open the mic only after connecting"
                    #   (the pre-warm). Open the mic first and connect second, and the ~1
                    #   second of audio recorded during the handshake piles up in the queue
                    #   and then floods in at once -- the audio timeline and the wall clock
                    #   are offset from then on, and **every latency number becomes
                    #   unusable** (measured on the first run: 1127ms handshake, 11 frames
                    #   backed up).
                    t_mic_epoch[0] = time.time()
                    await _call(on_ready)
                    ready.set()
                    continue
                if typ == "error":
                    m["errors"].append(ev)
                    break
                if typ == "transcript.done":
                    break

                # Everything else is treated as a transcript event, routed on **flags**
                # rather than on the type name -- so one extra type (say transcript.final)
                # cannot silently slip past.
                # ★The field name is settled by real events: it is `text`; `transcript` does
                #   not exist. The fallback `or` stays -- on the day of a rename it is the
                #   only buffer there is.
                text = event_text(ev)
                now = time.time()
                if text:
                    if m["ttfp_after_first_frame_ms"] is None and t_first_frame:
                        m["ttfp_after_first_frame_ms"] = int((now - t_first_frame) * 1000)
                    t_last_text = now
                # ★Pure pipeline latency: arrival time - the end of the audio this event
                #   claims to cover. This used to measure the gap from "previous text event"
                #   to "speech_final", which measured a constant 0 -- because final and
                #   speech_final are **two events arriving in the same millisecond**, so it
                #   was measuring the instrument itself.
                st, du = ev.get("start"), ev.get("duration")
                if text and st is not None and du is not None and t_mic_epoch[0]:
                    m["event_latency_ms"].append(
                        int((now - t_mic_epoch[0]) * 1000 - (st + du) * 1000))
                    # ★★Same formula, one extra number sent downstream, this one **for
                    #   control**. `_stop_wall` = the end of the audio this event covers,
                    #   converted to wall clock = the moment this person **actually stopped
                    #   speaking**. The silence gate is computed from it (see the
                    #   `SilenceCounter` class comment), which saves ~1.4s of opening latency.
                    #   ★This is the **only** place in the repo that holds the audio
                    #     timeline's origin -- which is why the conversion happens here, and
                    #     downstream receives a single wall-clock float without needing to
                    #     know what start/duration are.
                    ev["_stop_wall"] = t_mic_epoch[0] + st + du
                if text and du is not None:
                    # ★★**Stamp the reading only; do not block here.** This layer is a
                    #   faithful mirror of the wire protocol -- `stt_probe` diagnoses by
                    #   seeing every event, and blocking here would blind it in one eye.
                    #   The blocking happens in the wiring layer (`voice_cli`), which already
                    #   owns `mic_gate`.
                    ev["_cps"] = speech_rate(text, du)

                # ★Route the three states. This block **must stay at the same level as the
                #   latency statistics above** -- while editing the latency code once, the
                #   on_speech_final line got absorbed into the `if` above it, and **every
                #   event was then dispatched as speech_final**. `ast.parse` passed happily
                #   (syntactically legal, semantically wrong) and the kind field was
                #   corrupted across two whole recordings. -> The lesson: **a syntax check is
                #   not a behaviour check**, which is why dry_suite now has offline cases for
                #   route_event (assertions 14a-14d).
                kind = route_event(ev)
                await _call({"speech_final": on_speech_final, "final": on_final,
                             "partial": on_partial}[kind], text, ev)

        send_task = asyncio.create_task(sender(), name="stt-send")
        recv_task = asyncio.create_task(receiver(), name="stt-recv")
        try:
            # Normal order: the sender finishes first (sentinel -> audio.done), and the
            # receiver then gets transcript.done.
            await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            if recv_task.done():
                stop.set()              # receiver died first (error / done) -> do not leave
                                        # the sender blocked on queue.get()
            else:
                await asyncio.wait_for(asyncio.shield(recv_task), timeout=done_grace)
        except asyncio.TimeoutError:
            m["errors"].append({"type": "_no_done",
                                "detail": f"no transcript.done within {done_grace}s of audio.done"})
        finally:
            for t in (send_task, recv_task):
                t.cancel()
            for t, name in ((send_task, "send"), (recv_task, "recv")):
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    m["errors"].append({"type": f"_{name}_raised",
                                        "detail": f"{type(e).__name__}: {e}"})
    return m
