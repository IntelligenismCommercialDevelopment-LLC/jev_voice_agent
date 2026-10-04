"""tts_probe.py -- the TTS bench: connect for real and measure two numbers.

Usage (run it in your own terminal):
    cd jev_voice_agent && P=IAF_AI/dispatch/voice_out/tts_probe.py
    <venv>/bin/python $P                  # the lot: five samples + the clear protocol test + playback
    <venv>/bin/python $P --no-play        # no sound (numbers only)
    <venv>/bin/python $P --latency 2      # vary optimize_streaming_latency and compare first-sound
    <venv>/bin/python $P --voice ara --dump-events

**The two numbers it produces are the input to the real sink**:
    (1) `tts_ttfa_ms`      -- time to first audio
    (2) `chars_per_second` -- the real sink uses it to convert "how many seconds the player
       has actually played" into "how many characters have been spoken"

★**This probe does not open the microphone, so NO HEADPHONES ARE NEEDED**; speakers are
  fine. The "headphones are mandatory" rule applies to the real chain with the mic and TTS
  open at once (no AEC -> self-oscillation). Here there is a mouth and no ear.

★Cost: $15 per million characters. All five samples are ~250 characters ~ **$0.004**, so
  ten runs is under five cents.

★Naming: the report lands in `fixtures/tts_probe_*.json`. **It must not be called `stt_*`**
  -- the listener's `jev_bench.py` globs `fixtures/stt_*.jsonl` to collect its corpus, and a
  collision turns assertion 15h red, with the symptom "I only ran a TTS probe, why is the
  gate broken".
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time


def _find_shell_root(start):
    """Walk up to the shell root = the level that contains both `IAF_AI/` and
    `config_base/`.

    ★Do not count dirname levels and do not match on names (a cluster copy may be renamed
      or moved to a different depth). These lines are duplicated in several files on
      purpose (decoupling beats DRY).
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


HERE = os.path.dirname(os.path.abspath(__file__))        # = dispatch_dir
SHELL_ROOT = _find_shell_root(HERE)
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
if IAF_AI_ROOT not in sys.path:
    sys.path.insert(0, IAF_AI_ROOT)

from lib import tts_client        # noqa: E402

_TTY = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s):  return _c("1", s)
def dim(s):   return _c("2", s)
def green(s): return _c("32", s)
def yellow(s):return _c("33", s)
def red(s):   return _c("31", s)
def cyan(s):  return _c("36", s)


# ★The samples are deliberately graded by length: a short utterance's chars_per_second is
#   dragged down by its leading and trailing silence, so calibration must use the long
#   ones. The last two look like real replies (mixed script + punctuation pauses), not
#   tongue twisters.
SAMPLES = [
    ("tiny", "Right."),
    ("one line", "LanceDB is a vector database."),
    ("with pauses", "Well, this question really has two halves: one is latency, the other is cost."),
    ("mixed script", "We run Postgres and Redis on Kubernetes, and the vector side goes to LanceDB."),
    ("long", "When you move a service onto another machine, the part people most often "
             "overlook is the stateful half. A stateless service runs the moment you copy "
             "it, but the database, the hot data sitting in the cache, and those temporary "
             "files written to the local disk often behave differently once they have "
             "moved, and they do not complain about it straight away."),
]

CLEAR_TEXT = ("This is a deliberately long passage used to test text.clear. I will send "
              "text dot clear the moment the first block of audio arrives, and then watch "
              "whether the server answers with audio dot clear, and whether any audio "
              "leaks through after that. If you heard this sentence all the way to the "
              "end, the clear did not take effect.")


def load_tts_config(dispatch_dir=HERE, **overrides):
    """Read the `tts` block of this dispatch's `dispatch_config.json`. The `_`-prefixed
    annotations are filtered out by build_url."""
    with open(os.path.join(dispatch_dir, "dispatch_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tts = dict(cfg.get("tts") or {})
    tts.update({k: v for k, v in overrides.items() if v is not None})
    return tts, cfg.get("knobs") or {}


class Player:
    """Accumulate PCM and play it all at once when the run finishes.

    ★The probe **deliberately does not do streaming playback** -- a streaming player plus a
      jitter buffer plus a two-level clear is the real sink's job, and doing it here would
      be doing half of that work twice, after which the two copies would drift apart. This
      step only needs two numbers: time to first sound (computed from event timestamps, so
      playback is irrelevant) and characters per second (= character count / audio
      duration, also irrelevant to playback). Playing it back is only to verify one thing
      **with your ears**: whether the delivery sounds right.
    """

    def __init__(self, sample_rate, enabled=True):
        self.sr = sample_rate
        self.enabled = enabled
        self.buf = bytearray()
        self.cleared_at = None

    def feed(self, pcm):
        self.buf.extend(pcm)

    def flush_local(self):
        """The `on_clear` callback: clear the local playback queue -- the local half of the
        two-level clear."""
        self.cleared_at = len(self.buf)

    def save_wav(self, path):
        """Save the PCM received as a wav. ★This means the "does the delivery sound right"
        check does not require standing next to the machine -- run the probe and listen to
        the file afterwards. The audio also doubles as a reference sample for later
        comparisons when changing voice or speed."""
        import wave
        if not self.buf:
            return None
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)          # PCM16
            w.setframerate(self.sr)    # ★24000, not STT's 16000
            w.writeframes(bytes(self.buf))
        return len(self.buf) / (2 * self.sr)

    def play(self):
        if not self.enabled or not self.buf:
            return None
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as e:
            print(yellow(f"  (cannot play it back: {e}; --no-play turns this off)"))
            return None
        data = np.frombuffer(bytes(self.buf), dtype="<i2")
        # ★Use the server's sample rate (24000), not STT's 16000. Playing at 16k turns it
        #   into slow, deep speech.
        sd.play(data, self.sr)
        sd.wait()
        return len(data) / self.sr


async def run(args):
    tts_cfg, knobs = load_tts_config(
        voice=args.voice, language=args.language, speed=args.speed,
        optimize_streaming_latency=args.latency)
    key = tts_client.resolve_key(SHELL_ROOT, provider="xai")

    url, eff = tts_client.build_url(tts_cfg)
    print(bold("TTS probe"))
    print(dim(f"  shell root  {SHELL_ROOT}"))
    print(dim(f"  endpoint    {url.split('?')[0]}"))
    print(dim(f"  parameters  " + " ".join(f"{k}={v}" for k, v in eff.items())))
    print(dim(f"  playback    {'on' if args.play else 'off'}"
              f" (no mic here, so no headphones needed)"))
    print()

    player = Player(int(eff["sample_rate"]), enabled=args.play)
    events = []

    def on_event(ev):
        if args.dump_events:
            e = {k: v for k, v in ev.items() if k not in ("delta", "audio")}
            print(dim(f"    << {json.dumps(e, ensure_ascii=False)[:160]}"))
        if ev.get("type") not in ("audio.delta",):
            events.append({k: v for k, v in ev.items() if k not in ("delta", "audio")})

    rows = []
    t_start = time.time()
    async with tts_client.TtsConnection(
            tts_cfg, key, on_audio=player.feed, on_clear=player.flush_local,
            on_event=on_event) as tts:
        print(f"  handshake {tts.m['connect_ms']} ms")
        print()
        print(bold(f"  {'sample':<14}{'chars':>6}{'first ms':>10}{'audio s':>9}{'chars/s':>9}  "))
        print(dim("  " + "─" * 50))
        for label, text in SAMPLES:
            rec = await tts.say(text)
            if rec is None:
                print(red(f"  {label:<14} no reply"))
                continue
            rec["label"] = label
            rec["text"] = text
            rows.append(rec)
            cps = rec["chars_per_second"]
            print(f"  {label:<14}{rec['chars']:>6}{rec['ttfa_ms'] if rec['ttfa_ms'] is not None else '-':>10}"
                  f"{rec['audio_seconds']:>9.2f}{cps if cps else '-':>9}"
                  + ("" if rec["reason"] == "done" else red(f"  <- {rec['reason']}")))

        # -- the clear protocol test (the server-side half) --
        clear_rec = None
        if args.clear_test:
            print()
            print(bold("  clear protocol test (text.clear -> audio.clear)"))
            before = len(player.buf)
            await tts.speak_delta(CLEAR_TEXT)
            # Wait for the first block of audio (otherwise the clear goes out before any
            # audio does, and the "in flight" stretch cannot be measured)
            t0 = time.time()
            while len(player.buf) == before and time.time() - t0 < 10:
                await asyncio.sleep(0.02)
            got_first = len(player.buf) > before
            res = await tts.clear()
            await asyncio.sleep(0.5)      # give any leaked in-flight audio time to show up
            leaked = len(player.buf) - (player.cleared_at or 0)
            clear_rec = {**res, "got_first_audio": got_first,
                         "leaked_bytes_after_clear": leaked}
            print(f"    first audio block received  {green('yes') if got_first else red('no (clear went out too early; this round does not count)')}")
            print(f"    audio.clear                 {green('received') if res['ok'] else red('not received')}  "
                  f"{res['clear_ms']} ms")
            print(f"    discarded in flight         {res['dropped_after_clear']} blocks")
            print(f"    queued after on_clear       {leaked} bytes"
                  + ("  " + green("(the local level was cleared properly)") if leaked == 0
                     else "  " + red("<- the local queue level did not hold")))

        total_ms = int((time.time() - t_start) * 1000)
        metrics = dict(tts.m)
        cost = tts.cost_usd

    # -- summary --
    good = [r for r in rows if r["reason"] == "done" and r["chars_per_second"]]
    ttfas = [r["ttfa_ms"] for r in rows if r["ttfa_ms"] is not None]
    # ★Calibrate on the long samples only: in a short utterance the leading and trailing
    #   silence dominate and drag characters-per-second down.
    long_rows = [r for r in good if r["chars"] >= 20]
    print()
    print(bold("  -- the two numbers --"))
    if ttfas:
        print(f"  tts_ttfa_ms        p50 {int(statistics.median(ttfas))} · "
              f"min {min(ttfas)} · max {max(ttfas)}   "
              + dim(f"(budget is 100-300ms + RTT)"))
    else:
        print(red("  tts_ttfa_ms        not measured (not a single audio block arrived)"))
    if long_rows:
        cps_all = [r["chars_per_second"] for r in long_rows]
        rec_cps = round(statistics.median(cps_all), 2)
        print(f"  chars_per_second   p50 {rec_cps} · range {min(cps_all)}-{max(cps_all)}   "
              + dim(f"(only the {len(long_rows)} samples of >=20 characters)"))
        short = [r for r in good if r["chars"] < 20]
        if short:
            print(dim(f"                     short samples {[r['chars_per_second'] for r in short]} "
                      f"-- being low is correct (leading/trailing silence dominates); not used for calibration"))
        print()
        print(yellow(f"  -> put {rec_cps} into knobs.chars_per_second in "
                     f"config_base/voice_out/dispatch_config.json, then run sync_config.py"))
    else:
        print(red("  chars_per_second   not measured"))
    print()
    print(dim(f"  characters {metrics['chars_sent']} · audio "
              f"{metrics['audio_bytes'] / (2 * int(eff['sample_rate'])):.1f}s · "
              f"events {metrics['events']} · elapsed {total_ms}ms · cost ${cost:.5f}"))
    if metrics["errors"]:
        print(red(f"  errors: {json.dumps(metrics['errors'], ensure_ascii=False)[:500]}"))

    # -- write the report --
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(HERE, "fixtures", f"tts_probe_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report = {"stamp": stamp, "url_config": eff, "connect_ms": metrics["connect_ms"],
              "turns": rows, "clear_test": clear_rec,
              "chars_sent": metrics["chars_sent"], "audio_bytes": metrics["audio_bytes"],
              "cost_usd": round(cost, 6), "events": metrics["events"],
              "first_of_type": metrics["first_of_type"],
              "dropped_after_clear": metrics["dropped_after_clear"],
              "errors": metrics["errors"], "event_log": events[:200]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(dim(f"  report -> {os.path.relpath(out, SHELL_ROOT)}"))
    if args.wav:
        wav = os.path.join(HERE, "fixtures", f"tts_probe_{stamp}.wav")
        secs = player.save_wav(wav)
        if secs:
            print(dim(f"  audio  -> {os.path.relpath(wav, SHELL_ROOT)}  ({secs:.1f}s, "
                      f"{eff['sample_rate']} Hz mono)"))

    # -- play it back (to check the delivery sounds right) --
    if args.play:
        secs = player.play()
        if secs:
            print(dim(f"  played {secs:.1f}s"
                      f" (including what had been sent before the clear)" if args.clear_test else ""))
    return report


def main():
    ap = argparse.ArgumentParser(description="TTS probe: measure tts_ttfa_ms and chars_per_second")
    ap.add_argument("--voice", default=None, help="override the config's voice")
    ap.add_argument("--language", default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--latency", type=int, default=None,
                    help="optimize_streaming_latency: 0 voice-over / 1 live / 2 race for first audio")
    ap.add_argument("--no-play", dest="play", action="store_false",
                    help="do not play it back (numbers only)")
    ap.add_argument("--no-clear-test", dest="clear_test", action="store_false",
                    help="skip the clear protocol test")
    ap.add_argument("--no-wav", dest="wav", action="store_false",
                    help="do not save a wav (by default one is saved to fixtures/ so you can listen later)")
    ap.add_argument("--dump-events", action="store_true", help="print every non-audio event")
    ap.set_defaults(play=True, clear_test=True, wav=True)
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":
    main()
