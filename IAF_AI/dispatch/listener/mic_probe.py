"""mic_probe.py -- a capture probe (zero cost, zero network).

Usage:
    cd jev_voice_agent
    P=IAF_AI/dispatch/listener/mic_probe.py
    <venv>/bin/python $P                # all four questions (you have to speak)
    <venv>/bin/python $P --list-only    # list devices only (the one question needing no permission)
    <venv>/bin/python $P --device 1 --seconds 8

It answers four questions about the capture path. ★Three disciplines learned by
measurement; read them before changing this file:

  1. **Do not trust `sd.check_input_settings()`** -- measured here, it is **unstable**: on
     the first run it reported `-9986 Internal PortAudio error` for `16000·mono·int16`
     (AUHAL OpenStream failed), and the very next run on the same machine said it was fine.
     Meanwhile `sd.rec(samplerate=16000)` **recorded straight away both times**, with
     CoreAudio doing the resampling itself. Believing that -9986 pushes you down the
     "decimate 48k -> 16k yourself" path -- writing an extra FIR low-pass (there is no scipy
     in the venv) and taking on aliasing risk for nothing.
     **The only valid test is to actually record a short clip and look at what comes back.**

  2. **The test for permission is "all zeros", not "an error"** -- when macOS denies
     microphone permission it **hands back silence and raises nothing**. Without this test,
     "no permission" gets misdiagnosed as "the microphone is broken" or "nobody spoke".

  3. **There is no permission-free format check** -- any format negotiation really opens the
     device and therefore passes through the TCC gate (measured: the process sat in
     `Pa_IsFormatSupported -> AudioUnitSetProperty -> CoreAudio` for nearly 3 minutes,
     waiting on a dialog. The dialog belongs to **the parent app**, so running from an
     editor asks permission for that editor).
     The only genuinely permission-free call is `query_devices()`, hence the `--list-only`
     mode.

Aside: frame arithmetic, 100ms = 1600 samples x 2B = 3200 bytes (a hard requirement of the
streaming STT interface).
"""

import argparse
import math
import sys

import numpy as np
import sounddevice as sd

TARGET_SR = 16000
FRAME_MS = 100
FRAME_SAMPLES = TARGET_SR * FRAME_MS // 1000      # 1600
FRAME_BYTES = FRAME_SAMPLES * 2                   # 3200（int16）

_OK, _BAD, _WARN = "OK  ", "FAIL", "warn"


def dbfs(x):
    return -math.inf if x <= 0 else 20 * math.log10(x / 32768.0)


def _rec(dev, sr, seconds):
    buf = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16", device=dev)
    sd.wait()
    return buf.reshape(-1).astype(np.int32)


def _stats(x):
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    return {
        "n": int(x.size),
        "nonzero": int(np.count_nonzero(x)),
        "peak": int(np.max(np.abs(x))) if x.size else 0,
        "rms": rms,
        "clipped": int(np.count_nonzero(np.abs(x) >= 32700)),
    }


# -- (1) devices (the only question needing no permission) --------------------------
def q1_devices(want):
    print("(1) is an input device detected")
    ins = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if not ins:
        print(f"   [{_BAD}] no input device at all. Some machines have no built-in microphone -- is a USB interface plugged in?")
        return None
    for i, d in ins:
        print(f"   [{i}] in={d['max_input_channels']} sr={d['default_samplerate']:.0f} "
              f"{d['name']}" + (" ←" if i == want else ""))
    dev = sd.query_devices(want)
    print(f"   [{_OK}] using [{want}] {dev['name']!r}")
    if dev["max_input_channels"] > 1:
        print(f"   [{_WARN}] it reports {dev['max_input_channels']} input channels, but this "
              f"microphone has only one capsule -> everything below passes channels=1 "
              f"**explicitly** (reading it as 2 channels gives noise at the wrong speed)")
    try:
        out = sd.query_devices(sd.default.device[1])
        # ★The two match strings are **not display text**: they are how a speaker device is
        #   recognised by name. On a Chinese-locale macOS the device really is called
        #   "扬声器", so dropping it would silently stop detecting speakers there.
        extra = (" (★speakers. Use headphones, or enable the mic gate: without AEC this "
                 "feeds back -- the machine answers itself in a stable loop, and every lap "
                 "costs real money)"
                 if any(k in out["name"] for k in ("扬声器", "Speaker")) else "")
        print(f"   [info] default output = {out['name']!r}{extra}")
    except Exception:
        pass
    return want


# -- (2)+(3) one real recording settles both: permission + format
#    (check_input_settings cannot be used; see discipline 1 in the module header) -------
def q2_q3_probe(dev):
    print("\n(2)+(3) is permission granted / is the sample-rate format right "
          "(decided by **really recording 0.5s**, not by asking a negotiation API)")
    # Kept only as a point of comparison: let it say what it likes, we do not listen
    try:
        sd.check_input_settings(device=dev, channels=1, samplerate=TARGET_SR, dtype="int16")
        print(f"   [info] check_input_settings({TARGET_SR}) says: supported")
    except Exception as e:
        print(f"   [info] check_input_settings({TARGET_SR}) says: unsupported ({type(e).__name__})"
              f" <- ★a known false negative; not believed")

    sr_used = None
    for sr in (TARGET_SR, 48000):
        try:
            x = _rec(dev, sr, 0.5)
        except Exception as e:
            print(f"   [{_WARN}] {sr} Hz really failed to record: {type(e).__name__}: {e}")
            continue
        s = _stats(x)
        if s["nonzero"] == 0:
            print(f"   [{_BAD}] {sr} Hz captured {s['n']} samples but **all of them are 0** -- "
                  f"that is the signature of macOS denying microphone permission (silence, no error).")
            print(f"          Go to System Settings -> Privacy & Security -> Microphone and tick "
                  f"**the app that runs this process** (if you run it from an editor, that editor), "
                  f"then run again.")
            return None, False
        print(f"   [{_OK}] {sr} Hz · mono · int16 **really records** · non-zero {s['nonzero']}/{s['n']}"
              f" ({s['nonzero']/s['n']:.1%}) · noise floor {dbfs(s['rms']):+.1f} dBFS")
        sr_used = sr
        break

    if sr_used is None:
        print(f"   [{_BAD}] neither 16000 nor 48000 can record.")
        return None, False
    print(f"   [{_OK}] permission is granted (non-zero samples are the proof)")
    if sr_used == TARGET_SR:
        print(f"   [{_OK}] {TARGET_SR} connects directly -- **no** 48k->16k decimation and no FIR low-pass needed")
    else:
        print(f"   [{_WARN}] only {sr_used} Hz works -> you must do {sr_used//TARGET_SR}:1 decimation "
              f"yourself, **low-passing to {TARGET_SR//2} Hz first** (bare decimation folds everything "
              f"above 8kHz back into the speech band). There is no scipy in the venv.")
    print(f"   [{_OK}] frame arithmetic: {FRAME_MS}ms = {FRAME_SAMPLES} samples x 2B = {FRAME_BYTES} bytes"
          f" (the interface requires {FRAME_BYTES})")
    return sr_used, True


# -- (4) gain (the only question that needs you to speak) ---------------------------
def q4_gain(dev, sr, seconds):
    print(f"\n(4) is the gain free of clipping (recording {seconds}s -- **say a few sentences "
          f"now, at your normal volume**)", flush=True)
    x = _rec(dev, sr, seconds)
    s = _stats(x)
    print(f"   whole clip: peak {s['peak']} ({dbfs(s['peak']):+.1f} dBFS)"
          f" · RMS {s['rms']:.0f} ({dbfs(s['rms']):+.1f} dBFS) · clipped samples {s['clipped']}")

    # Per-second windows: even when the recording timing is off, you can still see which
    # second has speech and which stretch is the noise floor
    wins = []
    for k in range(seconds):
        w = x[k * sr:(k + 1) * sr]
        if w.size:
            wins.append((k, _stats(w)))
    print("   per second: " + " ".join(f"[{k}s {dbfs(t['rms']):+.0f}]" for k, t in wins) + " dBFS")

    quiet = min(wins, key=lambda t: t[1]["rms"])
    loud = max(wins, key=lambda t: t[1]["rms"])
    snr = dbfs(loud[1]["rms"]) - dbfs(quiet[1]["rms"])
    print(f"   noise floor (quietest, {quiet[0]}s) {dbfs(quiet[1]['rms']):+.1f} dBFS"
          f" · speech (loudest, {loud[0]}s) {dbfs(loud[1]['rms']):+.1f} dBFS · **SNR {snr:.1f} dB**")

    # ★Three states, not two. "No speech detected" must be reported separately from
    #   "tested and passed" -- a low SNR used to be only a warning, while the absolute-level
    #   gate (-40) could be scraped past by the noise floor alone, so a recording with
    #   **nobody speaking** still printed "verdict: all passed". This really happened:
    #   SNR 3.9 dB, per-second levels flat at -37 to -41, reported as all passed.
    #   ★Only SNR can decide "did anyone speak": the absolute level drifts with the noise
    #   floor (measured on one machine at -35.5 / -41 / -47.6 dBFS), so using it as the gate
    #   gives a false green.
    if snr < 10:
        print(f"   [{_BAD}] SNR is only {snr:.1f} dB and the per-second levels are flat -> "
              f"**there was no speech in those {seconds} seconds**, so (4) was not tested "
              f"(neither a pass nor a failure).")
        print(f"          Run it again and start speaking the moment the command is issued; or "
              f"run it in your own terminal, where the live prompt is visible -- this question "
              f"belongs there anyway.")
        return None

    good = True
    if s["clipped"]:
        print(f"   [{_BAD}] {s['clipped']} samples clipped -> turn the input gain down. Clipped audio costs STT words.")
        good = False
    elif dbfs(s["peak"]) > -3:
        print(f"   [{_WARN}] the peak is within 3 dB of full scale: not clipping yet, but no headroom -> lower the gain a little")
    if snr < 20:
        print(f"   [{_WARN}] SNR {snr:.1f} dB is low (>=20 dB is wanted): the gain is too low or "
              f"the room is too noisy. Usable, but STT quality will suffer.")
    else:
        print(f"   [{_OK}] both the level and the SNR are good enough for STT")
    return good


def main():
    ap = argparse.ArgumentParser(description="capture probe: four questions about the microphone path")
    ap.add_argument("--device", type=int, default=None, help="input device index (defaults to the system default input)")
    ap.add_argument("--seconds", type=int, default=8, help="how many seconds (4) records (default 8, leaving room to speak)")
    ap.add_argument("--list-only", action="store_true",
                    help="run only (1), the single question that needs no microphone permission")
    args = ap.parse_args()

    want = args.device if args.device is not None else sd.default.device[0]
    dev = q1_devices(want)
    if dev is None:
        return 1
    if args.list_only:
        print("\n(--list-only: (2)(3)(4) skipped -- they all really open the device, so they all need permission)")
        return 0
    sr, okperm = q2_q3_probe(dev)
    if not okperm:
        print("\nverdict: ★blocked on permission, see the FAIL above")
        return 1
    good = q4_gain(dev, sr, args.seconds)
    if good is None:
        print("\nverdict: (1)(2)(3) passed, **(4) detected no speech** -- not a pass (exit code 2)")
        return 2
    print(f"\nverdict: {'all passed' if good else '★(4) has a problem, see above'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
