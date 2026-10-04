"""stt_probe.py -- an acceptance harness: really connect to the streaming STT endpoint and
print the transcript word by word as it arrives.

Usage (run it in your own terminal; you have to speak):
    cd jev_voice_agent && P=IAF_AI/dispatch/listener/stt_probe.py
    <venv>/bin/python $P
    <venv>/bin/python $P --seconds 20 --language en
    <venv>/bin/python $P --dump-events     # print every raw event

Acceptance:
    say a sentence -> see it appear word by word + receive a speech_final

Cost: STT is billed per **audio hour** at $0.20, so 15 seconds is about $0.0008.
★But silence is billed by the duration sent too, so do not leave it running -- this probe
  shuts itself down when its time is up.

★It also serves to:
    - form a **qualitative view** of mixed-language quality (how proper nouns in a second
      language come out under a given `language` setting)
    - **store real word-by-word streams as fixtures** (corpus material for the offline
      harness) -> `--save` writes `fixtures/stt_YYYYmmdd-HHMMSS.jsonl`
"""

import argparse
import asyncio
import json
import os
import sys
import time

def _find_shell_root(start):
    """Walk up to the shell root = the level that holds both `IAF_AI/` and `config_base/`.

    ★It does not count dirname levels: scripts get moved, and counting levels breaks the
      moment they do -- in a way that points the path at somewhere that does not exist, so
      most call sites silently get empty content.
    ★It does not go by name either: a cluster copy must never depend on the root's name or
      location.
    ★These five lines appear in several files on purpose: boilerplate like "find the root /
      lock / atomic write" is better duplicated than hoisted into a shared script.
    """
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, "IAF_AI"))
                and os.path.isdir(os.path.join(d, "config_base"))):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("cannot find the shell root (no directory above holds both "
                               "IAF_AI/ and config_base/): " + str(start))
        d = p


# This file lives in IAF_AI/dispatch/listener/ -- this cell's own scripts are all here, so
# they are imported as siblings.
HERE = os.path.dirname(os.path.abspath(__file__))       # = dispatch_dir
SHELL_ROOT = _find_shell_root(HERE)
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
if IAF_AI_ROOT not in sys.path:
    sys.path.insert(0, IAF_AI_ROOT)

from lib import stt_client   # noqa: E402

_TTY = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s):   return _c("2", s)
def bold(s):  return _c("1", s)
def cyan(s):  return _c("36", s)
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)


async def main(args):
    key = stt_client.resolve_key(SHELL_ROOT)
    stt_cfg = {"language": args.language, "sample_rate": 16000}
    if args.keyterm:
        stt_cfg["keyterm"] = args.keyterm

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    dropped = []
    rows = []                      # the fixture stream
    # ★The timeline origin is **the moment the mic opens**, not process start. It used to be
    #   main()'s start, which folded the 1.1s handshake into every timestamp, so replaying a
    #   fixture by t was offset as a whole (the replayer keys on t).
    t0 = {"mic": None}

    def _on_frame(buf):
        # ★PortAudio's own thread -> call_soon_threadsafe is the only option
        loop.call_soon_threadsafe(q.put_nowait, buf)

    def _ms():
        base = t0["mic"] or time.time()
        return int((time.time() - base) * 1000)

    def _row(text, ev, kind):
        rows.append({"t": _ms(), "text": text, "is_final": bool(ev.get("is_final")),
                     "speech_final": bool(ev.get("speech_final")), "kind": kind,
                     # ★start/duration are the server's coordinates on the **audio timeline**
                     #   (confirmed present by measurement). Only with start can "when the
                     #   person began speaking" be subtracted from the latency -- without it,
                     #   ttfp mostly measures the seconds of silence before they spoke.
                     "start": ev.get("start"), "duration": ev.get("duration"),
                     # ★conf must go into the stream: the plan is to use confidence as a
                     #   free pre-gate, and setting its threshold needs paired
                     #   confidence-versus-completeness data. Without storing it the offline
                     #   replay harness cannot tune that gate -- the only way back is to
                     #   re-record, which costs money and someone's time.
                     "conf": ev.get("end_of_turn_confidence"),
                     "language": ev.get("language")})

    last_len = [0]

    def on_partial(text, ev):
        _row(text, ev, "partial")
        if _TTY and not args.plain:   # rewrite in place: an interim partial is meant to be rewritten
            pad = " " * max(0, last_len[0] - len(text))
            print(f"\r{dim('…')} {text}{pad}", end="", flush=True)
            last_len[0] = len(text)
        else:
            print(f"{dim('…')} {text}", flush=True)

    def on_final(text, ev):
        _row(text, ev, "final")
        # ★Finals with empty text really do arrive (3 of them in the first probe run).
        #   Printing a blank line reads as "words were dropped", so label it explicitly --
        #   it means "the text is locked but this segment has no content", not a bug.
        print(f"\r{cyan('final')} {text if text else dim('(empty)')}", flush=True)
        last_len[0] = 0

    def on_speech_final(text, ev):
        _row(text, ev, "speech_final")
        conf = ev.get("end_of_turn_confidence")
        tail = f"  {dim(f'end_of_turn_confidence={conf}')}" if conf is not None else ""
        print(f"\r{green('done')} {bold(text)}{tail}", flush=True)
        last_len[0] = 0

    def on_event(ev):
        if args.dump_events:
            print(dim(f"   ev {json.dumps(ev, ensure_ascii=False)[:300]}"), flush=True)

    print(bold(f"stt_probe · {args.seconds}s · language={args.language}"
               f"{' · keyterm=' + ','.join(args.keyterm) if args.keyterm else ''}"))
    print(dim("  connecting... (the handshake measures ~1.1s of TLS + WS, which is why it is worth pre-warming)"))

    mic = {"st": None}
    mic_ready = asyncio.Event()

    def _open_mic():
        # ★★The mic must not open until transcript.created arrives. Opening it first means
        #   the handshake second of audio piles up in the queue and then floods in at once --
        #   the audio timeline and the wall clock diverge from that point on and **every
        #   latency number becomes unusable** (measured on the first run: a 1127ms handshake,
        #   11 frames backed up, ttfp reading 1088ms but with an unknown offset inside it).
        #   This is what "pre-warm the connection" looks like on the client side.
        t0["mic"] = time.time()
        mic["st"] = stt_client.mic_stream(device=args.device, on_frame=_on_frame,
                                         on_status=lambda s: dropped.append(s))
        print(green("  connected · mic open -- talk now.")
              + dim(f" (it shuts down after {args.seconds} seconds)"), flush=True)
        mic_ready.set()

    async def stopper():
        await mic_ready.wait()     # ★counted from the mic opening, not from process start
        await asyncio.sleep(args.seconds)
        await q.put(None)          # the sentinel = end of audio -> the client sends audio.done

    try:
        m, _ = await asyncio.gather(
            stt_client.stream(stt_cfg, key, q, on_partial=on_partial, on_final=on_final,
                              on_speech_final=on_speech_final, on_event=on_event,
                              on_ready=_open_mic),
            stopper(),
        )
    finally:
        if mic["st"] is not None:
            mic["st"].stop()
            mic["st"].close()

    m["audio_status"] = dropped
    print()
    print(bold("-- metrics --"))
    print(f"  handshake {m['connect_ms']}ms · transcript.created +{m['created_ms']}ms")
    # ★Derived: subtract "the silence before the person spoke" from ttfp. The server gives
    #   start (on the audio timeline), so the true model latency = when the first partial
    #   arrived minus the speech start it claims.
    first = next((r for r in rows if r["kind"] == "partial" and r["text"]), None)
    if first and first.get("start") is not None:
        arrived = first["t"]
        speech_at = int(first["start"] * 1000)
        print("  " + bold(f"true first-partial latency {arrived - speech_at}ms")
              + dim(f" (arrived +{arrived}ms - speech start {speech_at}ms)"))
    lat = sorted(m.get("event_latency_ms") or [])
    if lat:
        print("  " + bold(f"pure pipeline delay n={len(lat)} · median {lat[len(lat)//2]}ms"
                          f" · range {lat[0]}–{lat[-1]}ms")
              + dim("  <- arrival -(start+duration), with the pre-speech silence removed"))
        print(dim(f"    against the budget: RTT 180-250 + STT interim results 200-500 = 380-750ms"))
    print(dim(f"  first partial (counted from the first frame) {m['ttfp_after_first_frame_ms']}ms"
              f" <- for comparison only; **do not budget latency with it** (it includes when the person spoke)"))
    print(f"  audio sent {m['audio_seconds']:.1f}s / {m['frames_sent']} frames"
          f" · about ${m['audio_seconds']/3600*0.20:.5f}")
    print(f"  event counts {m['events']}")
    if dropped:
        print(yellow(f"  ★PortAudio reported status {len(dropped)} time(s) (overflow = dropped frames): {set(dropped)}"))
    if m["errors"]:
        print(yellow(f"  ★errors {m['errors']}"))

    # ★The one required "print against the first real event" -- field names can only be
    #   established this way
    print(bold("\n-- the first occurrence of each event type (the authority on field names) --"))
    for typ, ev in m["first_of_type"].items():
        print(f"  {typ}: {json.dumps(ev, ensure_ascii=False)[:400]}")

    # -- acceptance criteria --
    got_partial = any(r["kind"] == "partial" and r["text"] for r in rows)
    got_sf = any(r["kind"] == "speech_final" for r in rows)
    print(bold("\n-- acceptance --"))
    print(f"  word-by-word on screen (a non-empty partial arrived): {'OK' if got_partial else '★no'}")
    print(f"  speech_final received                               : {'OK' if got_sf else '★no'}")
    n_partial = sum(1 for r in rows if r["kind"] == "partial")
    print(f"  {n_partial} partials in total ("
          f"{sum(1 for r in rows if r['kind']=='partial' and r['text'])} of them non-empty)")
    # ★Print the word-by-word progress of the last utterance -- \r rewriting looks good in a
    #   terminal but cannot be copied, and this block is the only pasteable evidence that
    #   word-by-word output really happened.
    tail = []
    for r in reversed(rows):
        tail.append(r)
        if r["kind"] == "speech_final" and len(tail) > 1:
            break
    tail = list(reversed(tail))
    if len(tail) > 1:
        print(dim("  word-by-word progress of the last utterance:"))
        for r in tail[:12]:
            print(dim(f"    +{r['t']:>5}ms {r['kind']:<12} {r['text']!r}"))
        if len(tail) > 12:
            print(dim(f"    ...({len(tail)} in total)"))
    print(f"  {len(rows)} rows in the stream"
          + (f" · including {sum(1 for r in rows if r['kind']=='speech_final')} speech_final segments" if rows else ""))

    # ★The report is written to disk **unconditionally**, with no --save needed. The reason
    #   is empirical: copy-pasting from a terminal is an unreliable channel -- across three
    #   consecutive runs the output was copied away before the metrics had printed, so the
    #   field names (which had to be read off a real event at least once) were never
    #   captured. Once it is on disk, anyone can read it afterwards.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    rp = os.path.join(HERE, "fixtures", f"stt_report_{stamp}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "rows": rows,
                   "args": {"seconds": args.seconds, "language": args.language,
                            "keyterm": args.keyterm}},
                  f, ensure_ascii=False, indent=2)
    print(bold(f"\n  ★full report written to {rp}"))
    print(dim("    (metrics + the first occurrence of each event type + the word-by-word stream are all in it)"))

    if args.save:
        p = os.path.join(HERE, "fixtures", f"stt_{stamp}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  fixture stream written to {p}")

    return 0 if (got_partial and got_sf) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="a live STT connection probe")
    ap.add_argument("--seconds", type=int, default=15, help="how many seconds the mic stays open (default 15)")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--keyterm", action="append", help="a keyterm; repeatable (<=100 of them, each <=50 characters)")
    ap.add_argument("--plain", action="store_true",
                    help="print each partial on its own line instead of rewriting in place -- "
                         "**use this when you want to paste the output somewhere** (the default "
                         "\\r rewriting looks good in a terminal but gets overwritten and cannot be copied)")
    ap.add_argument("--dump-events", action="store_true", help="print every raw event")
    ap.add_argument("--save", action="store_true", help="write the word-by-word stream to fixtures/*.jsonl")
    a = ap.parse_args()
    try:
        sys.exit(asyncio.run(main(a)))
    except KeyboardInterrupt:
        print()
