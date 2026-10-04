"""tts_live_check.py -- live acceptance for the mouth, all in one run (real LLM stream +
real speakers).

★★Once barge-in was removed and invalidation deleted along with it, this harness's third
  check (`text.clear` clearing two levels) **has no production path left at all**:
  `SpeakSink` never calls `clear()`, and the one route invalidation might have used went
  away with the mechanism. The only live caller of `tts_client.clear()` today is
  `tts_probe.py` (a manual probe that costs money). `--cut-after` therefore has exactly one
  honest behaviour left: refuse, and say why.
  The **statement order** of the two-level clear is guarded at source level by `dry_suite`
  assertion 19f.

Usage:
    cd jev_voice_agent && P=IAF_AI/dispatch/voice_out/tts_live_check.py
    <venv>/bin/python $P                     # makes sound for roughly 10-20 seconds

What it verifies:
  (1) `tts_ttfa_ms` has a value                      -- printed
  (2) **the input is a real LLM stream** (not a hard-coded string) -- it goes through
      chain.new_voice_session
  (3) `text.clear` clears two levels                 -- ★not verifiable in this harness any
      more (see the file header); only (1) and (2) remain

★**The microphone is not opened, so no headphones are needed.** Only the full voice chain
  (mic and mouth open at once) needs them.
★Cost: one LLM turn ~$0.001 + ~100 characters of TTS ~$0.0015.
"""

import argparse
import asyncio
import os
import sys
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
for _p in (SHELL_ROOT, os.path.join(SHELL_ROOT, "IAF_AI"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assemble_pool       # noqa: E402
import chain               # noqa: E402
import speak_sink          # noqa: E402

G, R, Y, D, B, Z = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def mark(b):
    return f"{G}✓{Z}" if b else f"{R}✗{Z}"


async def run(args):
    conn, player, make_sink = speak_sink.build(device=args.device, verbose=False)
    print(f"{B}Live acceptance (mouth){Z}")
    print(f"{D}  output {args.device or 'default device'} · {player.sr} Hz · mic stays closed, no headphones needed{Z}")
    player.start()
    lat = player.latency_s
    print(f"{D}  sound card reports {lat * 1000:.0f}ms latency (subtracted from the spoken-character count){Z}")

    checks = {}
    try:
        async with conn as tts:
            print(f"{D}  TTS handshake {tts.m['connect_ms']}ms{Z}")
            sink = make_sink(tts)
            drv, session_dir = chain.new_voice_session(peer=args.peer, sink=sink)

            before = assemble_pool.thread_path(args.peer)
            n_before = len(open(before, encoding="utf-8").read()) if os.path.exists(before) else 0

            drv.reservoir.add(args.ask)
            print(f"\n  {B}ask{Z} {args.ask}")
            t0 = time.time()
            task = asyncio.create_task(drv.pump())

            # ★With barge-in removed nothing can cut this off (--cut-after is rejected up
            #   in main), so this call necessarily runs to its natural end. There used to
            #   be measurements here for "did any audio leak through after the clear"
            #   (fed_after_clear / queued_after_clear / fed_late / cl, plus a 0.5s wait);
            #   those quantities only exist when something interrupts, so they went too.
            rec = await task

            t = sink.turn
            total_ms = int((time.time() - t0) * 1000)

            print(f"\n  {B}answer{Z} {rec['spoken']!r}")
            if rec["dropped_chars"]:
                print(f"  {D}({rec['dropped_chars']} characters were never spoken){Z}")

            # -- the checks --
            checks["(1) tts_ttfa_ms has a value"] = (
                t.get("ttfa_ms") is not None,
                f"first sound {t.get('ttfa_ms')}ms · LLM first chunk {rec['first_chunk_ms']}ms · "
                f"end to end {total_ms}ms")
            checks["(2) input is a real LLM stream (not hard-coded)"] = (
                bool(rec.get("generation_id")) and rec["total_ms"] > 0,
                f"generation_id={rec.get('generation_id')} · "
                f"usage={ {k: v for k, v in (rec.get('usage') or {}).items() if k in ('prompt_tokens', 'completion_tokens', 'cost')} }")
            # ★There used to be two more checks here (the two-level text.clear, and "it
            #   really was interrupted mid-way"), both of which depend on barge-in. With
            #   barge-in removed they are UNREACHABLE in this harness -- rather than leave
            #   hollow branches behind, they were deleted. The statement order of the
            #   two-level clear is guarded by `dry_suite` 19f (see the file header).
            checks["★playback has no dropouts (prefetch is sufficient)"] = (
                t.get("gaps", 0) == 0,
                f"{t.get('gaps', 0)} dropouts (raw underruns {player.underruns}, the vast "
                f"majority of which are idle callbacks before playback and after the turn, "
                f"inaudible)"
                + (f" · dropouts at {t.get('gap_log')} (seconds into playback, silence ms)"
                   if t.get("gap_log") else "")
                + ("" if t.get("gaps", 0) == 0 else f"  {Y}<- raise knobs.playback_lead_s{Z}"))
            print(f"\n  {D}per chunk: " + " | ".join(
                f"{c['chars']} chars gen {c['gen_ms']}ms audio {c['audio_s']:.1f}s queued {c['queued_s']}s"
                for c in t.get("per_chunk", [])) + Z)
            n_after = len(open(before, encoding="utf-8").read()) if os.path.exists(before) else 0
            checks["★the record keeps only the prefix that really was spoken"] = (
                n_after > n_before and rec["spoken"] == (sink.turn.get("_spoken_text") or rec["spoken"]),
                f"{assemble_pool.thread_path(args.peer)} grew by {n_after - n_before} characters")

            print(f"\n{D}  TTS {tts.m['chars_sent']} chars ${tts.cost_usd:.5f} · "
                  f"LLM ${(rec.get('usage') or {}).get('cost') or 0:.5f}{Z}")
            if tts.m["errors"]:
                print(f"  {R}tts errors: {tts.m['errors']}{Z}")
    finally:
        player.stop()

    print()
    bad = 0
    for k, (okk, ev) in checks.items():
        print(f"  {mark(okk)} {k}")
        print(f"      {D}{ev}{Z}")
        bad += 0 if okk else 1
    print()
    print((f"{G}{B}all passed{Z}" if bad == 0 else f"{R}{B}{bad} failed{Z}"))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="live acceptance for the mouth")
    ap.add_argument("--ask", default="in three or four sentences, what is a vector database for")
    ap.add_argument("--peer", default="live_check",
                    help="★deliberately not 'user': keep acceptance runs out of the real thread")
    # ★This flag is kept so that it can REFUSE EXPLICITLY rather than silently do nothing:
    #   anyone typing the old command should see the reason on the spot.
    ap.add_argument("--cut-after", type=float, default=0.0,
                    help="★no longer functional (barge-in was removed). Passing >0 refuses to run and explains why")
    ap.add_argument("--no-cut", dest="cut_after", action="store_const", const=0.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.cut_after > 0:
        print("★--cut-after no longer does anything: barge-in was removed and invalidation "
              "deleted with it, so **nothing can cut off a turn that has started speaking**.\n"
              "  Clearing (local queue + text.clear + the in-flight discard window) has no "
              "caller in the production path; the only live caller is tts_probe.py (a manual probe).\n"
              "  Its statement order is guarded at source level by dry_suite assertion 19f "
              "(offline, free).")
        sys.exit(2)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
