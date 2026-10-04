"""
jev_bench.py -- a fixture replay harness for the JEV#1 question.

Zero audio, zero microphone, repeatable. The corpus is the transcripts of real recorded
runs (fixtures/stt_*.jsonl), with the ground truth in fixtures/jev1_labels.json.

    # First look at the hand offline (no cost, no network):
    cd jev_voice_agent && B=IAF_AI/dispatch/listener/jev_bench.py
    <venv>/bin/python $B --dry
    # One real call, printing the server response verbatim (~$0.00001) -- the endpoint and
    # field names are established from a real reply:
    <venv>/bin/python $B --smoke
    # Full replay (23 cases + 1 warm-up, ~$0.0003):
    <venv>/bin/python $B

★Three disciplines, all learned the hard way

1. **The report is written to disk unconditionally** (fixtures/jev1_bench_<ts>.json) before
   anything is printed. Copy-pasting from a terminal is an unreliable channel -- over at
   stt_probe, three consecutive runs were copied away before the metrics had printed
   （build_plan §8.9）。
2. **The positive and negative examples in the question must not come from the corpus.**
   The 23 cases here are a held-out test set; the question's examples are separately
   hand-written new sentences of the same shape. Mix them and you are measuring recitation.
   `--dry` checks this.
3. **The threshold is swept, not pinned.** noul is a calibrated probability, so once one
   call has returned it, every threshold can be recomputed for free -> every run prints the
   whole sweep curve (this knob does not tune accuracy; it tunes how many times the LLM runs
   per utterance).
"""

import argparse
import asyncio
import glob
import json
import os
import re
import shutil
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
if IAF_AI_ROOT not in sys.path:                 # lib/jev_client + IAF_AI/reply_list
    sys.path.insert(0, IAF_AI_ROOT)

import listener_loop                                          # noqa: E402
import reply_list                                             # noqa: E402
from lib import jev_client                                    # noqa: E402
from lib.jev_client import JevClient, JevError                # noqa: E402

DISPATCH_DIR = HERE                             # this file lives in the listener cell
FIXTURES = os.path.join(HERE, "fixtures")       # the corpus travels with this cell
LABELS_PATH = os.path.join(FIXTURES, "jev1_labels.json")


# ---------------------------------------------------------- deriving cases from the corpus

def derive_cases(fixtures_dir=FIXTURES):
    """Derive replay cases from real transcripts by fixed rules (a pure function, testable
    offline).

    (1) every speech_final segment (= what the old fallback path really handed downstream)
    (2) the **cumulative U snapshot** at every non-empty is_final (cleared after a
        speech_final) -- this family is what JEV#1 actually sees when early submission buys
        the overlap.

    ★Finals and partials with empty text are always skipped: they really do arrive, the two
      guards in UserBuffer exist to block them, and the harness should not treat them as
      cases either.
    """
    cases = {}
    for path in sorted(glob.glob(os.path.join(fixtures_dir, "stt_*.jsonl"))):
        run = os.path.basename(path)
        u = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if row.get("speech_final"):
                    if text:
                        c = cases.setdefault(text, {"text": text, "families": set(), "runs": set()})
                        c["families"].add("speech_final")
                        c["runs"].add(run)
                    u = []
                elif row.get("is_final"):
                    if not text:
                        continue
                    u.append(text)
                    snap = " ".join(u)
                    c = cases.setdefault(snap, {"text": snap, "families": set(), "runs": set()})
                    c["families"].add("u_snapshot")
                    c["runs"].add(run)
    out = []
    for c in cases.values():
        out.append({"text": c["text"],
                    "families": sorted(c["families"]),
                    "runs": sorted(c["runs"])})
    return out


def load_labels(path=LABELS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["labels"]


def join_cases_labels(cases, labels):
    """Align cases with labels. ★A missing or surplus label is a hard error, never a
    silently smaller test run."""
    by_text = {x["text"]: x for x in labels}
    missing = [c["text"] for c in cases if c["text"] not in by_text]
    extra = [t for t in by_text if t not in {c["text"] for c in cases}]
    rows = []
    for c in cases:
        lab = by_text.get(c["text"])
        if lab is None:
            continue
        rows.append({**c, "label": bool(lab["label"]), "bucket": lab.get("bucket"),
                     "why": lab.get("why"), "debatable": bool(lab.get("debatable"))})
    return rows, missing, extra


def check_train_test_split(questions, cases):
    """Discipline 2: the question's positive and negative examples must not appear in the
    test set."""
    examples = []

    def walk(node, collecting):
        """★The two parameters cannot be merged into one: criteria has the shape
           {true: {rule, examples}} -- the walk has to descend everywhere (so "do not
           recurse into non-example keys" is wrong), but only strings underneath an
           examples key count as examples (so "collect every string you see" is wrong too).
           ★This was once written as a single-parameter recursion, which reached no example
           at all and still printed "none of them are in the test set" = a false green.
           Exactly the "it parses, therefore it works" class of bug."""
        if isinstance(node, str):
            if collecting:
                examples.append(node)
        elif isinstance(node, list):
            for x in node:
                walk(x, collecting)
        elif isinstance(node, dict):
            for k, v in node.items():
                # ★"例" is the key a Chinese-language question file uses. The shipped
                #   question now uses "examples"; the Chinese key is kept so a translated-back
                #   or Chinese question still has its examples checked.
                walk(v, collecting or k in ("例", "examples"))
    for q in questions.values():
        walk(q.get("criteria"), False)
        walk(q.get("instructions"), False)
    case_texts = {c["text"] for c in cases}
    norm = {re.sub(r"[\s\W_]+", "", t) for t in case_texts}
    leaks = [e for e in examples if e in case_texts or re.sub(r"[\s\W_]+", "", e) in norm]
    return examples, leaks


# -------------------------------------------------- 1 Hz word-by-word replay

async def replay(judge, tick_s=1.0, trim_units=1, fixtures_dir=FIXTURES):
    """★The replayer, and the only one. It advances U with partials/finals by `t`, ticks a
    **logical clock** every tick_s, hands the tail-shaved U to `judge`, **and feeds the
    verdict back into the buffer**.

    `judge(state) -> (noul|None, meaningful|None)`; --dry passes a fake judge that always
    returns (None, None) -> the same replay logic runs once at zero cost. **No second
    implementation exists** (two would inevitably drift).

    The difference from derive_cases (segment-level snapshots), and the entire reason this
    exists:
      derive_cases only snapshots at **final boundaries** -- whereas the U a 1 Hz ticker sees
      has **a live partial as its last segment**. Measured, 13 of 17 turns had just one final
      boundary in the whole turn, so measuring "early submission" at that granularity found
      only 2 of 17 -- **a lower bound, not the truth**.

    ★The verdict must be fed back: complete -> submit -> `U.clear()` (all or nothing is the
      only mode). Without feedback U keeps accumulating and the same turn is judged complete
      over and over -> "runs per turn" is over-counted (the first version made exactly this
      mistake: two turns reported as four).

    ★★Two things changed in 2026-09-27 to follow production (the speech_final fallback was
      deleted):

      (1) **`speech_final` no longer submits and no longer clears U**; it is only a time
          marker meaning "STT thinks this segment is finished". Without this change the
          harness measures a production path that **no longer exists**.
      (2) **Three-strikes clearing is now mandatory.** The old note "deliberately no three
          strikes here: it does not change JEV's answer, only when U is cleared" held while
          the fallback existed (every speech_final cleared U once). With the fallback gone,
          three strikes is the **only** cleanup path -- without it U accumulates without
          bound across the whole run, the state fed to JEV grows ever longer, and nothing
          measured means anything.
      ★Three strikes is **not reimplemented here**: `ListenerLoop._strike` is called as an
        unbound method against a shell holding only the few fields it touches. A
        reimplementation would inevitably drift from production -- and once it had, the
        harness would still be green, just measuring a different system.
    ★It does not really wait (the logical clock is advanced directly), so a whole batch runs
      instantly.
    """
    import collections
    import tempfile

    # ★The shell for three strikes: `_strike` only touches these names (u / knobs / strikes /
    #   _last_verdict_len / stats / _note), so this calls **the production implementation**
    #   with nothing copied.
    class _StrikeHost:
        def __init__(self, u, knobs):
            self.u, self.knobs = u, knobs
            self.strikes = 0
            self._last_verdict_len = None
            self.stats = collections.Counter()

        def _note(self, *a, **k):
            pass

    knobs = dict(listener_loop.DEFAULT_KNOBS)
    try:                                    # the same config as production, not a second set of numbers
        cfg = jev_client.load_dispatch_config(DISPATCH_DIR)
        for k, v in (cfg.get("knobs") or {}).items():
            if not k.startswith("_"):
                knobs[k] = v
    except Exception:
        pass

    events = []
    for path in sorted(glob.glob(os.path.join(fixtures_dir, "stt_*.jsonl"))):
        run = os.path.basename(path)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        rows.sort(key=lambda r: r["t"])
        tmp = tempfile.mkdtemp(prefix="jevbench_")
        u = reply_list.UserBuffer(tmp)
        host = _StrikeHost(u, knobs)
        turn, next_tick = 1, tick_s
        async def one_tick(at, turn_no):
            u_full = u.text()
            state = u.text_for_jev(trim_units)
            skip = jev_client.should_skip(state)
            ev = {"run": run, "turn": turn_no, "t": round(at, 2), "kind": "tick",
                  "state": state, "u_full": u_full, "skipped": skip,
                  "noul": None, "submitted": False}
            if not skip:
                noul, meaningful = await judge(state)
                ev["noul"] = noul
                if meaningful:
                    ev["submitted"] = True
                    u.clear()              # ★submitting clears everything; the verdict feeds back
                    host.strikes = 0
                    host._last_verdict_len = None
                elif meaningful is False:
                    # ★judged incomplete -> record a strike (the production implementation, see _StrikeHost above)
                    before = host.stats["strike_cleared"]
                    listener_loop.ListenerLoop._strike(host, len(u_full))
                    ev["strikes"] = host.strikes
                    if host.stats["strike_cleared"] > before:
                        ev["strike_cleared"] = True
            events.append(ev)
            return ev

        try:
            for r in rows:
                t = r["t"] / 1000.0
                while next_tick <= t:
                    await one_tick(next_tick, turn)
                    next_tick += tick_s
                text = (r.get("text") or "").strip()
                if r.get("speech_final"):
                    # ★★Purely a time marker: it does not submit and does not clear U (the
                    #   fallback was deleted). U carries on, waiting either for JEV#1 on the
                    #   ticker to pass it or for three strikes to clear it.
                    #   ★**Its own text is discarded too** -- across the recorded corpus,
                    #     17 of 17 had already been delivered by a preceding `is_final`
                    #     (13 equal to the last final, 4 cumulative over the whole turn).
                    left = u.text()
                    events.append({"run": run, "turn": turn, "t": round(t, 2),
                                   "kind": "speech_final", "state": left, "u_full": left,
                                   "skipped": None, "noul": None, "submitted": False})
                    turn += 1
                elif r.get("is_final"):
                    u.commit_final(text)
                else:
                    u.set_partial(text)

            # ★★The closing ticks **must** be played out: the corpus stops at the last
            #   speech_final, while production's ticker keeps turning. Without them the last
            #   utterance never receives a JEV#1 verdict -- so "segments never passed" is
            #   systematically over-counted and "pass lag" systematically under-counted, and
            #   the harness stays **green** while measuring something other than production.
            #   The cap is the strike threshold + 2 ticks: by then U has either passed or
            #   been cleared by three strikes.
            tail_cap = int(knobs["meaningless_strikes"]) + 2
            for _ in range(tail_cap):
                if not u.text().strip():
                    break
                await one_tick(next_tick, turn)
                next_tick += tick_s
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return events


def summarize_stream(events, threshold=None):
    """Per utterance: how many times JEV#1 passed, how many times three strikes cleared, and
    the lag from "STT done" to "passed".

    ★It reads only the `submitted` flag and never re-compares noul against a threshold here:
      submitting clears U, so the U seen by later ticks has already been changed by earlier
      verdicts, and **the threshold cannot be recomputed after the fact** (the segment-level
      branch can; this streaming one cannot). Changing the threshold means re-running.

    ★★The measure changed once the speech_final fallback was deleted:

      The old measure asked "how many milliseconds **earlier** than the fallback did early
      submission happen" (`early_lead_ms`), using the fallback as the reference. With the
      fallback gone there is no reference, and **the sign flips**: the question now is
      **how many milliseconds later than "STT done" JEV#1 passes** (`pass_lag_ms`), because
      that is the time-to-first-word cost this decision bought, measured on real material.

      ★`pass_lag_ms` is paired **across segments within a run**, not inside a segment: U now
        carries across a speech_final, so an utterance's pass **often happens inside a later
        segment** -- pairing within a segment records those as "never passed", leaving only
        the flattering half of the number.
      ★Negative values are meaningful: when someone finishes one sentence and carries
        straight on into the next, JEV#1 passes before STT is done (4 of 18 segments in the
        real material).
    """
    turns = {}
    for ev in events:
        d = turns.setdefault((ev["run"], ev["turn"]),
                             {"ticks": 0, "asked": 0, "passed": 0, "cleared": 0,
                              "first_pass_t": None, "stt_end_t": None,
                              "pass_lag_ms": None, "no_stt_end": True})
        if ev["kind"] == "tick":
            d["ticks"] += 1
            if ev["noul"] is not None:
                d["asked"] += 1
            if ev["submitted"]:
                d["passed"] += 1
                if d["first_pass_t"] is None:
                    d["first_pass_t"] = ev["t"]
            if ev.get("strike_cleared"):
                d["cleared"] += 1
        else:
            d["no_stt_end"] = False
            d["stt_end_t"] = ev["t"]

    # ★Cross-segment pairing: each "STT done" marker -> the **next** pass within the same run
    by_run = {}
    for ev in events:
        by_run.setdefault(ev["run"], []).append(ev)
    for run, evs in by_run.items():
        passes = [e["t"] for e in evs if e["kind"] == "tick" and e["submitted"]]
        for e in evs:
            if e["kind"] != "speech_final":
                continue
            nxt = next((t for t in passes if t >= e["t"]), None)
            d = turns[(run, e["turn"])]
            d["pass_lag_ms"] = round((nxt - e["t"]) * 1000) if nxt is not None else None
            if nxt is None:
                # This utterance was never passed before the run ended -- either three
                # strikes cleared it, or it is still sitting in U
                d["never_passed"] = True
    for d in turns.values():
        d["llm_runs"] = d["passed"]
    return turns


# ---------------------------------------------------------------- statistics

def score(rows, threshold, only_decided=True):
    """Compute the confusion matrix at a given threshold. Rows whose noul is None (skipped
    or failed) are excluded."""
    tp = fp = tn = fn = 0
    for r in rows:
        noul = r.get("noul")
        if noul is None:
            continue
        pred = noul > threshold
        if r["label"] and pred:
            tp += 1
        elif r["label"] and not pred:
            fn += 1
        elif not r["label"] and pred:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    return {"threshold": round(threshold, 3), "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": round((tp + tn) / n, 4) if n else None}


def sweep(rows, lo=0.05, hi=0.95, step=0.05):
    out = []
    t = lo
    while t <= hi + 1e-9:
        out.append(score(rows, t))
        t += step
    return out


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[i], 1)


# ---------------------------------------------------------------- main flow

async def run(args):
    cases = derive_cases()
    # ★★Fail with a sentence, not a stack trace. This repo ships an empty
    #   `fixtures/`, so "no corpus" is the state every new user starts in --
    #   it is not a bug and must not look like one.
    if not cases or not os.path.exists(LABELS_PATH):
        print("No corpus in fixtures/, so there is nothing to benchmark.")
        print(f"  expected: {FIXTURES}/stt_*.jsonl  +  {os.path.basename(LABELS_PATH)}")
        print("  record one with:  python stt_probe.py --save     (then hand-label it)")
        print("  why it is not shipped: the original corpus was real recordings of a")
        print("  specific person's speech, removed before open-sourcing.")
        return
    labels = load_labels()
    rows, missing, extra = join_cases_labels(cases, labels)

    print(f"corpus: {len(glob.glob(os.path.join(FIXTURES, 'stt_*.jsonl')))} runs -> {len(cases)} de-duplicated cases derived")
    if missing:
        print(f"[FAIL] {len(missing)} case(s) have no label (a silently smaller test run is not allowed):")
        for t in missing:
            print(f"        {t!r}")
        return 1
    if extra:
        print(f"[warn] {len(extra)} label(s) are not in the current corpus (did the corpus change?):")
        for t in extra:
            print(f"        {t!r}")

    qpath = args.question
    if qpath:
        with open(qpath, "r", encoding="utf-8") as f:
            questions = jev_client.extract_question_block(f.read())
    else:
        cfg = jev_client.load_dispatch_config(DISPATCH_DIR)
        questions = jev_client.load_questions(DISPATCH_DIR, cfg["agents"][args.agent])

    examples, leaks = check_train_test_split(questions, cases)
    print(f"question: {len(questions)} question(s) ({', '.join(questions)}) · {len(examples)} examples in criteria")
    if leaks:
        print(f"[FAIL] {len(leaks)} example(s) in the question overlap the test set -- that measures recitation:")
        for t in leaks:
            print(f"        {t!r}")
        return 1
    print("[OK  ] train/test separation: none of the question's examples are in the test set")

    n_true = sum(1 for r in rows if r["label"])
    print(f"ground truth: {n_true} complete · {len(rows) - n_true} incomplete · of which "
          f"{sum(1 for r in rows if r['debatable'])} are debatable")
    from collections import Counter
    print("            " + " · ".join(f"{k} {v}" for k, v in Counter(r["bucket"] for r in rows).items()))

    if args.dry and args.stream:
        async def no_judge(state):
            return None, None          # a fake judge: the same replay runs once at zero cost
        events = await replay(no_judge, args.tick, args.trim)
        only = [e for e in events if e["kind"] == "tick"]
        asked = [e for e in only if not e["skipped"]]
        print(f"\n--dry --stream: {len(only)} ticks ({args.tick} Hz logical clock) · "
              f"{len(only) - len(asked)} ticks with U empty or punctuation-only send nothing · "
              f"{len(asked)} real requests would be sent"
              f" ≈ ${len(asked) * 3.65e-05:.4f}")
        print("★Note this is an UPPER BOUND: the fake judge always says incomplete, so U is "
              "never submitted and cleared. A real run has fewer ticks (a pass clears U, and "
              "the state on following ticks gets shorter or empty)")
        for e in events:
            if e["kind"] != "tick":
                print(f"     t={e['t']:>6} ── speech_final ── {e['state']!r}")
            elif not e["skipped"]:
                trimmed = "" if e["state"] == e["u_full"] else f"   <- tail shaved from {e['u_full']!r}"
                print(f"  t={e['t']:>6} {e['state']!r}{trimmed}")
        return 0

    if args.dry:
        print("\n--dry: that is all, with no network and no cost. The case table follows\n")
        for i, r in enumerate(rows, 1):
            fam = "S" if r["families"] == ["speech_final"] else ("U" if r["families"] == ["u_snapshot"] else "SU")
            print(f"  {i:2d} [{fam}] {'complete' if r['label'] else 'fragment '}{'?' if r['debatable'] else ' '} "
                  f"{r['bucket']}  {r['text']!r}")
        return 0

    async with JevClient(DISPATCH_DIR, args.agent, threshold=args.threshold) as jev:
        print(f"\nendpoint {jev.endpoint}\nmodel {jev.model} · threshold {jev.threshold} · timeout {jev.timeout_s}s")

        if args.smoke:
            st = rows[0]["text"]
            print(f"\n--smoke: sending one request, {st!r}")
            r = await jev.ask(st)
            path = os.path.join(FIXTURES, f"jev1_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            print(f"written to {path}")
            print(json.dumps(r["raw"], ensure_ascii=False, indent=2))
            print(f"\nnoul={r['noul']}  meaningful={r['meaningful']}  {r['latency_ms']:.0f}ms")
            return 0

        print("\nwarming up the connection (the first request includes the TLS handshake, measured at ~700ms versus ~300ms after)...")
        warm_ms = await jev.warmup()
        print(f"  first request {warm_ms:.0f}ms")

        if args.stream:
            return await run_stream(jev, args, warm_ms)

        todo = rows[:args.limit] if args.limit else rows
        errors = []
        for i, r in enumerate(todo, 1):
            try:
                res = await jev.ask(r["text"])
            except JevError as e:
                r["noul"] = None
                r["error"] = str(e)
                errors.append(f"{r['text']!r}: {e}")
                print(f"  {i:2d} [FAIL] {e}")
                continue
            r["noul"] = res["noul"]
            r["meaningful"] = res["meaningful"]
            r["latency_ms"] = res["latency_ms"]
            r["skipped"] = res["skipped"]
            r["usage"] = res["usage"]
            ok = "✓" if res["meaningful"] == r["label"] else "✗"
            noul_s = "-skipped-" if res["noul"] is None else f"{res['noul']:.3f}"
            print(f"  {i:2d} {ok} noul={noul_s} truth={'complete' if r['label'] else 'fragment'} "
                  f"{r['bucket']}{'?' if r['debatable'] else ''}  {r['text'][:46]!r}")

        lat = [r["latency_ms"] for r in todo if r.get("latency_ms")]
        main = score(todo, jev.threshold)
        firm = score([r for r in todo if not r["debatable"]], jev.threshold)
        sw = sweep(todo)
        best = max((s for s in sw if s["acc"] is not None), key=lambda s: s["acc"], default=None)

        report = {
            "args": vars(args), "endpoint": jev.endpoint, "model": jev.model,
            "threshold": jev.threshold, "warmup_ms": round(warm_ms, 1),
            "calls": jev.calls, "input_tokens": jev.total_input_tokens,
            "cost_usd": jev.total_cost_usd,
            "latency_ms": {"n": len(lat), "p50": pct(lat, 50), "p90": pct(lat, 90),
                           "max": round(max(lat), 1) if lat else None},
            "at_threshold": main, "at_threshold_excl_debatable": firm,
            "sweep": sw, "best_threshold": best, "errors": errors,
            "buckets": {},
            "rows": todo,
        }
        for b in sorted({r["bucket"] for r in todo}):
            report["buckets"][b] = score([r for r in todo if r["bucket"] == b], jev.threshold)

        path = os.path.join(FIXTURES, f"jev1_bench_{time.strftime('%Y%m%d-%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n★report written to {path} (written unconditionally; copying from a terminal is unreliable)")

        print(f"\nthreshold {jev.threshold}: {main['tp']+main['tn']}/{main['n']} = {main['acc']}"
              f"   (excluding the 2 debatable cases: {firm['tp']+firm['tn']}/{firm['n']} = {firm['acc']})")
        print(f"  missed completes (fn) {main['fn']} · false completes (fp) {main['fp']}")
        print("  by bucket: " + " · ".join(
            f"{b} {s['tp']+s['tn']}/{s['n']}" for b, s in report["buckets"].items()))
        print(f"\nlatency n={len(lat)} p50 {pct(lat,50)}ms p90 {pct(lat,90)}ms max "
              f"{round(max(lat),1) if lat else None}ms (after warm-up)")
        print(f"cost: {jev.calls} calls · input {jev.total_input_tokens} tok · "
              f"server-reported ${jev.total_cost_usd:.6f}")
        print("\nthreshold sweep (noul is already in hand, so recomputing is free):")
        for s in sw:
            bar = "█" * int(round((s["acc"] or 0) * 40))
            mark = " <- best" if best and s["threshold"] == best["threshold"] else ""
            print(f"  {s['threshold']:.2f}  {s['acc']}  fn={s['fn']} fp={s['fp']}  {bar}{mark}")
        if errors:
            print(f"\n[FAIL] {len(errors)} error(s)")
            return 1
    return 0


async def run_stream(jev, args, warm_ms):
    """1 Hz word-by-word replay. ★Only this branch answers "how many times does it fire per
    utterance" (the distribution); the segment-level replay cannot."""
    print(f"\n{args.tick} Hz logical clock · tail shaved by {args.trim} unit(s) · a complete "
          f"verdict submits and clears U entirely")
    errors, lat = [], []

    async def judge(state):
        try:
            res = await jev.ask(state)
        except JevError as e:
            errors.append(str(e))
            print(f"        [FAIL] {e}")
            return None, None
        lat.append(res["latency_ms"])
        return res["noul"], res["meaningful"]

    events = await replay(judge, args.tick, args.trim)
    for ev in events:
        if ev["kind"] != "tick":
            mark = "U is empty (JEV#1 passed)" if not ev["u_full"].strip() else "U still holds, waiting for the next tick"
            print(f"     t={ev['t']:>6} ── speech_final ── {mark} {ev['state'][:44]!r}")
        elif ev["noul"] is not None:
            hit = "★complete -> submitted, U cleared" if ev["submitted"] else "                                  "
            trimmed = "  <- tail shaved" if ev["state"] != ev["u_full"] else ""
            print(f"  t={ev['t']:>6} noul={ev['noul']:.3f} {hit} {ev['state'][:44]!r}{trimmed}")

    turns = summarize_stream(events)
    runs = [d["llm_runs"] for d in turns.values()]
    lag = [d["pass_lag_ms"] for d in turns.values() if d["pass_lag_ms"] is not None]
    never = sum(1 for d in turns.values() if d.get("never_passed"))
    cleared = sum(d["cleared"] for d in turns.values())
    unterminated = sum(1 for d in turns.values() if d["no_stt_end"])
    from collections import Counter
    dist = Counter(runs)

    report = {"mode": "stream", "args": vars(args), "endpoint": jev.endpoint,
              "model": jev.model, "threshold": jev.threshold, "tick_s": args.tick,
              "trim_units": args.trim, "warmup_ms": round(warm_ms, 1),
              "calls": jev.calls, "input_tokens": jev.total_input_tokens,
              "cost_usd": jev.total_cost_usd,
              "latency_ms": {"n": len(lat), "p50": pct(lat, 50), "p90": pct(lat, 90)},
              "turns": {f"{k[0]}#{k[1]}": v for k, v in turns.items()},
              "unterminated_turns": unterminated,
              "llm_runs_per_turn": dict(sorted(dist.items())),
              "segments_total": len(turns),
              "segments_never_passed": never, "strike_cleared_total": cleared,
              "pass_lag_ms": {"n": len(lag), "p50": pct(lag, 50),
                              "p90": pct(lag, 90),
                              "min": min(lag) if lag else None,
                              "max": max(lag) if lag else None},
              "errors": errors, "events": events}
    path = os.path.join(FIXTURES, f"jev1_stream_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n★report written to {path}")

    print(f"\n★LLM runs per utterance (threshold {jev.threshold}, {len(turns)} segments, "
          f"{unterminated} of which never saw a speech_final at the end):")
    for k in sorted(dist):
        print(f"    {k} run(s) x {dist[k]} segment(s)  {'█' * dist[k]}")
    print(f"  mean {sum(runs)/len(runs):.2f} runs per segment")
    print(f"\n★★Lag from \"STT done\" to \"JEV#1 passed\" (= the time-to-first-word cost bought "
          f"by deleting the speech_final fallback, measured in the field):")
    if lag:
        print(f"    p50 {pct(lag,50)}ms · p90 {pct(lag,90)}ms · range {min(lag)}-{max(lag)}ms"
              f" · n={len(lag)}")
        print(f"    ★a negative value = JEV#1 passed BEFORE STT was done (one sentence finished and the next began)")
    else:
        print(f"    ★nothing paired at all -- no segment was ever passed; check the threshold first")
    print(f"\n★three strikes cleared {cleared} time(s) · {never}/{len(turns)} segments were "
          f"never passed before the run ended (dropped by three strikes, or still sitting in U)")
    print(f"\nlatency n={len(lat)} p50 {pct(lat,50)}ms p90 {pct(lat,90)}ms")
    print(f"cost: {jev.calls} calls · input {jev.total_input_tokens} tok · ${jev.total_cost_usd:.6f}")
    print(f"  -> {jev.total_cost_usd/max(1,len(turns)):.6f} USD per utterance")
    if errors:
        print(f"\n[FAIL] {len(errors)} error(s)")
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true", help="no network, no cost: only derive cases, check labels and train/test separation")
    p.add_argument("--smoke", action="store_true", help="send one real call and print the server response verbatim (to establish the endpoint and field names)")
    p.add_argument("--agent", default="jev1")
    p.add_argument("--question", help="run A/B against a different question file (defaults to the one in the config)")
    p.add_argument("--threshold", type=float, help="override knobs.meaningful_threshold")
    p.add_argument("--limit", type=int, help="run only the first N cases")
    p.add_argument("--stream", action="store_true",
                   help="★the real replay: advance U with partials/finals by t, tick a 1 Hz "
                        "logical clock, and shave the tail before feeding JEV. It measures how "
                        "many times the LLM runs per utterance")
    p.add_argument("--tick", type=float, default=1.0, help="tick interval in seconds (default 1.0)")
    p.add_argument("--trim", type=int, default=1, help="how many units to shave off the live segment before feeding JEV (default 1)")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
