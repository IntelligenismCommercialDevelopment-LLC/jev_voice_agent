import argparse
import asyncio
import collections
import contextlib
import json
import os
import statistics
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
            raise RuntimeError("cannot find the shell root (no directory above holds both IAF_AI/ and config_base/): "
                               + str(start))
        d = p


SHELL_ROOT = _find_shell_root(os.path.dirname(os.path.abspath(__file__)))
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
DISPATCH_DIR = os.path.join(IAF_AI_ROOT, "dispatch", "listener")
VOICE_OUT_DIR = os.path.join(IAF_AI_ROOT, "dispatch", "voice_out")
for _p in (SHELL_ROOT, IAF_AI_ROOT, DISPATCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assemble_pool
import chain
import listener_loop
from lib import stt_client
from lib.stt_client import implausible as _implausible, speech_rate as _speech_rate
from lib.jev_client import JevClient, load_dispatch_config
_TTY = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s):   return _c("2", s)


def bold(s):  return _c("1", s)


def cyan(s):  return _c("36", s)


def green(s): return _c("32", s)


def yellow(s): return _c("33", s)


class TypingSink:

    def __init__(self):
        self.fresh = True

    def new_turn(self):
        self.fresh = True

    async def __call__(self, chunk):
        if self.fresh:
            print("\n" + cyan("bot> "), end="", flush=True)
            self.fresh = False
        print(chunk, end="", flush=True)
        return None


class EchoSink:

    def __init__(self, inner, clear_line=None, gap_ms=400):
        self.inner = inner
        self._clear_line = clear_line or (lambda: None)
        self.gap_s = max(0.0, gap_ms / 1000.0)
        self.fresh = True
        self.chunks = 0
        self._t_first = None
        self._t_prev_done = None
        self._down = None

    @property
    def turn(self):
        return getattr(self.inner, "turn", None)

    def new_turn(self):
        self.fresh = True
        self.chunks = 0
        self._t_first = None
        self._t_prev_done = None
        self._down = None
        inner_new = getattr(self.inner, "new_turn", None)
        if inner_new is not None:
            inner_new()
        return self

    async def __call__(self, chunk):
        t0 = time.time()
        gen = None if self._t_prev_done is None else t0 - self._t_prev_done
        if self.fresh:
            self._clear_line()
            self._t_first = t0
            print("\n" + cyan("bot> ") + chunk, flush=True)
            self.fresh = False
        else:
            marks = []
            if self._down is not None and self._down >= self.gap_s:
                marks.append(f"play {self._down * 1000:.0f}ms")
            if gen is not None and gen >= self.gap_s:
                marks.append(f"gen {gen * 1000:.0f}ms")
            tag = yellow(" ⟨" + " + ".join(marks) + "⟩") if marks else ""
            print(dim(f"  +{t0 - self._t_first:5.2f}s") + tag + "  " + chunk, flush=True)
        self.chunks += 1
        n = await self.inner(chunk)
        self._down = time.time() - t0
        self._t_prev_done = time.time()
        return n

    async def settle(self):
        inner_settle = getattr(self.inner, "settle", None)
        return await inner_settle() if inner_settle is not None else None


def _spawn_reader(loop, queue):
    def _worker():
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return
            loop.call_soon_threadsafe(queue.put_nowait, line)

    t = threading.Thread(target=_worker, daemon=True, name="stdin-reader")
    t.start()
    return t


def _report(rec, sink, cuts=1):
    print()
    metrics_of = sink
    while getattr(metrics_of, "inner", None) is not None:
        metrics_of = metrics_of.inner
    t = dict(getattr(metrics_of, "turn", None) or {})
    gate_t = dict(getattr(sink, "turn", None) or {}) if metrics_of is not sink else {}
    sink.new_turn()
    u = rec.get("usage") or {}
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens")
    cost = u.get("cost")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")

    bits = [f"first text {rec.get('first_chunk_ms')}ms", f"total {rec.get('total_ms')}ms"]
    if t.get("chunks"):
        if t.get("ttfa_ms") is not None:
            bits.insert(1, cyan(f"first sound {t['ttfa_ms']}ms"))
        if t.get("spoken_chars") is not None:
            bits.append(f"spoken {t['spoken_chars']}/{t['chars']} chars")
        if t.get("gaps"):
            bits.append(yellow(f"★dropouts {t['gaps']} {t.get('gap_log')}"
                               f" (where it broke: between chunks -> raise playback_lead_s,"
                               f" mid-chunk -> raise tts.jitter_buffer_ms)"))
        if t.get("drain_timeout"):
            bits.append(yellow("★settle timed out (playback queue never drained)"))
        if t.get("clear"):
            cl = t["clear"]
            bits.append(("" if cl.get("ok") else yellow("★")) +
                        f"two-level clear local {cl.get('dropped_local_bytes', 0)}B/"
                        f"in-flight {cl.get('dropped_after_clear', 0)} chunks/"
                        f"audio.clear {'✓' if cl.get('ok') else '✗'}")
    if rec["dropped_chars"]:
        bits.append(yellow(f"dropped {rec['dropped_chars']} chars (settle reconciliation)"))
    if cuts > 1:
        bits.append(bold(f"drew {cuts} cuts"))
    if gate_t:
        g = [f"waited {gate_t.get('wait_ms')}ms"] if gate_t.get("wait_ms") is not None else []
        if gate_t.get("open_reason"):
            g.append(f"opened {gate_t['open_reason']}")
        if gate_t.get("asks"):
            g.append(f"asked JEV#2 {gate_t['asks']}")
        if g:
            bits.append("gate " + "/".join(g))
    if pt is not None:
        c = f"/cached {cached}" if cached else ""
        bits.append(f"in {pt}{c} · out {ct}")
    if cost is not None:
        bits.append(f"${cost:.6f}")
    print(dim("  [" + " · ".join(bits) + "]"))

    post = rec.get("post") or {}
    for r in post.get("ran", []):
        if "thread" in r:
            what = f"thread/{r['thread']}"
        else:
            what = "pool→semantic"
        note = r.get("skipped") or f"{r.get('chars')} chars"
        print(dim(f"  [compress {what} · {note}]"))
    if rec.get("post_error"):
        print(yellow(f"  [compression failed (this turn unaffected): {rec['post_error']}]"))


async def _drain(drv, sink):
    while len(drv.reservoir):
        try:
            rec = await drv.pump()
        except Exception as e:
            print(yellow(f"\n  [turn failed, skipped · {type(e).__name__}: {e}]"))
            if os.environ.get("TALK_CLI_TRACE"):
                import traceback; traceback.print_exc()
            sink.new_turn()
            continue
        if rec is None:
            await asyncio.sleep(0.02)
            continue
        _report(rec, sink, cuts=rec["cuts_consumed"])


class FakeJudge:

    def __init__(self, mode="len", n=12):
        self.mode, self.n, self.calls = mode, n, 0

    async def __call__(self, state):
        self.calls += 1
        if self.mode == "enough":
            return 0.99, True
        if self.mode == "never":
            return 0.01, False
        noul = min(0.99, len(state) / max(1, self.n) * 0.7)
        return round(noul, 3), len(state) >= self.n


def _show_u(lp, trim):
    full = lp.u.text()
    fed = lp.u.text_for_jev(trim)
    print(dim(f"  U({len(full)} chars) = {full!r}"))
    if fed != full:
        print(dim(f"  to JEV  = {fed!r}   <- tail shaved by {trim} unit(s): a live partial's tail can be revised downwards"))
    else:
        print(dim("  to JEV  = same (nothing to shave off the live segment)"))

HELP = """\
  ★Just talk into the microphone. The following are **side-channel** control
  commands (type them and press enter):
  enter  while idle = reconnect STT (while active it has no effect at all)
  /s   counters (ticks / submissions / strikes / silence / cost)
  /u   what U looks like right now (including the tail-shaved copy fed to JEV)
  /m   memory status (window size / how many old blocks await compression /
       how far from the automatic trigger)
  /compact  compress on demand: turn the blocks outside the retention window into a
            summary (the verbatim text is archived first)
            ★Note it does **not** make the per-turn injection smaller -- that is pinned
              by keep_recent_tokens. /compact buys a longer memory, at the cost of
              adding the summary to the injection as well.
  /h   /q"""

CONTROL_CONFIG = os.path.join(SHELL_ROOT, "control_config.json")
DEFAULT_CONTROL = {
    "sleep_after_ticks": 60,
    "announce_every_ticks": 60,
}


DEFAULT_MIC_GATE = {
    "enabled": True,
    "hangover_ms": 300,
    "feed_silence": True,
}


def _load_block(block, defaults, path=None):
    path = path or CONTROL_CONFIG
    out = dict(defaults)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(yellow(f"  ★{os.path.basename(path)} not found -> using built-in defaults {out}"))
        return out
    except (json.JSONDecodeError, OSError) as e:
        print(yellow(f"  ★{os.path.basename(path)} could not be read ({type(e).__name__}: {e}) "
                     f"-> using built-in defaults {out}"))
        return out
    for k, v in (raw.get(block) or {}).items():
        if k.startswith("_"):
            continue
        if k not in out:
            print(yellow(f"  ★{os.path.basename(path)}: {block}.{k} matches no parameter, ignored"))
            continue
        out[k] = v
    return out


DEFAULT_HALLUCINATION_GATE = {
    "enabled": True,
    "max_cps": 8.0,
    "min_units": 4,
}


def load_hallucination_gate(path=None):
    return _load_block("hallucination_gate", DEFAULT_HALLUCINATION_GATE, path)


def load_mic_gate(path=None):
    return _load_block("mic_gate", DEFAULT_MIC_GATE, path)


def load_control(path=None):
    return _load_block("silence", DEFAULT_CONTROL, path)

SERIAL_BUDGET_MS = 2430


class VoiceWiring:

    def __init__(self, lp, *, display=True, plain=False, hallucination_gate=None):
        self.lp = lp
        self.display = display
        self.plain = plain
        self.hg = hallucination_gate
        self.dropped = []
        self.rows = []
        self.empty_events = 0
        self.t_mic = None
        self.cycle = 1
        self.turn_start = time.time()
        self._last_len = 0

    def clear_line(self):
        if self._last_len and self.display:
            print("\r" + " " * (self._last_len + 2) + "\r", end="", flush=True)
        self._last_len = 0

    def _say(self, s):
        if self.display:
            print(s, flush=True)

    def mark_mic_open(self):
        now = time.time()
        if self.t_mic is None:
            self.t_mic = now
        self.turn_start = now

    def _ms(self):
        return int((time.time() - (self.t_mic or time.time())) * 1000)

    def _row(self, text, ev, kind):
        self.rows.append({
            "t": self._ms(), "cycle": self.cycle,
            "text": text, "is_final": bool(ev.get("is_final")),
            "speech_final": bool(ev.get("speech_final")), "kind": kind,
            "start": ev.get("start"), "duration": ev.get("duration"),
            "conf": ev.get("end_of_turn_confidence"),
            "language": ev.get("language"),
            "cps": ev.get("_cps"),
            "dropped": bool(ev.get("_dropped"))})

    def _hallucinated(self, text, ev, kind):
        hg = self.hg
        if not hg or not hg.get("enabled"):
            return False
        if not _implausible(text, ev.get("duration"),
                            max_cps=hg.get("max_cps", 8.0),
                            min_units=hg.get("min_units", 4)):
            return False
        ev["_dropped"] = True
        cps = ev.get("_cps") or _speech_rate(text, ev.get("duration"))
        ev["_cps"] = cps
        rec = {"kind": kind, "text": text, "cps": cps,
               "start": ev.get("start"), "duration": ev.get("duration"),
               "conf": ev.get("end_of_turn_confidence")}
        self.dropped.append(rec)
        self.clear_line()
        self._say(yellow(f"★blocked an impossible transcript ({cps:.1f} chars/s > "
                         f"{hg.get('max_cps', 8.0)}, audio window {ev.get('duration')}s):"
                         f" {text!r}"))
        return True

    def on_partial(self, text, ev):
        if self._hallucinated(text, ev, "partial"):
            self._row(text, ev, "partial")
            return
        self._row(text, ev, "partial")
        self.lp.on_partial(text, ev.get("_stop_wall"))
        if not (text or "").strip():
            self.empty_events += 1
            return
        if self.display and sys.stdout.isatty() and not self.plain:
            pad = " " * max(0, self._last_len - len(text))
            print(f"\r{dim('…')} {text}{pad}", end="", flush=True)
            self._last_len = len(text)
        else:
            self._say(f"{dim('…')} {text}")

    def on_final(self, text, ev):
        if self._hallucinated(text, ev, "final"):
            self._row(text, ev, "final")
            return
        self._row(text, ev, "final")
        self.lp.on_final(text, ev.get("_stop_wall"))
        if not (text or "").strip():
            self.empty_events += 1
            return
        self.clear_line()
        self._say(f"{cyan('final')} {text}")

    def on_speech_final(self, text, ev):
        self._row(text, ev, "speech_final")
        self.clear_line()
        conf = ev.get("end_of_turn_confidence")
        n_sub = len([r for r in self.lp.log
                     if r.get("submitted") and r["t"] >= self.turn_start])
        tail = (green(f"  <- JEV#1 has passed {n_sub} cut(s) of this utterance") if n_sub
                else dim("  (JEV#1 has not passed yet -- waiting for the next tick, or heading for three strikes)"))
        self._say(f"{green('STT done')} {bold(text)}"
                  + (dim(f"  conf={conf}") if conf is not None else "") + tail)
        self.lp.on_speech_final(text, stop_wall=ev.get("_stop_wall"))
        self.turn_start = time.time()


class MicGate:

    def __init__(self, inner, on_speak_start, on_speak_end):
        self.inner = inner
        self._on_start, self._on_end = on_speak_start, on_speak_end
        self.speaking = False

    @property
    def turn(self):
        return getattr(self.inner, "turn", None)

    @property
    def turns(self):
        return getattr(self.inner, "turns", [])

    def new_turn(self):
        inner_new = getattr(self.inner, "new_turn", None)
        if inner_new is not None:
            inner_new()
        return self

    async def __call__(self, chunk):
        if not self.speaking:
            self.speaking = True
            self._on_start()
        return await self.inner(chunk)

    async def settle(self):
        try:
            return await self.inner.settle()
        finally:
            if self.speaking:
                self.speaking = False
                self._on_end()


def headphone_gate(args):
    mg = load_mic_gate()
    out = "?"
    try:
        import sounddevice as sd
        out = sd.query_devices(kind="output")["name"]
    except Exception as e:
        print(dim(f"  (could not query the default output device: {type(e).__name__})"))
    print(bold("★Echo gate") + dim(f"  default output = {out!r}")
          + (green("  · mic gating: on") if mg["enabled"] else yellow("  · mic gating: ★off")))
    if mg["enabled"]:
        print(dim("  The mic is gated while the machine speaks (zero frames are sent)"
                  " -> **speakers are usable**, no headphones needed."))
        print(dim("  ★It only blocks the machine's own TTS: this is not AEC, and it cannot"
                  " block another person or another device playing sound."))
        print(dim("  ★Anything you say while the machine is speaking is DROPPED outright"
                  " (half duplex, the same direction as abolishing barge-in)."))
        ask = "  Have you checked the speaker setup? Type y to continue, anything else aborts: "
    else:
        print(dim("  PortAudio has no AEC. Speakers -> sound into the mic -> the machine"
                  " answers itself, a stable infinite loop."))
        print(dim("  ★Each lap = one full LLM call + one full TTS synthesis, and with barge-in"
                  " abolished it will not stop by itself."))
        ask = "  Are your headphones on? Type y to continue, anything else aborts: "
    if getattr(args, "yes_headphones", False):
        print(yellow("  --yes-headphones: confirmation skipped (on your own head)"))
        return True
    try:
        a = input(bold(ask)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if a not in ("y", "yes"):
        print(dim("  Aborted. (Put headphones on / check the speakers, or force it with --yes-headphones)"))
        return False
    return True


def load_stt_config(dispatch_dir=DISPATCH_DIR, language=None):
    cfg = dict(load_dispatch_config(dispatch_dir).get("stt") or {})
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if language:
        cfg["language"] = language
    return cfg


def print_memory(session_dir):
    cfg = assemble_pool.load_config()
    print(dim(f"  memory · retention window {cfg['keep_recent_tokens']} tok"
              f" (floor {cfg['min_keep_blocks']} blocks) · auto trigger at "
              f"{cfg['thread_trigger_tokens']} tok · knobs: control_config.json -> memory block"))
    peers = assemble_pool.active_peers()
    if not peers:
        print(dim("  (no thread file yet -- one appears after the first turn)"))
        return
    for peer in peers:
        full = assemble_pool._read(assemble_pool.thread_path(peer))
        keep = assemble_pool.keep_blocks_for(peer, cfg)
        _h, blocks = assemble_pool._split_blocks(full)
        tail_t = assemble_pool.est_tokens(assemble_pool.thread_tail(peer, keep))
        head_t = assemble_pool.est_tokens(assemble_pool.thread_head(peer, keep))
        older_t = assemble_pool.est_tokens(assemble_pool.thread_older(peer, keep))
        gap = cfg["thread_trigger_tokens"] - assemble_pool.est_tokens(full)
        print("  " + bold(peer)
              + dim(f"  total {assemble_pool.est_tokens(full)} tok / {len(blocks)} blocks"
                    f" · injected per turn = summary head {head_t} + window {tail_t} tok (last {keep} blocks)"
                    f" · {len(blocks) - keep} old blocks / {older_t} tok awaiting compression"
                    + (f" · {gap} tok short of the auto trigger" if gap > 0 else " · ★past the trigger")))


def print_gate(vo, gate, w, tts):
    if vo is None:
        return
    print(dim("  mouth " + " · ".join(f"{k}={v}" for k, v in vo.summary().items())))
    reasons = collections.Counter(t.get("open_reason") or "?" for t in gate.turns)
    if reasons:
        print(dim("  open reasons " + " · ".join(f"{k}×{v}" for k, v in reasons.items())))
    waits = [t["wait_ms"] for t in gate.turns if t.get("wait_ms") is not None]
    if waits:
        print(dim(f"  waited at the gate p50 {int(statistics.median(waits))}ms"
                  f" · range {min(waits)}–{max(waits)}ms"
                  + dim("   <- this overlaps the LLM's TTFT")))
    inner_turns = list(getattr(gate.inner, "turns", []))
    if getattr(gate.inner, "turn", {}).get("chunks"):
        inner_turns.append(gate.inner.turn)
    ttfa = [t["ttfa_ms"] for t in inner_turns if t.get("ttfa_ms") is not None]
    if ttfa:
        print("  " + bold(f"first sound p50 {int(statistics.median(ttfa))}ms")
              + dim(f" · n={len(ttfa)} · range {min(ttfa)}–{max(ttfa)}ms"
                    "   <- the moment the ear actually hears something"))
    gaps = sum(t.get("gaps") or 0 for t in inner_turns)
    if gaps:
        print("  " + yellow(f"★{gaps} dropouts in total")
              + dim(" -- check the gap_log timestamps: between chunks -> playback_lead_s, mid-chunk -> jitter_buffer_ms"))
    if tts is not None:
        print(dim(f"  TTS {tts.m['chars_sent']} chars · ${tts.cost_usd:.5f}"
                  + (f" · errors {tts.m['errors']}" if tts.m.get("errors") else "")))


def print_counters(lp, jev, w=None):
    sm = lp.summary()
    print(dim("  " + " · ".join(f"{k}={v}" for k, v in sm.items())))
    if w is not None and w.empty_events:
        print(dim(f"  {w.empty_events} empty events (silence-window heartbeats: not counted as speech, not added to U, not displayed)"))
    if sm.get("delivered_armed"):
        exp = sm.get("delivered_expired") or 0
        if exp:
            print(yellow(f"  ★the delivered watermark expired {exp} time(s) -- that piece was cut off for "
                         f"good (no finalisation). Same-piece content arriving afterwards had "
                         f"nothing trimmed, so it may have been delivered twice:"))
            for r in list(lp.u.delivered_expired)[:3]:
                print(yellow(f"      idle {r['idle_s']}s > ceiling {sm['delivered_ttl_s']}s · "
                             f"in hand {r['delivered'][:30]!r} · arrived {r['arrived'][:30]!r}"))
        else:
            print(dim(f"  watermark armed {sm['delivered_armed']} · disarmed normally "
                      f"{sm['delivered_disarmed']} · refreshed {sm['delivered_refreshed']} · "
                      f"expired 0 · longest idle {sm['delivered_max_idle_s']}s / ceiling "
                      f"{sm['delivered_ttl_s']}s (the smaller the margin, the stronger the case for keying on start)"))
    if w is not None and w.dropped:
        print(yellow(f"  ★the hallucination gate blocked {len(w.dropped)} transcript(s) (a speech rate no human could produce):"))
        for r in w.dropped[:5]:
            print(yellow(f"      {r['cps']:.1f} chars/s · window {r['duration']}s · "
                         f"conf={r['conf']} · {r['text'][:60]!r}"))
    said = sm.get("speech_finals", 0)
    passed = sm.get("submit_jev", 0)
    dropped = sm.get("strike_cleared", 0)
    if said or passed or dropped:
        print("  " + bold(f"STT done {said} / JEV#1 passed {passed} / three strikes dropped {dropped}")
              + dim(" (passing is the only way into the reply list; three strikes is the only way words are dropped)"))
        if said and passed + dropped < said:
            print("  " + dim(f"  ★{said - passed - dropped} segment(s) neither passed nor dropped -- "
                             f"still sitting in U (normal, mid-sentence) or unaccounted for"))
    if jev is not None:
        print(dim(f"  JEV {jev.calls} calls · input {jev.total_input_tokens} tok"
                  f" · ${jev.total_cost_usd:.6f}"))


async def main(args):
    async with contextlib.AsyncExitStack() as stack:
        return await _run(args, stack)


async def _run(args, stack):
    knobs = listener_loop.load_knobs(DISPATCH_DIR)
    if args.tick:
        knobs["tick_interval_s"] = args.tick
    if args.strikes:
        knobs["meaningless_strikes"] = args.strikes
    trim = int(knobs["jev_trim_units"])
    ctl = load_control()
    sleep_ticks = int(args.sleep_ticks if args.sleep_ticks is not None
                      else ctl["sleep_after_ticks"])
    announce_every = int(ctl["announce_every_ticks"])
    grace_s = sleep_ticks * max(0.0, float(knobs["tick_interval_s"]))
    mg = load_mic_gate()
    mic_gate = {"on": bool(mg["enabled"]), "muted": False, "gen": 0,
                "hangover_s": max(0.0, float(mg["hangover_ms"]) / 1000.0),
                "feed_silence": bool(mg["feed_silence"]), "muted_frames": 0}

    def _mute_mic():
        if not mic_gate["on"]:
            return
        mic_gate["gen"] += 1
        mic_gate["muted"] = True

    def _unmute_mic_later():
        if not mic_gate["on"]:
            return
        gen = mic_gate["gen"]

        async def _after():
            await asyncio.sleep(mic_gate["hangover_s"])
            if mic_gate["gen"] == gen:
                mic_gate["muted"] = False

        asyncio.get_event_loop().create_task(_after())

    stt_cfg = load_stt_config(DISPATCH_DIR, args.language)
    key = stt_client.resolve_key(SHELL_ROOT)

    gate = tts = player = None
    if args.tts:
        sys.path.insert(0, VOICE_OUT_DIR)
        import speak_sink
        import voice_out_loop
        conn, player, make_sink = speak_sink.build(device=args.out_device,
                                                   verbose=args.tts_verbose)
        player.start()
        stack.callback(player.stop)
        print(dim("  connecting TTS..."), end="", flush=True)
        tts = await stack.enter_async_context(conn)
        print(dim(f" handshake {tts.m['connect_ms']}ms · sound card reports "
                  f"{player.latency_s * 1000:.0f}ms"))
        gate = voice_out_loop.GateSink(
            MicGate(make_sink(tts), _mute_mic, _unmute_mic_later),
            verbose=args.tts_verbose)
        print(dim("  mic gating " + ("on" if mic_gate["on"] else "★off (headphones required)")
                  + f" · hangover {mg['hangover_ms']}ms · "
                  + ("zero frames while gated" if mic_gate["feed_silence"] else "★no frames while gated")))
        sink = gate if args.no_echo else EchoSink(gate, gap_ms=args.echo_gap_ms)
    else:
        sink = TypingSink()
    drv, session_dir = chain.new_voice_session(peer=args.peer, sink=sink)

    jev = None
    if args.fake_jev:
        judge = FakeJudge(args.fake_jev, args.fake_len)
        judge_name = f"fake verdict {args.fake_jev}" + (f"(>={args.fake_len} chars)" if args.fake_jev == "len" else "")
    else:
        jev = JevClient(DISPATCH_DIR, "jev1", threshold=knobs["meaningful_threshold"])
        await jev.__aenter__()
        judge_name = f"{jev.model} @ {jev.threshold}"

        async def judge(state):
            r = await jev.ask(state)
            return r["noul"], r["meaningful"]

    turns = []
    lp = listener_loop.ListenerLoop(drv, judge, knobs=knobs, verbose=True,
                                    on_turn=lambda rec: (turns.append(rec),
                                                         _report(rec, sink, cuts=rec["cuts_consumed"])))
    hg = load_hallucination_gate()
    w = VoiceWiring(lp, plain=args.plain, hallucination_gate=hg)
    if isinstance(sink, EchoSink):
        sink._clear_line = w.clear_line

    vo = jev2 = None
    vo_knobs = {}
    judge2_name = "—"
    if args.tts:
        vo_knobs = voice_out_loop.load_knobs(VOICE_OUT_DIR)
        if args.vo_tick:
            vo_knobs["tick_interval_s"] = args.vo_tick
        if args.fake_jev2:
            judge2 = FakeJudge(args.fake_jev2)
            judge2_name = f"fake verdict {args.fake_jev2}"
        else:
            jev2 = await stack.enter_async_context(
                voice_out_loop.open_jev2(VOICE_OUT_DIR, vo_knobs))
            judge2 = voice_out_loop.make_judge(jev2)
            judge2_name = f"{jev2.model} @ {jev2.threshold}"
        vo = voice_out_loop.VoiceOutLoop(drv, gate, judge2, knobs=vo_knobs, verbose=True)
        _vo_note = vo._note

        def _vnote(kind, msg):
            w.clear_line()
            _vo_note(kind, msg)

        vo._note = _vnote

    _orig_note = lp._note

    def _note(kind, msg):
        w.clear_line()
        _orig_note(kind, msg)

    lp._note = _note

    print(bold("voice_cli") + dim(f"  peer={args.peer}  session={os.path.basename(session_dir)}"))
    print(dim(f"  verdict = {judge_name}   tick {knobs['tick_interval_s']}s   "
              f"strikes {knobs['meaningless_strikes']}   tail shave {trim}"))
    print(dim(f"  LLM fires = JEV#1 passes **and** silence >= "
              f"{knobs['llm_min_silence_s']}s (listener_loop discipline 6)"))
    print(dim("  idle = " + (f"silence_ticks >= {sleep_ticks} ticks (~{sleep_ticks}s, "
                            f"tick={knobs['tick_interval_s']}s) -> cut STT, enter to reconnect"
                            if sleep_ticks > 0 else
                            "★off (STT stays up and silence is billed by wall clock too)")
              + f"   time limit = {'none (/q to finish)' if args.seconds <= 0 else f'{args.seconds}s'}"))
    print(dim(f"  STT = {stt_cfg.get('model')} · language={stt_cfg.get('language')}"
              f" · endpointing={stt_cfg.get('endpointing')}"
              f" · {len(stt_cfg.get('keyterm') or [])} keyterms"))
    print(dim("  hallucination gate " + (f"on · ceiling {hg['max_cps']} chars/s (under {hg['min_units']} units it does not judge)"
                            if hg["enabled"] else "★off (fabrications on very short audio go straight into U)")))
    if vo is not None:
        print(dim(f"  mouth = real TTS · JEV#2 {judge2_name} · gate "
                  f"floor {vo.min_silence_s}s / ceiling {vo.max_hold_s}s of true silence · poll "
                  f"{vo_knobs['tick_interval_s']}s / JEV#2 at most every "
                  f"{vo_knobs['jev2_min_interval_s']}s · prefetch "
                  f"{vo_knobs['playback_lead_s']}s"))
    else:
        print(dim("  mouth = terminal text (TypingSink) -- add --tts to speak out loud"))
    if isinstance(sink, EchoSink):
        print(dim(f"  reply text: one chunk per line (the leading +seconds is this chunk's time relative "
                  f"to the turn's first chunk; <play ...> = what the previous chunk spent in the "
                  f"gate/playback, <gen ...> = waiting for the LLM to emit this one; only marked "
                  f"at >={args.echo_gap_ms}ms)"))
    for name, cli in (("JEV#1", jev), ("JEV#2", jev2)):
        if cli is None:
            continue
        print(dim(f"  warming up the {name} connection..."), end="", flush=True)
        ms = await cli.warmup()
        print(dim(f" {ms:.0f}ms"))
    print(HELP)
    print_memory(session_dir)

    loop = asyncio.get_running_loop()
    cmd_q: asyncio.Queue = asyncio.Queue()
    dropped = []
    mic = {"st": None}
    mic_ready = asyncio.Event()
    stop_evt = asyncio.Event()
    sleep_evt = asyncio.Event()
    wake_evt = asyncio.Event()
    ear = {"q": None, "on": False, "cycle": 0, "armed": False}
    spoke = {"end_wall": time.monotonic()}

    def _idle_s():
        return min(drv.silence.silence_seconds(),
                   max(0.0, time.monotonic() - spoke["end_wall"]))

    async def _nap(sec):
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_evt.wait(), timeout=sec)
        return stop_evt.is_set()

    def _on_frame(buf):
        if not ear["on"]:
            return
        if mic_gate["muted"]:
            mic_gate["muted_frames"] += 1
            if not mic_gate["feed_silence"]:
                return
            buf = bytes(len(buf))
        q = ear["q"]
        if q is not None:
            loop.call_soon_threadsafe(q.put_nowait, buf)

    def _open_mic():
        w.mark_mic_open()
        mic["st"] = stt_client.mic_stream(device=args.device, on_frame=_on_frame,
                                         on_status=lambda s: dropped.append(s))
        tail = (f" (STT is cut if nobody speaks for {grace_s:.0f}s after the machine finishes; /q to finish)" if sleep_ticks > 0
                else " (it will never go idle by itself -- STT stays up and keeps billing; /q to finish)")
        print(green("  connected · mic open -- talk now.") + dim(tail), flush=True)
        mic_ready.set()

    def _close_mic():
        st, mic["st"] = mic["st"], None
        if st is not None:
            st.stop()
            st.close()

    async def _cut_stt():
        q = ear["q"]
        if q is not None:
            await q.put(None)

    async def _compact():
        cfg = assemble_pool.load_config()
        plan = assemble_pool.forced_plan()
        if not plan["threads"]:
            print(yellow("  /compact: nothing old to compress -- the whole conversation is still inside "
                         f"the retention window ({cfg['keep_recent_tokens']} tok)."))
            print_memory(session_dir)
            return
        snap = {}
        for peer in plan["threads"]:
            keep = assemble_pool.keep_blocks_for(peer, cfg)
            _h, blocks = assemble_pool._split_blocks(
                assemble_pool._read(assemble_pool.thread_path(peer)))
            snap[peer] = {
                "full": assemble_pool.est_tokens(
                    assemble_pool._read(assemble_pool.thread_path(peer))),
                "blocks": len(blocks), "keep": keep,
                "inject": assemble_pool.est_tokens(assemble_pool.thread_head(peer, keep))
                          + assemble_pool.est_tokens(assemble_pool.thread_tail(peer, keep))}
        print(bold(f"  /compact starting · compressing {plan['threads']}") + dim(" (an LLM call, a few seconds)"))
        t0 = time.time()
        rec = await asyncio.to_thread(chain.maybe_compress, session_dir, plan)
        print(dim(f"  took {time.time() - t0:.1f}s · receipt {rec.get('ran')}"))
        cfg = assemble_pool.load_config()
        for peer, b in snap.items():
            keep = assemble_pool.keep_blocks_for(peer, cfg)
            full_now = assemble_pool._read(assemble_pool.thread_path(peer))
            _h, blocks = assemble_pool._split_blocks(full_now)
            head_t = assemble_pool.est_tokens(assemble_pool.thread_head(peer, keep))
            tail_t = assemble_pool.est_tokens(assemble_pool.thread_tail(peer, keep))
            print("  " + bold(peer) + dim(
                f"  total {b['full']} -> {assemble_pool.est_tokens(full_now)} tok"
                f" · blocks {b['blocks']} -> {len(blocks)}"
                f" · ★injected per turn {b['inject']} -> {head_t + tail_t} tok"
                f" (summary head {head_t} + window {tail_t})"))
        ad = assemble_pool.archive_dir()
        if os.path.isdir(ad):
            print(dim(f"  verbatim text archived in {ad} ({len(os.listdir(ad))} files)"))

    async def commander():
        while not stop_evt.is_set():
            get = asyncio.create_task(cmd_q.get())
            waiter = asyncio.create_task(stop_evt.wait())
            done, pending = await asyncio.wait({get, waiter},
                                              return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if stop_evt.is_set():
                return
            line = get.result()
            s = (line or "").strip()
            if line is None or s in ("/q", "/quit"):
                stop_evt.set()
                wake_evt.set()
                await _cut_stt()
                return
            w.clear_line()
            if not s:
                wake_evt.set()
                continue
            if s in ("/h", "/help"):
                print(HELP)
            elif s == "/u":
                _show_u(lp, trim)
            elif s == "/s":
                print_counters(lp, jev, w)
                print_gate(vo, gate, w, tts)
            elif s == "/m":
                print_memory(session_dir)
            elif s in ("/compact", "/c"):
                await _compact()
            else:
                print(dim(f"  unknown command {s!r} -- /h for the list"))

    async def deadline():
        if args.seconds <= 0:
            return
        await mic_ready.wait()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_evt.wait(), timeout=args.seconds)
        if not stop_evt.is_set():
            w.clear_line()
            print(dim(f"  {args.seconds}s elapsed, finishing up..."))
        stop_evt.set()
        wake_evt.set()
        await _cut_stt()

    async def silence_watch():
        if sleep_ticks <= 0:
            return
        while not await _nap(max(0.2, float(knobs["tick_interval_s"]))):
            busy = drv.busy
            if busy:
                spoke["end_wall"] = time.monotonic()
            ticks = drv.silence.value()
            if ticks < sleep_ticks:
                ear["armed"] = True
            if not ear["on"] or sleep_evt.is_set() or busy:
                continue
            if time.monotonic() - spoke["end_wall"] < grace_s:
                continue
            if ticks < sleep_ticks:
                continue
            if not ear["armed"]:
                continue
            sleep_evt.set()
            await _cut_stt()

    async def sleep_heartbeat():
        if announce_every <= 0:
            return
        last = None
        while not await _nap(max(0.2, float(knobs["tick_interval_s"]))):
            if ear["on"]:
                last = None
                continue
            n = drv.silence.value()
            if last is None:
                last = n
            elif n - last >= announce_every:
                last = n
                w.clear_line()
                print(dim(f"  · idle · silence_ticks={n}"
                          f" · {_idle_s():.0f}s since anything was spoken · enter to wake"), flush=True)

    def _merge_metrics(dst, src):
        if not src:
            return
        dst["stt_sessions"] = dst.get("stt_sessions", 0) + 1
        for k in ("connect_ms", "created_ms"):
            if dst.get(k) is None:
                dst[k] = src.get(k)
        dst.setdefault("connect_ms_all", []).append(src.get("connect_ms"))
        for k in ("audio_seconds", "frames_sent"):
            dst[k] = (dst.get(k) or 0) + (src.get(k) or 0)
        dst.setdefault("event_latency_ms", []).extend(src.get("event_latency_ms") or [])
        dst.setdefault("errors", []).extend(src.get("errors") or [])
        dst.setdefault("first_of_type", {}).update(src.get("first_of_type") or {})
        if src.get("url_config"):
            dst["url_config"] = src["url_config"]
        s_ev, d_ev = src.get("events"), dst.setdefault("events", {})
        if isinstance(s_ev, dict):
            for k, v in s_ev.items():
                d_ev[k] = d_ev.get(k, 0) + v
        elif s_ev is not None:
            dst["events_nondict"] = s_ev

    async def ear_supervisor():
        while not stop_evt.is_set():
            ear["cycle"] += 1
            w.cycle = ear["cycle"]
            sleep_evt.clear()
            ear["armed"] = False
            ear["q"] = asyncio.Queue()
            ear["on"] = True
            one = {}
            try:
                one = await stt_client.stream(
                    stt_cfg, key, ear["q"],
                    on_partial=w.on_partial, on_final=w.on_final,
                    on_speech_final=w.on_speech_final, on_ready=_open_mic,
                    on_event=(lambda ev: print(dim("   ev " + json.dumps(
                        ev, ensure_ascii=False)[:300]), flush=True))
                    if args.dump_events else None)
            except Exception as e:
                one = {"errors": [{"type": "_stream_raised",
                                   "detail": f"{type(e).__name__}: {e}"}]}
            finally:
                ear["on"] = False
                _close_mic()
                _merge_metrics(m, one)
            if stop_evt.is_set():
                return
            w.clear_line()
            if sleep_evt.is_set():
                print("\n  " + bold(yellow("★gone idle"))
                      + dim(f" -- nobody spoke for {grace_s:.0f}s after the machine finished"
                            f" (silence_ticks={drv.silence.value()}; that number counts the time "
                            f"the machine spent speaking too). "
                            f"The STT connection has been cut (no audio sent = no billing)."))
            else:
                print("\n  " + bold(yellow("★STT dropped -- we did not cut it"))
                      + dim(f" (active period {ear['cycle']})"))
                errs = one.get("errors") or []
                print(dim(f"     errors={errs[-3:] if errs else '(none recorded)'}"))
                print(dim("     ★There is no automatic reconnect: a backoff policy is a separate piece of work. "
                          "At least it no longer spins on with a dead ear."))
            print("  " + bold("press enter to reconnect STT")
                  + dim("   /q to finish · silence_ticks keeps climbing while idle"))
            wake_evt.clear()
            await wake_evt.wait()
            if stop_evt.is_set():
                return
            drv.silence.mark_voice()
            w.clear_line()
            print(green(f"  <- woken (active period {ear['cycle'] + 1}) · reconnecting STT..."), flush=True)

    _spawn_reader(loop, cmd_q)
    runner = asyncio.create_task(lp.run())
    vo_runner = asyncio.create_task(vo.run()) if vo is not None else None
    m = {}
    try:
        await asyncio.gather(ear_supervisor(), commander(), deadline(),
                             silence_watch(), sleep_heartbeat())
    finally:
        _close_mic()
        t_d = time.time()
        while (drv.busy or len(drv.reservoir) > 0) and time.time() - t_d < args.drain:
            await asyncio.sleep(0.2)
        if drv.busy or len(drv.reservoir) > 0:
            print(yellow(f"  ★not drained within {args.drain}s (busy={drv.busy} · {len(drv.reservoir)} cuts in the reservoir)"))
        if vo is not None:
            await vo.aclose()
            vo_runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await vo_runner
        await lp.aclose()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
        if jev is not None:
            await jev.__aexit__(None, None, None)

    m["audio_status"] = dropped
    print()
    print(bold("-- pipeline metrics --"))
    n_sess = m.get("stt_sessions") or 0
    print(f"  STT handshake {m.get('connect_ms')}ms · transcript.created +{m.get('created_ms')}ms"
          + (dim(" (the first period's number = cold start)") if n_sess > 1 else ""))
    if n_sess > 1:
        print(f"  {n_sess} active periods · handshake each {m.get('connect_ms_all')}ms"
              + dim("   <- more than one = it went idle in between, or dropped the connection"))
    lat = sorted(m.get("event_latency_ms") or [])
    if lat:
        print(f"  STT pure pipeline delay n={len(lat)} · median {lat[len(lat) // 2]}ms"
              f" · range {lat[0]}–{lat[-1]}ms" + dim("   <- arrival -(start+duration)"))
    print(f"  audio sent {m.get('audio_seconds', 0):.1f}s / {m.get('frames_sent')} frames"
          f" · STT ≈ ${m.get('audio_seconds', 0) / 3600 * 0.20:.5f}")
    print(f"  event counts {m.get('events')}")
    if dropped:
        print(yellow(f"  ★PortAudio reported status {len(dropped)} time(s) (overflow = dropped frames): {set(dropped)}"))
    if m.get("errors"):
        print(yellow(f"  ★STT errors {m['errors']}"))

    print(bold("-- the first number to look at: time to first text --"))
    ttfts = [r["first_chunk_ms"] for r in turns if r.get("first_chunk_ms")]
    if ttfts:
        p50 = int(statistics.median(ttfts))
        verdict = (green("clearly better than the serial budget -> the overlap is working")
                   if p50 < SERIAL_BUDGET_MS - 300 else
                   yellow("at or above the serial budget -> early submission probably did not happen"))
        print(f"  {len(ttfts)} turns · p50 {p50}ms · range {min(ttfts)}–{max(ttfts)}ms"
              f"   against the serial budget {SERIAL_BUDGET_MS}ms -> {verdict}")
    else:
        print(dim("  no completed turns (nobody spoke? or everything was blocked as an empty submission)"))
    print_counters(lp, jev, w)
    if vo is not None:
        print(bold("-- the mouth side (speech gate + TTS) --"))
        print_gate(vo, gate, w, tts)
        if jev2 is not None:
            print(dim(f"  JEV#2 {jev2.calls} calls · ${jev2.total_cost_usd:.6f}"
                      f" · threshold {jev2.threshold}"
                      + yellow(" (★this threshold has never been swept over a real corpus)")))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    fx = os.path.join(DISPATCH_DIR, "fixtures")
    os.makedirs(fx, exist_ok=True)
    rp = os.path.join(fx, f"voice_{stamp}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "stt_config": stt_cfg, "knobs": knobs,
                   "metrics": m, "rows": w.rows, "ticks": list(lp.log),
                   "summary": lp.summary(),
                   "turns": [{k: v for k, v in t.items() if k != "post"} for t in turns],
                   "jev": None if jev is None else {"calls": jev.calls,
                                                    "cost_usd": jev.total_cost_usd},
                   "voice_out": None if vo is None else {
                       "knobs": vo_knobs, "summary": vo.summary(),
                       "ticks": list(vo.log), "gate_turns": gate.turns + [gate.turn],
                       "tts_turns": list(getattr(gate.inner, "turns", []))
                                    + [getattr(gate.inner, "turn", {})],
                       "jev2": None if jev2 is None else {
                           "calls": jev2.calls, "cost_usd": jev2.total_cost_usd,
                           "threshold": jev2.threshold}},
                   "tts": None if tts is None else {
                       "connect_ms": tts.m.get("connect_ms"),
                       "chars_sent": tts.m.get("chars_sent"),
                       "cost_usd": tts.cost_usd, "errors": tts.m.get("errors")}},
                  f, ensure_ascii=False, indent=2)
    print(bold(f"\n  ★full report written to {rp}"))
    jl = os.path.join(fx, f"voice_{stamp}.jsonl")
    with open(jl, "w", encoding="utf-8") as f:
        for r in w.rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dim(f"    per-event log {jl} (includes conf/language)"))
    print(dim("    ★Deliberately not named stt_*.jsonl: that glob is jev_bench's corpus, and "
              "dropping an unlabelled file in there turns dry_suite 15h (cases <-> labels "
              "one to one) red."))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description="full chain: mic -> STT -> JEV#1 -> LLM -> gate -> TTS")
    ap.add_argument("--peer", default="user")
    ap.add_argument("--seconds", type=float, default=0,
                    help="finish automatically N seconds after the mic opens. ★Default 0 = no limit "
                         "(stays up until /q). STT is billed by the duration sent, silence "
                         "included -- pass a positive number to cap it")
    ap.add_argument("--sleep-ticks", type=int, default=None,
                    help="temporarily override silence.sleep_after_ticks in control_config.json: "
                         "★STT is cut and the session goes idle only after this many ticks "
                         "with nobody speaking, counted from when the machine stops talking "
                         "(0 = never go idle). The same number is also the grace length")
    ap.add_argument("--drain", type=float, default=25.0, help="seconds to wait for the final turn to finish when shutting down")
    ap.add_argument("--language", default=None, help="override stt.language from dispatch_config")
    ap.add_argument("--device", type=int, default=None, help="input device index (see mic_probe --list-only)")
    ap.add_argument("--fake-jev", choices=["len", "enough", "never"],
                    help="fake verdict, no network (saves the JEV spend; the LLM still runs). Omit for real JEV")
    ap.add_argument("--fake-len", type=int, default=12, help="character threshold for --fake-jev len")
    ap.add_argument("--tts", action="store_true",
                    help="★speak out loud: the mouth becomes GateSink(SpeakSink) + the three-stage gate "
                         "+ JEV#2 (= a full voice conversation; without it, terminal text only)")
    ap.add_argument("--out-device", default=None, help="output device (used with --tts)")
    ap.add_argument("--tts-verbose", action="store_true", help="print gate and settle details every turn")
    ap.add_argument("--fake-jev2", choices=["enough", "never"],
                    help="fake JEV#2, no network (saves the mouth side's JEV spend; TTS still speaks and "
                         "still costs). enough=pass every tick / never=never pass, which forces "
                         "the max_hold_s backstop")
    ap.add_argument("--vo-tick", type=float, help="override the mouth loop's tick_interval_s")
    ap.add_argument("--tick", type=float, help="override knobs.tick_interval_s")
    ap.add_argument("--strikes", type=int, help="override knobs.meaningless_strikes")
    ap.add_argument("--plain", action="store_true", help="one line per partial (no in-place rewriting)")
    ap.add_argument("--no-echo", action="store_true",
                    help="with --tts, do not print the reply text (it is printed by default, one chunk per line)")
    ap.add_argument("--echo-gap-ms", type=int, default=400,
                    help="threshold in ms for annotating timings in the reply text (default 400)")
    ap.add_argument("--dump-events", action="store_true", help="print every raw event")
    ap.add_argument("--yes-headphones", action="store_true",
                    help="★skip the echo gate. Without headphones it will feed back, and every lap costs real money")
    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if not headphone_gate(_args):
        sys.exit(2)
    try:
        sys.exit(asyncio.run(main(_args)) or 0)
    except KeyboardInterrupt:
        pass
