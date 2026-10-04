"""
assemble_pool.py -- information-pool storage and compression assembly (deterministic script).

Division of labour, and this one is not negotiable:
  **the script only reads / joins / reorders by recency / counts tokens / decides triggers /
  files compression output**; the **content** of both compressions is produced by an LLM
  (the thread_compressor / pool_compressor dispatches), called by chain.py and written to
  disk here. The LLM never orchestrates and never touches the raw files.

The information pool has two layers:
  - pool/semantic.md          the integrated semantic layer (a de-personalised narrative,
                              overwritten by pool_compressor) = where connectivity lives
  - pool/threads/{peer}.md    the per-counterpart verbatim buffer (the canonical and only
                              raw store); thread_compressor compresses its "old blocks"

★These two layers are the speaker's entire memory. Under the single-shot pattern the
session history is always exactly one turn, so "remembering what was said last time" rests
entirely on these files being re-injected every turn through context_files.

Triggers (nothing is really compressed until the queue is empty; chain decides and calls):
  - thread: a pool/threads/{peer}.md with est_tokens >= thread_trigger -> compress the
            blocks outside the most recent keep_recent_tokens tokens (the boundary is
            computed by keep_blocks_for(), the only place in the module that computes it)
  - pool  : [semantic + every active tail] est_tokens >= pool_trigger -> re-compress
            semantic as a blend

Thresholds come from mailbox/pool_config.json (overriding DEFAULT_CONFIG; a smoke test can
force a trigger with absurdly low thresholds).
"""

import json
import os
import re
import threading
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


SHELL_ROOT = _find_shell_root(os.path.dirname(os.path.abspath(__file__)))
POOL_DIR = os.path.join(SHELL_ROOT, "pool")
THREADS_DIR = os.path.join(POOL_DIR, "threads")
SEMANTIC = os.path.join(POOL_DIR, "semantic.md")
MAILBOX = os.path.join(SHELL_ROOT, "mailbox")
# ★The same file that holds voice_cli's silence thresholds (the designated place for
#   human-tunable knobs). Only its `memory` block is read here.
CONTROL_CONFIG = os.path.join(SHELL_ROOT, "control_config.json")
# Keys that control_config.json is allowed to override. Anything else is ignored **and
# complained about**.
_MEMORY_KEYS = ("thread_trigger_tokens", "thread_target_tokens",
                "keep_recent_tokens", "min_keep_blocks", "pool_trigger_tokens")

DEFAULT_CONFIG = {
    # ★★2026-09-27, decided: 50000 -> 8000. The old value **structurally disabled
    #   compression** for voice: measured at 78 tok/turn, 50000 needs ~640 turns to reach,
    #   and the first trigger would hand ~48,000 tok of 600 turns in one go to a compressor
    #   with max_tokens=8000. 8000 means compressing roughly every 100 turns, with ~6k tok
    #   of input each time.
    "thread_trigger_tokens": 8000,    # per-counterpart conversation above this -> conventional compression
    "thread_target_tokens": 5000,     # target ceiling for compressed old blocks (soft, steered by the prompt)
    # ★★The retention window (= the part excluded from compression and injected every turn)
    #   is measured as a **token budget**, not a block count (decided 2026-09-27; the
    #   reasoning is in `keep_blocks_for`: "a turn" has no clean definition at this layer).
    #   The `thread_keep_blocks` key was **deleted**, not left unused -- a key that looks
    #   like a knob but nothing reads is exactly the "I changed the config and nothing
    #   happened" trap this module keeps falling into.
    "keep_recent_tokens": 2000,       # the most recent N tokens of verbatim blocks: never compressed, injected every turn
    "min_keep_blocks": 6,             # ★a floor, not decoration: one 5000-tok monologue makes the budget compute 0 blocks
    "pool_trigger_tokens": 70000,     # new material below the pool narrative above this -> blended compression
}

# ★The two paths that write a thread file (append / overwrite after compression) must be
#   mutually exclusive: since 2026-09-27 compression runs on a **worker thread**
#   (`chain._after_turn` goes through asyncio.to_thread, because a blocking LLM call would
#   otherwise stall the event loop and nothing would consume mic frames). Meanwhile
#   `append_thread` is read-modify-write and `_write` truncates before writing -- a
#   collision loses blocks, in either direction, and raises nothing.
_THREAD_LOCK = threading.RLock()
_warned = set()


def _warn_once(key, msg):
    """Complain about the same configuration problem only once. ★But it must complain:
    `load_config()` runs every turn, and a silent except would make "control_config.json is
    wrong" permanently undiagnosable (an old trap in this module)."""
    if key not in _warned:
        _warned.add(key)
        print(f"[assemble_pool] ⚠ {msg}")

_BLOCK_RE = re.compile(r"(?m)^### \[step ")
_COMPRESSED_HEAD_RE = re.compile(r"^### \[step 0\] compressed\b")

# Where CJK starts: CJK radical supplements, Han, fullwidth punctuation, kana, hangul.
# Used by est_tokens to count the two segments separately.
_CJK_START = 0x2E7F


# -- basics --------------------------------------------------------------------------
def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return default


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def est_tokens(text):
    """Token estimate. Used only for threshold decisions, so it need not be exact -- but
    **the direction of its error must be right**.

    ★Re-calibrated by measurement 2026-09-26. The upstream template used `len(text)//4`,
    which is an English calibration and under-counts Chinese by 2.4-2.9x -- and under-counting
    means the pool grows past the context budget before compression ever fires.

    Method: send twice to the same provider (1 character, then 833) and difference the
    prompt_tokens to cancel out the chat template overhead.
      Chinese prose   0.735 tok/char (one provider)   0.589 tok/char (another)
      English prose   0.141 tok/char (both agreed, ~7 characters per token)

    Chosen: 0.75 for CJK, 0.25 for everything else -- both **above** the measured values.
    That bias is deliberate: over-estimating compresses early, under-estimating blows the
    context window. The cost of over-estimating is one extra compression.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) > _CJK_START)
    return int(cjk * 0.75 + (len(text) - cjk) * 0.25)


def load_config():
    """DEFAULT_CONFIG <- the `memory` block of the shell root's `control_config.json`
    <- `mailbox/pool_config.json`.

    ★`control_config.json` is **where a human tunes these numbers** (the same file as the
      silence thresholds). It lives at the shell root, stays out of `config_base/`, and is
      not subject to the mirror rule -- not one character of it reaches the LLM.
    ★`mailbox/pool_config.json` stays as a last override layer (it is written to disk, so a
      test must clean it up; changing DEFAULT_CONFIG in-process does not persist).
    """
    cfg = dict(DEFAULT_CONFIG)
    raw = _read(CONTROL_CONFIG)
    if raw.strip():
        try:
            mem = (json.loads(raw).get("memory") or {})
        except (json.JSONDecodeError, ValueError) as e:
            mem = {}
            _warn_once("ctl_bad", f"{CONTROL_CONFIG} is not valid JSON ({e}) -> these numbers "
                                  f"are using built-in defaults: { {k: cfg[k] for k in _MEMORY_KEYS} }")
        for k, v in mem.items():
            if k.startswith("_"):
                continue
            if k not in _MEMORY_KEYS:
                _warn_once("ctl_" + k, f"memory.{k} in control_config.json matches no "
                                       f"parameter and was ignored (known: {sorted(_MEMORY_KEYS)})")
                continue
            cfg[k] = v
    override = _read(os.path.join(MAILBOX, "pool_config.json"))
    if override.strip():
        try:
            cfg.update(json.loads(override))
        except (json.JSONDecodeError, ValueError) as e:
            _warn_once("mb_bad", f"mailbox/pool_config.json is not valid JSON ({e}) -> ignored")
    return cfg


def keep_blocks_for(peer, cfg=None):
    """How many verbatim blocks this counterpart should keep right now (= the part excluded
    from compression and injected every turn).

    ★★Decided 2026-09-27: measured as a **token budget**, not a block count. Why not "per
      turn": `reply_list._persist` is **two independent ifs** -- one `in` block when the
      person spoke, one `out` block when the machine did. So a turn is 1 or 2 blocks, never
      fixed. A real thread file confirms it: 79 blocks = 41 `in` / 38 `out`, with steps
      22/23/24 being three `in` blocks in a row (three turns where the machine said nothing).
      -> Counting blocks means counting in an unstable unit; counting tokens is both stable
        and exactly the quantity actually cared about (context budget / cost / latency), and
        it automatically keeps fewer turns when someone speaks in long sentences, leaving
        the injected volume unchanged.

    ★`min_keep_blocks` is a **floor, not decoration**: a pure token budget facing one
      5000-tok monologue computes 0 blocks -- not even the sentence just spoken gets
      injected, which in the field reads as "it forgot what I just said".
    ★★**This is the only place in the module that computes this boundary.** Injection
      (`assemble_voice`), the old blocks fed to the compressor (`thread_older`) and the
      verbatim text kept after compression (`place_thread_compression`) must all use the
      **same** boundary, or the blocks in between end up in neither the summary nor the
      verbatim text -- history silently lost.
    """
    cfg = cfg or load_config()
    budget = int(cfg["keep_recent_tokens"])
    floor = max(1, int(cfg["min_keep_blocks"]))
    _h, blocks = _split_blocks(_read(thread_path(peer)))
    if not blocks:
        return floor
    n = used = 0
    for b in reversed(blocks):
        t = est_tokens(b)
        if n >= floor and used + t > budget:
            break
        used += t
        n += 1
    return n


# -- per-counterpart thread storage (the canonical, only raw store) ------------------
def _safe(peer):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(peer or "unknown"))


def thread_path(peer):
    return os.path.join(THREADS_DIR, f"{_safe(peer)}.md")


def append_thread(peer, direction, step, body):
    """Append one block to a counterpart's thread (direction is 'in' or 'out'). Verbatim and
    append-only; old blocks are gathered up by compression."""
    p = thread_path(peer)
    block = f"### [step {step}] {direction} · {peer}\n{(body or '').rstrip()}\n\n"
    with _THREAD_LOCK:              # ★mutually exclusive with the post-compression overwrite
        prev = _read(p)
        if not prev.strip():
            prev = f"# thread · {peer}\n\n"
        _write(p, prev.rstrip() + "\n\n" + block.rstrip() + "\n")


def next_step(peer):
    """The next free step number for this counterpart = the largest existing step + 1.

    ★Why not "number of blocks + 1": after compression `place_thread_compression` writes the
      summary as `### [step 0] compressed`, so the block count shrinks and a count-based
      number would **collide with a surviving verbatim block**. The symptom of a collision
      is the same step appearing twice, and nothing raises when sorting or reconciling. So
      scan for the maximum.
    """
    _h, blocks = _split_blocks(_read(thread_path(peer)))
    mx = 0
    for b in blocks:
        m = re.match(r"### \[step (\d+)\]", b)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def _split_blocks(text):
    """Split a thread body into (header, [block, ...]). A block starts at '### [step '."""
    if not text.strip():
        return "", []
    idxs = [m.start() for m in _BLOCK_RE.finditer(text)]
    if not idxs:
        return text, []
    header = text[:idxs[0]]
    blocks = []
    for i, s in enumerate(idxs):
        e = idxs[i + 1] if i + 1 < len(idxs) else len(text)
        blocks.append(text[s:e].rstrip())
    return header, blocks


def thread_tail(peer, keep_blocks):
    """A counterpart's most recent keep_blocks blocks (for injection; bounded, which is what
    bounds the pool's token count)."""
    _h, blocks = _split_blocks(_read(thread_path(peer)))
    return "\n\n".join(blocks[-keep_blocks:]) if blocks else ""


def thread_head(peer, keep_blocks):
    """A thread's **compression head** (the step 0 block). Returns an empty string when
    nothing has been compressed, or when the tail already contains it.

    ★Why it has to be injected separately (a hole found during full-chain acceptance on
      2026-09-26): `thread_tail` takes only the last keep_blocks blocks, while
      `place_thread_compression` puts the summary in **block 0** -- which always falls
      outside the tail. Without injecting it, the agent goes **hard amnesiac** about
      anything older the moment the first thread compression runs, while a faithful summary
      sits at the top of the very same file.
    ★The upstream template does not inject the compression head because it relies on
      `semantic.md` for long-term memory; but with a single counterpart in voice, the pool
      compression trigger is structurally unreachable (pool_tok = semantic + a bounded tail,
      which never grows), so semantic stays empty forever.
    ★It is self-converging: the next compression's input, `thread_older`, includes this
      block again and re-compresses it, so its length is held by the word limit in the
      compression skill prompt and it does not grow without bound.
    """
    _h, blocks = _split_blocks(_read(thread_path(peer)))
    if not blocks:
        return ""
    if len(blocks) <= keep_blocks:
        return ""                      # the tail already holds every block; injecting again duplicates
    if not _COMPRESSED_HEAD_RE.match(blocks[0]):
        return ""
    return blocks[0]


def thread_older(peer, keep_blocks):
    """A counterpart's blocks outside the most recent keep_blocks (thread_compressor's input)."""
    _h, blocks = _split_blocks(_read(thread_path(peer)))
    if len(blocks) <= keep_blocks:
        return ""
    return "\n\n".join(blocks[:-keep_blocks])


def active_peers():
    """Threads with content, most recently modified first -- whoever spoke last comes first."""
    if not os.path.isdir(THREADS_DIR):
        return []
    files = []
    for fn in os.listdir(THREADS_DIR):
        if not fn.endswith(".md"):
            continue                    # ★`_archive/` is a directory, and this line excludes it
        full = os.path.join(THREADS_DIR, fn)
        if _read(full).strip():
            files.append((os.path.getmtime(full), fn[:-3]))
    files.sort(reverse=True)   # newest first
    return [peer for _mt, peer in files]


# -- token accounting + trigger decisions --------------------------------------------
def compression_plan():
    """Return {'threads':[peer,...], 'pool':bool, '_debug':{...}}. Called by chain after a turn."""
    cfg = load_config()
    over_threads = []
    tails_tok = 0
    per_thread = {}
    keeps = {}
    for peer in active_peers():
        keep = keeps[peer] = keep_blocks_for(peer, cfg)   # ★each counterpart has its own boundary
        full = _read(thread_path(peer))
        tok = est_tokens(full)
        per_thread[peer] = tok
        if tok >= cfg["thread_trigger_tokens"] and thread_older(peer, keep).strip():
            over_threads.append(peer)
        tails_tok += est_tokens(thread_tail(peer, keep))
    pool_tok = est_tokens(_read(SEMANTIC)) + tails_tok
    return {
        "threads": over_threads,
        "pool": pool_tok >= cfg["pool_trigger_tokens"],
        "_debug": {"pool_tok": pool_tok, "per_thread_tok": per_thread,
                   "thread_trigger": cfg["thread_trigger_tokens"],
                   "pool_trigger": cfg["pool_trigger_tokens"],
                   "keep_blocks": keeps, "keep_recent_tokens": cfg["keep_recent_tokens"]},
    }


def forced_plan(include_pool=False):
    """The plan used by `/compact`: **ignore the thresholds** and list every counterpart that
    really has old blocks to compress right now.

    ★It still goes through the `thread_older(...)` test -- a counterpart whose retention
      window is not yet full has no old blocks at all, and forcing it would ask the LLM to
      summarise an empty text. The "empty output is never written to disk" guard in
      `maybe_compress` only blocks an empty **return**, not an input that was empty to begin
      with (in that case the LLM invents something, which is then filed as a real summary
      and replaces the verbatim text).
    ★★**When all that remains outside the window is the previous summary, it is compressed
      again -- this is deliberate design, decided 2026-09-27, not a bug.** Repeatedly
      re-compressing the summary is the **convergence mechanism**: the older something is,
      the more abstractly it is held, and the length is bounded by the word limit in the
      compression skill prompt. A guard was once added here on the theory that "a summary of
      a summary loses information"; it was rejected on the spot as the intended design --
      do not add it back.
    ★`include_pool` defaults to False: the trigger formula for the pool layer
      (semantic.md) is still an open design decision, and force-compressing it would make an
      LLM-written integrated narrative part of every turn's injection from then on -- not
      something a convenience command should decide.
    """
    cfg = load_config()
    peers = []
    for peer in active_peers():
        if thread_older(peer, keep_blocks_for(peer, cfg)).strip():
            peers.append(peer)
    return {"threads": peers, "pool": bool(include_pool), "_debug": {"forced": True}}


# -- assembling the voice envelope (the speaker's entrance to memory) ----------------
def assemble_voice(session_dir, peer):
    """Build `workspace/current/envelope.md` = [integrated semantic layer] + [this
    counterpart's recent verbatim tail].

    ★This is the **only** entrance to the speaker's memory. Under the single-shot pattern
      session history is always exactly one turn (verified: on the second turn it cannot
      recall a name given on the first), so "remembering what we talked about" rests
      entirely on this file being re-injected every turn through context_files.

    ★Why context_files is not simply pointed at `pool/threads/` (the engine does support
      directory injection): directory injection reads **whole files**, and a thread file's
      ceiling is `thread_trigger_tokens`. In voice, time to first word scales with input
      size, so injecting that much is suicide. The design was always to **inject the tail
      only**: a bounded tail bounds the pool. Hence this layer.

    ★Three sections, not two (fixed 2026-09-26): semantic + the **thread compression head**
      + the tail. The compression head was added during acceptance testing -- see
      thread_head().
    """
    cfg = load_config()
    semantic = _read(SEMANTIC, "(The integrated semantic layer is still empty.)").strip()
    keep = keep_blocks_for(peer, cfg)
    head = thread_head(peer, keep).strip()
    tail = thread_tail(peer, keep).strip()

    # ★Section order = stable -> slow-changing -> changes every turn (prefix caching wants
    #   the stable prefix first): persona (in the config, most stable) -> semantic (changes
    #   only on compression) -> compression head (same) -> tail (every turn).
    lines = [
        "# Earlier conversation (injected by the system; not what the other person just said)", "",
        "## What I understand about this person and these matters", "",
        semantic or "(Nothing yet.)", "",
    ]
    if head:
        lines += ["## Talked about earlier (compressed into a summary, not verbatim)", "", head, ""]
    lines += [
        f"## Recent verbatim exchanges (newest at the bottom · the last {keep} blocks, "
        f"within about {cfg['keep_recent_tokens']} tokens)", "",
        tail or "(Nothing said yet.)", "",
        "---", "",
        "Everything above is background. The user message below is what the other person is "
        "saying **now**; answer that.",
    ]
    envelope = "\n".join(lines)
    _write(os.path.join(session_dir, "envelope.md"), envelope)
    return envelope


# -- staging compression input / filing output (chain calls one before and one after) --
def stage_thread_input(session_dir, peer):
    """Stage a counterpart's old blocks into `compress_input.md` (thread_compressor's
    context_file).

    ★★Returns **`n_older` = the number of blocks actually handed to the compressor**, and
      the caller must pass it straight back to `place_thread_compression`. The reason is
      concurrency (2026-09-27): compression runs on a worker thread and takes seconds, and
      the main thread may append new blocks meanwhile. If filing recomputed the boundary as
      "keep the last keep blocks", those new blocks would end up **neither in the summary
      (they were inside the retention window when staging happened) nor in the verbatim text
      that is kept (new blocks pushed them out)** -- history silently lost. Locating by
      block count cannot drift: the summary always covers exactly the first n_older blocks.
    """
    with _THREAD_LOCK:
        keep = keep_blocks_for(peer)
        _h, blocks = _split_blocks(_read(thread_path(peer)))
        n_older = max(0, len(blocks) - keep)
        older = "\n\n".join(blocks[:n_older])
    _write(os.path.join(session_dir, "compress_input.md"),
           f"# To compress: the older part of this agent's conversation with {peer} "
           f"(the most recent turns stay where they are and are not included here)\n\n{older}\n")
    return n_older


def archive_dir():
    return os.path.join(THREADS_DIR, "_archive")


def place_thread_compression(peer, compressed_text, n_older):
    """thread_compressor's output -> filed back: the compressed block goes on top, and
    `blocks[n_older:]` is kept verbatim.

    ★★**This function is lossy: it overwrites the thread file, and that file is the only
      store of the verbatim text.** So before filing, the whole pre-overwrite file is copied
      to `threads/_archive/{peer}_{timestamp}.md` (added 2026-09-27). However faithful a
      summary is, it is still a summary, and there is **no second copy of the verbatim text**
      anywhere on this chain -- the STT fixtures only hold the recognition stream from the
      listening side, never what the machine said.
    ★Uses `n_older` rather than recomputing the boundary: see `stage_thread_input`.
    """
    with _THREAD_LOCK:
        full = _read(thread_path(peer))
        _h, blocks = _split_blocks(full)
        n_older = max(0, min(int(n_older), len(blocks)))
        recent = blocks[n_older:]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if full.strip():
            _write(os.path.join(archive_dir(), f"{_safe(peer)}_{stamp}.md"), full)
        new = (f"# thread · {peer}\n\n"
               f"### [step 0] compressed · {peer}\n"
               f"({n_older} older blocks were compressed on {time.strftime('%Y-%m-%d %H:%M')}"
               f"; the originals are in _archive/{_safe(peer)}_{stamp}.md. "
               f"The most recent {len(recent)} blocks are kept verbatim below.)\n\n"
               f"{compressed_text.strip()}\n\n"
               + "\n\n".join(recent) + "\n")
        _write(thread_path(peer), new)


def stage_pool_input(session_dir):
    """Stage [semantic + every active tail] into compress_input.md (pool_compressor's
    context_file)."""
    cfg = load_config()
    parts = [f"# The previous integrated semantic layer\n\n{_read(SEMANTIC).strip()}", "",
             "# Recent additions per counterpart (to be folded into the semantic layer)"]
    for peer in active_peers():
        t = thread_tail(peer, keep_blocks_for(peer, cfg))
        if t.strip():
            parts.append(f"\n## With {peer}\n\n{t}")
    _write(os.path.join(session_dir, "compress_input.md"), "\n".join(parts) + "\n")


def place_pool_compression(new_semantic_text):
    """pool_compressor's output -> overwrite semantic.md (the tails stay in the thread files
    as the canonical store and the injection source)."""
    _write(SEMANTIC, new_semantic_text.strip() + "\n")


if __name__ == "__main__":
    # Self-check: print the trigger decision and a token-estimate reconciliation (no LLM spend)
    print("--- est_tokens self-check (measured: Chinese 0.589-0.735 tok/char, English 0.141 tok/char) ---")
    for label, sample in (("100 CJK characters", "今天天气不错我们出去走走吧" * 8),
                          ("100 ASCII characters", "the quarterly revenue forecast was revised downward by "),
                          ("empty", "")):
        print(f"  {label:22s} len={len(sample):4d}  est_tokens={est_tokens(sample)}")
    print("\n--- compression_plan ---")
    print(json.dumps(compression_plan(), ensure_ascii=False, indent=2))
