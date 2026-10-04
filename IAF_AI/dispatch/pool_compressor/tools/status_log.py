"""status_log.py -- a live, per-event STATUS.md trail.

The canonical copy lives in tools/_shared/ and is copied by sync into each dispatch's tools/
(so explore_tools can import it as a sibling); chain.py imports this file directly too. Both
share the same append logic and target the same file:
    <shell_root>/workspace/current/STATUS.md
(workspace/current is the symlink chain points at this cycle's session_{id}/ folder, so the
trail lands inside this cycle's archive.)

★One unified version: byte-identical across shells. Two things are kept outside the file so
  that the body no longer forks per organ --
  - the per-organ attention-ledger watch roots live in this shell's
    tools/_shared/ledger_roots.json (pure data; missing = the ledger is off).
  - alongside writing this shell's STATUS.md, every event also appends a line prefixed with
    [organ] to total_status.md at the cluster root (a live confluence of every IAF's internal
    events; `tail -f total_status.md` to watch). Best-effort, and never interrupts the role.

Design points:
- **Append per event** (append-only), not one dump at the end of a cycle -- so a kill still
  leaves behind the trail of what already happened.
- **Best-effort**: any write failure is swallowed silently and never interrupts the role or
  a tool call.
- Event shapes: ### Turn block -> 🧠 llm_started -> ▶/✓ tool -> 💬 response.
"""

import json
import os
import time

# -- Locate shell_root: walk up from this file to the level that holds workspace/ --------
# (This file may sit in tools/_shared/ or have been synced to IAF_AI/dispatch/<role>/tools/,
#  so the depth is unknown and it has to be probed by walking up.)
def _find_shell_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "workspace")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # Fallback: assume the tools/_shared/ layout (../..)
    return os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


_SHELL_ROOT = _find_shell_root()
_ORGAN = os.path.basename(_SHELL_ROOT).replace("_iaf", "")   # the organ tag used in the confluence


# -- Locate the cluster root (where total_status.md lands): walk up from shell_root looking
#    for config_base/cluster_manifest.json (a name-free marker) --
def _find_cluster_root():
    d = _SHELL_ROOT
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "config_base", "cluster_manifest.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # No marker found -> this shell is standalone, not inside a cluster. The confluence
    # file then belongs to this shell itself.
    # ★Never fall back to a directory *above* the shell root: a standalone clone would
    #   write total_status.md outside its own folder (two levels up, i.e. into whatever
    #   happens to contain the checkout). Best-effort writes must still stay inside.
    return _SHELL_ROOT


_TOTAL_STATUS = os.path.join(_find_cluster_root(), "total_status.md")


def _status_path():
    # Write workspace/current/STATUS.md (the symlink target = this cycle's session folder)
    return os.path.join(_SHELL_ROOT, "workspace", "current", "STATUS.md")


# -- the attention op-ledger ------------------------------------------------------------
# chain and the tool _wrap share this module instance within one process, so _STATE is shared
# naturally: cycle_header stores session/step, turn_header stores role, tool_start parks the
# path, and tool_done's success branch emits one machine-readable record.
# Self-contained (it does not import an attention library, because this file gets synced into
# each dispatch/tools and such a library would not travel with it).
_STATE = {"session_id": "", "step": 0, "role": "", "_pending": {}}

_LEDGER_TOOLS = {"read_file", "read_docx", "write_file", "edit_file",
                 "delete_file", "list_dir", "grep"}


def _load_ledger_roots():
    """Per-organ: which directory prefixes the attention ledger watches (relative paths as
    the LLM sees them). Kept outside the code in this shell's tools/_shared/ledger_roots.json
    (pure data, different per organ). Missing or empty -> returns (), which turns the ledger
    off (the case for organs with no consumer and no output, such as the speaker here)."""
    try:
        with open(os.path.join(_SHELL_ROOT, "tools", "_shared", "ledger_roots.json"), encoding="utf-8") as f:
            roots = json.load(f)
        return tuple(roots) if isinstance(roots, list) else ()
    except Exception:
        return ()


_LEDGER_ROOTS = _load_ledger_roots()
# ★Cross-file contract: these four prefixes must match the strings explore_tools.py returns
#   on a soft failure. Change one side only and failure detection silently stops working --
#   a soft failure then gets counted as a success, with no error anywhere.
_LEDGER_FAIL_PREFIXES = ("[OUT OF BOUNDS]", "[ERROR]", "[DENIED]", "[MISSING DEP]")


def _ledger_path():
    return os.path.join(_SHELL_ROOT, "world", "attention_ledger.jsonl")


def _norm_path(path):
    if not path:
        return ""
    p = str(path).strip().strip("'\"").replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p.rstrip("/ ").strip()


def _emit_ledger(tool, path, ok, ts):
    """Append one op-ledger record (best-effort; any exception is swallowed silently and
    never interrupts a tool).
    An empty _LEDGER_ROOTS means startswith(()) is always False, so this returns immediately
    (the ledger is off)."""
    try:
        path = _norm_path(path)
        if tool not in _LEDGER_TOOLS or not path.startswith(_LEDGER_ROOTS):
            return
        rec = {"session_id": _STATE["session_id"], "step": _STATE["step"],
               "role": _STATE["role"], "tool": tool, "path": path,
               "ok": bool(ok), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        p = _ledger_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append(line):
    """Best-effort: append one line to this shell's STATUS.md, and also append an
    [organ]-prefixed copy to total_status.md at the cluster root.
    The two writes have independent try blocks; either failing is swallowed silently and
    never interrupts the role. total_status is the live confluence of every IAF's events
    (watch it with tail -f)."""
    try:
        p = _status_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)   # if the current symlink exists, this writes into its target
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        tagged = line.replace("\n", "\n[%s] " % _ORGAN)   # tag every physical line, so grep "[organ]" filters cleanly
        with open(_TOTAL_STATUS, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (_ORGAN, tagged))
    except Exception:
        pass


def _ts():
    return time.strftime("%H:%M:%S")


# -- cycle and turn level ---------------------------------------------------------------
def cycle_header(session_id, world_time, step):
    _STATE["session_id"] = session_id
    _STATE["step"] = step
    _STATE["_pending"] = {}
    append(f"# STATUS · {session_id}\n\nworld time: {world_time} · step {step}\n"
           f"(a live per-event trail -- an interruption still leaves what already happened.)")


def turn_header(role, detail=""):
    _STATE["role"] = role
    line = f"\n---\n### Turn [{_ts()}]  role={role}"
    if detail:
        line += f"\n  {detail}"
    append(line)


def llm_started(role, model=""):
    append(f"  - [{_ts()}] 🧠 {role} llm_started · model={model}")


def response(role, text):
    snip = str(text or "").replace("\n", " ")[:180]
    append(f"  - [{_ts()}] 💬 {role}: {snip}")


def error(msg):
    append(f"  - [{_ts()}] ✗ ERROR: {str(msg)[:180]}")


def footer(note=""):
    append(f"\n---\narchived at {_ts()}. {note}")


# -- tool level (explore_tools calls one of these on entry and one on exit per handler) --
def tool_start(tool, inp):
    # No truncation here: the full inp dict is available, so park the real path directly.
    try:
        _STATE["_pending"][tool] = (inp or {}).get("path") if isinstance(inp, dict) else None
    except Exception:
        pass
    s = str(inp if inp is not None else "")
    if len(s) > 300:
        s = s[:300] + "…"
    append(f"  - [{_ts()}] ▶ `{tool}` {s}")


def tool_done(tool, dur_s=None, out=None, err=None):
    d = f" ({dur_s:.1f}s)" if isinstance(dur_s, (int, float)) else ""
    path = _STATE["_pending"].pop(tool, None) if isinstance(_STATE.get("_pending"), dict) else None
    if err is not None:
        _emit_ledger(tool, path, ok=False, ts=_ts())
        append(f"  - [{_ts()}] ✗ `{tool}` failed{d}: {str(err)[:120]}")
        return
    meta = ""
    head = ""
    if out is not None:
        s = str(out)
        meta = f" → {len(s)}c"
        # The first line gives a semantic hint (e.g. [WRITTEN] / [OUT OF BOUNDS] / ===== filename)
        head = s.strip().splitlines()[0][:60] if s.strip() else ""
        if head:
            meta += f" · {head}"
    # A soft failure is detected from the head prefix: when a handler returns a string like
    # [OUT OF BOUNDS]/[ERROR] it does not raise and still arrives here, and must not count as success.
    ok = not head.startswith(_LEDGER_FAIL_PREFIXES)
    _emit_ledger(tool, path, ok=ok, ts=_ts())
    append(f"  - [{_ts()}] ✓ `{tool}` done{d}{meta}")
