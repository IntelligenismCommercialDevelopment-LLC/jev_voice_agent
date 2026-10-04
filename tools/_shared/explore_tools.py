"""explore_tools.py -- the per-role fenced tool factory (loaded by the engine's free_search mode).

The canonical copy lives in tools/_shared/ and is copied by the sync step into each
dispatch's tools/ subdirectory; every role shares the same code, and **behaviour is driven
by each dispatch_config.json's free_search block** (read_allow / write_allow /
max_read_tokens / can_delete) -- so one file ends up with different read/write fences in
different dispatches.

The engine's calling convention (the free_search branch of IAF_AI/dispatch/*/dispatch.py):
    make_tools(read_roots, write_root) -> {name: {description, parameters, handler}}
This factory **ignores** the read_roots/write_root the engine passes in and resolves the
absolute fences itself from the sibling dispatch_config.json's free_search block (more
controllable, and consistent with reviewing the config).

🔒 Security rule (a hard-coded denylist that no allow can override):
    config_base/ · IAF_AI/ · world/ · prompt_base/_golden/ · memory/experience_ledger.jsonl · *_retired/
"""

import glob as _glob
import json
import os

# -- locate shell_root and this dispatch's config ------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # …/IAF_AI/dispatch/<role>/tools
_DISPATCH_DIR = os.path.dirname(_THIS_DIR)                        # …/IAF_AI/dispatch/<role>
_SHELL_ROOT = os.path.realpath(os.path.join(_DISPATCH_DIR, "..", "..", ".."))  # the shell root
_CONFIG_PATH = os.path.join(_DISPATCH_DIR, "dispatch_config.json")

# 🔒 Never touchable (reading may be opened up a little, but writing and deleting never)
_DENY_ALWAYS = [
    os.path.realpath(os.path.join(_SHELL_ROOT, p)) for p in [
        "config_base", "IAF_AI", "world",
        "prompt_base/_golden", "prompt_base/_retired",
        "memory/experience_ledger.jsonl", "memory/_retired",
    ]
]

READ_CAP_CHARS = 200_000       # per-file read ceiling (so one huge file cannot blow the context)
GREP_CAP_HITS = 80


def _load_free_search():
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    for _k, a in (cfg.get("agents") or {}).items():
        if a.get("role") == "core" and a.get("free_search"):
            return a["free_search"]
    return {}


def _abs_roots(rel_list):
    return [os.path.realpath(os.path.join(_SHELL_ROOT, p)) for p in (rel_list or [])]


def _under(path_abs, roots):
    for r in roots:
        if path_abs == r or path_abs.startswith(r + os.sep):
            return True
    return False


def _denied(path_abs):
    for d in _DENY_ALWAYS:
        if path_abs == d or path_abs.startswith(d + os.sep):
            return True
    return False


def _resolve(path, roots):
    """Normalise a caller-supplied path (possibly relative to shell_root) to an absolute one
    and check it is inside roots and not in the denylist.
    Returns (abs_path, err). A non-empty err means it is out of bounds."""
    if os.path.isabs(path):
        p = os.path.realpath(path)
    else:
        p = os.path.realpath(os.path.join(_SHELL_ROOT, path))
    if _denied(p):
        return None, f"[DENIED] path hits the security denylist: {path}"
    if not _under(p, roots):
        allowed = ", ".join(os.path.relpath(r, _SHELL_ROOT) for r in roots) or "(none)"
        return None, f"[OUT OF BOUNDS] {path} is outside the allowed range. Allowed: {allowed}"
    return p, None


def _rel(p):
    try:
        return os.path.relpath(p, _SHELL_ROOT)
    except Exception:
        return p


def make_tools(read_roots=None, write_root=None):
    """The engine entry point. Ignores the arguments passed in and resolves the fences from
    this dispatch config's free_search block."""
    fs = _load_free_search()
    READ_ROOTS = _abs_roots(fs.get("read_allow") or fs.get("free_search_path") or [])
    WRITE_ROOTS = _abs_roots(fs.get("write_allow") or [])
    MAX_READ_TOKENS = int(fs.get("max_read_tokens", 120_000))
    CAN_DELETE = bool(fs.get("can_delete", False))

    budget = {"read_tokens": 0}   # a mutable budget counter in the closure (cumulative for this run)

    def _budget_note():
        used = budget["read_tokens"]
        if used >= MAX_READ_TOKENS:
            return (f"\n\n⛔ The reading budget is exhausted (about {used} / {MAX_READ_TOKENS} tokens read). "
                    f"**Stop exploring now and start producing**: write your deliverable, then finish.")
        return ""

    def _over_budget():
        return budget["read_tokens"] >= MAX_READ_TOKENS

    # -- read-only exploration ------------------------------------------------------
    def list_dir(inp):
        path = (inp or {}).get("path", ".")
        p, err = _resolve(path, READ_ROOTS)
        if err:
            return err
        if not os.path.isdir(p):
            return f"[ERROR] not a directory: {_rel(p)}"
        rows = []
        for name in sorted(os.listdir(p)):
            fp = os.path.join(p, name)
            if _denied(fp):
                continue
            if os.path.isdir(fp):
                rows.append(f"  {name}/")
            else:
                try:
                    kb = os.path.getsize(fp) / 1024
                    rows.append(f"  {name}  ({kb:.1f} KB)")
                except OSError:
                    rows.append(f"  {name}")
        return f"directory {_rel(p)}/:\n" + ("\n".join(rows) or "  (empty)")

    def read_file(inp):
        if _over_budget():
            return _budget_note().strip()
        path = (inp or {}).get("path", "")
        p, err = _resolve(path, READ_ROOTS)
        if err:
            return err
        if not os.path.isfile(p):
            return f"[ERROR] file does not exist: {_rel(p)}"
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read(READ_CAP_CHARS + 1)
        except Exception as e:
            return f"[ERROR] read failed: {e}"
        truncated = len(text) > READ_CAP_CHARS
        text = text[:READ_CAP_CHARS]
        budget["read_tokens"] += len(text) // 4         # a chars/4 estimate
        head = f"===== {_rel(p)} ====="
        if truncated:
            head += f"\n(over {READ_CAP_CHARS} characters; truncated)"
        return f"{head}\n{text}{_budget_note()}"

    def read_docx(inp):
        if _over_budget():
            return _budget_note().strip()
        path = (inp or {}).get("path", "")
        p, err = _resolve(path, READ_ROOTS)
        if err:
            return err
        if not os.path.isfile(p):
            return f"[ERROR] file does not exist: {_rel(p)}"
        try:
            import docx  # python-docx
        except ImportError:
            return ("[MISSING DEP] reading .docx needs python-docx (pip install python-docx). "
                    "This file cannot be read for now; use .md/.txt material instead or skip it.")
        try:
            d = docx.Document(p)
            text = "\n".join(par.text for par in d.paragraphs)
        except Exception as e:
            return f"[ERROR] failed to parse the docx: {e}"
        text = text[:READ_CAP_CHARS]
        budget["read_tokens"] += len(text) // 4
        return f"===== {_rel(p)} (docx) =====\n{text}{_budget_note()}"

    def grep(inp):
        inp = inp or {}
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        if not pattern:
            return "[ERROR] pattern is required"
        p, err = _resolve(path, READ_ROOTS)
        if err:
            return err
        import re
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"[ERROR] invalid regex: {e}"
        files = []
        if os.path.isfile(p):
            files = [p]
        else:
            for fp in _glob.glob(os.path.join(p, "**", "*"), recursive=True):
                if os.path.isfile(fp) and not _denied(fp):
                    files.append(fp)
        hits = []
        for fp in files:
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{_rel(fp)}:{i}: {line.strip()[:200]}")
                            if len(hits) >= GREP_CAP_HITS:
                                break
            except Exception:
                continue
            if len(hits) >= GREP_CAP_HITS:
                hits.append(f"...(hit the cap of {GREP_CAP_HITS} matches)")
                break
        return "\n".join(hits) if hits else f"({pattern} had no matches in {_rel(p)})"

    # -- writing (constrained by write_allow) ----------------------------------------
    def write_file(inp):
        inp = inp or {}
        path = inp.get("path", "")
        content = inp.get("content", "")
        p, err = _resolve(path, WRITE_ROOTS)
        if err:
            return err
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content if content.endswith("\n") else content + "\n")
        except Exception as e:
            return f"[ERROR] write failed: {e}"
        return f"[WRITTEN] {_rel(p)} ({len(content)} characters)"

    def edit_file(inp):
        inp = inp or {}
        path = inp.get("path", "")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        p, err = _resolve(path, WRITE_ROOTS)
        if err:
            return err
        if not os.path.isfile(p):
            return f"[ERROR] file does not exist, cannot edit: {_rel(p)} (use write_file to create it)"
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return f"[ERROR] read failed: {e}"
        if old and old not in text:
            return f"[ERROR] old_string was not found in {_rel(p)} (it must match character for character)"
        cnt = text.count(old) if old else 0
        if old and cnt > 1:
            return f"[ERROR] old_string matched {cnt} times and is not unique -- give more surrounding context"
        newtext = text.replace(old, new, 1) if old else new
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(newtext)
        except Exception as e:
            return f"[ERROR] writing back failed: {e}"
        return f"[EDITED] {_rel(p)}"

    tools = {
        "list_dir": {
            "description": "List the contents of a directory within the allowed range (subdirectories/files + size). path is relative to the shell root.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "directory path (relative to the shell root)"}}},
            "handler": list_dir,
        },
        "read_file": {
            "description": "Read a whole text file within the allowed range (once the reading budget is exceeded it refuses and tells you to start producing).",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
            "handler": read_file,
        },
        "read_docx": {
            "description": "Read the body text of a .docx file within the allowed range.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
            "handler": read_docx,
        },
        "grep": {
            "description": "Search by regex within the allowed range (a file, or a directory recursively). Returns path:lineno: content.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"]},
            "handler": grep,
        },
    }
    # No readable roots (read_allow and free_search_path are both empty) -> expose no
    # read-only exploration tools at all, so the model does not burn tool loops on read tools
    # that will "always be refused" (the read-side fence mirrors the write-side WRITE_ROOTS).
    if not READ_ROOTS:
        tools.clear()
    if WRITE_ROOTS:
        tools["write_file"] = {
            "description": "Create or overwrite a file within the allowed write range. path is relative to the shell root.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]},
            "handler": write_file,
        }
        tools["edit_file"] = {
            "description": "Make one unique string replacement in an existing file within the allowed write range (old_string -> new_string).",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"}},
                "required": ["path", "new_string"]},
            "handler": edit_file,
        }
    if WRITE_ROOTS and CAN_DELETE:
        def delete_file(inp):
            path = (inp or {}).get("path", "")
            p, err = _resolve(path, WRITE_ROOTS)
            if err:
                return err
            if not os.path.isfile(p):
                return f"[ERROR] file does not exist: {_rel(p)}"
            try:
                os.remove(p)
            except Exception as e:
                return f"[ERROR] delete failed: {e}"
            return f"[DELETED] {_rel(p)}"
        tools["delete_file"] = {
            "description": "Delete a file within the allowed write range (use with care).",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
            "handler": delete_file,
        }

    # -- live STATUS trail: wrap every handler so one line is appended on entry and exit
    #    (best-effort, with no change to the engine).
    # status_log is copied into this directory by sync; when missing it degrades silently
    # (the tools still work, there is just no trail).
    try:
        import time as _time
        import status_log as _sl
    except Exception:
        _sl = None
    if _sl is not None:
        def _wrap(_h, _nm):
            def _inner(inp):
                _sl.tool_start(_nm, inp)
                _t0 = _time.time()
                try:
                    out = _h(inp)
                except Exception as e:
                    _sl.tool_done(_nm, _time.time() - _t0, err=e)
                    raise
                _sl.tool_done(_nm, _time.time() - _t0, out=out)
                return out
            return _inner
        for _name, _spec in tools.items():
            _spec["handler"] = _wrap(_spec["handler"], _name)

    return tools
