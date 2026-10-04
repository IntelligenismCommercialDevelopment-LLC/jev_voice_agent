"""
Session I/O — Container-based session management for Chat Dispatch.

Each session is a self-contained directory under sessions/{sid}/.
This module does NOT depend on dispatch/roundtable/session_manager.py
because that module hard-codes a flat {sid}.jsonl layout.

Public API:
    new_session(dispatch_root, model, provider) -> sid
    append_history(dispatch_root, sid, record)
    load_history(dispatch_root, sid) -> list[dict]
    list_active_sessions(dispatch_root) -> list[dict]      # for UI session list
    load_meta(dispatch_root, sid) -> dict
    update_meta(dispatch_root, sid, **fields)
    log_call(dispatch_root, sid, event, **fields)
    delete_session(dispatch_root, sid)

Path helpers (used by dispatch.py for resolving "session:" prefixed paths):
    session_dir(dispatch_root, sid) -> str
    session_staging_dir(dispatch_root, sid) -> str
    session_search_archive_dir(dispatch_root, sid) -> str
"""

import json
import os
import shutil
import time
import uuid


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _sessions_root(dispatch_root):
    return os.path.join(dispatch_root, "sessions")


def session_dir(dispatch_root, sid):
    return os.path.join(_sessions_root(dispatch_root), sid)


def session_staging_dir(dispatch_root, sid):
    return os.path.join(session_dir(dispatch_root, sid), "staging")


def session_search_archive_dir(dispatch_root, sid):
    return os.path.join(session_dir(dispatch_root, sid), "search_archive")


def _history_path(dispatch_root, sid):
    return os.path.join(session_dir(dispatch_root, sid), "history.jsonl")


def _call_log_path(dispatch_root, sid):
    return os.path.join(session_dir(dispatch_root, sid), "call_log.jsonl")


def _meta_path(dispatch_root, sid):
    return os.path.join(session_dir(dispatch_root, sid), "meta.json")


# ---------------------------------------------------------------------------
# Time / ID helpers
# ---------------------------------------------------------------------------

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _generate_sid():
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}-{short}"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def new_session(dispatch_root, model="", provider=""):
    """Create a new session container with initial meta + empty state.
    Returns the session_id."""
    sid = _generate_sid()
    sdir = session_dir(dispatch_root, sid)

    os.makedirs(sdir, exist_ok=True)
    os.makedirs(session_staging_dir(dispatch_root, sid), exist_ok=True)
    os.makedirs(session_search_archive_dir(dispatch_root, sid), exist_ok=True)

    # Initial history with session_start record
    start_record = {
        "type": "session_start",
        "session_id": sid,
        "timestamp": _now_iso(),
    }
    with open(_history_path(dispatch_root, sid), "w", encoding="utf-8") as f:
        f.write(json.dumps(start_record, ensure_ascii=False) + "\n")

    # Empty call log
    open(_call_log_path(dispatch_root, sid), "w", encoding="utf-8").close()

    # Initial meta
    meta = {
        "session_id": sid,
        "title_hint": "New session",
        "created_at": _now_iso(),
        "last_activity": _now_iso(),
        "status": "active",
        "round_count": 0,
        "model": model,
        "provider": provider,
    }
    _write_json(_meta_path(dispatch_root, sid), meta)

    return sid


def delete_session(dispatch_root, sid):
    """Remove the entire session directory. Used for abandoning unwanted sessions.
    For Phase 5 archive, use a separate close-and-archive operation instead."""
    sdir = session_dir(dispatch_root, sid)
    if os.path.isdir(sdir):
        shutil.rmtree(sdir)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def append_history(dispatch_root, sid, record):
    """Append one record to history.jsonl. Also updates last_activity in meta
    and (for the first user_input) fills title_hint."""
    if "timestamp" not in record:
        record["timestamp"] = _now_iso()

    path = _history_path(dispatch_root, sid)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Session history not found: {path}")

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Meta updates
    meta = load_meta(dispatch_root, sid)
    meta["last_activity"] = record["timestamp"]

    if record.get("type") == "user_input":
        # Fill title_hint if it is still default
        if meta.get("title_hint") in ("", "New session", None):
            content = (record.get("content") or "").strip().replace("\n", " ")
            meta["title_hint"] = content[:30] or "New session"

    if record.get("type") == "agent_response":
        meta["round_count"] = int(meta.get("round_count", 0)) + 1

    _write_json(_meta_path(dispatch_root, sid), meta)


def load_history(dispatch_root, sid):
    """Return all records from history.jsonl as a list of dicts.
    Returns [] if session does not exist."""
    path = _history_path(dispatch_root, sid)
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def load_meta(dispatch_root, sid):
    path = _meta_path(dispatch_root, sid)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def update_meta(dispatch_root, sid, **fields):
    meta = load_meta(dispatch_root, sid)
    meta.update(fields)
    meta["last_activity"] = _now_iso()
    _write_json(_meta_path(dispatch_root, sid), meta)


# ---------------------------------------------------------------------------
# Session list (for UI)
# ---------------------------------------------------------------------------

def list_active_sessions(dispatch_root):
    """Return list of active session summaries, most-recently-active first.

    Each dict: {sid, title_hint, last_activity, status, round_count}
    'active' means: present under sessions/ and meta.status != 'closing'.
    """
    root = _sessions_root(dispatch_root)
    if not os.path.isdir(root):
        return []

    summaries = []
    for entry in os.listdir(root):
        sdir = os.path.join(root, entry)
        if not os.path.isdir(sdir):
            continue
        meta = load_meta(dispatch_root, entry)
        if not meta:
            # Directory exists but meta missing — skip
            continue
        if meta.get("status") in ("closing", "archived"):
            continue
        summaries.append({
            "sid": entry,
            "title_hint": meta.get("title_hint", "New session"),
            "last_activity": meta.get("last_activity", ""),
            "status": meta.get("status", "active"),
            "round_count": meta.get("round_count", 0),
        })

    summaries.sort(key=lambda s: s["last_activity"], reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------

def log_call(dispatch_root, sid, event, **fields):
    """Append one JSON line to the session's call_log.jsonl.
    Silently no-ops if sid is falsy or file cannot be written."""
    if not sid:
        return
    entry = {"timestamp": _now_utc(), "event": event}
    entry.update(fields)
    try:
        with open(_call_log_path(dispatch_root, sid), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except IOError:
        pass


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _write_json(path, data):
    """Atomic write via temp file + rename."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
