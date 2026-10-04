"""
Session Manager — JSONL read/write for dispatch collaboration records

Each session is a single .jsonl file in the dispatch's sessions/ directory.
One JSON object per line. This is the persistent storage — staging files
are ephemeral, session files are permanent.

Record types:
    session_start  — marks creation, stores config snapshot
    user_input     — user's message that triggered a round
    agent_response — one agent's response in one round
    round_complete — marks end of a full turn cycle

File safety: writes use temp file + rename (atomic on same filesystem)
to avoid corruption if the process is interrupted mid-write.
"""

import json
import os
import tempfile
import time
import uuid


def create_session(sessions_dir):
    """
    Create a new session. Returns session_id.
    Creates an empty .jsonl file in sessions_dir.

    Args:
        sessions_dir: Absolute path to the sessions/ directory

    Returns:
        session_id string
    """
    os.makedirs(sessions_dir, exist_ok=True)
    session_id = _generate_session_id()
    filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")

    # Write the session_start record
    start_record = {
        "type": "session_start",
        "session_id": session_id,
        "timestamp": _now(),
    }
    _atomic_append(filepath, start_record)

    return session_id


def append_to_session(sessions_dir, session_id, record):
    """
    Append a record to an existing session file.

    Args:
        sessions_dir: Absolute path to the sessions/ directory
        session_id:   The session to append to
        record:       Dict to write as one JSON line

    The record is automatically stamped with a timestamp if not present.
    """
    filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session file not found: {filepath}")

    if "timestamp" not in record:
        record["timestamp"] = _now()

    _atomic_append(filepath, record)


def load_session(sessions_dir, session_id):
    """
    Load all records from a session file.

    Args:
        sessions_dir: Absolute path to the sessions/ directory
        session_id:   The session to load

    Returns:
        List of dicts, one per JSONL line
    """
    filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session file not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[session] Warning: bad JSON on line {line_num} "
                    f"in {session_id}.jsonl: {e}"
                )
    return records


def list_sessions(sessions_dir):
    """
    List all session IDs in the sessions directory, sorted by
    modification time (most recent first).

    Args:
        sessions_dir: Absolute path to the sessions/ directory

    Returns:
        List of session_id strings
    """
    if not os.path.exists(sessions_dir):
        return []

    sessions = []
    for filename in os.listdir(sessions_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(sessions_dir, filename)
            mtime = os.path.getmtime(filepath)
            session_id = filename[:-6]  # strip .jsonl
            sessions.append((session_id, mtime))

    sessions.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in sessions]


def delete_session(sessions_dir, session_id):
    """
    Delete a session file.

    Args:
        sessions_dir: Absolute path to the sessions/ directory
        session_id:   The session to delete

    Raises:
        FileNotFoundError if session does not exist.
    """
    filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session file not found: {filepath}")
    os.remove(filepath)


def format_session_history(records):
    """
    Format session records into a readable markdown string for injection
    into staging/session_history.md.

    This is the bridge between persistent storage (JSONL) and the
    context_injector (which reads .md files).

    Args:
        records: List of dicts from load_session()

    Returns:
        Formatted string ready to write to session_history.md
    """
    if not records:
        return ""

    parts = ["## Discussion History\n"]

    for record in records:
        rtype = record.get("type", "")

        if rtype == "user_input":
            content = record.get("content", "")
            parts.append(f"[User]: {content}\n")

        elif rtype == "agent_response":
            name = record.get("display_name", record.get("agent_id", "Agent"))
            round_num = record.get("round", "?")
            content = record.get("content", "")
            parts.append(f"[{name}] (Round {round_num}): {content}\n")

        elif rtype == "round_complete":
            round_num = record.get("round", "?")
            parts.append(f"--- End of Round {round_num} ---\n")

        elif rtype == "trimmed_marker":
            content = record.get("content", "[earlier rounds omitted]")
            parts.append(f"*{content}*\n")

    return "\n".join(parts)


# --- Internal helpers ---


def _generate_session_id():
    """Generate a session ID: timestamp prefix + short uuid for uniqueness."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{ts}-{short_id}"


def _now():
    """ISO 8601 timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_append(filepath, record):
    """
    Append a JSON record to a file. Uses temp file + rename for new files,
    direct append for existing files (append is atomic on most filesystems
    for reasonably sized writes).
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"

    if not os.path.exists(filepath):
        # New file: write to temp then rename
        dir_name = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            os.write(fd, line.encode("utf-8"))
            os.close(fd)
            os.rename(tmp_path, filepath)
        except Exception:
            os.close(fd)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    else:
        # Existing file: append
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)
