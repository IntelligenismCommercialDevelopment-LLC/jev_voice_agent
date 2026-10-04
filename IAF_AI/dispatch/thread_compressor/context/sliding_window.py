"""
Sliding Window — Dispatch-level context trimming strategy

When session history grows too long (many rounds of multi-agent discussion),
this module trims older rounds to keep the total context within the
model's window.

Trimming behavior is controlled by trim_strategy in dispatch_config.json:
    {
      "keep_first_user_input": true,   // preserve original question
      "keep_last_user_input": true,    // preserve latest follow-up
      "drop_order": "oldest_first"     // which rounds to drop first
    }

Token estimation uses a simple character-based heuristic (~4 chars per
token for English, ~2 for Chinese). Good enough for trimming decisions.
"""

# Default trim strategy — used if config doesn't specify one
_DEFAULT_STRATEGY = {
    "keep_first_user_input": True,
    "keep_last_user_input": True,
    "drop_order": "oldest_first",
}


def estimate_tokens(text):
    """
    Estimate token count from text. Uses a blended heuristic:
    count ASCII chars / 4 + non-ASCII chars / 2.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars / 4 + non_ascii / 2)


def trim_records(records, max_tokens=3000, trim_strategy=None):
    """
    Trim session records to fit within a token budget.

    Args:
        records:       List of session record dicts
        max_tokens:    Token budget for session history portion
        trim_strategy: Dict from dispatch_config.json, or None for defaults

    Returns:
        Trimmed list of records. If no trimming needed, returns original list.
    """
    if not records:
        return records

    strategy = dict(_DEFAULT_STRATEGY)
    if trim_strategy:
        strategy.update(trim_strategy)

    # Estimate current total
    total_text = " ".join(r.get("content", "") for r in records)
    current_tokens = estimate_tokens(total_text)

    if current_tokens <= max_tokens:
        return records

    # --- Categorize records ---
    keep_first = strategy["keep_first_user_input"]
    keep_last = strategy["keep_last_user_input"]
    drop_order = strategy["drop_order"]

    protected_head = []   # session_start + first user_input (if protected)
    rounds = {}           # round_num -> [records]
    protected_tail = []   # last user_input (if protected)
    current_round = 0

    # Find first and last user_input indices
    first_user_idx = None
    last_user_idx = None
    for i, record in enumerate(records):
        if record.get("type") == "user_input":
            if first_user_idx is None:
                first_user_idx = i
            last_user_idx = i

    # Categorize each record
    for i, record in enumerate(records):
        rtype = record.get("type", "")

        # session_start always protected
        if rtype == "session_start":
            protected_head.append(record)
            continue

        # First user_input
        if i == first_user_idx and keep_first:
            protected_head.append(record)
            continue

        # Last user_input (only if different from first)
        if i == last_user_idx and keep_last and last_user_idx != first_user_idx:
            protected_tail.append(record)
            continue

        # Agent responses grouped by round
        if rtype == "agent_response":
            rnd = record.get("round", current_round)
            current_round = rnd
            if rnd not in rounds:
                rounds[rnd] = []
            rounds[rnd].append(record)
            continue

        if rtype == "round_complete":
            rnd = record.get("round", current_round)
            if rnd not in rounds:
                rounds[rnd] = []
            rounds[rnd].append(record)
            continue

        # Unprotected user_input (middle ones) — attach to nearest round
        if rtype == "user_input":
            next_round = current_round + 1
            if next_round not in rounds:
                rounds[next_round] = []
            rounds[next_round].insert(0, record)
            continue

        # Anything else — try to keep
        if current_round in rounds:
            rounds[current_round].append(record)
        else:
            protected_head.append(record)

    # --- Drop rounds until under budget ---
    sorted_rounds = sorted(rounds.keys())

    if drop_order == "oldest_first":
        drop_candidates = list(sorted_rounds)
    elif drop_order == "newest_first":
        drop_candidates = list(reversed(sorted_rounds))
    else:
        # Default to oldest_first for unknown strategies
        drop_candidates = list(sorted_rounds)

    kept_rounds = list(sorted_rounds)
    dropped_count = 0

    while kept_rounds:
        kept_records = _assemble(
            protected_head, rounds, kept_rounds, protected_tail, dropped_count
        )
        text = " ".join(r.get("content", "") for r in kept_records)
        if estimate_tokens(text) <= max_tokens:
            break

        # Drop next candidate
        if drop_order == "oldest_first":
            to_drop = kept_rounds.pop(0)
        elif drop_order == "newest_first":
            to_drop = kept_rounds.pop(-1)
        else:
            to_drop = kept_rounds.pop(0)

        dropped_count += 1

    return _assemble(
        protected_head, rounds, kept_rounds, protected_tail, dropped_count
    )


def _assemble(protected_head, rounds, kept_rounds, protected_tail, dropped_count):
    """Assemble final record list from components."""
    result = list(protected_head)

    if dropped_count > 0:
        result.append({
            "type": "trimmed_marker",
            "content": f"[{dropped_count} earlier round(s) omitted for brevity]",
        })

    for rnd in kept_rounds:
        result.extend(rounds[rnd])

    result.extend(protected_tail)
    return result
