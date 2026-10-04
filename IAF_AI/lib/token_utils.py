"""
Shared token estimation utility.
Used by context trimming strategies across all agents.
"""


def estimate_tokens(text):
    """Rough estimate: ~3 chars per token for mixed Chinese/English."""
    if isinstance(text, str):
        return len(text) // 3
    if isinstance(text, list):
        return sum(len(item.get("text", "")) // 3 for item in text)
    return 0
