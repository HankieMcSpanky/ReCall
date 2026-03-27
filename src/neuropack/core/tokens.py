"""Token estimation utilities."""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic."""
    return max(1, len(text) // 4)


def estimate_tokens_for_list(items: list[str]) -> int:
    """Estimate tokens for a list of strings (e.g., L2 facts)."""
    return estimate_tokens(" ".join(items)) if items else 1
