"""Session summary generation."""
from __future__ import annotations

from neuropack.compression.extractive import ExtractiveCompressor, _split_sentences
from neuropack.types import MemoryRecord


def _categorize_sentence(sentence: str) -> str:
    s = sentence.lower()
    if any(w in s for w in ("found", "discovered", "investigated", "explored", "searched", "analyzed", "looked")):
        return "investigated"
    if any(w in s for w in ("learned", "understood", "realized", "noted", "observed")):
        return "learned"
    if any(w in s for w in ("completed", "finished", "fixed", "created", "implemented", "built", "added", "resolved")):
        return "completed"
    if any(w in s for w in ("todo", "next", "should", "need to", "will", "plan")):
        return "next_steps"
    return "learned"


def generate_session_summary(memories: list[MemoryRecord]) -> dict:
    """Generate a structured summary from a list of session memories.

    Returns dict with keys: investigated, learned, completed, next_steps, summary.
    """
    if not memories:
        return {
            "investigated": [],
            "learned": [],
            "completed": [],
            "next_steps": [],
            "summary": "Empty session",
        }

    all_parts: list[str] = []
    for m in memories:
        all_parts.append(m.l3_abstract)
        all_parts.extend(m.l2_facts)

    result: dict[str, list[str]] = {
        "investigated": [],
        "learned": [],
        "completed": [],
        "next_steps": [],
    }

    for part in all_parts:
        for sentence in _split_sentences(part):
            category = _categorize_sentence(sentence)
            if sentence not in result[category]:
                result[category].append(sentence)

    compressor = ExtractiveCompressor()
    combined = ". ".join(m.l3_abstract for m in memories if m.l3_abstract)
    summary = compressor.compress_l3(combined) if combined else "Session with no content"

    return {**result, "summary": summary}
