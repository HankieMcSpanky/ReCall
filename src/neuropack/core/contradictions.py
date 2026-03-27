"""Contradiction detection between memories.

Surfaces conflicts when a new memory contradicts existing ones.
Uses embedding similarity to find candidates, then keyword heuristics
to detect logical conflicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from neuropack.types import MemoryRecord, RecallResult


# Patterns that indicate assertions
_ASSERTION_PATTERNS = [
    re.compile(r"\b(?:is|are|was|were|has|have|does|do)\b", re.IGNORECASE),
    re.compile(r"\b(?:always|never|must|should|cannot|can't)\b", re.IGNORECASE),
]

# Negation patterns
_NEGATION_WORDS = {
    "not", "no", "never", "don't", "doesn't", "didn't", "won't",
    "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "shouldn't", "wouldn't",
    "disable", "disabled", "remove", "removed", "stop", "stopped",
}

# Opposite pairs
_OPPOSITES = [
    ("enable", "disable"),
    ("true", "false"),
    ("yes", "no"),
    ("always", "never"),
    ("start", "stop"),
    ("add", "remove"),
    ("allow", "deny"),
    ("accept", "reject"),
    ("increase", "decrease"),
    ("use", "avoid"),
    ("prefer", "avoid"),
    ("best", "worst"),
]


@dataclass
class Contradiction:
    """A detected contradiction between two memories."""
    existing_id: str
    existing_content: str
    reason: str
    confidence: float  # 0.0 to 1.0


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful terms from text (lowercased, no stopwords)."""
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "but", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "it", "its", "this", "that", "with",
        "from", "by", "as", "i", "we", "you", "they", "he", "she",
    }
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return words - stopwords


def _has_negation(text: str) -> bool:
    """Check if text contains negation."""
    words = set(text.lower().split())
    return bool(words & _NEGATION_WORDS)


def _check_opposite_pairs(text1: str, text2: str) -> Optional[str]:
    """Check if texts contain opposite word pairs."""
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    for a, b in _OPPOSITES:
        if (a in w1 and b in w2) or (b in w1 and a in w2):
            return f"Opposite terms: '{a}' vs '{b}'"
    return None


def detect_contradictions(
    new_content: str,
    candidates: list[RecallResult],
    similarity_threshold: float = 0.6,
) -> list[Contradiction]:
    """Detect contradictions between new content and existing similar memories.

    Args:
        new_content: The new memory content being stored.
        candidates: Similar memories found via recall (sorted by similarity).
        similarity_threshold: Minimum similarity to consider as contradiction candidate.

    Returns:
        List of detected contradictions.
    """
    contradictions = []
    new_terms = _extract_key_terms(new_content)
    new_neg = _has_negation(new_content)

    for result in candidates:
        if result.score < similarity_threshold:
            continue

        existing = result.record
        existing_terms = _extract_key_terms(existing.content)

        # High overlap means they talk about the same topic
        overlap = new_terms & existing_terms
        if len(overlap) < 2:
            continue

        confidence = 0.0
        reasons = []

        # Check 1: One has negation, the other doesn't
        existing_neg = _has_negation(existing.content)
        if new_neg != existing_neg and len(overlap) >= 3:
            confidence += 0.4
            reasons.append("Negation mismatch on overlapping topic")

        # Check 2: Opposite word pairs
        opposite = _check_opposite_pairs(new_content, existing.content)
        if opposite:
            confidence += 0.5
            reasons.append(opposite)

        # Check 3: Same subject, different predicate (very high similarity but different)
        if result.score > 0.85 and new_content.strip() != existing.content.strip():
            # Very similar but not identical -- likely an update/contradiction
            confidence += 0.3
            reasons.append(f"Very similar memory (score={result.score:.2f}) with different content")

        if confidence >= 0.3 and reasons:
            contradictions.append(Contradiction(
                existing_id=existing.id,
                existing_content=existing.content[:200],
                reason="; ".join(reasons),
                confidence=min(confidence, 1.0),
            ))

    return contradictions
