"""Centralized input validation for NeuroPack."""
from __future__ import annotations

import re

from neuropack.exceptions import ValidationError

# Patterns
_ALPHANUM_HYPHENS = re.compile(r"^[a-zA-Z0-9_-]+$")
_NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

MAX_TAGS = 20
MAX_TAG_LENGTH = 50
MAX_NAMESPACE_LENGTH = 64


def validate_tags(tags: list[str]) -> list[str]:
    """Validate and normalize tags."""
    if len(tags) > MAX_TAGS:
        raise ValidationError("tags", f"Too many tags ({len(tags)}). Maximum is {MAX_TAGS}.")
    for tag in tags:
        if len(tag) > MAX_TAG_LENGTH:
            raise ValidationError(
                "tags", f"Tag '{tag[:20]}...' exceeds {MAX_TAG_LENGTH} chars."
            )
        if not _ALPHANUM_HYPHENS.match(tag):
            raise ValidationError(
                "tags", f"Tag '{tag}' contains invalid characters. Use alphanumeric, hyphens, underscores."
            )
    return tags


def validate_namespace(namespace: str) -> str:
    """Validate a namespace identifier."""
    if len(namespace) > MAX_NAMESPACE_LENGTH:
        raise ValidationError(
            "namespace", f"Namespace exceeds {MAX_NAMESPACE_LENGTH} chars."
        )
    if not _NAMESPACE_PATTERN.match(namespace):
        raise ValidationError(
            "namespace", f"Namespace '{namespace}' contains invalid characters. "
            "Use alphanumeric, hyphens, underscores, dots."
        )
    return namespace


def validate_priority(priority: float) -> float:
    """Validate priority is between 0 and 1."""
    if not (0.0 <= priority <= 1.0):
        raise ValidationError(
            "priority", f"Priority must be between 0.0 and 1.0, got {priority}."
        )
    return priority


_FTS_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once that this these those i me my myself we our ours he him "
    "his she her it its they them their what which who whom how where when why "
    "all each every both few more most other some such no nor not only own same "
    "so than too very and but or if while about up down just also there here "
    "because although however still yet".split()
)


def sanitize_fts_query(query: str) -> str:
    """Sanitize an FTS5 query string to prevent syntax errors.

    Strips stop words and joins remaining content terms with OR so that
    documents matching *any* content word are returned.  BM25 ranking still
    prioritises documents with more matches.
    """
    # Strip FTS5 special operators
    sanitized = re.sub(r'[*^{}():"]', " ", query)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if not sanitized:
        return '""'
    # Keep only content words (drop stop words)
    tokens = [t for t in sanitized.split() if t.lower() not in _FTS_STOP_WORDS]
    if not tokens:
        # All words were stop words — fall back to original
        tokens = sanitized.split()
    if len(tokens) <= 1:
        return tokens[0] if tokens else '""'
    return " OR ".join(tokens)
