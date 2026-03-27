"""Privacy tag processing for NeuroPack."""
from __future__ import annotations

import re
from enum import Enum

_PRIVATE_PATTERN = re.compile(r"<private>(.*?)</private>", re.DOTALL)


class PrivacyMode(str, Enum):
    STRIP = "strip"
    FULL = "full"
    REDACT = "redact"


def process_privacy(text: str, mode: PrivacyMode) -> str:
    """Process text according to privacy mode.

    Returns text suitable for L3/L2 compression and embedding.
    The raw text (for L1) is always stored unmodified.
    """
    if mode == PrivacyMode.FULL:
        return text
    if mode == PrivacyMode.REDACT:
        return _PRIVATE_PATTERN.sub("[REDACTED]", text).strip()
    # Default: strip
    return _PRIVATE_PATTERN.sub("", text).strip()


def strip_private_from_preview(text: str) -> str:
    """Always strip private content from previews."""
    return _PRIVATE_PATTERN.sub("", text).strip()


def has_private_content(text: str) -> bool:
    """Check if text contains private tags."""
    return bool(_PRIVATE_PATTERN.search(text))
