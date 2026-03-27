from __future__ import annotations

import re

from neuropack.compression.extractive import STOPWORDS


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords, min length 2."""
    words = re.findall(r"\b[a-z0-9]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def bigrams(tokens: list[str]) -> list[str]:
    """Generate bigram tokens for richer representation."""
    return [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
