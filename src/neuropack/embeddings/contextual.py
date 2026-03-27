"""Contextual embedding wrapper — prepends heuristic context to content before embedding."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from neuropack.compression.extractive import STOPWORDS
from neuropack.embeddings.base import Embedder


# Additional stopwords for context generation (beyond the base set)
_CONTEXT_STOPWORDS = STOPWORDS | frozenset({
    "also", "just", "like", "would", "could", "should", "might", "will",
    "shall", "may", "much", "many", "make", "made", "get", "got", "going",
    "goes", "went", "come", "came", "take", "took", "give", "gave", "say",
    "said", "tell", "told", "ask", "asked", "know", "knew", "think",
    "thought", "want", "wanted", "need", "needed", "use", "used", "try",
    "tried", "thing", "things", "way", "ways", "time", "times", "right",
    "good", "well", "new", "first", "last", "long", "great", "little",
    "own", "old", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "sure", "really", "actually", "yes",
    "see", "seen", "look", "looked", "let", "help", "helped", "keep",
    "kept", "still", "something", "nothing", "everything", "anything",
    "someone", "everyone", "anyone", "here", "there", "now", "then",
    "today", "yesterday", "tomorrow", "user", "assistant",
})


def _extract_named_entities(text: str) -> list[str]:
    """Extract likely named entities via capitalization patterns.

    Finds sequences of capitalized words that are not at sentence starts.
    """
    entities: list[str] = []

    # Match capitalized multi-word phrases (e.g. "Mario Rossi", "New York")
    for match in re.finditer(r"(?<![.!?\n]\s)(?<!\A)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        entities.append(match.group(0))

    # Match single capitalized words that are not sentence-initial
    # Split into sentences, then pick capitalized words that aren't first
    sentences = re.split(r"[.!?\n]+", text)
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) < 2:
            continue
        for word in words[1:]:
            cleaned = re.sub(r"[^A-Za-z']", "", word)
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                low = cleaned.lower()
                if low not in _CONTEXT_STOPWORDS and low not in {"the", "this", "that", "it", "is"}:
                    entities.append(cleaned)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def _extract_key_nouns(text: str, top_n: int = 5) -> list[str]:
    """Extract the most frequent non-stopword tokens as key topics."""
    words = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text.lower())
    filtered = [w for w in words if w not in _CONTEXT_STOPWORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _detect_conversation_topic(text: str) -> str | None:
    """Detect if text contains conversation markers and extract topic hints."""
    # Look for role markers like [user], [assistant], or similar
    role_pattern = re.compile(r"\[(user|assistant|human|ai|system)\]", re.IGNORECASE)
    if not role_pattern.search(text):
        return None

    # Extract what comes after [user] markers as topic hints
    user_parts: list[str] = []
    for match in re.finditer(r"\[(?:user|human)\]\s*:?\s*(.+?)(?=\[|$)", text, re.IGNORECASE | re.DOTALL):
        snippet = match.group(1).strip()[:200]
        if snippet:
            user_parts.append(snippet)

    if not user_parts:
        return None

    # Extract key nouns from user parts
    combined = " ".join(user_parts)
    nouns = _extract_key_nouns(combined, top_n=3)
    if nouns:
        return ", ".join(nouns)
    return None


def generate_context(content: str, session_context: str = "") -> str:
    """Generate a short context prefix that situates the content.

    Uses heuristic extraction only (no LLM calls).

    Args:
        content: The text content to generate context for.
        session_context: Optional broader session context string.

    Returns:
        A 1-2 sentence context prefix string.
    """
    if session_context:
        # Extract key topics from the session context
        session_nouns = _extract_key_nouns(session_context, top_n=3)
        content_nouns = _extract_key_nouns(content, top_n=3)
        entities = _extract_named_entities(content)

        topic_part = ", ".join(session_nouns) if session_nouns else "various topics"
        subjects = list(dict.fromkeys(content_nouns + [e for e in entities[:2]]))
        if subjects:
            subject_part = ", ".join(subjects[:4])
            return f"This is part of a conversation about {topic_part}. The user discussed {subject_part}."
        return f"This is part of a conversation about {topic_part}."

    # No session context — extract from the content itself
    entities = _extract_named_entities(content)
    nouns = _extract_key_nouns(content, top_n=5)
    conversation_topic = _detect_conversation_topic(content)

    parts: list[str] = []

    # Build the topic portion
    if conversation_topic:
        parts.append(f"Discussion about {conversation_topic}")
    elif nouns:
        parts.append(f"Discussion about {', '.join(nouns[:3])}")

    # Add entities if found
    if entities:
        entity_str = ", ".join(entities[:3])
        if parts:
            parts[0] += f" involving {entity_str}"
        else:
            parts.append(f"Mentions {entity_str}")

    if not parts:
        return ""

    return parts[0] + "."


class ContextualEmbeddingWrapper(Embedder):
    """Wraps any existing embedder to prepend heuristic context before embedding.

    This improves retrieval by adding situating information (key entities,
    topics) to the embedded text, making vectors more distinctive.
    """

    def __init__(
        self,
        base_embedder: Embedder,
        context_generator: object | None = None,
    ):
        """Initialize the contextual wrapper.

        Args:
            base_embedder: The underlying embedder to delegate to.
            context_generator: Optional custom context generator callable.
                If None, uses the default ``generate_context`` function.
        """
        self._base = base_embedder
        self._context_generator = context_generator or generate_context

    @property
    def dim(self) -> int:
        return self._base.dim

    def generate_context(self, content: str, session_context: str = "") -> str:
        """Generate a short context snippet that situates the content.

        Args:
            content: The text to generate context for.
            session_context: Optional session-level context string.

        Returns:
            A context prefix string (1-2 sentences).
        """
        return self._context_generator(content, session_context)

    def embed_with_context(self, content: str, session_context: str = "") -> np.ndarray:
        """Generate context, prepend it to content, then embed the combined text.

        Args:
            content: The text to embed.
            session_context: Optional session context for richer prefixes.

        Returns:
            L2-normalized embedding vector.
        """
        prefix = self.generate_context(content, session_context)
        if prefix:
            combined = f"{prefix} {content}"
        else:
            combined = content
        return self._base.embed(combined)

    def embed(self, text: str) -> np.ndarray:
        """Delegate to base embedder for backwards compatibility."""
        return self._base.embed(text)

    def update_idf(self, text: str) -> None:
        """Delegate IDF updates to the base embedder."""
        self._base.update_idf(text)

    def save_state(self) -> str:
        """Delegate state persistence to the base embedder."""
        return self._base.save_state()

    def load_state(self, state_json: str) -> None:
        """Delegate state loading to the base embedder."""
        self._base.load_state(state_json)
