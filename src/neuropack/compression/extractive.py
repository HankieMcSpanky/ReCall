from __future__ import annotations

import re

STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "couldn",
    "couldnt", "did", "didn", "didnt", "do", "does", "doesn", "doesnt", "doing",
    "don", "dont", "down", "during", "each", "few", "for", "from", "further",
    "get", "got", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven",
    "havent", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "however", "if", "in", "into", "is", "isn", "isnt", "it", "its",
    "itself", "just", "let", "like", "ll", "may", "me", "might", "mightn", "more",
    "most", "mustn", "mustnt", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "same", "shall", "shan", "shant", "she", "should",
    "shouldn", "shouldnt", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "us", "ve", "very",
    "was", "wasn", "wasnt", "we", "were", "weren", "werent", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "wont", "would", "wouldn", "wouldnt", "you", "your", "yours", "yourself",
    "yourselves",
})


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str) -> list[str]:
    """Extract lowercase words, removing stopwords and short tokens."""
    words = re.findall(r"\b[a-z0-9]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _score_sentences(sentences: list[str]) -> list[tuple[float, int, str]]:
    """Score sentences by word frequency. Returns (score, original_index, sentence)."""
    all_tokens: list[str] = []
    sentence_tokens: list[list[str]] = []
    for s in sentences:
        tokens = _tokenize(s)
        sentence_tokens.append(tokens)
        all_tokens.extend(tokens)

    freq: dict[str, int] = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1

    scored: list[tuple[float, int, str]] = []
    for i, (s, tokens) in enumerate(zip(sentences, sentence_tokens)):
        if not tokens:
            scored.append((0.0, i, s))
            continue
        score = sum(freq.get(t, 0) for t in tokens) / len(tokens)
        scored.append((score, i, s))

    return scored


class ExtractiveCompressor:
    """Extractive fallback compressor using frequency-based sentence scoring."""

    def compress_l3(self, text: str) -> str:
        """Return the highest-scoring sentence as a one-line abstract."""
        sentences = _split_sentences(text)
        if not sentences:
            return text[:200] if text else ""
        if len(sentences) == 1:
            return sentences[0][:200]

        scored = _score_sentences(sentences)
        best = max(scored, key=lambda x: x[0])
        return best[2][:200]

    def compress_l2(self, text: str) -> list[str]:
        """Return top-N sentences as structured facts in original order."""
        sentences = _split_sentences(text)
        if not sentences:
            return [text[:500]] if text else []
        if len(sentences) <= 2:
            return sentences

        scored = _score_sentences(sentences)
        top_n = min(5, max(1, len(sentences) // 3))
        top_scored = sorted(scored, key=lambda x: x[0], reverse=True)[:top_n]
        # Re-sort by original position to preserve narrative order
        top_scored.sort(key=lambda x: x[1])
        return [s for _, _, s in top_scored]
