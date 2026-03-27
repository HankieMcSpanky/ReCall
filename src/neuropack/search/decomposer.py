"""Rule-based query decomposition for multi-hop retrieval."""
from __future__ import annotations

import re


# Conjunctions and temporal markers that signal multi-part queries
_SPLIT_CONJUNCTIONS = re.compile(
    r"\b(and also|and then|as well as|in addition to|along with)\b",
    re.IGNORECASE,
)

_SIMPLE_CONJUNCTIONS = re.compile(
    r"\band\b",
    re.IGNORECASE,
)

_TEMPORAL_MARKERS = re.compile(
    r"\b(when|while|after|before|during|at the same time as|meanwhile)\b",
    re.IGNORECASE,
)

_COMPARATIVE_MARKERS = re.compile(
    r"\b(compare|compared|comparison|versus|vs\.?|differ|difference|between)\b",
    re.IGNORECASE,
)

# Pattern: "What did X say about Y?" or "How does X relate to Y?"
_ENTITY_ABOUT = re.compile(
    r"(?:what|how|why|when|where|did|does|do|has|have|had|is|was|were|are)\b.+?\b"
    r"([\w]+(?:\s+[\w]+)?)\b.+?\b(?:about|regarding|concerning|on|with)\b\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)

# Pattern for temporal + factual: "What happened in March?" or "What did I do last week?"
_TEMPORAL_FACTUAL = re.compile(
    r"\b(?:in|during|on|at|last|this|next)\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|week|month|year|spring|summer|fall|winter|quarter"
    r"|morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)

# Entity patterns (capitalized words)
_CAPITALIZED_ENTITY = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _count_entities(query: str) -> int:
    """Count likely named entities in the query."""
    entities = _CAPITALIZED_ENTITY.findall(query)
    # Filter out common sentence-start words
    skip = {"what", "who", "where", "when", "why", "how", "the", "this", "that",
            "which", "does", "did", "has", "have", "had", "was", "were", "are",
            "tell", "show", "give", "can", "could", "would", "should", "will",
            "do", "is", "it", "my", "me", "we", "our", "they", "their",
            "please", "also", "just"}
    return sum(1 for e in entities if e.lower() not in skip)


class QueryDecomposer:
    """Decomposes complex queries into focused sub-queries using rule-based heuristics.

    No LLM calls are made. Splitting is based on conjunctions, temporal markers,
    comparative language, and entity isolation patterns.
    """

    def __init__(self) -> None:
        pass

    def should_decompose(self, query: str) -> bool:
        """Determine if a query would benefit from decomposition.

        Returns True if the query has:
        - Multiple named entities
        - Temporal + factual components
        - Comparative language
        - Multi-clause structure with conjunctions
        """
        # Comparative queries
        if _COMPARATIVE_MARKERS.search(query):
            return True

        # Temporal markers combined with other content
        if _TEMPORAL_MARKERS.search(query):
            # Only decompose if there's substantial content beyond the marker
            cleaned = _TEMPORAL_MARKERS.sub("", query).strip()
            words = [w for w in cleaned.split() if len(w) > 2]
            if len(words) >= 3:
                return True

        # Multi-clause conjunctions
        if _SPLIT_CONJUNCTIONS.search(query):
            return True

        # Simple "and" with substantial clauses on both sides
        if _SIMPLE_CONJUNCTIONS.search(query):
            parts = _SIMPLE_CONJUNCTIONS.split(query)
            substantial = [p.strip() for p in parts if len(p.strip().split()) >= 3]
            if len(substantial) >= 2:
                return True

        # Multiple entities
        if _count_entities(query) >= 2:
            return True

        # Entity-about pattern
        if _ENTITY_ABOUT.search(query):
            return True

        return False

    def decompose(self, query: str) -> list[str]:
        """Break a query into 1-3 focused sub-queries.

        The original query is always returned as the first item.
        Returns a list of 1 item if no decomposition is needed.

        Strategies applied in order:
        1. Split on conjunctions
        2. Separate temporal from factual
        3. Entity isolation
        """
        if not self.should_decompose(query):
            return [query]

        sub_queries: list[str] = []

        # Strategy 1: Split on compound conjunctions first
        conj_match = _SPLIT_CONJUNCTIONS.search(query)
        if conj_match:
            parts = _SPLIT_CONJUNCTIONS.split(query)
            # Filter out the conjunction matches and keep substantial parts
            cleaned = [p.strip() for p in parts if p.strip() and not _SPLIT_CONJUNCTIONS.match(p)]
            for part in cleaned[:2]:
                if len(part.split()) >= 2:
                    sub_queries.append(part.rstrip("?").strip())
            if sub_queries:
                return self._finalize(query, sub_queries)

        # Strategy 1b: Split on simple "and" if both sides are substantial
        and_match = _SIMPLE_CONJUNCTIONS.search(query)
        if and_match:
            parts = _SIMPLE_CONJUNCTIONS.split(query)
            substantial = [p.strip() for p in parts if len(p.strip().split()) >= 3]
            if len(substantial) >= 2:
                for part in substantial[:2]:
                    sub_queries.append(part.rstrip("?").strip())
                return self._finalize(query, sub_queries)

        # Strategy 2: Separate temporal from factual
        temporal_match = _TEMPORAL_MARKERS.search(query)
        temporal_period = _TEMPORAL_FACTUAL.search(query)

        if temporal_match:
            marker = temporal_match.group(0)
            parts = query.split(marker, 1)
            if len(parts) == 2:
                before = parts[0].strip().rstrip("?").strip()
                after = parts[1].strip().rstrip("?").strip()
                if len(before.split()) >= 2:
                    sub_queries.append(before)
                if len(after.split()) >= 2:
                    sub_queries.append(after)
                if sub_queries:
                    return self._finalize(query, sub_queries)

        if temporal_period:
            period = temporal_period.group(0)
            # Create a time-focused sub-query and a topic-focused sub-query
            non_temporal = _TEMPORAL_FACTUAL.sub("", query).strip()
            non_temporal = re.sub(r"\s+", " ", non_temporal).strip("? ")
            if len(non_temporal.split()) >= 2:
                sub_queries.append(non_temporal)
                sub_queries.append(f"events {period}")
                return self._finalize(query, sub_queries)

        # Strategy 3: Entity isolation
        entity_match = _ENTITY_ABOUT.search(query)
        if entity_match:
            entity = entity_match.group(1).strip()
            topic = entity_match.group(2).strip().rstrip("?").strip()
            if entity and topic:
                sub_queries.append(f"{entity}")
                sub_queries.append(f"{topic}")
                return self._finalize(query, sub_queries)

        # Comparative: split on vs/versus/compared/between...and
        if _COMPARATIVE_MARKERS.search(query):
            # Try "between X and Y"
            between_match = re.search(
                r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", query, re.IGNORECASE
            )
            if between_match:
                sub_queries.append(between_match.group(1).strip())
                sub_queries.append(between_match.group(2).strip())
                return self._finalize(query, sub_queries)

            # Try "X vs Y" or "X versus Y"
            vs_match = re.search(
                r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)", query, re.IGNORECASE
            )
            if vs_match:
                sub_queries.append(vs_match.group(1).strip())
                sub_queries.append(vs_match.group(2).strip())
                return self._finalize(query, sub_queries)

        # Multiple entities fallback
        entities = _CAPITALIZED_ENTITY.findall(query)
        skip = {"what", "who", "where", "when", "why", "how", "the", "this", "that",
                "which", "does", "did", "has", "have", "had", "was", "were", "are",
                "tell", "show", "give", "can", "could", "would", "should", "will",
                "do", "is", "it", "my", "me", "we", "our", "they", "their",
                "please", "also", "just"}
        real_entities = [e for e in entities if e.lower() not in skip]
        if len(real_entities) >= 2:
            for entity in real_entities[:2]:
                sub_queries.append(entity)
            return self._finalize(query, sub_queries)

        # No decomposition possible
        return [query]

    def _finalize(self, original: str, sub_queries: list[str]) -> list[str]:
        """Ensure original query is first, deduplicate, and cap at 3."""
        result = [original]
        seen = {original.lower().strip()}
        for sq in sub_queries:
            sq = sq.strip()
            if sq and sq.lower() not in seen:
                seen.add(sq.lower())
                result.append(sq)
        return result[:3]
