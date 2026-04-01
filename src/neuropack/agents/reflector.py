"""Observer/Reflector — intelligent memory consolidation agent.

Watches incoming memories, extracts atomic facts, then periodically deduplicates,
resolves contradictions (keeps newest), and merges related facts into summaries.

Key improvement over Mastra: runs INCREMENTALLY — only processes NEW facts
since last reflection, not the entire store.

Usage:
    from neuropack.agents.reflector import MemoryReflector
    reflector = MemoryReflector(store, llm_provider=ollama)
    reflector.observe(memory_record)
    result = reflector.reflect()
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from neuropack.core.consolidation import find_clusters
from neuropack.core.contradictions import _extract_key_terms, _has_negation, _check_opposite_pairs

logger = logging.getLogger(__name__)


@dataclass
class AtomicFact:
    """A single atomic fact extracted from a memory."""
    text: str
    source_id: str
    timestamp: float
    superseded: bool = False
    merged_into: str = ""


@dataclass
class ReflectionResult:
    """Stats from a reflection pass."""
    deduped: int = 0
    merged: int = 0
    superseded: int = 0
    new_summaries: int = 0
    facts_processed: int = 0


_EXTRACT_PROMPT = (
    "Extract atomic facts from this text. One fact per line, no opinions.\n\n"
    "Text:\n{content}\n\nFacts (one per line):"
)
_MERGE_PROMPT = (
    "Combine these related facts into one concise summary. Remove redundancy.\n\n"
    "Facts:\n{facts}\n\nConsolidated summary (1-2 sentences):"
)


def _llm_call(llm: Any, prompt: str, max_tokens: int = 500) -> str:
    """Call an LLM provider with generate() or chat() interface."""
    if hasattr(llm, "generate"):
        return llm.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    if hasattr(llm, "chat"):
        return llm.chat([{"role": "user", "content": prompt}],
                        max_tokens=max_tokens, temperature=0.0)
    raise TypeError("LLM provider must have generate() or chat()")


class MemoryReflector:
    """Observer/Reflector agent for incremental memory consolidation."""

    def __init__(self, store: Any, llm_provider: Any = None) -> None:
        self._store = store
        self._llm = llm_provider
        self._pending: list[AtomicFact] = []
        self._all_facts: list[AtomicFact] = []
        self._summaries: list[str] = []
        self._last_reflect_idx: int = 0

    def observe(self, record: Any) -> list[AtomicFact]:
        """Extract atomic facts from a memory record and queue for reflection."""
        content = record if isinstance(record, str) else getattr(record, "content", str(record))
        source_id = getattr(record, "id", "")
        facts = self._extract_atomic_facts(content, source_id)
        self._pending.extend(facts)
        self._all_facts.extend(facts)
        return facts

    def reflect(self) -> ReflectionResult:
        """Full reflection over all pending facts: dedup, contradictions, merge."""
        if not self._pending:
            return ReflectionResult()
        facts = list(self._pending)
        self._pending.clear()
        self._last_reflect_idx = len(self._all_facts)  # sync incremental cursor
        return self._run_reflection(facts)

    def reflect_incremental(self) -> ReflectionResult:
        """Only process facts added since the last reflection (efficient)."""
        new_facts = self._all_facts[self._last_reflect_idx:]
        self._last_reflect_idx = len(self._all_facts)
        self._pending.clear()
        if not new_facts:
            return ReflectionResult()
        return self._run_reflection(new_facts)

    def _run_reflection(self, facts: list[AtomicFact]) -> ReflectionResult:
        result = ReflectionResult(facts_processed=len(facts))
        # Step 1: Deduplicate
        for group in self._find_duplicates(facts):
            if len(group) > 1:
                group.sort(key=lambda f: f.timestamp, reverse=True)
                for dup in group[1:]:
                    dup.superseded = True
                    result.deduped += 1
        # Step 2: Resolve contradictions
        result.superseded += self._resolve_contradictions(
            [f for f in facts if not f.superseded])
        # Step 3: Merge related
        active = [f for f in facts if not f.superseded]
        result.merged, result.new_summaries = self._merge_related(active)
        return result

    # --- Duplicate detection ---

    def _find_duplicates(self, facts: list[AtomicFact]) -> list[list[AtomicFact]]:
        """Group facts by similarity (embeddings if available, else keywords)."""
        if not facts:
            return []
        try:
            return self._find_duplicates_embedding(facts)
        except Exception:
            return self._find_duplicates_keyword(facts)

    def _find_duplicates_embedding(self, facts: list[AtomicFact]) -> list[list[AtomicFact]]:
        import numpy as np
        embedder = getattr(self._store, "_embedder", None)
        if embedder is None:
            raise RuntimeError("No embedder")
        embeddings = np.array([embedder.embed(f.text) for f in facts], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings /= norms
        _Dummy = type("_R", (), {"namespace": "default"})
        clusters = find_clusters([_Dummy() for _ in facts], embeddings,
                                 threshold=0.88, min_cluster_size=2)
        return [[facts[i] for i in c] for c in clusters]

    def _find_duplicates_keyword(self, facts: list[AtomicFact]) -> list[list[AtomicFact]]:
        """Group facts by Jaccard keyword overlap >= 0.6."""
        n = len(facts)
        terms = [_extract_key_terms(f.text) for f in facts]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if not terms[i] or not terms[j]:
                    continue
                jaccard = len(terms[i] & terms[j]) / len(terms[i] | terms[j])
                if jaccard >= 0.6:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        return [[facts[i] for i in g] for g in clusters.values() if len(g) >= 2]

    # --- Contradiction resolution ---

    def _resolve_contradictions(self, facts: list[AtomicFact]) -> int:
        """Find contradicting facts, keep newest, mark older as superseded."""
        count = 0
        n = len(facts)
        for i in range(n):
            if facts[i].superseded:
                continue
            ti, ni = _extract_key_terms(facts[i].text), _has_negation(facts[i].text)
            for j in range(i + 1, n):
                if facts[j].superseded:
                    continue
                tj = _extract_key_terms(facts[j].text)
                overlap = ti & tj
                if len(overlap) < 2:
                    continue
                nj = _has_negation(facts[j].text)
                if (ni != nj and len(overlap) >= 3) or \
                   _check_opposite_pairs(facts[i].text, facts[j].text):
                    (facts[j] if facts[i].timestamp >= facts[j].timestamp
                     else facts[i]).superseded = True
                    count += 1
        return count

    # --- Merge related facts ---

    def _merge_related(self, facts: list[AtomicFact]) -> tuple[int, int]:
        """Combine related facts into summaries. Returns (merged, summaries)."""
        if len(facts) < 3:
            return 0, 0
        groups = [g for g in self._find_duplicates_keyword(facts) if len(g) >= 3]
        if not groups:
            return 0, 0
        merged, created = 0, 0
        for group in groups:
            texts = [f.text for f in group]
            if self._llm:
                try:
                    summary = _llm_call(self._llm,
                                        _MERGE_PROMPT.format(facts="\n".join(f"- {t}" for t in texts)),
                                        max_tokens=200)
                except Exception:
                    summary = "; ".join(dict.fromkeys(t.strip() for t in texts))
            else:
                summary = "; ".join(dict.fromkeys(t.strip() for t in texts))
            sid = f"summary_{len(self._summaries)}"
            self._summaries.append(summary)
            for f in group:
                f.merged_into = sid
            merged += len(group)
            created += 1
        return merged, created

    # --- Atomic fact extraction ---

    def _extract_atomic_facts(self, content: str, source_id: str) -> list[AtomicFact]:
        """Split content into individual atomic facts (LLM or heuristic)."""
        if self._llm:
            try:
                response = _llm_call(self._llm,
                                     _EXTRACT_PROMPT.format(content=content[:2000]))
                now = time.time()
                facts = [AtomicFact(text=ln.lstrip("- ").strip(), source_id=source_id,
                                    timestamp=now)
                         for ln in response.strip().splitlines()
                         if ln.strip() and len(ln.strip()) > 5]
                if facts:
                    return facts
            except Exception:
                pass
        # Heuristic fallback: split on sentence boundaries
        now = time.time()
        return [AtomicFact(text=s.strip(), source_id=source_id, timestamp=now)
                for s in re.split(r'(?<=[.!?])\s+', content.strip())
                if len(s.strip()) > 10]
