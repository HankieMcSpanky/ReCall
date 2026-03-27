"""Memory reflection and synthesis: generate insights across recalled memories."""
from __future__ import annotations

import json
import re


class MemoryReflector:
    """Uses an LLM to synthesize insights from multiple recalled memories."""

    def __init__(self, llm_provider):
        self._llm = llm_provider

    def synthesize(self, query: str, memories: list[dict]) -> dict:
        """Generate a unified insight from a set of recalled memories.

        Args:
            query: The original recall query.
            memories: List of dicts with keys: id, content, l3_abstract, tags.

        Returns:
            Dict with: insight, patterns, confidence, source_ids.
        """
        if not memories:
            return {
                "insight": "No memories available to synthesize.",
                "patterns": [],
                "confidence": 0.0,
                "source_ids": [],
            }

        memory_block = "\n\n".join(
            f"[Memory {i+1}] (id={m['id']}, tags={m.get('tags', [])})\n"
            f"Summary: {m.get('l3_abstract', '')}\n"
            f"Content: {m.get('content', '')[:500]}"
            for i, m in enumerate(memories[:10])
        )

        prompt = (
            f"You are analyzing memories relevant to: {query}\n\n"
            f"Memories:\n{memory_block}\n\n"
            f"Provide a JSON response with:\n"
            f'- "insight": A unified synthesis (2-4 sentences) connecting these memories\n'
            f'- "patterns": Array of recurring patterns or themes found\n'
            f'- "confidence": Float 0-1 indicating how well the memories answer the query\n'
            f"\nRespond with ONLY valid JSON."
        )

        try:
            response = self._llm.complete(prompt)
            parsed = _parse_json(response)
            parsed["source_ids"] = [m["id"] for m in memories[:10]]
            return parsed
        except Exception:
            return {
                "insight": "Synthesis unavailable (LLM error).",
                "patterns": [],
                "confidence": 0.0,
                "source_ids": [m["id"] for m in memories[:10]],
            }

    def reflect(self, memory: dict, related: list[dict]) -> dict:
        """Reflect on a single memory in context of related memories.

        Returns: dict with reflection, contradictions, evolution.
        """
        related_block = "\n".join(
            f"- {m.get('l3_abstract', m.get('content', '')[:200])}"
            for m in related[:5]
        )

        prompt = (
            f"Reflect on this memory:\n"
            f"Content: {memory.get('content', '')[:500]}\n"
            f"Summary: {memory.get('l3_abstract', '')}\n\n"
            f"Related memories:\n{related_block}\n\n"
            f"Provide a JSON response with:\n"
            f'- "reflection": Your analysis of this memory\'s significance\n'
            f'- "contradictions": Array of any contradictions with related memories\n'
            f'- "evolution": How this topic has evolved based on the memories\n'
            f"\nRespond with ONLY valid JSON."
        )

        try:
            response = self._llm.complete(prompt)
            return _parse_json(response)
        except Exception:
            return {
                "reflection": "Reflection unavailable (LLM error).",
                "contradictions": [],
                "evolution": "",
            }


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"error": "Could not parse LLM response"}
