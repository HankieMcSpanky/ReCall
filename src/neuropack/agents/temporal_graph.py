"""Temporal Knowledge Graph — facts as time-scoped graph edges.

Enhances the existing KnowledgeGraph with:
1. Automatic supersession via temporal edges
2. Cross-entity reasoning ("Alice works with Bob at Google" → "Alice is at Google")
3. Entity timeline queries
4. Fact validity windows (valid_from → valid_until)

Usage:
    from neuropack.agents.temporal_graph import TemporalKnowledgeGraph

    tkg = TemporalKnowledgeGraph(store)
    tkg.add_fact("User", "camera", "Canon EOS R5", date="2023/05/20")
    tkg.add_fact("User", "camera", "Sony A7IV", date="2023/07/15")
    # Automatically supersedes Canon → Sony

    current = tkg.get_current("User", "camera")
    # → "Sony A7IV"

    history = tkg.get_history("User", "camera")
    # → [("Canon EOS R5", "2023/05/20", "2023/07/15"), ("Sony A7IV", "2023/07/15", None)]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TemporalFact:
    """A fact with time-scoped validity."""
    entity: str
    attribute: str
    value: str
    valid_from: str  # date when this became true
    valid_until: str | None = None  # None = still current
    source_memory_id: str = ""
    confidence: float = 1.0
    superseded_by: str | None = None  # ID of the superseding fact


class TemporalKnowledgeGraph:
    """Knowledge graph with temporal fact tracking."""

    def __init__(self, store: Any = None) -> None:
        self._store = store
        # Main storage: (entity, attribute) → list of TemporalFact (ordered by valid_from)
        self._facts: dict[tuple[str, str], list[TemporalFact]] = {}
        # Entity index: entity → set of attributes
        self._entity_attrs: dict[str, set[str]] = {}
        # Relationship index: entity → set of (relation, target_entity)
        self._relationships: dict[str, set[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Core: Add and query facts
    # ------------------------------------------------------------------

    def add_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        date: str = "",
        memory_id: str = "",
        confidence: float = 1.0,
    ) -> TemporalFact:
        """Add a fact. Auto-supersedes conflicting older facts."""
        key = (entity.lower(), attribute.lower())

        fact = TemporalFact(
            entity=entity,
            attribute=attribute,
            value=value,
            valid_from=date,
            source_memory_id=memory_id,
            confidence=confidence,
        )

        if key not in self._facts:
            self._facts[key] = []

        # Check for supersession
        for existing in self._facts[key]:
            if (existing.valid_until is None
                    and existing.value.lower() != value.lower()
                    and existing.valid_from <= date):
                # Supersede the old fact
                existing.valid_until = date
                existing.superseded_by = f"{entity}_{attribute}_{date}"
                logger.debug(
                    "Superseded: %s.%s = '%s' → '%s'",
                    entity, attribute, existing.value, value,
                )

        self._facts[key].append(fact)
        self._facts[key].sort(key=lambda f: f.valid_from)

        # Update entity index
        if entity.lower() not in self._entity_attrs:
            self._entity_attrs[entity.lower()] = set()
        self._entity_attrs[entity.lower()].add(attribute.lower())

        return fact

    def add_relationship(
        self,
        entity1: str,
        relation: str,
        entity2: str,
    ) -> None:
        """Add a relationship between entities."""
        e1 = entity1.lower()
        e2 = entity2.lower()
        if e1 not in self._relationships:
            self._relationships[e1] = set()
        self._relationships[e1].add((relation, e2))
        # Bidirectional for some relations
        if relation in ("works_with", "friends_with", "related_to"):
            if e2 not in self._relationships:
                self._relationships[e2] = set()
            self._relationships[e2].add((relation, e1))

    def get_current(self, entity: str, attribute: str) -> str | None:
        """Get the CURRENT value of an attribute. Returns None if not found."""
        key = (entity.lower(), attribute.lower())
        facts = self._facts.get(key, [])
        for fact in reversed(facts):
            if fact.valid_until is None:
                return fact.value
        return None

    def get_history(self, entity: str, attribute: str) -> list[tuple[str, str, str | None]]:
        """Get the full history of an attribute.

        Returns: [(value, valid_from, valid_until), ...]
        """
        key = (entity.lower(), attribute.lower())
        facts = self._facts.get(key, [])
        return [(f.value, f.valid_from, f.valid_until) for f in facts]

    def get_all_current(self, entity: str) -> dict[str, str]:
        """Get all current attributes for an entity."""
        result = {}
        attrs = self._entity_attrs.get(entity.lower(), set())
        for attr in attrs:
            val = self.get_current(entity, attr)
            if val is not None:
                result[attr] = val
        return result

    def get_entity_timeline(self, entity: str) -> list[dict]:
        """Get a chronological timeline of ALL changes for an entity."""
        timeline = []
        attrs = self._entity_attrs.get(entity.lower(), set())
        for attr in attrs:
            key = (entity.lower(), attr)
            for fact in self._facts.get(key, []):
                timeline.append({
                    "date": fact.valid_from,
                    "attribute": fact.attribute,
                    "value": fact.value,
                    "superseded": fact.valid_until is not None,
                    "valid_until": fact.valid_until,
                })
        timeline.sort(key=lambda t: t["date"])
        return timeline

    # ------------------------------------------------------------------
    # Cross-entity reasoning
    # ------------------------------------------------------------------

    def infer(self, entity: str, attribute: str, max_hops: int = 2) -> str | None:
        """Try to infer a fact through relationships.

        If "Alice works_with Bob" and "Bob works_at Google",
        then infer "Alice works_at Google".
        """
        # Direct lookup first
        direct = self.get_current(entity, attribute)
        if direct is not None:
            return direct

        # Try 1-hop inference
        if max_hops <= 0:
            return None

        rels = self._relationships.get(entity.lower(), set())
        for relation, target in rels:
            # Check if the target entity has the attribute
            val = self.get_current(target, attribute)
            if val is not None:
                return val
            # Try deeper (2-hop)
            if max_hops > 1:
                val = self.infer(target, attribute, max_hops - 1)
                if val is not None:
                    return val

        return None

    def find_related_entities(self, entity: str) -> list[tuple[str, str]]:
        """Find all entities related to the given entity.

        Returns: [(relation, target_entity), ...]
        """
        return list(self._relationships.get(entity.lower(), set()))

    # ------------------------------------------------------------------
    # Query support for benchmark
    # ------------------------------------------------------------------

    def answer_knowledge_query(self, query: str) -> str | None:
        """Try to answer a knowledge question directly from the graph.

        Parses simple patterns:
        - "How many X?" → lookup count attribute
        - "What is my X?" → lookup attribute
        - "Where did X move?" → lookup location attribute
        """
        import re
        q = query.lower()

        # "How many X have I Y?" — look for count attributes
        count_match = re.search(r'how many\s+(.+?)(?:\s+(?:have|did|do)\s+I|\?)', q)
        if count_match:
            topic = count_match.group(1).strip().replace(" ", "_")
            # Search all user attributes for matching topic
            for attr in self._entity_attrs.get("user", set()):
                if topic in attr or attr in topic:
                    val = self.get_current("user", attr)
                    if val:
                        return val

        # "What is my X?" — direct attribute lookup
        what_match = re.search(r'what (?:is|are) my\s+(.+?)[\?]', q)
        if what_match:
            attr = what_match.group(1).strip().replace(" ", "_")
            val = self.get_current("user", attr)
            if val:
                return val

        # "Where did X move?" — location lookup
        where_match = re.search(r'where did\s+(\w+)\s+move', q)
        if where_match:
            entity = where_match.group(1)
            val = self.get_current(entity, "location")
            return val

        return None

    def build_context_for_query(self, query: str) -> str:
        """Build a context string from the graph for a given query.

        Extracts entities from the query, finds all their current facts
        and relationships, and formats as text.
        """
        import re
        # Extract potential entity names (capitalized words)
        entities = set()
        entities.add("user")  # always include user
        for word in query.split():
            clean = word.strip("?,!.;:\"'()[]")
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.add(clean.lower())

        lines = []
        for entity in entities:
            current = self.get_all_current(entity)
            if current:
                lines.append(f"## {entity.title()}")
                for attr, val in sorted(current.items()):
                    lines.append(f"- {attr}: {val}")

                # Add relationships
                rels = self.find_related_entities(entity)
                if rels:
                    for rel, target in rels:
                        lines.append(f"- {rel}: {target}")

                # Add changes
                timeline = self.get_entity_timeline(entity)
                changes = [t for t in timeline if t["superseded"]]
                if changes:
                    lines.append("Changes:")
                    for ch in changes[-3:]:  # last 3 changes
                        lines.append(
                            f"  - {ch['attribute']}: was '{ch['value']}' "
                            f"(until {ch['valid_until']})"
                        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return graph statistics."""
        total_facts = sum(len(v) for v in self._facts.values())
        current_facts = sum(
            1 for facts in self._facts.values()
            for f in facts if f.valid_until is None
        )
        return {
            "total_facts": total_facts,
            "current_facts": current_facts,
            "superseded_facts": total_facts - current_facts,
            "entities": len(self._entity_attrs),
            "relationships": sum(len(v) for v in self._relationships.values()),
            "attributes": len(self._facts),
        }
