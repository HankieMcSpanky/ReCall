"""Bi-Temporal Fact Tracker — tracks fact versions with validity windows.

Every fact has four timestamps:
  valid_from:   when the fact became true in the real world
  valid_until:  when the fact was superseded (None = still current)
  created_at:   when the system ingested it
  expired_at:   when the system learned it was outdated

At query time, only facts where valid_until IS NULL are returned.
When a new fact contradicts an old one, the old one gets valid_until = now.

Usage:
    tracker = FactTracker()
    tracker.add("User", "camera", "Canon EOS R5", session_date="2023/05/20")
    tracker.add("User", "camera", "Sony A7IV", session_date="2023/07/15")
    # Canon automatically superseded

    tracker.get_current("User", "camera")  # → "Sony A7IV"
    tracker.get_changes("User", "camera")  # → [("Canon EOS R5", "Sony A7IV", "2023/07/15")]

    # For benchmark: inject current facts into prompt
    context = tracker.build_knowledge_context(query="How many cameras?")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackedFact:
    entity: str
    attribute: str
    value: str
    session_date: str       # when the conversation happened
    event_expression: str = ""  # raw time expression ("last Saturday")
    valid_from: str = ""    # when fact became true
    valid_until: str | None = None  # None = still current
    source_id: str = ""


class FactTracker:
    """Lightweight bi-temporal fact tracker."""

    def __init__(self) -> None:
        # (entity_lower, attribute_lower) → list of TrackedFact
        self._facts: dict[tuple[str, str], list[TrackedFact]] = {}

    def add(
        self,
        entity: str,
        attribute: str,
        value: str,
        session_date: str = "",
        event_expression: str = "",
        source_id: str = "",
    ) -> TrackedFact:
        """Add a fact. Auto-supersedes older conflicting facts."""
        key = (entity.lower(), attribute.lower())
        fact = TrackedFact(
            entity=entity,
            attribute=attribute,
            value=value,
            session_date=session_date,
            event_expression=event_expression,
            valid_from=session_date,
            source_id=source_id,
        )

        if key not in self._facts:
            self._facts[key] = []

        # Check for supersession
        for existing in self._facts[key]:
            if (existing.valid_until is None
                    and existing.value.lower() != value.lower()
                    and existing.session_date <= session_date):
                existing.valid_until = session_date

        self._facts[key].append(fact)
        return fact

    def get_current(self, entity: str, attribute: str) -> str | None:
        """Get the current value of an attribute."""
        key = (entity.lower(), attribute.lower())
        for fact in reversed(self._facts.get(key, [])):
            if fact.valid_until is None:
                return fact.value
        return None

    def get_changes(self, entity: str, attribute: str) -> list[tuple[str, str, str]]:
        """Get all value changes for an attribute.

        Returns: [(old_value, new_value, change_date), ...]
        """
        key = (entity.lower(), attribute.lower())
        facts = self._facts.get(key, [])
        changes = []
        for f in facts:
            if f.valid_until is not None:
                # Find what superseded it
                for f2 in facts:
                    if f2.valid_from == f.valid_until and f2.valid_until is None:
                        changes.append((f.value, f2.value, f.valid_until))
                        break
        return changes

    def extract_and_track(self, observations: str, session_date: str = "", source_id: str = "") -> int:
        """Extract facts from observation text and track them.

        Parses observation bullet points for entity-attribute-value tuples.
        Returns count of facts tracked.
        """
        count = 0
        for line in observations.split("\n"):
            line = line.strip().lstrip("-•* ")
            if not line or len(line) < 10:
                continue

            # Extract counts: "4 Korean restaurants", "5 model kits"
            count_match = re.search(
                r'(?:now |total |has tried |tried |have |has |owns? )'
                r'(\d+)\s+(.+?)(?:\.|,|$)',
                line, re.IGNORECASE,
            )
            if count_match:
                value = count_match.group(1)
                attr = count_match.group(2).strip().lower().replace(" ", "_")[:40]
                self.add("User", attr, value, session_date=session_date, source_id=source_id)
                count += 1
                continue

            # Extract "changed from X to Y" patterns
            change_match = re.search(
                r'(?:changed|switched|moved|upgraded|went)\s+from\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)',
                line, re.IGNORECASE,
            )
            if change_match:
                old_val = change_match.group(1).strip()
                new_val = change_match.group(2).strip()
                # Try to extract attribute from context
                attr = re.sub(r'\b(from|to|the|a|an)\b', '', line[:30]).strip().lower().replace(" ", "_")[:30]
                self.add("User", attr, new_val, session_date=session_date, source_id=source_id)
                count += 1
                continue

            # Extract "User [verb] [value]" patterns
            user_match = re.search(
                r'[Uu]ser\s+(?:uses?|prefers?|owns?|has|likes?|works? at|lives? in)\s+'
                r'(?:a |an |the )?(.+?)(?:\.|,|;|$)',
                line, re.IGNORECASE,
            )
            if user_match:
                value = user_match.group(1).strip()[:60]
                verb_match = re.search(r'[Uu]ser\s+(\w+)', line)
                attr = verb_match.group(1).lower() if verb_match else "attribute"
                self.add("User", attr, value, session_date=session_date, source_id=source_id)
                count += 1

            # Extract raw time expressions
            time_match = re.search(
                r'(last (?:Saturday|Sunday|Monday|Tuesday|Wednesday|Thursday|Friday)'
                r'|last (?:week|month|weekend)'
                r'|\d+ (?:days?|weeks?|months?) ago'
                r'|yesterday|Valentine|Christmas|two weeks ago'
                r'|a couple of days ago)',
                line, re.IGNORECASE,
            )
            if time_match:
                expr = time_match.group(0)
                # Store the event with its raw expression
                event_text = line[:80]
                self.add("User", "event", event_text,
                         session_date=session_date,
                         event_expression=expr,
                         source_id=source_id)
                count += 1

        return count

    def build_knowledge_context(self, query: str = "") -> str:
        """Build a context string showing current facts and changes.

        Used to inject into LLM prompts for knowledge_update questions.
        """
        lines = []

        # Show all detected changes
        all_changes = []
        for key, facts in self._facts.items():
            for f in facts:
                if f.valid_until is not None:
                    # Find superseding fact
                    for f2 in facts:
                        if f2.valid_from == f.valid_until and f2.valid_until is None:
                            all_changes.append({
                                "entity": f.entity,
                                "attribute": f.attribute,
                                "old": f.value,
                                "new": f2.value,
                                "date": f.valid_until,
                            })
                            break

        if all_changes:
            lines.append("## Fact Changes Detected (bi-temporal tracking)")
            for ch in all_changes:
                lines.append(
                    f"- {ch['entity']}.{ch['attribute']}: "
                    f"OLD: {ch['old']} → CURRENT: {ch['new']} "
                    f"(changed {ch['date']})"
                )

        # Show current facts relevant to query
        if query:
            query_words = set(query.lower().split())
            relevant = []
            for key, facts in self._facts.items():
                for f in facts:
                    if f.valid_until is None:
                        fact_text = f"{f.entity} {f.attribute} {f.value}".lower()
                        score = sum(1 for w in query_words if w in fact_text)
                        if score >= 1:
                            relevant.append((score, f))
            relevant.sort(key=lambda x: -x[0])
            if relevant:
                lines.append("\n## Current Facts (relevant to query)")
                for _, f in relevant[:10]:
                    lines.append(f"- {f.entity}.{f.attribute} = {f.value} [{f.session_date}]")

        return "\n".join(lines)

    def stats(self) -> dict:
        total = sum(len(v) for v in self._facts.values())
        current = sum(1 for facts in self._facts.values() for f in facts if f.valid_until is None)
        return {
            "total_facts": total,
            "current": current,
            "superseded": total - current,
            "attributes": len(self._facts),
        }
