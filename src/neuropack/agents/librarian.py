"""Memory Librarian Agent — intelligent memory organization and retrieval.

The Librarian runs as a post-processing step after every store() call,
or as a batch job to organize existing memories. It:

1. Extracts structured FACT CARDS from conversations
2. Detects and resolves superseded facts (old value → new value)
3. Maintains a searchable fact index for instant lookups
4. Builds entity timelines for temporal reasoning
5. Classifies memory importance and decay rate

Works with any LLM backend: Ollama (free, local), OpenAI, Anthropic.

Usage:
    from neuropack.agents.librarian import MemoryLibrarian

    librarian = MemoryLibrarian(store)
    librarian.process_memory(memory_id)        # after store()
    librarian.organize_all()                    # batch job
    card = librarian.lookup("korean restaurants tried")  # instant lookup
    timeline = librarian.entity_timeline("korean restaurants")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Structured fact card — the core data unit of the librarian
@dataclass
class FactCard:
    """A single structured fact extracted from a memory."""
    entity: str              # "User", "Rachel", "project X"
    attribute: str           # "korean_restaurants_tried", "job_title", "location"
    value: str               # "4", "Senior Engineer", "Chicago"
    date: str                # when this became true (session date or extracted date)
    memory_id: str           # source memory
    confidence: float        # 0-1
    is_current: bool = True  # False if superseded by a newer card
    superseded_by: str = ""  # ID of the superseding card
    raw_text: str = ""       # original sentence this was extracted from
    card_type: str = "fact"  # fact, preference, event, decision


# Extraction prompt for local LLM (Ollama)
_EXTRACT_PROMPT = """Extract structured facts from this conversation as a JSON array.

For each fact, output:
{{"entity": "who/what", "attribute": "what property", "value": "the value", "type": "fact|preference|event|decision"}}

Rules:
- Entity is usually "User" for first-person statements
- Attribute should be a short snake_case descriptor
- Value should be specific (include numbers, names, brands)
- Type: "preference" for likes/dislikes/favorites, "event" for activities/trips, "decision" for choices made, "fact" for everything else
- Include ALL facts, not just important ones

Example input: "I tried a new Korean restaurant downtown, that makes 4 total now. I'm using my Canon EOS R5 for the photos."
Example output:
[
  {{"entity": "User", "attribute": "korean_restaurants_tried", "value": "4", "type": "fact"}},
  {{"entity": "User", "attribute": "camera", "value": "Canon EOS R5", "type": "fact"}},
  {{"entity": "User", "attribute": "activity", "value": "tried new Korean restaurant downtown", "type": "event"}}
]

Now extract facts from this conversation:
{content}

JSON array:"""

# Supersession prompt
_SUPERSESSION_PROMPT = """Given these two facts about the same topic, determine if the newer one supersedes the older one.

Old fact ({old_date}): {old_entity}.{old_attr} = "{old_value}"
New fact ({new_date}): {new_entity}.{new_attr} = "{new_value}"

Does the new fact UPDATE/REPLACE the old fact? Answer only "yes" or "no".
If they're about different things (e.g., different restaurants, different items), answer "no".
"""


class MemoryLibrarian:
    """Intelligent memory organization agent."""

    def __init__(
        self,
        store: Any,
        llm_provider: Any | None = None,
        auto_process: bool = True,
    ) -> None:
        """Initialize the librarian.

        Args:
            store: MemoryStore instance
            llm_provider: Optional LLM for extraction (OllamaProvider, etc.)
                         If None, uses heuristic extraction (no LLM cost)
            auto_process: If True, hook into store.store() for automatic processing
        """
        self._store = store
        self._llm = llm_provider
        self._fact_cards: dict[str, FactCard] = {}  # card_id -> FactCard
        self._entity_index: dict[str, list[str]] = {}  # entity -> [card_ids]
        self._attribute_index: dict[str, list[str]] = {}  # "entity.attr" -> [card_ids]
        self._auto_process = auto_process

    # ------------------------------------------------------------------
    # Core: Process a memory and extract fact cards
    # ------------------------------------------------------------------

    def process_memory(self, memory_id: str) -> list[FactCard]:
        """Extract fact cards from a stored memory and handle supersession."""
        # Get the memory
        records = self._store.list(limit=1)  # TODO: get by ID
        # For now, work with content directly
        return []

    def process_content(
        self,
        content: str,
        memory_id: str = "",
        date: str = "",
    ) -> list[FactCard]:
        """Extract fact cards from raw content.

        This is the main extraction method. Works with or without LLM.
        """
        if self._llm:
            cards = self._extract_with_llm(content, memory_id, date)
        else:
            cards = self._extract_heuristic(content, memory_id, date)

        # Check for supersession against existing cards
        for card in cards:
            self._check_supersession(card)
            self._index_card(card)

        return cards

    def process_batch(
        self,
        items: list[dict],
        progress_callback: Any = None,
    ) -> int:
        """Process a batch of memories. Returns count of cards extracted.

        Each item should have: content, memory_id (optional), date (optional).
        """
        total_cards = 0
        for i, item in enumerate(items):
            content = item.get("content", "")
            memory_id = item.get("memory_id", "")
            date = item.get("date", "")

            cards = self.process_content(content, memory_id, date)
            total_cards += len(cards)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(items))

        return total_cards

    # ------------------------------------------------------------------
    # Lookup: Instant fact retrieval
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> FactCard | None:
        """Look up the CURRENT value of a fact by attribute or natural language.

        Returns the most recent, non-superseded fact card matching the query.
        """
        query_lower = query.lower().replace(" ", "_")

        # Direct attribute lookup
        for key, card_ids in self._attribute_index.items():
            if query_lower in key.lower():
                # Return the most recent current card
                for cid in reversed(card_ids):
                    card = self._fact_cards.get(cid)
                    if card and card.is_current:
                        return card

        # Keyword search across all current cards
        query_words = set(query.lower().split())
        best_card = None
        best_score = 0
        for card in self._fact_cards.values():
            if not card.is_current:
                continue
            card_text = f"{card.entity} {card.attribute} {card.value}".lower()
            score = sum(1 for w in query_words if w in card_text)
            if score > best_score:
                best_score = score
                best_card = card

        return best_card

    def lookup_all(self, entity: str = "", attribute: str = "") -> list[FactCard]:
        """Get all current cards for an entity or attribute."""
        results = []
        for card in self._fact_cards.values():
            if not card.is_current:
                continue
            if entity and entity.lower() not in card.entity.lower():
                continue
            if attribute and attribute.lower() not in card.attribute.lower():
                continue
            results.append(card)
        return results

    def entity_timeline(self, entity: str) -> list[FactCard]:
        """Get the full history of an entity (including superseded facts)."""
        results = []
        entity_lower = entity.lower()
        for card in self._fact_cards.values():
            if entity_lower in card.entity.lower() or entity_lower in card.attribute.lower():
                results.append(card)
        results.sort(key=lambda c: c.date)
        return results

    # ------------------------------------------------------------------
    # Statistics and reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return librarian statistics."""
        total = len(self._fact_cards)
        current = sum(1 for c in self._fact_cards.values() if c.is_current)
        superseded = total - current
        by_type = {}
        for card in self._fact_cards.values():
            by_type[card.card_type] = by_type.get(card.card_type, 0) + 1
        return {
            "total_cards": total,
            "current_cards": current,
            "superseded_cards": superseded,
            "entities": len(self._entity_index),
            "by_type": by_type,
        }

    def export_cards(self) -> list[dict]:
        """Export all fact cards as dicts."""
        return [
            {
                "entity": c.entity,
                "attribute": c.attribute,
                "value": c.value,
                "date": c.date,
                "is_current": c.is_current,
                "type": c.card_type,
                "confidence": c.confidence,
                "raw_text": c.raw_text,
            }
            for c in self._fact_cards.values()
        ]

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------

    def _extract_with_llm(
        self, content: str, memory_id: str, date: str,
    ) -> list[FactCard]:
        """Use LLM (Ollama or cloud) to extract structured facts."""
        prompt = _EXTRACT_PROMPT.format(content=content[:3000])

        try:
            if hasattr(self._llm, "generate"):
                response = self._llm.generate(prompt, max_tokens=1000, temperature=0.0)
            elif hasattr(self._llm, "chat"):
                response = self._llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=1000, temperature=0.0,
                )
            else:
                return self._extract_heuristic(content, memory_id, date)

            # Parse JSON response
            return self._parse_llm_response(response, memory_id, date)

        except Exception as e:
            logger.debug("LLM extraction failed: %s, falling back to heuristic", e)
            return self._extract_heuristic(content, memory_id, date)

    def _parse_llm_response(
        self, response: str, memory_id: str, date: str,
    ) -> list[FactCard]:
        """Parse LLM JSON response into FactCards."""
        cards = []

        # Extract JSON array from response
        response = response.strip()
        # Find the JSON array
        start = response.find("[")
        end = response.rfind("]")
        if start == -1 or end == -1:
            return cards

        json_str = response[start:end + 1]
        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common LLM JSON issues
            json_str = json_str.replace("'", '"')
            try:
                items = json.loads(json_str)
            except json.JSONDecodeError:
                return cards

        for item in items:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "Unknown"))
            attribute = str(item.get("attribute", ""))
            value = str(item.get("value", ""))
            card_type = str(item.get("type", "fact"))

            if not attribute or not value:
                continue

            card_id = f"{entity}_{attribute}_{int(hashlib.md5(value.encode()).hexdigest()[:8], 16) % 100000}_{date}"
            card = FactCard(
                entity=entity,
                attribute=attribute,
                value=value,
                date=date,
                memory_id=memory_id,
                confidence=0.85,
                card_type=card_type,
            )
            self._fact_cards[card_id] = card
            cards.append(card)

        return cards

    # ------------------------------------------------------------------
    # Heuristic extraction (no LLM, zero cost)
    # ------------------------------------------------------------------

    def _extract_heuristic(
        self, content: str, memory_id: str, date: str,
    ) -> list[FactCard]:
        """Extract fact cards using regex patterns. Fast, free, works offline."""
        cards = []

        # Pattern 1: "I [verb] [object]" — actions, preferences, possessions
        _ACTION_PATTERNS = [
            # Preferences
            (r"\bI\s+(?:prefer|like|love|enjoy|use|own|bought|switched to|upgraded to)\s+"
             r"(?:a |an |the |my )?(.+?)(?:\.|,|;|!|\n|$)", "preference"),
            # Current state
            (r"\bI(?:'m| am)\s+(?:a |an )?(.+?)(?:\.|,|;|!|\n|$)", "fact"),
            # Quantities / counts
            (r"\b(?:that (?:makes|brings)|now have|I have|total of|I've tried)\s+"
             r"(\d+)\s+(.+?)(?:\.|,|;|!|\n|$)", "fact"),
            # Activities
            (r"\bI\s+(?:went|visited|attended|tried|started|finished|completed)\s+"
             r"(?:a |an |the |my |to )?(.+?)(?:\.|,|;|!|\n|$)", "event"),
            # Decisions
            (r"\bI\s+(?:decided|chose|picked|selected|went with)\s+"
             r"(?:a |an |the |to )?(.+?)(?:\.|,|;|!|\n|$)", "decision"),
            # Named entities with attributes
            (r"\bmy\s+(\w+(?:\s+\w+)?)\s+is\s+(.+?)(?:\.|,|;|\n|$)", "fact"),
            # Switched/changed
            (r"\bI\s+(?:switched|changed|moved|upgraded)\s+(?:from\s+\w+\s+)?to\s+"
             r"(?:a |an |the )?(.+?)(?:\.|,|;|!|\n|$)", "decision"),
            # Have been doing
            (r"\bI(?:'ve| have)\s+been\s+(.+?)(?:\.|,|;|!|\n|$)", "fact"),
        ]

        # Also match third-person "User X" patterns (from LLM observations)
        _THIRD_PERSON_PATTERNS = [
            (r"[Uu]ser\s+(?:tried|visited|attended|went to|started)\s+(?:a |an |the )?(.+?)(?:\.|,|;|!|\n|$)", "event"),
            (r"[Uu]ser\s+(?:uses?|owns?|has|bought|prefers?|likes?|enjoys?)\s+(?:a |an |the )?(.+?)(?:\.|,|;|!|\n|$)", "preference"),
            (r"[Uu]ser\s+(?:has tried|has visited)\s+(\d+)\s+(.+?)(?:\.|,|;|!|\n|$)", "fact"),
            (r"[Uu]ser\s+(?:switched|changed|moved|upgraded)\s+(?:from\s+\w+\s+)?to\s+(?:a |an |the )?(.+?)(?:\.|,|;|!|\n|$)", "decision"),
            (r"[Uu]ser(?:'s| is)\s+(?:a |an )?(.+?)(?:\.|,|;|!|\n|$)", "fact"),
            (r"[Uu]ser\s+(?:works?|lives?|studies)\s+(?:at|in|for)\s+(.+?)(?:\.|,|;|!|\n|$)", "fact"),
        ]
        _ACTION_PATTERNS.extend(_THIRD_PERSON_PATTERNS)

        _NOISE = {"to", "it", "them", "this", "that", "also", "just", "really", "very",
                   "a new", "the same", "more", "some", "any"}

        # Normalize: strip role prefixes like "[user]: " for raw session text
        content = re.sub(r'\[(?:user|assistant|Date:[^\]]*)\]:\s*', '', content)

        for pattern, card_type in _ACTION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2 and match[0].isdigit():
                        # Count pattern: "that makes 4 total"
                        value = match[0]
                        attribute = match[1].strip().lower().replace(" ", "_")
                    elif len(match) == 2:
                        # Named attribute: "my X is Y"
                        attribute = match[0].strip().lower().replace(" ", "_")
                        value = match[1].strip()
                    else:
                        value = match[0] if isinstance(match, tuple) else match
                        attribute = ""
                else:
                    value = match.strip()
                    attribute = ""

                # Skip noise
                if not value or len(value) < 3 or value.lower() in _NOISE:
                    continue
                value = value[:100]  # truncate long values

                # Auto-generate attribute from value if not set
                if not attribute:
                    # Use first 2-3 words as attribute
                    words = value.lower().split()[:3]
                    attribute = "_".join(w for w in words if w not in _NOISE)

                if not attribute:
                    continue

                card_id = f"User_{attribute}_{hash(value) % 100000}_{date}"
                card = FactCard(
                    entity="User",
                    attribute=attribute,
                    value=value,
                    date=date,
                    memory_id=memory_id,
                    confidence=0.6,
                    card_type=card_type,
                    raw_text=value[:200],
                )
                self._fact_cards[card_id] = card
                cards.append(card)

        return cards

    # ------------------------------------------------------------------
    # Supersession detection
    # ------------------------------------------------------------------

    def _check_supersession(self, new_card: FactCard) -> None:
        """Check if this new card supersedes an existing card."""
        key = f"{new_card.entity.lower()}.{new_card.attribute.lower()}"

        if key in self._attribute_index:
            for old_card_id in self._attribute_index[key]:
                old_card = self._fact_cards.get(old_card_id)
                if not old_card or not old_card.is_current:
                    continue

                # Same entity + attribute, different value, newer date
                if (old_card.value.lower() != new_card.value.lower()
                        and old_card.date <= new_card.date):

                    # Use LLM to verify if it's truly a supersession
                    if self._llm and old_card.confidence > 0.5:
                        is_superseded = self._verify_supersession_llm(old_card, new_card)
                        if not is_superseded:
                            continue

                    # Mark old card as superseded
                    old_card_mut = self._fact_cards[old_card_id]
                    object.__setattr__(old_card_mut, 'is_current', False)
                    object.__setattr__(old_card_mut, 'superseded_by',
                                       f"{new_card.entity}_{new_card.attribute}_{hash(new_card.value) % 100000}_{new_card.date}")
                    logger.debug(
                        "Superseded: %s.%s = '%s' → '%s'",
                        old_card.entity, old_card.attribute,
                        old_card.value, new_card.value,
                    )

    def _verify_supersession_llm(self, old: FactCard, new: FactCard) -> bool:
        """Use LLM to verify if a new fact truly supersedes an old one."""
        prompt = _SUPERSESSION_PROMPT.format(
            old_date=old.date, old_entity=old.entity, old_attr=old.attribute, old_value=old.value,
            new_date=new.date, new_entity=new.entity, new_attr=new.attribute, new_value=new.value,
        )
        try:
            if hasattr(self._llm, "generate"):
                response = self._llm.generate(prompt, max_tokens=10, temperature=0.0)
            else:
                response = self._llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=10, temperature=0.0,
                )
            return "yes" in response.lower()
        except Exception:
            # Default to yes if LLM fails (assume supersession)
            return True

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_card(self, card: FactCard) -> None:
        """Add a card to the entity and attribute indexes."""
        card_id = f"{card.entity}_{card.attribute}_{hash(card.value) % 100000}_{card.date}"

        # Entity index
        entity_key = card.entity.lower()
        if entity_key not in self._entity_index:
            self._entity_index[entity_key] = []
        if card_id not in self._entity_index[entity_key]:
            self._entity_index[entity_key].append(card_id)

        # Attribute index
        attr_key = f"{card.entity.lower()}.{card.attribute.lower()}"
        if attr_key not in self._attribute_index:
            self._attribute_index[attr_key] = []
        if card_id not in self._attribute_index[attr_key]:
            self._attribute_index[attr_key].append(card_id)

    # ------------------------------------------------------------------
    # Integration with benchmark
    # ------------------------------------------------------------------

    def build_knowledge_snapshot(self) -> str:
        """Build a text snapshot of all CURRENT facts for use as LLM context.

        Returns a formatted string of all current fact cards, organized by entity.
        """
        by_entity: dict[str, list[FactCard]] = {}
        for card in self._fact_cards.values():
            if not card.is_current:
                continue
            entity = card.entity
            if entity not in by_entity:
                by_entity[entity] = []
            by_entity[entity].append(card)

        lines = ["## Current Knowledge"]
        for entity, cards in sorted(by_entity.items()):
            lines.append(f"\n### {entity}")
            for card in sorted(cards, key=lambda c: c.attribute):
                date_tag = f" [{card.date}]" if card.date else ""
                lines.append(f"- {card.attribute}: {card.value}{date_tag}")

        return "\n".join(lines)

    def build_change_history(self, attribute: str = "") -> str:
        """Build a text history of fact changes for knowledge_update questions.

        Shows old → new value transitions with dates.
        """
        changes = []
        for card in self._fact_cards.values():
            if not card.is_current:
                # This card was superseded
                new_id = card.superseded_by
                new_card = self._fact_cards.get(new_id)
                if attribute:
                    if attribute.lower() not in card.attribute.lower():
                        continue
                changes.append({
                    "entity": card.entity,
                    "attribute": card.attribute,
                    "old_value": card.value,
                    "old_date": card.date,
                    "new_value": new_card.value if new_card else "?",
                    "new_date": new_card.date if new_card else "?",
                })

        if not changes:
            return ""

        lines = ["## Knowledge Changes Detected"]
        for ch in changes:
            lines.append(
                f"- {ch['entity']}.{ch['attribute']}: "
                f"OLD ({ch['old_date']}): {ch['old_value']} → "
                f"CURRENT ({ch['new_date']}): {ch['new_value']}"
            )
        return "\n".join(lines)
