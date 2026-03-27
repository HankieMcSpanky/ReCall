from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from neuropack.storage.database import Database
from neuropack.types import MemoryRecord


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    name: str
    entity_type: str = "concept"
    namespace: str = "default"
    mention_count: int = 1
    first_seen: str = ""
    last_seen: str = ""
    status: str = "active"


@dataclass(frozen=True, slots=True)
class Relationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str = "related_to"
    memory_id: str = ""
    namespace: str = "default"
    weight: float = 1.0
    created_at: str = ""
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None


# Patterns for entity extraction
_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_URL_RE = re.compile(r'https?://[^\s<>"]+')
_VERSION_RE = re.compile(r'\b[vV]?\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b')
_PATH_RE = re.compile(r'(?:[A-Za-z]:)?(?:/[\w.-]+){2,}')
_QUOTED_RE = re.compile(r'"([^"]{2,60})"')
_CAPITALIZED_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+(?:[a-z]+\s+)?[A-Z][a-z]+)+)\b')
_SINGLE_CAP_RE = re.compile(r'\b([A-Z][a-z]{2,})\b')

# Verb patterns for relationship extraction
_VERB_PATTERNS = [
    (re.compile(r'(\S+)\s+uses?\s+(\S+)', re.IGNORECASE), "uses"),
    (re.compile(r'(\S+)\s+created?\s+(\S+)', re.IGNORECASE), "created"),
    (re.compile(r'(\S+)\s+is\s+(?:a\s+)?(\S+)', re.IGNORECASE), "is"),
    (re.compile(r'(\S+)\s+(?:depends?\s+on|requires?)\s+(\S+)', re.IGNORECASE), "depends_on"),
    (re.compile(r'(\S+)\s+(?:built|built\s+with)\s+(\S+)', re.IGNORECASE), "built_with"),
    (re.compile(r'(\S+)\s+(?:implements?|supports?)\s+(\S+)', re.IGNORECASE), "implements"),
]

# Skip these common words as entities
_STOP_ENTITIES = {
    "The", "This", "That", "These", "Those", "There", "Here",
    "What", "When", "Where", "Which", "While", "With", "Without",
    "About", "After", "Before", "Between", "Because", "However",
    "Also", "Each", "Every", "Some", "Many", "Most", "Other",
    "First", "Last", "Next", "New", "Old", "Good", "Bad",
    "Just", "Only", "Very", "Still", "Already", "Always",
}

# Temporal marker patterns for detecting time-scoped facts
_TEMPORAL_CURRENT = re.compile(
    r'\b(?:as of|since|starting|from|currently|now)\b', re.IGNORECASE
)
_TEMPORAL_ENDED = re.compile(
    r'\b(?:until|no longer|previously|formerly|was|used to|stopped|ended|left|quit)\b',
    re.IGNORECASE,
)
_DATE_MARKER = re.compile(
    r'\b(\d{4}[-/]\d{2}(?:[-/]\d{2})?|\w+ \d{4})\b'
)


def detect_temporal_markers(text: str) -> dict:
    """Detect temporal signals in text.

    Returns dict with keys: valid_from (str|None), ended (bool),
    date_hint (str|None).
    """
    result: dict = {"valid_from": None, "ended": False, "date_hint": None}

    dates = _DATE_MARKER.findall(text)
    if dates:
        result["date_hint"] = dates[0]

    if _TEMPORAL_ENDED.search(text):
        result["ended"] = True
        if dates:
            result["valid_from"] = dates[0]
    elif _TEMPORAL_CURRENT.search(text):
        if dates:
            result["valid_from"] = dates[0]

    return result


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract entities from text without NLP dependencies.

    Returns list of (name, entity_type) tuples.
    """
    entities: dict[str, str] = {}

    # Emails
    for m in _EMAIL_RE.finditer(text):
        entities[m.group()] = "email"

    # URLs
    for m in _URL_RE.finditer(text):
        entities[m.group()] = "url"

    # Version numbers
    for m in _VERSION_RE.finditer(text):
        entities[m.group()] = "version"

    # File paths
    for m in _PATH_RE.finditer(text):
        entities[m.group()] = "path"

    # Quoted terms
    for m in _QUOTED_RE.finditer(text):
        val = m.group(1).strip()
        if len(val) >= 2:
            entities[val] = "concept"

    # Multi-word capitalized phrases (proper nouns)
    for m in _CAPITALIZED_RE.finditer(text):
        name = m.group(1)
        if name not in _STOP_ENTITIES and not all(w in _STOP_ENTITIES for w in name.split()):
            entities[name] = "proper_noun"

    # Single capitalized words (at least 3 chars, not at sentence start)
    for m in _SINGLE_CAP_RE.finditer(text):
        name = m.group(1)
        if name in _STOP_ENTITIES:
            continue
        # Check it's not at sentence start
        pos = m.start()
        if pos > 0 and text[pos - 1] not in '.!?\n':
            if name not in entities:
                entities[name] = "concept"

    return list(entities.items())


def extract_relationships(
    text: str, entities: list[tuple[str, str]]
) -> list[tuple[str, str, str]]:
    """Extract relationships between entities.

    Uses co-occurrence within sentences + verb-pattern heuristics.
    Returns list of (source_name, target_name, relation_type) tuples.
    """
    relationships: list[tuple[str, str, str]] = []
    entity_names = {name for name, _ in entities}

    # Split into sentences
    sentences = re.split(r'[.!?\n]+', text)

    for sentence in sentences:
        # Find which entities appear in this sentence
        present = [name for name in entity_names if name in sentence]

        # Co-occurrence: entities in same sentence are related
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                relationships.append((present[i], present[j], "co_occurs"))

        # Verb pattern matching
        for pattern, rel_type in _VERB_PATTERNS:
            match = pattern.search(sentence)
            if match:
                src, tgt = match.group(1), match.group(2)
                if src in entity_names and tgt in entity_names:
                    relationships.append((src, tgt, rel_type))

    # Deduplicate
    seen = set()
    unique = []
    for rel in relationships:
        key = (rel[0], rel[1], rel[2])
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    return unique


class KnowledgeGraph:
    """Zero-dependency knowledge graph built from memory content."""

    def __init__(
        self, db: Database, namespace: str = "default", temporal_tracking: bool = True
    ):
        self._db = db
        self._namespace = namespace
        self._temporal = temporal_tracking

    def process_memory(self, record: MemoryRecord) -> None:
        """Extract entities and relationships from a memory and persist."""
        text = record.content
        entities = extract_entities(text)
        relationships = extract_relationships(text, entities)
        now = datetime.now(timezone.utc).isoformat()
        ns = record.namespace

        # Temporal marker detection
        temporal = detect_temporal_markers(text) if self._temporal else {}

        conn = self._db.connect()

        for name, etype in entities:
            row = conn.execute(
                "SELECT id, mention_count FROM entities WHERE name = ? AND namespace = ?",
                (name, ns),
            ).fetchone()

            if row:
                d = dict(row)
                conn.execute(
                    "UPDATE entities SET mention_count = ?, last_seen = ? WHERE id = ?",
                    (d["mention_count"] + 1, now, d["id"]),
                )
            else:
                eid = uuid.uuid4().hex
                conn.execute(
                    """INSERT INTO entities (id, name, entity_type, namespace, mention_count, first_seen, last_seen)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (eid, name, etype, ns, now, now),
                )

        conn.commit()

        # Insert relationships with temporal data
        for src_name, tgt_name, rel_type in relationships:
            src_row = conn.execute(
                "SELECT id FROM entities WHERE name = ? AND namespace = ?",
                (src_name, ns),
            ).fetchone()
            tgt_row = conn.execute(
                "SELECT id FROM entities WHERE name = ? AND namespace = ?",
                (tgt_name, ns),
            ).fetchone()

            if src_row and tgt_row:
                src_id = dict(src_row)["id"]
                tgt_id = dict(tgt_row)["id"]
                rid = uuid.uuid4().hex

                valid_from = temporal.get("date_hint") or (now if temporal.get("valid_from") else None)
                valid_until = now if temporal.get("ended") else None

                # Auto-supersede: if new rel contradicts existing active one
                if self._temporal and rel_type not in ("co_occurs",):
                    self._auto_supersede(conn, src_id, tgt_id, rel_type, rid, now)

                conn.execute(
                    """INSERT INTO relationships
                       (id, source_entity_id, target_entity_id, relation_type,
                        memory_id, namespace, weight, created_at, valid_from, valid_until)
                       VALUES (?, ?, ?, ?, ?, ?, 1.0, ?, ?, ?)""",
                    (rid, src_id, tgt_id, rel_type, record.id, ns, now, valid_from, valid_until),
                )

        conn.commit()

    def _auto_supersede(
        self,
        conn,
        src_id: str,
        tgt_id: str,
        rel_type: str,
        new_rel_id: str,
        now: str,
    ) -> None:
        """Supersede existing active relationships between the same entities with the same type."""
        old_rels = conn.execute(
            """SELECT id FROM relationships
               WHERE source_entity_id = ? AND target_entity_id = ?
               AND relation_type = ? AND valid_until IS NULL AND superseded_by IS NULL""",
            (src_id, tgt_id, rel_type),
        ).fetchall()

        for row in old_rels:
            old_id = dict(row)["id"]
            conn.execute(
                "UPDATE relationships SET valid_until = ?, superseded_by = ? WHERE id = ?",
                (now, new_rel_id, old_id),
            )

    def query_entity(self, name: str, as_of: str | None = None) -> dict:
        """Look up an entity and its relationships.

        If *as_of* is given (ISO timestamp or date string), only return
        relationships that were valid at that point in time.
        """
        conn = self._db.connect()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        ).fetchone()

        if row is None:
            return {"found": False, "name": name}

        entity = dict(row)

        if as_of:
            rels = conn.execute(
                """SELECT r.*, e1.name as source_name, e2.name as target_name
                   FROM relationships r
                   JOIN entities e1 ON r.source_entity_id = e1.id
                   JOIN entities e2 ON r.target_entity_id = e2.id
                   WHERE (r.source_entity_id = ? OR r.target_entity_id = ?)
                   AND (r.valid_from IS NULL OR r.valid_from <= ?)
                   AND (r.valid_until IS NULL OR r.valid_until > ?)""",
                (entity["id"], entity["id"], as_of, as_of),
            ).fetchall()
        else:
            rels = conn.execute(
                """SELECT r.*, e1.name as source_name, e2.name as target_name
                   FROM relationships r
                   JOIN entities e1 ON r.source_entity_id = e1.id
                   JOIN entities e2 ON r.target_entity_id = e2.id
                   WHERE r.source_entity_id = ? OR r.target_entity_id = ?""",
                (entity["id"], entity["id"]),
            ).fetchall()

        return {
            "found": True,
            "name": entity["name"],
            "entity_type": entity["entity_type"],
            "namespace": entity["namespace"],
            "mention_count": entity["mention_count"],
            "first_seen": entity["first_seen"],
            "last_seen": entity["last_seen"],
            "status": entity.get("status", "active"),
            "relationships": [
                {
                    "source": dict(r)["source_name"],
                    "target": dict(r)["target_name"],
                    "relation_type": dict(r)["relation_type"],
                    "weight": dict(r)["weight"],
                    "valid_from": dict(r).get("valid_from"),
                    "valid_until": dict(r).get("valid_until"),
                    "superseded_by": dict(r).get("superseded_by"),
                }
                for r in rels
            ],
        }

    def get_current_facts(self, entity_name: str) -> dict:
        """Return only currently valid relationships (valid_until IS NULL)."""
        conn = self._db.connect()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?", (entity_name,)
        ).fetchone()

        if row is None:
            return {"found": False, "name": entity_name}

        entity = dict(row)
        rels = conn.execute(
            """SELECT r.*, e1.name as source_name, e2.name as target_name
               FROM relationships r
               JOIN entities e1 ON r.source_entity_id = e1.id
               JOIN entities e2 ON r.target_entity_id = e2.id
               WHERE (r.source_entity_id = ? OR r.target_entity_id = ?)
               AND r.valid_until IS NULL AND r.superseded_by IS NULL""",
            (entity["id"], entity["id"]),
        ).fetchall()

        return {
            "found": True,
            "name": entity["name"],
            "current_facts": [
                {
                    "source": dict(r)["source_name"],
                    "target": dict(r)["target_name"],
                    "relation_type": dict(r)["relation_type"],
                    "valid_from": dict(r).get("valid_from"),
                }
                for r in rels
            ],
        }

    def fact_timeline(self, entity_name: str) -> dict:
        """Show how relationships for an entity evolved over time."""
        conn = self._db.connect()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?", (entity_name,)
        ).fetchone()

        if row is None:
            return {"found": False, "name": entity_name}

        entity = dict(row)
        rels = conn.execute(
            """SELECT r.*, e1.name as source_name, e2.name as target_name
               FROM relationships r
               JOIN entities e1 ON r.source_entity_id = e1.id
               JOIN entities e2 ON r.target_entity_id = e2.id
               WHERE r.source_entity_id = ? OR r.target_entity_id = ?
               ORDER BY r.created_at ASC""",
            (entity["id"], entity["id"]),
        ).fetchall()

        timeline = []
        for r in rels:
            d = dict(r)
            entry = {
                "source": d["source_name"],
                "target": d["target_name"],
                "relation_type": d["relation_type"],
                "created_at": d["created_at"],
                "valid_from": d.get("valid_from"),
                "valid_until": d.get("valid_until"),
                "superseded_by": d.get("superseded_by"),
                "active": d.get("valid_until") is None and d.get("superseded_by") is None,
            }
            timeline.append(entry)

        return {
            "found": True,
            "name": entity["name"],
            "timeline": timeline,
        }

    def supersede_fact(self, old_rel_id: str, new_rel_id: str) -> bool:
        """Manually supersede an old relationship with a new one."""
        conn = self._db.connect()
        now = datetime.now(timezone.utc).isoformat()

        old = conn.execute(
            "SELECT id FROM relationships WHERE id = ?", (old_rel_id,)
        ).fetchone()
        new = conn.execute(
            "SELECT id FROM relationships WHERE id = ?", (new_rel_id,)
        ).fetchone()

        if not old or not new:
            return False

        conn.execute(
            "UPDATE relationships SET valid_until = ?, superseded_by = ? WHERE id = ?",
            (now, new_rel_id, old_rel_id),
        )
        conn.commit()
        return True

    def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """Search entities by name (LIKE match)."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? ORDER BY mention_count DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            {
                "name": dict(r)["name"],
                "entity_type": dict(r)["entity_type"],
                "namespace": dict(r)["namespace"],
                "mention_count": dict(r)["mention_count"],
            }
            for r in rows
        ]

    def get_related(self, entity_name: str, depth: int = 1, limit: int = 20) -> dict:
        """Traverse graph from an entity to find related entities."""
        result = self.query_entity(entity_name)
        if not result.get("found"):
            return result

        if depth > 1:
            # Get names of directly related entities
            related_names = set()
            for rel in result.get("relationships", []):
                related_names.add(rel["source"])
                related_names.add(rel["target"])
            related_names.discard(entity_name)

            # Query each related entity (depth-1)
            related_details = []
            for name in list(related_names)[:limit]:
                detail = self.query_entity(name)
                if detail.get("found"):
                    related_details.append(detail)
            result["related_entities"] = related_details

        return result

    def entity_stats(self) -> dict:
        """Get entity and relationship counts + top entities."""
        conn = self._db.connect()
        e_count = dict(conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone())["cnt"]
        r_count = dict(conn.execute("SELECT COUNT(*) as cnt FROM relationships").fetchone())["cnt"]

        top = conn.execute(
            "SELECT name, mention_count FROM entities ORDER BY mention_count DESC LIMIT 10"
        ).fetchall()

        return {
            "total_entities": e_count,
            "total_relationships": r_count,
            "top_entities": [
                {"name": dict(r)["name"], "mentions": dict(r)["mention_count"]}
                for r in top
            ],
        }

    def delete_for_memory(self, memory_id: str) -> None:
        """Remove relationships associated with a memory."""
        conn = self._db.connect()
        conn.execute("DELETE FROM relationships WHERE memory_id = ?", (memory_id,))
        conn.commit()
