"""Graph-based retrieval: find memories linked via knowledge graph entities."""
from __future__ import annotations

from neuropack.core.knowledge_graph import extract_entities
from neuropack.storage.database import Database


class GraphRetriever:
    """Walk the knowledge graph to find memory_ids related to a query."""

    def __init__(self, db: Database, max_hops: int = 2):
        self._db = db
        self._max_hops = max_hops

    def retrieve(self, query: str, limit: int = 30) -> list[tuple[str, float]]:
        """Extract entities from query, walk graph, return (memory_id, score) pairs.

        Score = 1/(1+hop_distance) * relationship_weight * mention_boost.
        """
        entities = extract_entities(query)
        if not entities:
            return []

        conn = self._db.connect()
        entity_names = [name for name, _ in entities]

        # Resolve entity ids + mention counts
        entity_map: dict[str, tuple[str, int]] = {}  # name -> (id, mention_count)
        for name in entity_names:
            row = conn.execute(
                "SELECT id, mention_count FROM entities WHERE name = ? AND status = 'active'",
                (name,),
            ).fetchone()
            if row:
                d = dict(row)
                entity_map[name] = (d["id"], d["mention_count"])

        if not entity_map:
            return []

        # Collect memory_ids with scores via graph walk
        memory_scores: dict[str, float] = {}

        for name, (eid, mention_count) in entity_map.items():
            mention_boost = min(mention_count / 5.0, 2.0)  # cap boost at 2x
            self._walk(conn, eid, 0, mention_boost, memory_scores, set())

        # Sort by score descending
        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def _walk(
        self,
        conn,
        entity_id: str,
        hop: int,
        mention_boost: float,
        memory_scores: dict[str, float],
        visited: set[str],
    ) -> None:
        if hop > self._max_hops or entity_id in visited:
            return
        visited.add(entity_id)

        # Get all active relationships involving this entity
        rows = conn.execute(
            """SELECT memory_id, weight, source_entity_id, target_entity_id
               FROM relationships
               WHERE (source_entity_id = ? OR target_entity_id = ?)
               AND valid_until IS NULL AND superseded_by IS NULL""",
            (entity_id, entity_id),
        ).fetchall()

        distance_factor = 1.0 / (1.0 + hop)

        for row in rows:
            d = dict(row)
            mid = d["memory_id"]
            weight = d["weight"]
            score = distance_factor * weight * mention_boost

            if mid in memory_scores:
                memory_scores[mid] = max(memory_scores[mid], score)
            else:
                memory_scores[mid] = score

            # Walk to the other entity at hop+1
            if hop + 1 <= self._max_hops:
                other_id = (
                    d["target_entity_id"]
                    if d["source_entity_id"] == entity_id
                    else d["source_entity_id"]
                )
                self._walk(conn, other_id, hop + 1, mention_boost * 0.5, memory_scores, visited)
