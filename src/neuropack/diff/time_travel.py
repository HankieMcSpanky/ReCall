from __future__ import annotations

from datetime import datetime, timezone

from neuropack.types import MemoryRecord


class TimeTravelEngine:
    """Recall memories as they existed at a past point in time."""

    def recall_as_of(
        self,
        store,
        query: str,
        as_of: datetime,
        limit: int = 10,
    ) -> list[dict]:
        """Recall memories that existed at as_of, using historical versions.

        Searches memories created before as_of and reconstructs their state
        at that point using version history when the current version was
        modified after as_of.

        Args:
            store: MemoryStore instance
            query: Semantic search query
            as_of: Point in time to recall from
            limit: Max results
        """
        # First, do a normal recall to get candidates
        results = store.recall(query=query, limit=limit * 3)

        as_of_results: list[dict] = []
        for r in results:
            record = r.record
            # Skip memories created after as_of
            if record.created_at > as_of:
                continue

            # If the memory was last updated before as_of, use current state
            if record.updated_at <= as_of:
                as_of_results.append({
                    "id": record.id,
                    "l3_abstract": record.l3_abstract,
                    "tags": record.tags,
                    "score": round(r.score, 4),
                    "source": record.source,
                    "created_at": record.created_at.isoformat(),
                    "as_of_version": "current",
                })
            else:
                # Memory was updated after as_of; reconstruct from versions
                version = store._repo.get_version_at(record.id, as_of)
                if version is not None:
                    as_of_results.append({
                        "id": record.id,
                        "l3_abstract": version.l3_abstract,
                        "tags": version.tags,
                        "score": round(r.score, 4),
                        "source": record.source,
                        "created_at": record.created_at.isoformat(),
                        "as_of_version": f"v{version.version}",
                    })
                else:
                    # No version history, but memory existed; use current as fallback
                    as_of_results.append({
                        "id": record.id,
                        "l3_abstract": record.l3_abstract,
                        "tags": record.tags,
                        "score": round(r.score, 4),
                        "source": record.source,
                        "created_at": record.created_at.isoformat(),
                        "as_of_version": "current (no version history)",
                    })

            if len(as_of_results) >= limit:
                break

        return as_of_results

    def snapshot_at(
        self,
        store,
        as_of: datetime,
    ) -> list[dict]:
        """Return all memories as they existed at a point in time.

        Args:
            store: MemoryStore instance
            as_of: Point in time to snapshot
        """
        as_of_iso = as_of.isoformat()
        conn = store._db.connect()

        # Get all memories that existed at as_of (created before or at as_of)
        rows = conn.execute(
            "SELECT id, l3_abstract, tags, source, created_at, updated_at "
            "FROM memories WHERE created_at <= ? ORDER BY created_at DESC",
            (as_of_iso,),
        ).fetchall()

        import json

        snapshot: list[dict] = []
        for row in rows:
            d = dict(row)
            memory_id = d["id"]
            updated_at = datetime.fromisoformat(d["updated_at"])
            tags = json.loads(d["tags"])

            if updated_at <= as_of:
                # Current state is valid for as_of
                snapshot.append({
                    "id": memory_id,
                    "l3_abstract": d["l3_abstract"],
                    "tags": tags,
                    "source": d["source"],
                    "created_at": d["created_at"],
                    "version": "current",
                })
            else:
                # Reconstruct from version history
                version = store._repo.get_version_at(memory_id, as_of)
                if version is not None:
                    snapshot.append({
                        "id": memory_id,
                        "l3_abstract": version.l3_abstract,
                        "tags": version.tags,
                        "source": d["source"],
                        "created_at": d["created_at"],
                        "version": f"v{version.version}",
                    })
                else:
                    # No version history; use current state as best effort
                    snapshot.append({
                        "id": memory_id,
                        "l3_abstract": d["l3_abstract"],
                        "tags": tags,
                        "source": d["source"],
                        "created_at": d["created_at"],
                        "version": "current (no version history)",
                    })

        return snapshot
