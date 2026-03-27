from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from neuropack.diff.models import TimelineEntry


def build_timeline(
    store,
    entity: str | None = None,
    tag: str | None = None,
    granularity: str = "day",
) -> list[TimelineEntry]:
    """Build a timeline of memory changes grouped by period.

    Args:
        store: MemoryStore instance
        entity: Optional entity name to filter by (via knowledge graph)
        tag: Optional tag to filter by
        granularity: "day", "week", or "month"
    """
    conn = store._db.connect()

    # Determine the SQL date format for grouping
    if granularity == "week":
        # ISO week: YYYY-Www
        date_fmt = "%Y-W%W"
    elif granularity == "month":
        date_fmt = "%Y-%m"
    else:
        date_fmt = "%Y-%m-%d"

    # Build conditions for the memories query
    conditions: list[str] = []
    params: list[object] = []

    if tag:
        conditions.append("m.tags LIKE ?")
        params.append(f"%{tag}%")

    if entity:
        # Filter by entity: find memory IDs linked to this entity via relationships
        entity_rows = conn.execute(
            "SELECT id FROM entities WHERE LOWER(name) = LOWER(?)", (entity,)
        ).fetchall()
        if entity_rows:
            entity_ids = [dict(r)["id"] for r in entity_rows]
            placeholders = ",".join("?" * len(entity_ids))
            rel_rows = conn.execute(
                f"SELECT DISTINCT memory_id FROM relationships "
                f"WHERE source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders})",
                entity_ids + entity_ids,
            ).fetchall()
            memory_ids = [dict(r)["memory_id"] for r in rel_rows]
            if memory_ids:
                placeholders = ",".join("?" * len(memory_ids))
                conditions.append(f"m.id IN ({placeholders})")
                params.extend(memory_ids)
            else:
                return []
        else:
            return []

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Count created memories per period
    created_query = f"""
        SELECT strftime(?, m.created_at) as period, COUNT(*) as cnt, m.tags
        FROM memories m {where}
        GROUP BY period
        ORDER BY period
    """
    # We need a different approach since we want tags per period
    # First get raw data
    raw_query = f"""
        SELECT m.created_at, m.updated_at, m.tags
        FROM memories m {where}
        ORDER BY m.created_at
    """
    rows = conn.execute(raw_query, params).fetchall()

    # Also get deletions from audit_log
    audit_conditions: list[str] = ["a.action = 'delete'"]
    audit_params: list[object] = []
    audit_rows = conn.execute(
        "SELECT a.timestamp, a.memory_id FROM audit_log a WHERE a.action = 'delete' ORDER BY a.timestamp"
    ).fetchall()

    # Also get modifications from memory_versions
    version_conditions: list[str] = []
    version_params: list[object] = []
    if tag:
        version_conditions.append("mv.tags LIKE ?")
        version_params.append(f"%{tag}%")

    version_where = ""
    if version_conditions:
        version_where = "WHERE " + " AND ".join(version_conditions)

    version_rows = conn.execute(
        f"SELECT mv.saved_at, mv.tags FROM memory_versions mv {version_where} ORDER BY mv.saved_at",
        version_params,
    ).fetchall()

    # Group data by period
    period_data: dict[str, dict] = defaultdict(lambda: {
        "added": 0, "modified": 0, "deleted": 0, "tags": Counter()
    })

    def _format_period(iso_ts: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_ts)
        except ValueError:
            return iso_ts[:10]
        if granularity == "week":
            return dt.strftime("%Y-W%W")
        elif granularity == "month":
            return dt.strftime("%Y-%m")
        else:
            return dt.strftime("%Y-%m-%d")

    def _period_label(period: str) -> str:
        if granularity == "week":
            return f"Week {period}"
        elif granularity == "month":
            parts = period.split("-")
            if len(parts) == 2:
                months = [
                    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                ]
                try:
                    return f"{months[int(parts[1])]} {parts[0]}"
                except (ValueError, IndexError):
                    return period
            return period
        else:
            return period

    # Process created memories
    for row in rows:
        d = dict(row)
        period = _format_period(d["created_at"])
        period_data[period]["added"] += 1
        tags = json.loads(d["tags"])
        for t in tags:
            period_data[period]["tags"][t] += 1

    # Process modifications (version saves indicate a modification happened)
    for row in version_rows:
        d = dict(row)
        period = _format_period(d["saved_at"])
        period_data[period]["modified"] += 1

    # Process deletions
    for row in audit_rows:
        d = dict(row)
        period = _format_period(d["timestamp"])
        period_data[period]["deleted"] += 1

    # Build timeline entries
    entries: list[TimelineEntry] = []
    for period in sorted(period_data.keys()):
        data = period_data[period]
        top_tags = [tag for tag, _ in data["tags"].most_common(5)]
        entries.append(TimelineEntry(
            period=period,
            period_label=_period_label(period),
            added=data["added"],
            modified=data["modified"],
            deleted=data["deleted"],
            top_tags=top_tags,
        ))

    return entries
