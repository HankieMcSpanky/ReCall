from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from neuropack.diff.models import (
    DiffStats,
    MemoryChangeSummary,
    MemoryDiff,
    MemoryUpdate,
)


def parse_relative_date(text: str) -> datetime:
    """Parse a relative date string or ISO date into a datetime.

    Supports:
    - "now"
    - "yesterday"
    - "last week", "last month"
    - "N days ago", "N hours ago", "N weeks ago", "N months ago"
    - ISO dates like "2026-03-01" or "2026-03-01T12:00:00"
    """
    text = text.strip().lower()
    now = datetime.now(timezone.utc)

    if text == "now":
        return now

    if text == "yesterday":
        return now - timedelta(days=1)

    if text == "last week":
        return now - timedelta(weeks=1)

    if text == "last month":
        return now - timedelta(days=30)

    # "N days ago", "N hours ago", "N weeks ago", "N months ago"
    match = re.match(r"(\d+)\s+(day|hour|week|month)s?\s+ago", text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return now - timedelta(days=amount)
        elif unit == "hour":
            return now - timedelta(hours=amount)
        elif unit == "week":
            return now - timedelta(weeks=amount)
        elif unit == "month":
            return now - timedelta(days=amount * 30)

    # Try ISO format
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    raise ValueError(f"Cannot parse date: {text!r}")


class MemoryDiffEngine:
    """Compute diffs between two points in time."""

    def diff_since(
        self,
        store,
        since: datetime,
        until: datetime | None = None,
    ) -> MemoryDiff:
        """Compute a diff of all memory changes in [since, until].

        Args:
            store: MemoryStore instance (used to access repo)
            since: Start of the time window
            until: End of the time window (default: now)
        """
        if until is None:
            until = datetime.now(timezone.utc)
        return self._compute_diff(store, since, until)

    def diff_between(
        self,
        store,
        t1: datetime,
        t2: datetime,
    ) -> MemoryDiff:
        """Compute a diff between two absolute timestamps.

        Args:
            store: MemoryStore instance
            t1: Earlier timestamp
            t2: Later timestamp
        """
        if t1 > t2:
            t1, t2 = t2, t1
        return self._compute_diff(store, t1, t2)

    def _compute_diff(
        self,
        store,
        since: datetime,
        until: datetime,
    ) -> MemoryDiff:
        repo = store._repo
        since_iso = since.isoformat()
        until_iso = until.isoformat()

        # New memories created in range
        created = repo.list_created_between(since_iso, until_iso)
        new_memories = [
            MemoryChangeSummary(
                id=r.id,
                l3_abstract=r.l3_abstract,
                tags=r.tags,
                created_at=r.created_at,
                source=r.source,
            )
            for r in created
        ]

        # Updated memories (updated_at in range but created_at before range)
        updated_records = repo.list_updated_between(since_iso, until_iso)
        created_ids = {r.id for r in created}
        updated_memories: list[MemoryUpdate] = []
        for record in updated_records:
            if record.id in created_ids:
                continue  # Skip newly created ones
            # Look up the version that was saved just before or at 'since'
            version = repo.get_version_at(record.id, since)
            old_l3 = version.l3_abstract if version else record.l3_abstract
            old_tags = version.tags if version else record.tags
            updated_memories.append(
                MemoryUpdate(
                    id=record.id,
                    old_l3=old_l3,
                    new_l3=record.l3_abstract,
                    old_tags=old_tags,
                    new_tags=record.tags,
                    changed_at=record.updated_at,
                )
            )

        # Deleted memories (from audit_log)
        deleted_entries = repo.get_deleted_between(since_iso, until_iso)
        deleted_ids = [entry["memory_id"] for entry in deleted_entries if entry.get("memory_id")]

        # Compute stats
        all_new_tags: set[str] = set()
        for m in new_memories:
            all_new_tags.update(m.tags)
        for u in updated_memories:
            all_new_tags.update(set(u.new_tags) - set(u.old_tags))

        removed_tags: set[str] = set()
        for u in updated_memories:
            removed_tags.update(set(u.old_tags) - set(u.new_tags))

        stats = DiffStats(
            added=len(new_memories),
            modified=len(updated_memories),
            deleted=len(deleted_ids),
            topics_added=sorted(all_new_tags),
            topics_removed=sorted(removed_tags),
        )

        return MemoryDiff(
            since=since,
            until=until,
            new_memories=new_memories,
            updated_memories=updated_memories,
            deleted_ids=deleted_ids,
            stats=stats,
        )
