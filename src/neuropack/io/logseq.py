"""Logseq sync — bidirectional sync with Logseq graphs.

Logseq uses markdown files in a `pages/` directory with bullet-point format.
Each memory becomes a page, and pages can be imported back as memories.

Usage:
    from neuropack.io.logseq import LogseqSync

    sync = LogseqSync("/path/to/logseq-graph", store)
    sync.sync_to_graph()     # export memories to Logseq
    sync.sync_from_graph()   # import Logseq pages as memories
"""

from __future__ import annotations

import re
from pathlib import Path


class LogseqSync:
    """Bidirectional sync with Logseq graphs."""

    def __init__(self, graph_path: str, store, folder: str = "ReCall"):
        self._graph = Path(graph_path)
        self._store = store
        self._folder = folder

    @property
    def _pages_dir(self) -> Path:
        return self._graph / "pages"

    @property
    def _sync_dir(self) -> Path:
        return self._pages_dir / self._folder

    def sync_to_graph(self, tags: list[str] | None = None, limit: int = 100) -> int:
        """Export memories to Logseq as markdown pages. Returns count exported."""
        self._sync_dir.mkdir(parents=True, exist_ok=True)

        tag = tags[0] if tags and len(tags) == 1 else None
        records = self._store.list(limit=limit, tag=tag)
        if tags and len(tags) > 1:
            records = [r for r in records if any(t in r.tags for t in tags)]

        count = 0
        for r in records:
            md = self._memory_to_logseq(r)
            # Logseq uses page titles as filenames
            safe_title = re.sub(r'[^\w\s-]', '', r.l3_abstract or r.id)[:60].strip()
            safe_title = safe_title.replace(' ', '_') or r.id
            file_path = self._sync_dir / f"{safe_title}____{r.id[:8]}.md"
            file_path.write_text(md, encoding="utf-8")
            count += 1

        return count

    def sync_from_graph(self) -> int:
        """Import Logseq pages from the sync folder as memories. Returns count imported."""
        if not self._sync_dir.exists():
            return 0

        count = 0
        for f in sorted(self._sync_dir.glob("*.md")):
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                continue

            # Extract content from Logseq bullet format
            lines = []
            tags = ["logseq"]
            for line in content.split("\n"):
                stripped = line.strip()
                # Skip Logseq metadata
                if stripped.startswith("tags::"):
                    raw_tags = stripped.replace("tags::", "").strip()
                    tags.extend(t.strip().strip("#") for t in raw_tags.split(",") if t.strip())
                    continue
                # Remove bullet prefixes
                if stripped.startswith("- "):
                    stripped = stripped[2:]
                if stripped and not stripped.startswith("id::") and not stripped.startswith("collapsed::"):
                    lines.append(stripped)

            if not lines:
                continue

            text = "\n".join(lines)
            self._store.store(
                content=text,
                tags=tags,
                source="logseq",
            )
            count += 1

        return count

    @staticmethod
    def _memory_to_logseq(record) -> str:
        """Convert a memory record to Logseq page format."""
        lines = []

        # Title as first bullet
        lines.append(f"- **{record.l3_abstract or 'Memory'}**")
        lines.append(f"  id:: {record.id}")

        # Tags
        if record.tags:
            tag_str = ", ".join(f"#{t}" for t in record.tags)
            lines.append(f"  tags:: {tag_str}")

        # Key facts as sub-bullets
        if record.l2_facts:
            for fact in record.l2_facts:
                lines.append(f"  - {fact}")

        # Content as collapsed block
        if record.content:
            lines.append(f"  - Full content")
            lines.append(f"    collapsed:: true")
            for content_line in record.content.split("\n")[:20]:
                if content_line.strip():
                    lines.append(f"    - {content_line.strip()}")

        # Metadata
        lines.append(f"  - type:: {record.memory_type or 'general'}")
        lines.append(f"  - staleness:: {record.staleness or 'stable'}")

        return "\n".join(lines)
