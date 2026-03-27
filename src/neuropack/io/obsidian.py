from __future__ import annotations

import re
from pathlib import Path

from neuropack.types import MemoryRecord


class ObsidianSync:
    """Bidirectional sync with Obsidian vaults."""

    def __init__(self, vault_path: str, store, folder: str = "NeuroPack"):
        self._vault = Path(vault_path)
        self._store = store
        self._folder = folder

    @property
    def _sync_dir(self) -> Path:
        return self._vault / self._folder

    def sync_to_vault(self, tags: list[str] | None = None, limit: int = 100) -> int:
        """Write memories to vault as markdown files. Returns count exported."""
        self._sync_dir.mkdir(parents=True, exist_ok=True)

        tag = tags[0] if tags and len(tags) == 1 else None
        records = self._store.list(limit=limit, tag=tag)
        if tags and len(tags) > 1:
            records = [r for r in records if any(t in r.tags for t in tags)]

        count = 0
        for r in records:
            md = self._memory_to_md(r)
            file_path = self._sync_dir / f"{r.id}.md"
            file_path.write_text(md, encoding="utf-8")
            count += 1

        return count

    def sync_from_vault(self) -> int:
        """Import markdown files from vault folder. Returns count imported."""
        if not self._sync_dir.exists():
            return 0

        count = 0
        for f in sorted(self._sync_dir.glob("*.md")):
            parsed = self._md_to_memory(f)
            if parsed is None:
                continue

            # Skip if memory ID already exists
            existing_id = parsed.get("id")
            if existing_id and self._store.get(existing_id) is not None:
                continue

            self._store.store(
                content=parsed["content"],
                tags=parsed.get("tags", ["obsidian"]),
                source=parsed.get("source", "obsidian"),
                priority=parsed.get("priority", 0.5),
            )
            count += 1

        return count

    def full_sync(self) -> dict:
        """Bidirectional sync. Returns {exported, imported}."""
        exported = self.sync_to_vault()
        imported = self.sync_from_vault()
        return {"exported": exported, "imported": imported}

    def _memory_to_md(self, record: MemoryRecord) -> str:
        """Convert a memory record to markdown with YAML frontmatter."""
        fm = "---\n"
        fm += f"id: {record.id}\n"
        fm += f"tags: [{', '.join(record.tags)}]\n"
        fm += f"priority: {record.priority}\n"
        fm += f"source: {record.source}\n"
        fm += f"created_at: {record.created_at.isoformat()}\n"
        fm += f"l3_abstract: {record.l3_abstract}\n"
        fm += "---\n\n"
        return fm + record.content

    def _md_to_memory(self, file_path: Path) -> dict | None:
        """Parse a markdown file with frontmatter into a memory dict."""
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return None

        tags: list[str] = ["obsidian"]
        priority = 0.5
        source = "obsidian"
        memory_id = None
        content = text

        # Parse YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                content = parts[2].strip()

                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("id:"):
                        memory_id = line[3:].strip()
                    elif line.startswith("tags:"):
                        val = line[5:].strip().strip("[]")
                        if val:
                            tags = [t.strip().strip("'\"") for t in val.split(",") if t.strip()]
                            if "obsidian" not in tags:
                                tags.append("obsidian")
                    elif line.startswith("priority:"):
                        try:
                            priority = float(line[9:].strip())
                        except ValueError:
                            pass
                    elif line.startswith("source:"):
                        source = line[7:].strip()

        if not content or len(content) < 5:
            return None

        result: dict = {
            "content": content,
            "tags": tags,
            "source": source,
            "priority": priority,
        }
        if memory_id:
            result["id"] = memory_id

        return result
