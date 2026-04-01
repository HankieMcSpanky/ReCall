"""Notion import — import from Notion export (markdown/CSV).

Notion exports are ZIP files containing markdown files with metadata.
This module imports them as memories.

Usage:
    from neuropack.io.notion import NotionImporter

    importer = NotionImporter(store)
    count = importer.import_export("/path/to/notion-export.zip")
    # or from extracted directory:
    count = importer.import_directory("/path/to/notion-export/")
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


class NotionImporter:
    """Import memories from Notion export files."""

    def __init__(self, store):
        self._store = store

    def import_export(self, zip_path: str) -> int:
        """Import from a Notion export ZIP file. Returns count imported."""
        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".md"):
                    content = zf.read(name).decode("utf-8", errors="replace")
                    imported = self._import_markdown(content, source_file=name)
                    if imported:
                        count += 1
                elif name.endswith(".csv"):
                    content = zf.read(name).decode("utf-8", errors="replace")
                    count += self._import_csv(content, source_file=name)
        logger.info("Imported %d items from Notion export", count)
        return count

    def import_directory(self, dir_path: str) -> int:
        """Import from an extracted Notion export directory."""
        count = 0
        root = Path(dir_path)

        for md_file in root.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8", errors="replace")
            if self._import_markdown(content, source_file=str(md_file)):
                count += 1

        for csv_file in root.rglob("*.csv"):
            content = csv_file.read_text(encoding="utf-8", errors="replace")
            count += self._import_csv(content, source_file=str(csv_file))

        logger.info("Imported %d items from Notion directory", count)
        return count

    def _import_markdown(self, content: str, source_file: str = "") -> bool:
        """Import a single Notion markdown page as a memory."""
        content = content.strip()
        if not content or len(content) < 20:
            return False

        # Extract title (first heading)
        title = ""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Clean Notion-specific artifacts
        content = self._clean_notion_markdown(content)

        if len(content.strip()) < 20:
            return False

        # Extract tags from content
        tags = ["notion"]
        # Notion uses database properties that sometimes appear as key: value
        prop_matches = re.findall(r'^(\w+):\s*(.+)$', content, re.MULTILINE)
        for key, value in prop_matches[:5]:
            if key.lower() in ("tags", "category", "type", "status"):
                tags.extend(t.strip() for t in value.split(",") if t.strip())

        self._store.store(
            content=content,
            tags=tags,
            source="notion",
            l3_override=title if title else None,
        )
        return True

    def _import_csv(self, content: str, source_file: str = "") -> int:
        """Import a Notion database CSV export. Returns count imported."""
        count = 0
        reader = csv.DictReader(io.StringIO(content))

        for row in reader:
            # Combine all non-empty fields into content
            parts = []
            title = ""
            tags = ["notion"]

            for key, value in row.items():
                if not value or not value.strip():
                    continue
                if key.lower() in ("name", "title"):
                    title = value.strip()
                    parts.insert(0, f"# {value.strip()}")
                elif key.lower() in ("tags", "category"):
                    tags.extend(t.strip() for t in value.split(",") if t.strip())
                else:
                    parts.append(f"**{key}**: {value.strip()}")

            if not parts:
                continue

            text = "\n".join(parts)
            if len(text) < 20:
                continue

            self._store.store(
                content=text,
                tags=tags,
                source="notion",
                l3_override=title if title else None,
            )
            count += 1

        return count

    @staticmethod
    def _clean_notion_markdown(content: str) -> str:
        """Remove Notion-specific formatting artifacts."""
        # Remove Notion block IDs
        content = re.sub(r'\n[a-f0-9]{32}\n', '\n', content)
        # Remove empty links
        content = re.sub(r'\[([^\]]*)\]\(\)', r'\1', content)
        # Remove Notion callout syntax
        content = re.sub(r'^[💡📌⚠️ℹ️🔥]\s*', '', content, flags=re.MULTILINE)
        return content
