"""Structured data connectors: CSV and JSON array import."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def parse_csv(
    path: str,
    content_column: str = "content",
    tag_columns: list[str] | None = None,
    source_column: str | None = None,
) -> list[dict]:
    """Parse a CSV file into memory-ready dicts.

    Args:
        path: Path to CSV file.
        content_column: Column name containing the text content.
        tag_columns: Column names to extract as tags.
        source_column: Column name to use as source.

    Returns list of dicts with keys: content, tags, source, priority.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    memories: list[dict] = []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            content = row.get(content_column, "").strip()
            if not content or len(content) < 5:
                continue

            tags: list[str] = ["csv", "imported"]
            if tag_columns:
                for tc in tag_columns:
                    val = row.get(tc, "").strip()
                    if val:
                        tags.append(val)

            source = "csv"
            if source_column and row.get(source_column):
                source = f"csv:{row[source_column].strip()}"

            memories.append({
                "content": content,
                "tags": tags,
                "source": source,
                "priority": 0.5,
            })

    return memories


def parse_json_array(
    path: str,
    content_field: str = "content",
    tag_field: str | None = "tags",
    source_field: str | None = "source",
) -> list[dict]:
    """Parse a JSON array file into memory-ready dicts.

    Expects a JSON file with a top-level array of objects.

    Args:
        path: Path to JSON file.
        content_field: Field name for text content.
        tag_field: Field name for tags (list of strings).
        source_field: Field name for source.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]

    memories: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        content = item.get(content_field, "")
        if not isinstance(content, str) or len(content.strip()) < 5:
            continue

        tags: list[str] = ["json", "imported"]
        if tag_field and isinstance(item.get(tag_field), list):
            tags.extend(str(t) for t in item[tag_field])

        source = "json"
        if source_field and item.get(source_field):
            source = str(item[source_field])

        memories.append({
            "content": content.strip(),
            "tags": tags,
            "source": source,
            "priority": item.get("priority", 0.5),
        })

    return memories
