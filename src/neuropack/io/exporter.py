from __future__ import annotations

import json
from pathlib import Path

from neuropack.types import MemoryRecord


def export_jsonl(records: list[MemoryRecord], file_path: str) -> None:
    """Export memories as JSONL (one JSON object per line)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for r in records:
        obj = {
            "id": r.id,
            "content": r.content,
            "l3_abstract": r.l3_abstract,
            "l2_facts": r.l2_facts,
            "tags": r.tags,
            "source": r.source,
            "priority": r.priority,
            "namespace": r.namespace,
            "created_at": r.created_at.isoformat(),
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_markdown(records: list[MemoryRecord], dir_path: str) -> None:
    """Export memories as markdown files with YAML frontmatter."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

    for r in records:
        frontmatter = "---\n"
        frontmatter += f"id: {r.id}\n"
        frontmatter += f"tags: [{', '.join(r.tags)}]\n"
        frontmatter += f"priority: {r.priority}\n"
        frontmatter += f"source: {r.source}\n"
        frontmatter += f"created_at: {r.created_at.isoformat()}\n"
        frontmatter += f"l3_abstract: {r.l3_abstract}\n"
        frontmatter += "---\n\n"

        content = frontmatter + r.content
        file_path = path / f"{r.id}.md"
        file_path.write_text(content, encoding="utf-8")


def export_json(records: list[MemoryRecord], file_path: str) -> None:
    """Export memories as a single JSON array."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "id": r.id,
            "content": r.content,
            "l3_abstract": r.l3_abstract,
            "l2_facts": r.l2_facts,
            "tags": r.tags,
            "source": r.source,
            "priority": r.priority,
            "namespace": r.namespace,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
