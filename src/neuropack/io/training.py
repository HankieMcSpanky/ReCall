from __future__ import annotations

import json
from pathlib import Path

from neuropack.types import MemoryRecord


def export_openai_finetune(records: list[MemoryRecord], file_path: str) -> None:
    """Export as OpenAI fine-tuning JSONL format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for r in records:
        if not r.l3_abstract or not r.content:
            continue
        obj = {
            "messages": [
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": r.l3_abstract},
                {"role": "assistant", "content": r.content},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def export_alpaca(records: list[MemoryRecord], file_path: str) -> None:
    """Export as Alpaca instruction-tuning JSONL format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for r in records:
        if not r.l3_abstract or not r.content:
            continue
        obj = {
            "instruction": f"Recall: {r.l3_abstract}",
            "input": "",
            "output": r.content,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def export_knowledge_qa(records: list[MemoryRecord], file_path: str) -> None:
    """Export as Q&A pairs from L2 facts, grouped by tag."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group facts by primary tag
    tag_facts: dict[str, list[str]] = {}
    for r in records:
        primary_tag = r.tags[0] if r.tags else "general"
        if primary_tag not in tag_facts:
            tag_facts[primary_tag] = []
        tag_facts[primary_tag].extend(r.l2_facts)

    lines = []
    for tag, facts in tag_facts.items():
        if not facts:
            continue
        obj = {
            "messages": [
                {"role": "user", "content": f"What do you know about {tag}?"},
                {"role": "assistant", "content": " ".join(facts)},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def export_embeddings_dataset(records: list[MemoryRecord], file_path: str) -> None:
    """Export for embedding fine-tuning (text + label pairs)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for r in records:
        if not r.content:
            continue
        label = r.tags[0] if r.tags else "general"
        obj = {
            "text": r.content,
            "label": label,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
