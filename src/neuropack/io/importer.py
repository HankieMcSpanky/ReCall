from __future__ import annotations

import json
from pathlib import Path


def parse_chatgpt_export(file_path: str) -> list[dict]:
    """Parse ChatGPT data export (conversations.json).

    Extracts assistant messages as memories.
    """
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    memories: list[dict] = []

    # ChatGPT export is a list of conversations
    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        title = conv.get("title", "")
        mapping = conv.get("mapping", {})

        for node in mapping.values():
            msg = node.get("message")
            if msg is None:
                continue
            if msg.get("author", {}).get("role") != "assistant":
                continue

            parts = msg.get("content", {}).get("parts", [])
            text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
            if not text or len(text) < 10:
                continue

            memories.append({
                "content": text,
                "tags": ["chatgpt", "imported"],
                "source": f"chatgpt:{title}" if title else "chatgpt",
                "priority": 0.5,
            })

    return memories


def parse_claude_export(file_path: str) -> list[dict]:
    """Parse Claude's JSON export format.

    Handles both conversation array format and single conversation.
    """
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    memories: list[dict] = []

    # Normalize to list
    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        # Claude exports can have different structures
        messages = conv.get("chat_messages", conv.get("messages", []))
        title = conv.get("name", conv.get("title", ""))

        for msg in messages:
            # Claude format uses "sender" or "role"
            role = msg.get("sender", msg.get("role", ""))
            if role not in ("assistant", "AI"):
                continue

            # Content can be string or structured
            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", str(p)) for p in content
                    if isinstance(p, (str, dict))
                )
            if not isinstance(content, str) or len(content.strip()) < 10:
                continue

            memories.append({
                "content": content.strip(),
                "tags": ["claude", "imported"],
                "source": f"claude:{title}" if title else "claude",
                "priority": 0.5,
            })

    return memories


def parse_markdown_files(path: str) -> list[dict]:
    """Parse markdown files from a directory or single file.

    Each file becomes one memory. YAML frontmatter parsed for tags/priority.
    """
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("**/*.md"))

    memories: list[dict] = []

    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue

        tags: list[str] = ["markdown", "imported"]
        priority = 0.5
        source = f.stem
        content = text

        # Parse YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                content = parts[2].strip()

                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("tags:"):
                        val = line[5:].strip().strip("[]")
                        if val:
                            tags.extend(t.strip().strip("'\"") for t in val.split(",") if t.strip())
                    elif line.startswith("priority:"):
                        try:
                            priority = float(line[9:].strip())
                        except ValueError:
                            pass
                    elif line.startswith("source:"):
                        source = line[7:].strip()

        if content and len(content) >= 5:
            memories.append({
                "content": content,
                "tags": tags,
                "source": source,
                "priority": priority,
            })

    return memories


def parse_jsonl(file_path: str) -> list[dict]:
    """Parse JSONL file. One JSON object per line.

    Expected format: {content, tags?, source?, priority?}
    """
    path = Path(file_path)
    memories: list[dict] = []

    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        content = obj.get("content", "")
        if not content or len(content) < 5:
            continue

        memories.append({
            "content": content,
            "tags": obj.get("tags", ["imported"]),
            "source": obj.get("source", "jsonl"),
            "priority": obj.get("priority", 0.5),
        })

    return memories
