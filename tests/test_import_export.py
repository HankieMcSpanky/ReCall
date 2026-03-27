from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.io.importer import parse_chatgpt_export, parse_markdown_files, parse_jsonl
from neuropack.io.exporter import export_jsonl, export_markdown, export_json


@pytest.fixture
def io_store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


def test_parse_chatgpt_export(tmp_path):
    data = [
        {
            "title": "Test Conv",
            "mapping": {
                "1": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["This is a helpful response about Python programming."]},
                    }
                },
                "2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Tell me about Python"]},
                    }
                },
            },
        }
    ]
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(data))

    memories = parse_chatgpt_export(str(path))
    assert len(memories) == 1
    assert "chatgpt" in memories[0]["tags"]
    assert "Python" in memories[0]["content"]


def test_parse_markdown_files(tmp_path):
    md_dir = tmp_path / "notes"
    md_dir.mkdir()

    (md_dir / "note1.md").write_text("---\ntags: [python, dev]\npriority: 0.8\n---\n\nPython is great for scripting.")
    (md_dir / "note2.md").write_text("Simple note without frontmatter. This is a test note.")

    memories = parse_markdown_files(str(md_dir))
    assert len(memories) == 2
    assert any(m["priority"] == 0.8 for m in memories)


def test_parse_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"content": "Memory one is about testing.", "tags": ["test"]}),
        json.dumps({"content": "Memory two is about coding.", "source": "dev"}),
    ]
    path.write_text("\n".join(lines))

    memories = parse_jsonl(str(path))
    assert len(memories) == 2
    assert memories[0]["tags"] == ["test"]


def test_export_jsonl(io_store, tmp_path):
    io_store.store("Export test memory one", tags=["export"])
    io_store.store("Export test memory two", tags=["export"])

    records = io_store.list()
    path = str(tmp_path / "export.jsonl")
    export_jsonl(records, path)

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert "content" in obj
    assert "l3_abstract" in obj


def test_export_markdown(io_store, tmp_path):
    io_store.store("Markdown export test memory", tags=["md"])

    records = io_store.list()
    dir_path = str(tmp_path / "md_export")
    export_markdown(records, dir_path)

    files = list(Path(dir_path).glob("*.md"))
    assert len(files) == 1

    content = files[0].read_text()
    assert "---" in content  # frontmatter
    assert "Markdown export test memory" in content


def test_export_json(io_store, tmp_path):
    io_store.store("JSON export test", tags=["json"])

    records = io_store.list()
    path = str(tmp_path / "export.json")
    export_json(records, path)

    data = json.loads(Path(path).read_text())
    assert isinstance(data, list)
    assert len(data) == 1


def test_round_trip_jsonl(io_store, tmp_path):
    io_store.store("Round trip test content for verification", tags=["roundtrip"])

    records = io_store.list()
    path = str(tmp_path / "roundtrip.jsonl")
    export_jsonl(records, path)

    imported = parse_jsonl(path)
    assert len(imported) == 1
    assert "Round trip test" in imported[0]["content"]


def test_import_memories_via_store(io_store, tmp_path):
    path = tmp_path / "import.jsonl"
    lines = [
        json.dumps({"content": "Imported memory about testing.", "tags": ["test"]}),
        json.dumps({"content": "Imported memory about coding.", "tags": ["dev"]}),
    ]
    path.write_text("\n".join(lines))

    count = io_store.import_memories(format="jsonl", path=str(path))
    assert count == 2
    assert io_store.stats().total_memories == 2


def test_export_memories_via_store(io_store, tmp_path):
    io_store.store("Python is a programming language with dynamic typing and garbage collection", tags=["export"])
    io_store.store("SQLite is a lightweight database engine used for local storage", tags=["export"])

    path = str(tmp_path / "out.jsonl")
    count = io_store.export_memories(format="jsonl", path=path)
    assert count == 2
