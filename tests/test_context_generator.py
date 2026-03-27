"""Tests for context file generation (CLAUDE.md)."""
import pytest
from pathlib import Path

from neuropack.config import NeuropackConfig
from neuropack.core.context_generator import generate_context_markdown
from neuropack.core.store import MemoryStore


def test_generate_empty():
    md = generate_context_markdown([])
    assert "# NeuroPack Memory Context" in md
    assert "0 memories indexed" in md


def test_generate_with_memories(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        store.store(content="Python is a programming language", tags=["tech", "python"])
        store.store(content="Tokyo is the capital of Japan", tags=["travel", "japan"])
        md = store.generate_context(limit=50)
        assert "## Recent Activity" in md
        assert "## Tags" in md
    finally:
        store.close()


def test_tag_cloud(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        store.store(content="First memory", tags=["tech"])
        store.store(content="Second memory about technology", tags=["tech"])
        store.store(content="Travel memory", tags=["travel"])
        md = store.generate_context()
        assert "`tech` (2)" in md
        assert "`travel` (1)" in md
    finally:
        store.close()


def test_key_facts(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        store.store(
            content="Python is great for AI development. It has many libraries for machine learning.",
            tags=["tech"],
            priority=0.9,
        )
        md = store.generate_context()
        assert "## Key Facts" in md
    finally:
        store.close()


def test_custom_title():
    md = generate_context_markdown([], title="My Custom Title")
    assert "# My Custom Title" in md


def test_cli_generate_context(tmp_path):
    from click.testing import CliRunner
    from neuropack.cli.main import cli

    db_path = str(tmp_path / "test.db")
    output_path = str(tmp_path / "CLAUDE.md")
    runner = CliRunner()

    # Store a memory first
    runner.invoke(cli, ["--db", db_path, "store", "Test memory for context generation", "-t", "test"])

    # Generate context
    result = runner.invoke(cli, ["--db", db_path, "generate-context", "-o", output_path])
    assert result.exit_code == 0
    assert Path(output_path).exists()
    content = Path(output_path).read_text(encoding="utf-8")
    assert "NeuroPack Memory Context" in content


def test_generate_context_with_tag_filter(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        store.store(content="Tech memory about Python programming", tags=["tech"])
        store.store(content="Travel memory about Paris in spring", tags=["travel"])
        md = store.generate_context(tags=["tech"])
        # Should include tech memory
        assert "tech" in md.lower() or "python" in md.lower() or "programming" in md.lower()
    finally:
        store.close()
