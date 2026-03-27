from __future__ import annotations

from pathlib import Path

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.io.obsidian import ObsidianSync


@pytest.fixture
def obs_store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def vault_path(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return str(vault)


def test_sync_to_vault(obs_store, vault_path):
    obs_store.store("Memory for Obsidian sync test", tags=["test"])
    obs_store.store("Another memory for vault", tags=["test"])

    sync = ObsidianSync(vault_path=vault_path, store=obs_store)
    count = sync.sync_to_vault()

    assert count == 2
    files = list(Path(vault_path, "NeuroPack").glob("*.md"))
    assert len(files) == 2


def test_sync_from_vault(obs_store, vault_path):
    # Create markdown files in vault
    sync_dir = Path(vault_path) / "NeuroPack"
    sync_dir.mkdir()

    (sync_dir / "note1.md").write_text(
        "---\ntags: [vault, note]\npriority: 0.7\n---\n\nThis is a vault note for testing."
    )
    (sync_dir / "note2.md").write_text(
        "A simple note without frontmatter content for import."
    )

    sync = ObsidianSync(vault_path=vault_path, store=obs_store)
    count = sync.sync_from_vault()

    assert count == 2
    assert obs_store.stats().total_memories == 2


def test_full_sync(obs_store, vault_path):
    obs_store.store("Existing memory for bidirectional sync", tags=["test"])

    # Create a vault note too
    sync_dir = Path(vault_path) / "NeuroPack"
    sync_dir.mkdir()
    (sync_dir / "vault_note.md").write_text(
        "---\ntags: [vault]\n---\n\nVault-originated note for testing."
    )

    sync = ObsidianSync(vault_path=vault_path, store=obs_store)
    result = sync.full_sync()

    assert result["exported"] >= 1
    assert result["imported"] >= 1


def test_yaml_frontmatter_parsing(obs_store, vault_path):
    sync_dir = Path(vault_path) / "NeuroPack"
    sync_dir.mkdir()

    (sync_dir / "fm.md").write_text(
        "---\ntags: [python, dev]\npriority: 0.9\nsource: obsidian\n---\n\nPython development notes from Obsidian vault."
    )

    sync = ObsidianSync(vault_path=vault_path, store=obs_store)
    count = sync.sync_from_vault()

    assert count == 1
    records = obs_store.list()
    assert len(records) == 1
    assert "python" in records[0].tags or "dev" in records[0].tags


def test_skip_existing_ids(obs_store, vault_path):
    record = obs_store.store("Already exists in store", tags=["test"])

    sync_dir = Path(vault_path) / "NeuroPack"
    sync_dir.mkdir()

    # Create a file with the same ID
    (sync_dir / f"{record.id}.md").write_text(
        f"---\nid: {record.id}\ntags: [test]\n---\n\nAlready exists in store"
    )

    sync = ObsidianSync(vault_path=vault_path, store=obs_store)
    count = sync.sync_from_vault()

    assert count == 0  # Should skip because ID exists


def test_empty_vault(obs_store, vault_path):
    sync = ObsidianSync(vault_path=vault_path, store=obs_store)
    count = sync.sync_from_vault()
    assert count == 0
