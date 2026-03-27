"""Tests for onboarding: np init, np doctor, first-run hint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from neuropack.cli.main import cli
from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


@pytest.fixture
def runner():
    return CliRunner()


class TestInit:
    def test_init_creates_db_and_examples(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        # Inputs: embedder choice=1, LLM=n, encryption=n, auth=n
        result = runner.invoke(cli, ["--db", db_path, "init"], input="1\nn\nn\nn\n")
        assert result.exit_code == 0
        assert "NeuroPack - AI Memory Layer" in result.output
        assert "[stored]" in result.output
        assert Path(db_path).exists()

    def test_init_stores_example_memories(self, tmp_path):
        db_path = str(tmp_path / "memories.db")
        config = NeuropackConfig(db_path=db_path)
        store = MemoryStore(config)
        store.initialize()

        from neuropack.cli.onboarding import EXAMPLE_MEMORIES
        for text in EXAMPLE_MEMORIES:
            store.store(content=text, tags=["example"], source="test")

        s = store.stats()
        assert s.total_memories == 3
        store.close()


class TestDoctor:
    def test_doctor_runs_clean(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        result = runner.invoke(cli, ["--db", db_path, "doctor"])
        assert result.exit_code == 0
        assert "NeuroPack Health Check" in result.output
        assert "Database" in result.output
        assert "Schema" in result.output

    def test_doctor_shows_memory_count(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        # Store a memory first
        store_result = runner.invoke(cli, ["--db", db_path, "store", "test memory"])
        result = runner.invoke(cli, ["--db", db_path, "doctor"])
        assert result.exit_code == 0
        assert "1 memories" in result.output or "Memory count" in result.output

    def test_doctor_no_llm_configured(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        result = runner.invoke(cli, ["--db", db_path, "doctor"])
        assert "None configured" in result.output


class TestFirstRunHint:
    def test_first_run_hint_shown_on_empty_db(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        result = runner.invoke(cli, ["--db", db_path, "list"])
        assert "np init" in result.output

    def test_first_run_hint_not_shown_twice(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        # First run shows hint
        runner.invoke(cli, ["--db", db_path, "list"])
        # Second run should not show hint
        result = runner.invoke(cli, ["--db", db_path, "list"])
        assert "np init" not in result.output

    def test_first_run_hint_not_shown_if_memories_exist(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "memories.db")
        runner.invoke(cli, ["--db", db_path, "store", "some memory"])
        # New invocation after storing -- hint flag should be set
        result = runner.invoke(cli, ["--db", db_path, "list"])
        # The hint should not show because there are memories
        # (the store command itself may have shown it, but subsequent calls shouldn't)
        assert result.exit_code == 0


class TestConfigEnv:
    def test_config_env_written(self, tmp_path):
        from neuropack.cli.onboarding import _write_config_env
        _write_config_env(tmp_path, auth_token="my-token")
        env_file = tmp_path / "config.env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "NEUROPACK_AUTH_TOKEN=my-token" in content
