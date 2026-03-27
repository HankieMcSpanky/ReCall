"""Tests for multi-agent learning commands."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from neuropack.cli.agents import _auto_tag
from neuropack.cli.main import cli
from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


@pytest.fixture
def runner():
    return CliRunner()


class TestAutoTag:
    def test_mistake_detected(self):
        assert _auto_tag("Shorted BTC at 65k, lost 2% - mistake in timing") == "mistake"

    def test_win_detected(self):
        assert _auto_tag("Longed ETH at 3200, gained 5% on breakout") == "win"

    def test_observation_default(self):
        assert _auto_tag("Market is consolidating around 60k support level") == "observation"

    def test_error_is_mistake(self):
        assert _auto_tag("Error in position sizing calculation") == "mistake"

    def test_profit_is_win(self):
        assert _auto_tag("Made a profit on the trade today") == "win"


class TestAgentCreate:
    def test_create_agent(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        result = runner.invoke(cli, ["--db", db_path, "agent", "create", "trader1"])
        assert result.exit_code == 0
        assert "Created agent 'trader1'" in result.output

    def test_create_duplicate_agent(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        runner.invoke(cli, ["--db", db_path, "agent", "create", "trader1"])
        result = runner.invoke(cli, ["--db", db_path, "agent", "create", "trader1"])
        assert "already exists" in result.output


class TestAgentLog:
    def test_log_auto_tags_mistake(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot1"])
        result = runner.invoke(
            cli, ["--db", db_path, "agent", "log", "bot1", "Lost 3% on bad entry"]
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["tag"] == "mistake"

    def test_log_auto_tags_win(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot1"])
        result = runner.invoke(
            cli, ["--db", db_path, "agent", "log", "bot1", "Gained 5% on momentum trade"]
        )
        output = json.loads(result.output)
        assert output["tag"] == "win"


class TestAgentList:
    def test_list_agents(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot1"])
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot2"])
        result = runner.invoke(cli, ["--db", db_path, "agent", "list"])
        assert result.exit_code == 0
        agents = json.loads(result.output)
        names = [a["name"] for a in agents]
        assert "bot1" in names
        assert "bot2" in names

    def test_list_empty(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        result = runner.invoke(cli, ["--db", db_path, "agent", "list"])
        assert "No agents" in result.output


class TestAgentScoreboard:
    def test_scoreboard(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot1"])
        runner.invoke(cli, ["--db", db_path, "agent", "log", "bot1", "Won big profit today"])
        runner.invoke(cli, ["--db", db_path, "agent", "log", "bot1", "Made a loss on trade"])
        result = runner.invoke(cli, ["--db", db_path, "agent", "scoreboard"])
        assert result.exit_code == 0
        board = json.loads(result.output)
        assert len(board) == 1
        assert board[0]["agent"] == "bot1"
        assert board[0]["wins"] == 1
        assert board[0]["mistakes"] == 1
        assert board[0]["win_ratio"] == 0.5


class TestAgentShare:
    def test_share_to_shared_namespace(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot1"])
        log_result = runner.invoke(
            cli, ["--db", db_path, "agent", "log", "bot1", "Great win on BTC trade"]
        )
        memory_id = json.loads(log_result.output)["id"]
        share_result = runner.invoke(
            cli, ["--db", db_path, "agent", "share", "bot1", memory_id]
        )
        assert share_result.exit_code == 0
        output = json.loads(share_result.output)
        assert output["namespace"] == "shared"


class TestAgentLearn:
    def test_learn_from_shared(self, tmp_path, runner: CliRunner):
        db_path = str(tmp_path / "test.db")
        # Create two agents
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot1"])
        runner.invoke(cli, ["--db", db_path, "agent", "create", "bot2"])
        # bot1 logs and shares
        log_result = runner.invoke(
            cli, ["--db", db_path, "agent", "log", "bot1", "Profit from momentum strategy"]
        )
        memory_id = json.loads(log_result.output)["id"]
        runner.invoke(cli, ["--db", db_path, "agent", "share", "bot1", memory_id])
        # bot2 learns
        result = runner.invoke(cli, ["--db", db_path, "agent", "learn", "bot2"])
        assert result.exit_code == 0
        # bot2 should see the shared memory (not from bot1's agent source)
