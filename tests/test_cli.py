"""Tests for Click CLI."""
import json

from click.testing import CliRunner

from neuropack.cli.main import cli


def test_store_and_recall(tmp_path):
    db = str(tmp_path / "test.db")
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", db, "store", "Python is great for scripting"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "id" in data

    result = runner.invoke(cli, ["--db", db, "recall", "Python scripting"])
    assert result.exit_code == 0
    assert "Python" in result.output


def test_list(tmp_path):
    db = str(tmp_path / "test.db")
    runner = CliRunner()
    runner.invoke(cli, ["--db", db, "store", "Memory one"])
    runner.invoke(cli, ["--db", db, "store", "Memory two, totally different content"])

    result = runner.invoke(cli, ["--db", db, "list"])
    assert result.exit_code == 0


def test_inspect(tmp_path):
    db = str(tmp_path / "test.db")
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", db, "store", "Inspectable memory content"])
    data = json.loads(result.output)
    mid = data["id"]

    result = runner.invoke(cli, ["--db", db, "inspect", mid])
    assert result.exit_code == 0
    detail = json.loads(result.output)
    assert detail["raw_text"] == "Inspectable memory content"


def test_forget(tmp_path):
    db = str(tmp_path / "test.db")
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", db, "store", "To be forgotten"])
    data = json.loads(result.output)
    mid = data["id"]

    result = runner.invoke(cli, ["--db", db, "forget", mid])
    assert result.exit_code == 0
    assert "Deleted" in result.output


def test_stats(tmp_path):
    db = str(tmp_path / "test.db")
    runner = CliRunner()
    runner.invoke(cli, ["--db", db, "store", "Stats test memory"])

    result = runner.invoke(cli, ["--db", db, "stats"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total_memories"] == 1


def test_forget_nonexistent(tmp_path):
    db = str(tmp_path / "test.db")
    runner = CliRunner()
    # Initialize db first
    runner.invoke(cli, ["--db", db, "stats"])
    result = runner.invoke(cli, ["--db", db, "forget", "nonexistent-id"])
    assert result.exit_code != 0
