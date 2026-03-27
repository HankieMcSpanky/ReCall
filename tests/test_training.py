from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.io.training import (
    export_openai_finetune,
    export_alpaca,
    export_knowledge_qa,
    export_embeddings_dataset,
)


@pytest.fixture
def train_store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


def test_openai_format(train_store, tmp_path):
    train_store.store("Python is a programming language.", tags=["tech"])
    train_store.store("SQLite is a database engine.", tags=["tech"])

    records = train_store.list()
    path = str(tmp_path / "openai.jsonl")
    export_openai_finetune(records, path)

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 2

    obj = json.loads(lines[0])
    assert "messages" in obj
    assert len(obj["messages"]) == 3
    assert obj["messages"][0]["role"] == "system"
    assert obj["messages"][1]["role"] == "user"
    assert obj["messages"][2]["role"] == "assistant"


def test_alpaca_format(train_store, tmp_path):
    train_store.store("Neural networks learn from data.", tags=["ml"])

    records = train_store.list()
    path = str(tmp_path / "alpaca.jsonl")
    export_alpaca(records, path)

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 1

    obj = json.loads(lines[0])
    assert "instruction" in obj
    assert "input" in obj
    assert "output" in obj
    assert obj["instruction"].startswith("Recall: ")


def test_knowledge_qa(train_store, tmp_path):
    train_store.store("Python has list comprehensions.", tags=["python"])
    train_store.store("Python supports decorators.", tags=["python"])
    train_store.store("SQLite uses B-trees.", tags=["database"])

    records = train_store.list()
    path = str(tmp_path / "qa.jsonl")
    export_knowledge_qa(records, path)

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) >= 1  # At least one tag group

    for line in lines:
        obj = json.loads(line)
        assert "messages" in obj
        assert obj["messages"][0]["role"] == "user"
        assert "What do you know about" in obj["messages"][0]["content"]


def test_embeddings_dataset(train_store, tmp_path):
    train_store.store("Text for embeddings training.", tags=["ml"])

    records = train_store.list()
    path = str(tmp_path / "embeddings.jsonl")
    export_embeddings_dataset(records, path)

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 1

    obj = json.loads(lines[0])
    assert "text" in obj
    assert "label" in obj
    assert obj["label"] == "ml"


def test_empty_records(tmp_path):
    path = str(tmp_path / "empty.jsonl")
    export_openai_finetune([], path)
    content = Path(path).read_text()
    assert content.strip() == ""


def test_export_training_via_store(train_store, tmp_path):
    train_store.store("Store export test data.", tags=["test"])

    path = str(tmp_path / "train.jsonl")
    count = train_store.export_training(format="openai", path=path)
    assert count == 1

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 1


def test_tag_filtering(train_store, tmp_path):
    train_store.store("Python data for export", tags=["python"])
    train_store.store("Java data for export", tags=["java"])

    path = str(tmp_path / "filtered.jsonl")
    count = train_store.export_training(format="openai", path=path, tags=["python"])
    assert count == 1


def test_invalid_format(train_store, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        train_store.export_training(format="invalid", path=str(tmp_path / "bad.jsonl"))
