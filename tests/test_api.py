"""Tests for FastAPI HTTP endpoints."""
import pytest
from fastapi.testclient import TestClient

from neuropack.api.app import create_app
from neuropack.config import NeuropackConfig


@pytest.fixture
def app(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"), auth_token="test-secret")
    app = create_app(config)
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-secret"}


def test_health_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_store_requires_auth(client):
    resp = client.post("/v1/memories", json={"content": "test"})
    assert resp.status_code == 401


def test_store_and_get(client, auth_headers):
    resp = client.post(
        "/v1/memories",
        json={"content": "User loves Python", "tags": ["prefs"]},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["l3_abstract"]

    # Get it back
    resp = client.get(f"/v1/memories/{data['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["content"] == "User loves Python"


def test_recall(client, auth_headers):
    client.post(
        "/v1/memories",
        json={"content": "Python is the best programming language for data science"},
        headers=auth_headers,
    )
    resp = client.post(
        "/v1/recall",
        json={"query": "programming language", "limit": 5},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1


def test_list_memories(client, auth_headers):
    client.post("/v1/memories", json={"content": "Memory one"}, headers=auth_headers)
    client.post("/v1/memories", json={"content": "Memory two"}, headers=auth_headers)
    resp = client.get("/v1/memories", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_update_memory(client, auth_headers):
    resp = client.post(
        "/v1/memories",
        json={"content": "Original", "tags": ["v1"]},
        headers=auth_headers,
    )
    mid = resp.json()["id"]

    resp = client.patch(
        f"/v1/memories/{mid}",
        json={"tags": ["v2"], "priority": 0.9},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["tags"] == ["v2"]
    assert resp.json()["priority"] == 0.9


def test_delete_memory(client, auth_headers):
    resp = client.post("/v1/memories", json={"content": "To delete"}, headers=auth_headers)
    mid = resp.json()["id"]

    resp = client.delete(f"/v1/memories/{mid}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp = client.get(f"/v1/memories/{mid}", headers=auth_headers)
    assert resp.status_code == 404


def test_delete_nonexistent(client, auth_headers):
    resp = client.delete("/v1/memories/nonexistent", headers=auth_headers)
    assert resp.status_code == 404


def test_stats(client, auth_headers):
    resp = client.get("/v1/stats", headers=auth_headers)
    assert resp.status_code == 200
    assert "total_memories" in resp.json()


def test_batch_store(client, auth_headers):
    resp = client.post(
        "/v1/memories/batch",
        json={"items": [
            {"content": "Batch item 1"},
            {"content": "Batch item 2"},
            {"content": "Batch item 3"},
        ]},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    assert len(resp.json()) == 3


def test_caller_provided_l3_l2(client, auth_headers):
    resp = client.post(
        "/v1/memories",
        json={
            "content": "Some content",
            "l3": "Custom abstract",
            "l2": ["Fact 1", "Fact 2"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["l3_abstract"] == "Custom abstract"
    assert data["l2_facts"] == ["Fact 1", "Fact 2"]


def test_no_auth_when_token_empty(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "noauth.db"), auth_token="")
    app = create_app(config)
    with TestClient(app) as client:
        resp = client.post("/v1/memories", json={"content": "No auth needed"})
        assert resp.status_code == 201
