"""Tests for mobile access: workspace endpoints, chat, mobile UI, PWA routes."""
import pytest
from fastapi.testclient import TestClient

from neuropack.api.app import create_app
from neuropack.config import NeuropackConfig


@pytest.fixture
def app(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"), auth_token="test-secret")
    return create_app(config)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth():
    return {"Authorization": "Bearer test-secret"}


# --- Mobile UI & PWA (all public, no auth required) ---


def test_mobile_html(client):
    resp = client.get("/mobile")
    assert resp.status_code == 200
    assert "NeuroPack" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_mobile_html_no_auth_needed(client):
    """Mobile page is served without auth headers."""
    resp = client.get("/mobile")
    assert resp.status_code == 200


def test_pwa_manifest(client):
    resp = client.get("/mobile/manifest.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "NeuroPack"
    assert data["start_url"] == "/mobile"
    assert data["display"] == "standalone"
    assert "scope" in data
    assert "orientation" in data


def test_service_worker(client):
    resp = client.get("/mobile/sw.js")
    assert resp.status_code == 200
    assert "application/javascript" in resp.headers["content-type"]
    assert "np-mobile-v" in resp.text


def test_pwa_icon(client):
    resp = client.get("/mobile/icon.svg")
    assert resp.status_code == 200
    assert "image/svg+xml" in resp.headers["content-type"]


def test_pwa_routes_no_auth(client):
    """All PWA routes must be accessible without auth."""
    for path in ["/mobile", "/mobile/manifest.json", "/mobile/sw.js", "/mobile/icon.svg"]:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"


# --- Workspace endpoints ---


def test_list_workspaces_empty(client, auth):
    resp = client.get("/v1/workspaces", headers=auth)
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_create_workspace(client, auth):
    resp = client.post(
        "/v1/workspaces",
        json={"name": "Test WS", "goal": "Test goal"},
        headers=auth,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Test WS"
    assert data["status"] == "active"
    assert "id" in data


def test_list_workspaces_after_create(client, auth):
    """Creating a workspace should make it appear in the list."""
    client.post("/v1/workspaces", json={"name": "WS1", "goal": ""}, headers=auth)
    client.post("/v1/workspaces", json={"name": "WS2", "goal": ""}, headers=auth)

    resp = client.get("/v1/workspaces", headers=auth)
    assert resp.status_code == 200
    assert resp.json()["count"] == 2
    names = {ws["name"] for ws in resp.json()["workspaces"]}
    assert names == {"WS1", "WS2"}


def test_workspace_tasks_lifecycle(client, auth):
    # Create workspace
    resp = client.post(
        "/v1/workspaces",
        json={"name": "Task WS", "goal": "Tasks"},
        headers=auth,
    )
    ws_id = resp.json()["id"]

    # List tasks - empty
    resp = client.get(f"/v1/workspaces/{ws_id}/tasks", headers=auth)
    assert resp.status_code == 200
    assert resp.json()["count"] == 0

    # Create task
    resp = client.post(
        f"/v1/workspaces/{ws_id}/tasks",
        json={"title": "Buy milk", "description": "2% fat"},
        headers=auth,
    )
    assert resp.status_code == 201
    task_id = resp.json()["id"]
    assert resp.json()["status"] == "open"

    # Claim task
    resp = client.post(
        f"/v1/workspaces/tasks/{task_id}/claim",
        json={"agent_name": "phone-user"},
        headers=auth,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "claimed"
    assert resp.json()["assigned_to"] == "phone-user"

    # Complete task
    resp = client.post(
        f"/v1/workspaces/tasks/{task_id}/complete",
        json={"agent_name": "phone-user"},
        headers=auth,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"


def test_double_claim_task(client, auth):
    """Claiming an already-claimed task should return 409."""
    resp = client.post("/v1/workspaces", json={"name": "DC WS", "goal": ""}, headers=auth)
    ws_id = resp.json()["id"]

    resp = client.post(f"/v1/workspaces/{ws_id}/tasks", json={"title": "Task"}, headers=auth)
    task_id = resp.json()["id"]

    # First claim succeeds
    resp = client.post(
        f"/v1/workspaces/tasks/{task_id}/claim",
        json={"agent_name": "agent-a"},
        headers=auth,
    )
    assert resp.status_code == 200

    # Second claim by different agent should fail
    resp = client.post(
        f"/v1/workspaces/tasks/{task_id}/claim",
        json={"agent_name": "agent-b"},
        headers=auth,
    )
    assert resp.status_code == 409


def test_claim_nonexistent_task(client, auth):
    """Claiming a task that doesn't exist should return 404."""
    resp = client.post(
        "/v1/workspaces/tasks/nonexistent/claim",
        json={"agent_name": "test"},
        headers=auth,
    )
    assert resp.status_code in (404, 409)


def test_complete_nonexistent_task(client, auth):
    """Completing a task that doesn't exist should return 404."""
    resp = client.post(
        "/v1/workspaces/tasks/nonexistent/complete",
        json={"agent_name": "test"},
        headers=auth,
    )
    assert resp.status_code in (404, 409)


def test_workspace_not_found(client, auth):
    resp = client.get("/v1/workspaces/nonexistent/tasks", headers=auth)
    assert resp.status_code == 404


def test_create_task_ws_not_found(client, auth):
    resp = client.post(
        "/v1/workspaces/nonexistent/tasks",
        json={"title": "Test"},
        headers=auth,
    )
    assert resp.status_code == 404


def test_workspace_requires_auth(client):
    resp = client.get("/v1/workspaces")
    assert resp.status_code == 401


def test_task_filter_by_status(client, auth):
    """Filter tasks by status query parameter."""
    resp = client.post("/v1/workspaces", json={"name": "Filter WS", "goal": ""}, headers=auth)
    ws_id = resp.json()["id"]

    # Create two tasks
    client.post(f"/v1/workspaces/{ws_id}/tasks", json={"title": "Task 1"}, headers=auth)
    resp = client.post(f"/v1/workspaces/{ws_id}/tasks", json={"title": "Task 2"}, headers=auth)
    task2_id = resp.json()["id"]

    # Claim one
    client.post(f"/v1/workspaces/tasks/{task2_id}/claim", json={}, headers=auth)

    # Filter by open
    resp = client.get(f"/v1/workspaces/{ws_id}/tasks?status=open", headers=auth)
    assert resp.json()["count"] == 1
    assert resp.json()["tasks"][0]["title"] == "Task 1"


def test_claim_with_default_body(client, auth):
    """Claim endpoint works with empty body (uses default agent_name)."""
    resp = client.post("/v1/workspaces", json={"name": "Def WS", "goal": ""}, headers=auth)
    ws_id = resp.json()["id"]
    resp = client.post(f"/v1/workspaces/{ws_id}/tasks", json={"title": "T"}, headers=auth)
    task_id = resp.json()["id"]

    resp = client.post(f"/v1/workspaces/tasks/{task_id}/claim", json={}, headers=auth)
    assert resp.status_code == 200
    assert resp.json()["assigned_to"] == "mobile-user"


# --- Chat endpoint ---


def test_chat_no_llm_returns_503(client, auth):
    """Chat returns 503 when no LLM is configured."""
    resp = client.post(
        "/v1/chat",
        json={"message": "Hello"},
        headers=auth,
    )
    assert resp.status_code == 503
    assert "No LLM" in resp.json()["detail"]


def test_chat_requires_auth(client):
    resp = client.post("/v1/chat", json={"message": "Hello"})
    assert resp.status_code == 401


def test_chat_validates_message(client, auth):
    resp = client.post("/v1/chat", json={"message": ""}, headers=auth)
    assert resp.status_code == 422


def test_chat_invalid_role_in_history(client, auth):
    """Invalid role in history should fail validation."""
    resp = client.post(
        "/v1/chat",
        json={
            "message": "Hi",
            "history": [{"role": "system", "content": "test"}],
        },
        headers=auth,
    )
    assert resp.status_code == 422


def test_chat_with_valid_history(client, auth):
    """Chat with valid history structure (still 503 without LLM, but validates schema)."""
    resp = client.post(
        "/v1/chat",
        json={
            "message": "And what about X?",
            "history": [
                {"role": "user", "content": "What is Y?"},
                {"role": "assistant", "content": "Y is..."},
            ],
        },
        headers=auth,
    )
    # 503 because no LLM configured, but NOT 422 (schema is valid)
    assert resp.status_code == 503
