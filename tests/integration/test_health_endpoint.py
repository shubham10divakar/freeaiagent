import pytest


@pytest.mark.integration
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["active_backend"] == "ollama"
    assert data["default_model"] == "llama3.2:3b"


@pytest.mark.integration
def test_health_degraded_when_no_backend(isolated_config, isolated_db, monkeypatch):
    async def _fail(override_model=None):
        raise RuntimeError("No backend available")

    monkeypatch.setattr("freeaiagent.router.resolve", _fail)

    from freeaiagent.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "degraded"
    assert "No backend available" in data["error"]


@pytest.mark.integration
def test_models_returns_list(client):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "llama3.2:3b" in data["models"]


@pytest.mark.integration
def test_models_empty_when_backend_down(isolated_config, isolated_db, monkeypatch):
    async def _empty():
        return []

    monkeypatch.setattr("freeaiagent.router.available_models", _empty)

    from freeaiagent.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/models")
    assert r.json()["models"] == []
