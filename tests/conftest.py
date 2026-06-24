import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect config reads/writes to a temp directory."""
    monkeypatch.setattr("freeaiagent.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("freeaiagent.config.CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect SQLite DB to a temp file so tests never touch ~/.freeaiagent/."""
    db = tmp_path / "context.db"
    monkeypatch.setattr("freeaiagent.context.DB_FILE", db)
    return db


@pytest.fixture
def mock_backend():
    """AsyncMock backend that is always available and returns a fixed response."""
    b = AsyncMock()
    b.is_available.return_value = True
    b.chat.return_value = "Hello from mock backend"
    b.available_models.return_value = ["llama3.2:3b", "mistral:7b"]

    async def _stream(messages, model):
        for token in ["Hello", " from", " mock", " stream"]:
            yield token

    b.stream = _stream
    return b


@pytest.fixture
def patched_router(mock_backend, monkeypatch):
    """Patch router.resolve and router.available_models to use mock_backend."""
    async def _resolve(override_model=None, override_backend=None, override_max_messages=None):
        if override_backend and override_backend != "ollama":
            raise RuntimeError(f"Backend '{override_backend}' is not configured. Available: ['ollama']")
        # Honour the real per-call → backend → global resolution against config.
        from freeaiagent import config as _cfg, router as _router
        mm = _router._max_messages(_cfg.load(), override_backend or "ollama", override_max_messages)
        return mock_backend, override_model or "Llama-3.2-1B-Instruct", mm

    async def _models(backend_name=None):
        return await mock_backend.available_models()

    monkeypatch.setattr("freeaiagent.router.resolve", _resolve)
    monkeypatch.setattr("freeaiagent.router.available_models", _models)
    return mock_backend


@pytest.fixture
def isolated_tools(tmp_path, monkeypatch):
    """Redirect the tool registry file to a temp location."""
    monkeypatch.setattr("freeaiagent.tools.TOOLS_FILE", tmp_path / "tools.json")
    return tmp_path / "tools.json"


@pytest.fixture
def client(isolated_config, isolated_db, isolated_tools, patched_router):
    """FastAPI TestClient with mocked backend and isolated storage."""
    from freeaiagent.main import app
    with TestClient(app) as c:
        yield c
