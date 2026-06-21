import pytest
from unittest.mock import AsyncMock, patch
import freeaiagent.router as router_mod


def _make_backend(available: bool = True, response: str = "ok"):
    b = AsyncMock()
    b.is_available.return_value = available
    b.chat.return_value = response
    b.available_models.return_value = ["llama3.2:3b"]
    return b


@pytest.mark.unit
async def test_resolve_returns_default_backend(isolated_config, monkeypatch):
    ollama = _make_backend(available=True)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama},
    )
    backend, model = await router_mod.resolve()
    assert backend is ollama
    assert model == "llama3.2:3b"


@pytest.mark.unit
async def test_resolve_falls_back_when_default_unavailable(isolated_config, monkeypatch):
    ollama = _make_backend(available=False)
    groq = _make_backend(available=True, response="groq response")

    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama, "groq": groq},
    )
    backend, model = await router_mod.resolve()
    assert backend is groq


@pytest.mark.unit
async def test_resolve_raises_when_all_backends_unavailable(isolated_config, monkeypatch):
    ollama = _make_backend(available=False)
    groq = _make_backend(available=False)

    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama, "groq": groq},
    )
    with pytest.raises(RuntimeError, match="No backend available"):
        await router_mod.resolve()


@pytest.mark.unit
async def test_resolve_uses_override_model(isolated_config, monkeypatch):
    ollama = _make_backend(available=True)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama},
    )
    _, model = await router_mod.resolve(override_model="mistral:7b")
    assert model == "mistral:7b"


@pytest.mark.unit
async def test_available_models_delegates_to_active_backend(isolated_config, monkeypatch):
    ollama = _make_backend(available=True)
    ollama.available_models.return_value = ["llama3.2:3b", "phi3"]
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama},
    )
    models = await router_mod.available_models()
    assert "llama3.2:3b" in models
    assert "phi3" in models


@pytest.mark.unit
async def test_available_models_empty_when_backend_down(isolated_config, monkeypatch):
    ollama = _make_backend(available=False)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama},
    )
    assert await router_mod.available_models() == []
