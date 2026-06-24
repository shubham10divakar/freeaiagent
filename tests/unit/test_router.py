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
def test_openai_compat_preset_inert_without_key():
    cfg = {"backends": {"gemini": {"type": "openai_compat", "base_url": "https://x", "api_key": ""}}}
    assert "gemini" not in router_mod._build_backends(cfg)


@pytest.mark.unit
def test_openai_compat_preset_built_with_key():
    cfg = {"backends": {"gemini": {
        "type": "openai_compat", "base_url": "https://x", "api_prefix": "", "api_key": "AIza-key"}}}
    backends = router_mod._build_backends(cfg)
    assert "gemini" in backends
    assert backends["gemini"].api_prefix == ""
    assert backends["gemini"].api_key == "AIza-key"


@pytest.mark.unit
def test_local_openai_compat_still_built_without_key():
    # local servers default api_key to "not-needed" and must still build
    cfg = {"backends": {"lmstudio": {"type": "openai_compat", "base_url": "http://localhost:1234"}}}
    assert "lmstudio" in router_mod._build_backends(cfg)


@pytest.mark.unit
async def test_resolve_returns_default_backend(isolated_config, monkeypatch):
    llamafile = _make_backend(available=True)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"llamafile": llamafile},
    )
    backend, model, _ = await router_mod.resolve()
    assert backend is llamafile
    assert model == "llama-3.2-3b"


@pytest.mark.unit
async def test_resolve_falls_back_when_default_unavailable(isolated_config, monkeypatch):
    ollama = _make_backend(available=False)
    groq = _make_backend(available=True, response="groq response")

    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama, "groq": groq},
    )
    backend, model, _ = await router_mod.resolve()
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
    _, model, _ = await router_mod.resolve(override_model="mistral:7b")
    assert model == "mistral:7b"


# ── per-backend context limits (_max_messages) ───────────────────────────────

@pytest.mark.unit
def test_max_messages_falls_back_to_global():
    cfg = {"max_messages": 20, "backends": {"ollama": {}}}
    assert router_mod._max_messages(cfg, "ollama", None) == 20


@pytest.mark.unit
def test_max_messages_backend_level_overrides_global():
    cfg = {"max_messages": 20, "backends": {"ollama": {"max_messages": 50}}}
    assert router_mod._max_messages(cfg, "ollama", None) == 50


@pytest.mark.unit
def test_max_messages_per_call_override_wins():
    cfg = {"max_messages": 20, "backends": {"ollama": {"max_messages": 50}}}
    assert router_mod._max_messages(cfg, "ollama", 5) == 5


@pytest.mark.unit
def test_max_messages_backend_zero_is_respected():
    # explicit 0 (unlimited) at backend level must override a nonzero global
    cfg = {"max_messages": 20, "backends": {"groq": {"max_messages": 0}}}
    assert router_mod._max_messages(cfg, "groq", None) == 0


@pytest.mark.unit
async def test_resolve_returns_backend_level_max_messages(isolated_config, monkeypatch):
    ollama = _make_backend(available=True)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama},
    )
    monkeypatch.setattr(
        "freeaiagent.router.load",
        lambda: {"default_backend": "ollama", "fallback_order": ["ollama"],
                 "default_model": "m", "max_messages": 7,
                 "backends": {"ollama": {"max_messages": 99}}},
    )
    _, _, max_messages = await router_mod.resolve()
    assert max_messages == 99


@pytest.mark.unit
async def test_available_models_delegates_to_active_backend(isolated_config, monkeypatch):
    llamafile = _make_backend(available=True)
    llamafile.available_models.return_value = ["Llama-3.2-1B-Instruct", "phi3"]
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"llamafile": llamafile},
    )
    models = await router_mod.available_models()
    assert "Llama-3.2-1B-Instruct" in models
    assert "phi3" in models


@pytest.mark.unit
async def test_available_models_empty_when_backend_down(isolated_config, monkeypatch):
    ollama = _make_backend(available=False)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama},
    )
    assert await router_mod.available_models() == []


@pytest.mark.unit
async def test_resolve_backend_override(isolated_config, monkeypatch):
    ollama = _make_backend(available=True)
    groq = _make_backend(available=True)
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama, "groq": groq},
    )
    backend, _, _ = await router_mod.resolve(override_backend="groq")
    assert backend is groq


@pytest.mark.unit
async def test_resolve_backend_override_unknown_raises(isolated_config, monkeypatch):
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": _make_backend()},
    )
    with pytest.raises(RuntimeError, match="not configured"):
        await router_mod.resolve(override_backend="nonexistent")


@pytest.mark.unit
async def test_resolve_backend_override_unavailable_raises(isolated_config, monkeypatch):
    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"lmstudio": _make_backend(available=False)},
    )
    with pytest.raises(RuntimeError, match="not reachable"):
        await router_mod.resolve(override_backend="lmstudio")


@pytest.mark.unit
async def test_build_backends_openai_compat(isolated_config, monkeypatch):
    from freeaiagent.backends.openai_compat import OpenAICompatibleBackend
    config = {
        "backends": {
            "lmstudio": {
                "type": "openai_compat",
                "base_url": "http://localhost:1234",
                "models": ["mistral-7b"],
            }
        }
    }
    backends = router_mod._build_backends(config)
    assert "lmstudio" in backends
    assert isinstance(backends["lmstudio"], OpenAICompatibleBackend)


@pytest.mark.unit
async def test_available_models_specific_backend(isolated_config, monkeypatch):
    ollama = _make_backend(available=True)
    ollama.available_models.return_value = ["llama3.2:3b"]
    groq = _make_backend(available=True)
    groq.available_models.return_value = ["llama-3.1-8b-instant"]

    monkeypatch.setattr(
        "freeaiagent.router._build_backends",
        lambda cfg: {"ollama": ollama, "groq": groq},
    )
    models = await router_mod.available_models(backend_name="groq")
    assert "llama-3.1-8b-instant" in models
