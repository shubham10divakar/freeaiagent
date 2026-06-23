import pytest

from freeaiagent import (
    Client, PullProgress, BackendUnavailable, DownloadInProgress,
)
from freeaiagent import pull as pull_mod


@pytest.fixture
def sdk(client):
    """SDK Client wired to the in-process app via the Starlette TestClient.

    `client` (from conftest) is a sync httpx-based TestClient with isolated
    config/db/tools and a patched router, so it doubles as the SDK transport.
    """
    return Client(name="tester", _http=client)


# ── chat / stream / task ─────────────────────────────────────────────────────

@pytest.mark.integration
def test_chat_returns_response(sdk):
    assert sdk.chat("hello") == "Hello from mock backend"


@pytest.mark.integration
def test_chat_uses_caller_session(sdk):
    sdk.chat("hello")
    # name="tester" -> X-Caller-ID resolves the session to "tester"
    msgs = sdk.context.get(session="tester")
    assert any(m["content"] == "hello" for m in msgs)


@pytest.mark.integration
def test_stream_yields_tokens(sdk):
    assert list(sdk.stream("hi")) == ["Hello", " from", " mock", " stream"]


@pytest.mark.integration
def test_task_returns_result(sdk):
    assert sdk.task("summarize", input="text") == "Hello from mock backend"


@pytest.mark.integration
def test_chat_backend_override_unavailable_raises(sdk):
    with pytest.raises(BackendUnavailable):
        sdk.chat("hi", backend="nonexistent")


# ── models / sessions / context / config / tools ─────────────────────────────

@pytest.mark.integration
def test_models_list(sdk):
    assert sdk.models.list() == ["llama3.2:3b", "mistral:7b"]


@pytest.mark.integration
def test_models_catalog_and_installed(sdk):
    cat = sdk.models.catalog()
    assert any(m["name"] == "llama-3.2-3b" for m in cat)
    assert isinstance(sdk.models.installed(), list)


@pytest.mark.integration
def test_models_active(sdk):
    # health's default_model comes from patched_router's resolve
    assert sdk.models.active() == "Llama-3.2-1B-Instruct"


@pytest.mark.integration
def test_sessions_crud(sdk):
    created = sdk.sessions.create("work", title="Work")
    assert created["id"] == "work"
    assert any(s["id"] == "work" for s in sdk.sessions.list())
    renamed = sdk.sessions.rename("work", "Work v2")
    assert renamed["title"] == "Work v2"
    assert sdk.sessions.delete("work")["deleted"] == "work"


@pytest.mark.integration
def test_context_get_and_clear(sdk):
    sdk.chat("remember this", session="s1")
    assert len(sdk.context.get(session="s1")) >= 2
    cleared = sdk.context.clear(session="s1")
    assert cleared >= 2
    assert sdk.context.get(session="s1") == []


@pytest.mark.integration
def test_config_get_and_set(sdk):
    assert "default_backend" in sdk.config.get()
    sdk.config.set("default_backend", "groq")
    assert sdk.config.get()["default_backend"] == "groq"


@pytest.mark.integration
def test_tools_register_list_unregister(sdk):
    sdk.tools.register("get_weather", description="Weather",
                       endpoint="http://localhost:9000/w",
                       parameters={"type": "object", "properties": {}})
    assert any(t["name"] == "get_weather" for t in sdk.tools.list())
    assert sdk.tools.unregister("get_weather")["deleted"] == "get_weather"


# ── health ───────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_health_and_is_running(sdk):
    assert sdk.health()["status"] == "ok"
    assert sdk.is_running() is True


# ── search ───────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_search_term_and_repo(sdk, monkeypatch):
    monkeypatch.setattr("freeaiagent.hf.search_models", lambda q, limit=20: [{"id": "x"}])
    monkeypatch.setattr("freeaiagent.hf.list_gguf_files", lambda r: [{"path": "a.gguf"}])
    assert sdk.search("qwen") == [{"id": "x"}]
    assert sdk.search("owner/repo") == [{"path": "a.gguf"}]


# ── pull ─────────────────────────────────────────────────────────────────────

class _FakeBackend:
    def __init__(self, chunks):
        self.chunks = chunks

    def download(self, force=False, on_chunk=None):
        for done, total, phase in self.chunks:
            if on_chunk:
                on_chunk(done, total, phase)
        return "/models/qwen.gguf"


@pytest.mark.integration
def test_pull_yields_progress_and_done(sdk, monkeypatch):
    backend = _FakeBackend([(50, 100, "model"), (100, 100, "model")])
    monkeypatch.setattr(
        "freeaiagent.pull.resolve_target",
        lambda target, port=8080: pull_mod.PullTarget(backend, "qwen", 4.7, 8, True),
    )
    events = list(sdk.pull("qwen2.5-7b"))
    assert all(isinstance(e, PullProgress) for e in events)
    assert events[0].type == "start"
    assert events[-1].type == "done"
    assert events[-1].path.endswith("qwen.gguf")


@pytest.mark.integration
def test_pull_on_progress_callback(sdk, monkeypatch):
    backend = _FakeBackend([(100, 100, "model")])
    monkeypatch.setattr(
        "freeaiagent.pull.resolve_target",
        lambda target, port=8080: pull_mod.PullTarget(backend, "qwen", 4.7, 8, True),
    )
    seen = []
    list(sdk.pull("qwen2.5-7b", on_progress=seen.append))
    assert seen and isinstance(seen[0], PullProgress)


@pytest.mark.integration
def test_pull_409_when_busy(sdk, monkeypatch):
    backend = _FakeBackend([(100, 100, "model")])
    monkeypatch.setattr(
        "freeaiagent.pull.resolve_target",
        lambda target, port=8080: pull_mod.PullTarget(backend, "qwen", 4.7, 8, True),
    )
    from freeaiagent.endpoints import models as models_ep
    models_ep._download_lock.acquire()
    try:
        with pytest.raises(DownloadInProgress):
            list(sdk.pull("qwen2.5-7b"))
    finally:
        models_ep._download_lock.release()
