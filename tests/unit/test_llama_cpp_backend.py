import pytest

from freeaiagent.backends.llama_cpp import LlamaCppBackend


class FakeLlama:
    instances = 0

    def __init__(self, model_path, **kw):
        FakeLlama.instances += 1
        self.model_path = model_path
        self.kw = kw

    def create_chat_completion(self, messages, stream=False):
        if stream:
            return iter([
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {}}]},  # no content -> skipped
            ])
        return {"choices": [{"message": {"content": "Hello"}}]}


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    monkeypatch.setattr("freeaiagent.backends.llama_cpp._INSTANCES", {})
    FakeLlama.instances = 0


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("freeaiagent.backends.llama_cpp.MODELS_DIR", tmp_path)
    return tmp_path


def _with_llama(monkeypatch, cls=FakeLlama):
    monkeypatch.setattr("freeaiagent.backends.llama_cpp._import_llama", lambda: cls)


def _no_llama(monkeypatch):
    monkeypatch.setattr("freeaiagent.backends.llama_cpp._import_llama", lambda: None)


# ── path resolution ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resolve_path_catalog_gguf(models_dir):
    backend = LlamaCppBackend(model="qwen2.5-7b")
    assert backend._resolve_path() == models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"


@pytest.mark.unit
def test_resolve_path_explicit(tmp_path):
    p = tmp_path / "custom.gguf"
    backend = LlamaCppBackend(model_path=str(p))
    assert backend._resolve_path() == p


@pytest.mark.unit
def test_resolve_path_fused_model_is_none(models_dir):
    # fused llamafile (not gguf) has no in-process path
    backend = LlamaCppBackend(model="llama-3.2-3b")
    assert backend._resolve_path() is None


# ── is_available ─────────────────────────────────────────────────────────────

@pytest.mark.unit
async def test_unavailable_without_package(models_dir, monkeypatch):
    (models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    _no_llama(monkeypatch)
    backend = LlamaCppBackend(model="qwen2.5-7b")
    assert await backend.is_available() is False


@pytest.mark.unit
async def test_unavailable_when_model_missing(models_dir, monkeypatch):
    _with_llama(monkeypatch)
    backend = LlamaCppBackend(model="qwen2.5-7b")
    assert await backend.is_available() is False


@pytest.mark.unit
async def test_available_with_package_and_model(models_dir, monkeypatch):
    (models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    _with_llama(monkeypatch)
    backend = LlamaCppBackend(model="qwen2.5-7b")
    assert await backend.is_available() is True


# ── chat / stream ────────────────────────────────────────────────────────────

@pytest.mark.unit
async def test_chat_returns_content_and_caches(models_dir, monkeypatch):
    (models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    _with_llama(monkeypatch)
    backend = LlamaCppBackend(model="qwen2.5-7b")

    assert await backend.chat([{"role": "user", "content": "hi"}], "qwen2.5-7b") == "Hello"
    await backend.chat([{"role": "user", "content": "again"}], "qwen2.5-7b")
    assert FakeLlama.instances == 1  # model loaded once, reused


@pytest.mark.unit
async def test_stream_yields_content_tokens(models_dir, monkeypatch):
    (models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    _with_llama(monkeypatch)
    backend = LlamaCppBackend(model="qwen2.5-7b")

    tokens = [t async for t in backend.stream([{"role": "user", "content": "hi"}], "qwen2.5-7b")]
    assert tokens == ["Hel", "lo"]


@pytest.mark.unit
async def test_chat_raises_helpful_error_without_package(models_dir, monkeypatch):
    (models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    _no_llama(monkeypatch)
    backend = LlamaCppBackend(model="qwen2.5-7b")
    with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
        await backend.chat([{"role": "user", "content": "hi"}], "qwen2.5-7b")


@pytest.mark.unit
async def test_available_models(models_dir, monkeypatch):
    (models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    backend = LlamaCppBackend(model="qwen2.5-7b")
    assert await backend.available_models() == ["qwen2.5-7b"]


# ── router wiring ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_router_builds_llama_cpp_backend():
    import freeaiagent.router as router_mod
    cfg = {"default_model": "qwen2.5-7b",
           "backends": {"llama_cpp": {"type": "llama_cpp", "n_ctx": 2048}}}
    backends = router_mod._build_backends(cfg)
    assert isinstance(backends["llama_cpp"], LlamaCppBackend)
    assert backends["llama_cpp"].n_ctx == 2048
