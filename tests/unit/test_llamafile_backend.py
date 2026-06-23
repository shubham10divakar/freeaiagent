import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from freeaiagent.backends.llamafile import LlamafileBackend, DEFAULT_MODEL


@pytest.fixture
def backend():
    return LlamafileBackend(port=8080, auto_download=True, auto_start=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_available_true_when_already_running(backend):
    with patch.object(backend, "_running", return_value=True):
        assert await backend.is_available() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_available_starts_if_binary_exists_but_not_running(backend, tmp_path):
    bin_path = tmp_path / "model.llamafile"
    bin_path.touch()
    with (
        patch.object(backend, "_running", side_effect=[False, True]),
        patch.object(backend, "_bin", return_value=bin_path),
        patch.object(backend, "_start") as mock_start,
    ):
        result = await backend.is_available()
    assert result is True
    mock_start.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_available_downloads_if_binary_missing(backend, tmp_path):
    bin_path = tmp_path / "model.llamafile"

    def fake_download():
        bin_path.touch()  # simulate creating the file

    with (
        patch.object(backend, "_running", side_effect=[False, True]),
        patch.object(backend, "_bin", return_value=bin_path),
        patch.object(backend, "download", side_effect=fake_download) as mock_dl,
        patch.object(backend, "_start"),
    ):
        result = await backend.is_available()
    assert result is True
    mock_dl.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_available_false_when_auto_start_disabled(backend):
    backend.auto_start = False
    with patch.object(backend, "_running", return_value=False):
        assert await backend.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_available_false_when_download_fails(backend, tmp_path):
    bin_path = tmp_path / "model.llamafile"
    with (
        patch.object(backend, "_running", return_value=False),
        patch.object(backend, "_bin", return_value=bin_path),
        patch.object(backend, "download", side_effect=OSError("network error")),
    ):
        assert await backend.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_skipped_when_binary_exists(backend, tmp_path):
    bin_path = tmp_path / "model.llamafile"
    bin_path.touch()
    backend.auto_download = True
    with (
        patch.object(backend, "_running", side_effect=[False, True]),
        patch.object(backend, "_bin", return_value=bin_path),
        patch.object(backend, "download") as mock_dl,
        patch.object(backend, "_start"),
    ):
        await backend.is_available()
    mock_dl.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_returns_content(backend):
    messages = [{"role": "user", "content": "hello"}]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "hi there"}}]
    }
    with patch("httpx2.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client
        result = await backend.chat(messages, "Llama-3.2-1B-Instruct")
    assert result == "hi there"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_available_models_from_api(backend):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "Llama-3.2-1B-Instruct"}]}
    with patch("httpx2.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client
        models = await backend.available_models()
    assert models == ["Llama-3.2-1B-Instruct"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_available_models_fallback_on_error(backend):
    with patch("httpx2.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client_cls.return_value = mock_client
        models = await backend.available_models()
    assert models == [DEFAULT_MODEL]


@pytest.mark.unit
def test_bin_path_adds_exe_on_windows(tmp_path):
    import platform
    backend = LlamafileBackend()
    with patch("freeaiagent.backends.llamafile.LLAMAFILE_DIR", tmp_path):
        with patch.object(platform, "system", return_value="Windows"):
            path = backend._bin()
    assert path.suffix == ".exe"


@pytest.mark.unit
def test_bin_path_no_exe_on_linux(tmp_path):
    import platform
    backend = LlamafileBackend()
    with patch("freeaiagent.backends.llamafile.LLAMAFILE_DIR", tmp_path):
        with patch.object(platform, "system", return_value="Linux"):
            path = backend._bin()
    assert path.suffix != ".exe"


# ── Engine/weights split (GGUF models) ───────────────────────────────────────

@pytest.mark.unit
def test_gguf_model_detected_and_routed_to_models_dir(tmp_path):
    from freeaiagent.backends import llamafile as lf
    backend = LlamafileBackend(model="qwen2.5-7b")
    assert backend._is_gguf() is True
    with patch.object(lf, "MODELS_DIR", tmp_path):
        path = backend._bin()
    assert path.name.endswith(".gguf")  # gguf keeps its extension, even on Windows
    assert path.parent == tmp_path


@pytest.mark.unit
def test_fused_model_is_not_gguf():
    backend = LlamafileBackend(model="llama-3.2-3b")
    assert backend._is_gguf() is False


@pytest.mark.unit
def test_command_uses_engine_with_m_for_gguf():
    backend = LlamafileBackend(model="qwen2.5-7b", port=9001)
    cmd = backend._command()
    assert cmd[0] == str(backend._engine_path())
    assert "-m" in cmd and str(backend._bin()) in cmd
    assert "--server" in cmd
    assert "9001" in cmd


@pytest.mark.unit
def test_command_runs_fused_file_directly():
    backend = LlamafileBackend(model="llama-3.2-3b")
    cmd = backend._command()
    assert cmd[0] == str(backend._bin())
    assert "-m" not in cmd


@pytest.mark.unit
def test_installed_requires_engine_for_gguf(tmp_path, monkeypatch):
    from freeaiagent.backends import llamafile as lf
    monkeypatch.setattr(lf, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(lf, "ENGINE_DIR", tmp_path / "engine")
    backend = LlamafileBackend(model="qwen2.5-7b")

    gguf = backend._bin()
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.touch()
    # GGUF present but engine missing -> not installed
    assert backend._installed() is False

    engine = backend._engine_path()
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.touch()
    assert backend._installed() is True


@pytest.mark.unit
def test_download_fetches_engine_then_model_for_gguf(tmp_path, monkeypatch):
    from freeaiagent.backends import llamafile as lf
    backend = LlamafileBackend(model="qwen2.5-7b")
    calls = []

    def fake_dl(url, dest, force=False, make_exec=False, on_chunk=None, phase="model"):
        calls.append(url)
        return dest

    monkeypatch.setattr(backend, "_download_file", fake_dl)
    # point engine at a path that doesn't exist so download() fetches it
    monkeypatch.setattr(backend, "_engine_path", lambda: tmp_path / "engine-missing")
    backend.download()

    assert calls[0] == lf.ENGINE_URL          # engine fetched first
    assert calls[1].endswith(".gguf")         # then the model weights


# ── Download progress callback (Phase 5 Step 1) ──────────────────────────────

@pytest.mark.unit
def test_download_file_invokes_on_chunk_with_phase(tmp_path):
    backend = LlamafileBackend(model="llama-3.2-3b")
    dest = tmp_path / "model.bin"

    class FakeResp:
        headers = {"Content-Length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __init__(self):
            self._chunks = [b"a", b"b", b"c"]

        def read(self, _n):
            return self._chunks.pop(0) if self._chunks else b""

    events = []
    with patch("freeaiagent.backends.llamafile.urllib.request.urlopen", return_value=FakeResp()):
        backend._download_file(
            "http://x/model.bin", dest,
            on_chunk=lambda d, t, p: events.append((d, t, p)),
            phase="model",
        )

    assert events == [(1, 3, "model"), (2, 3, "model"), (3, 3, "model")]
    assert dest.exists()


@pytest.mark.unit
def test_download_file_falls_back_to_print_progress_without_callback(tmp_path):
    backend = LlamafileBackend(model="llama-3.2-3b")
    dest = tmp_path / "model.bin"

    class FakeResp:
        headers = {"Content-Length": "1"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __init__(self):
            self._chunks = [b"a"]

        def read(self, _n):
            return self._chunks.pop(0) if self._chunks else b""

    with (
        patch("freeaiagent.backends.llamafile.urllib.request.urlopen", return_value=FakeResp()),
        patch.object(backend, "_print_progress") as mock_print,
    ):
        backend._download_file("http://x/model.bin", dest)

    mock_print.assert_called_once_with(1, 1)


@pytest.mark.unit
def test_download_threads_callback_through_engine_and_model_phases(tmp_path, monkeypatch):
    backend = LlamafileBackend(model="qwen2.5-7b")
    phases = []

    def fake_dl(url, dest, force=False, make_exec=False, on_chunk=None, phase="model"):
        phases.append(phase)
        return dest

    monkeypatch.setattr(backend, "_download_file", fake_dl)
    monkeypatch.setattr(backend, "_engine_path", lambda: tmp_path / "engine-missing")
    backend.download(on_chunk=lambda d, t, p: None)

    assert phases == ["engine", "model"]
