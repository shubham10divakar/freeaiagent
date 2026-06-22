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
