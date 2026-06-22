import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from freeaiagent.backends.openai_compat import OpenAICompatibleBackend


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *exc):
        return False


@pytest.mark.unit
async def test_stream_yields_deltas():
    backend = OpenAICompatibleBackend("http://localhost:1234")
    lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
    ]
    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.stream = MagicMock(return_value=_FakeStreamCtx(lines))
        out = [d async for d in backend.stream([{"role": "user", "content": "hi"}], "m")]
    assert out == ["Hi", " there"]
    # streaming request must set stream=True
    assert mock_client.stream.call_args[1]["json"]["stream"] is True


@pytest.mark.unit
async def test_is_available_true_when_models_endpoint_200():
    backend = OpenAICompatibleBackend("http://localhost:1234")
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        assert await backend.is_available() is True


@pytest.mark.unit
async def test_is_available_false_when_connection_refused():
    backend = OpenAICompatibleBackend("http://localhost:9999")
    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = Exception("Connection refused")
        assert await backend.is_available() is False


@pytest.mark.unit
async def test_chat_returns_content():
    backend = OpenAICompatibleBackend("http://localhost:1234")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello from openai compat"}}]
    }

    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        result = await backend.chat([{"role": "user", "content": "hi"}], "mistral-7b")
        assert result == "Hello from openai compat"


@pytest.mark.unit
async def test_chat_sends_correct_payload():
    backend = OpenAICompatibleBackend("http://localhost:1234", api_key="test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }

    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        await backend.chat(messages, "phi3")

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["model"] == "phi3"
        assert call_kwargs["json"]["messages"] == messages
        assert "Bearer test-key" in call_kwargs["headers"]["Authorization"]


@pytest.mark.unit
async def test_available_models_from_v1_models():
    backend = OpenAICompatibleBackend("http://localhost:1234")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"id": "mistral-7b"}, {"id": "phi3-mini"}]
    }

    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        models = await backend.available_models()
        assert models == ["mistral-7b", "phi3-mini"]


@pytest.mark.unit
async def test_default_api_prefix_uses_v1():
    backend = OpenAICompatibleBackend("http://localhost:1234")
    assert backend._url("/chat/completions") == "http://localhost:1234/v1/chat/completions"


@pytest.mark.unit
async def test_empty_api_prefix_for_gemini_style_url():
    backend = OpenAICompatibleBackend(
        "https://generativelanguage.googleapis.com/v1beta/openai", api_prefix=""
    )
    assert backend._url("/chat/completions") == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


@pytest.mark.unit
async def test_available_models_falls_back_to_config_list():
    backend = OpenAICompatibleBackend(
        "http://localhost:1234", model_list=["fallback-model"]
    )
    with patch("freeaiagent.backends.openai_compat.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = Exception("unreachable")
        models = await backend.available_models()
        assert models == ["fallback-model"]
