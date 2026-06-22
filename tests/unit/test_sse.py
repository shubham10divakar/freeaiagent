import pytest

from freeaiagent.backends._sse import openai_sse_deltas


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.mark.unit
async def test_extracts_content_and_stops_at_done():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"ignored after done"}}]}',
    ]
    out = [d async for d in openai_sse_deltas(FakeResponse(lines))]
    assert out == ["Hello", " world"]


@pytest.mark.unit
async def test_skips_malformed_empty_and_role_only_deltas():
    lines = [
        "",
        ": this is an sse comment",
        "data: not-json",
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',  # no content
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
    ]
    out = [d async for d in openai_sse_deltas(FakeResponse(lines))]
    assert out == ["ok"]
