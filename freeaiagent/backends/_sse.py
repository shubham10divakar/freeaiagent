"""Shared parsing helpers for streaming chat responses."""
import json
from typing import AsyncIterator


async def openai_sse_deltas(response) -> AsyncIterator[str]:
    """Yield content deltas from an OpenAI-style `stream=True` SSE response.

    Each event looks like `data: {"choices":[{"delta":{"content":"..."}}]}`,
    terminated by `data: [DONE]`. Malformed lines are skipped.
    """
    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if delta:
            yield delta
