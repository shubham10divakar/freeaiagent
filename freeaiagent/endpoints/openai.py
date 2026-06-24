"""OpenAI-compatible wire protocol.

Speaking the OpenAI shape on the outside lets any app already using the OpenAI
SDK, LangChain, or LlamaIndex point ``base_url`` at freeaiagent with no code
change — internally we route to whatever backend is active.
"""
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import router as llm_router

api = APIRouter(tags=["openai"])


class _Message(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    messages: List[_Message]
    model: Optional[str] = None       # used as the model override; None = default
    stream: bool = False
    # Accepted for compatibility and ignored (backends manage their own defaults):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


def _completion_envelope(model: str, content: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        # Token accounting isn't tracked locally; report zeros for shape parity.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chunk(cid: str, created: int, model: str, *, delta: dict, finish_reason=None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@api.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions, with optional streaming."""
    messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
    try:
        backend, model, _ = await llm_router.resolve(override_model=req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not req.stream:
        content = await backend.chat(messages, model)
        return _completion_envelope(model, content)

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    async def event_stream():
        # First chunk announces the assistant role (OpenAI convention).
        yield _chunk(cid, created, model, delta={"role": "assistant"})
        try:
            async for token in backend.stream(messages, model):
                yield _chunk(cid, created, model, delta={"content": token})
        except Exception as e:
            yield _chunk(cid, created, model, delta={"content": f"[error: {e}]"})
        yield _chunk(cid, created, model, delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@api.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list (active backend's models)."""
    models = await llm_router.available_models()
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": now, "owned_by": "freeaiagent"}
            for m in models
        ],
    }
