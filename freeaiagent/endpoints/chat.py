import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from .. import context, router, summarize, tools as tool_registry
from ..caller import resolve_session, CALLER_HEADER
from ..config import load as load_config

api = APIRouter(tags=["chat"])


async def _apply_context_strategy(backend, model: str, session_id: str) -> None:
    """Run the configured context strategy (currently: optional summarization)."""
    cfg = load_config()
    if cfg.get("context_strategy") == "summarize":
        await summarize.maybe_summarize(
            backend, model, session_id,
            threshold=cfg.get("summarize_threshold", 40),
            batch=cfg.get("summarize_batch", 30),
            summarize_model=cfg.get("summarize_model"),
        )


def _build_messages(req: "ChatRequest", session_id: str, max_messages: int) -> list:
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages += context.as_llm_messages(session_id=session_id, max_messages=max_messages)
    messages.append({"role": "user", "content": req.message})
    return messages


class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    session_id: str = "default"
    tools: bool = False  # let the model call registered tools mid-conversation
    max_messages: Optional[int] = None  # per-call context window override


@api.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    Send a message. Conversation history is preserved per session.

    The session is resolved from the body `session_id`, then the
    `X-Caller-ID` header, then `"default"` — so an app can set the header
    once and get its own context thread automatically.
    Optionally override the model or backend for this single message.
    """
    session_id = resolve_session(req.session_id, request.headers.get(CALLER_HEADER))

    try:
        backend, model, max_messages = await router.resolve(
            override_model=req.model,
            override_backend=req.backend,
            override_max_messages=req.max_messages,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    await _apply_context_strategy(backend, model, session_id)
    messages = _build_messages(req, session_id, max_messages)
    context.append("user", req.message, session_id=session_id)

    if req.tools and tool_registry.all_tools():
        response = await tool_registry.run(backend, model, messages)
    else:
        response = await backend.chat(messages, model)

    context.append(
        "assistant", response,
        session_id=session_id,
        model=model,
        backend=type(backend).__name__,
    )

    return {
        "response": response,
        "model": model,
        "backend": type(backend).__name__,
        "session_id": session_id,
        "context_length": context.count(session_id=session_id),
    }


@api.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    Same as /chat but streams the reply as Server-Sent Events:
        data: {"token": "Hello"}\\n\\n
        data: {"token": " there"}\\n\\n
        data: [DONE]\\n\\n

    The full response is persisted to the session once streaming completes.
    """
    session_id = resolve_session(req.session_id, request.headers.get(CALLER_HEADER))

    try:
        backend, model, max_messages = await router.resolve(
            override_model=req.model,
            override_backend=req.backend,
            override_max_messages=req.max_messages,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    await _apply_context_strategy(backend, model, session_id)
    messages = _build_messages(req, session_id, max_messages)
    context.append("user", req.message, session_id=session_id)

    async def event_stream():
        parts: list[str] = []
        try:
            async for token in backend.stream(messages, model):
                parts.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:  # surface backend errors to the client as an SSE event
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            context.append(
                "assistant", "".join(parts),
                session_id=session_id,
                model=model,
                backend=type(backend).__name__,
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
