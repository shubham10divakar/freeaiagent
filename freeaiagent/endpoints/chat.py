import base64
import json
import os
import tempfile
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from .. import context, ensemble, router, summarize, tools as tool_registry
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
    # Ensemble: true = use config models; a list = use those models; null = off
    # (unless config ensemble.enabled). Needs >= 2 models or it's a normal chat.
    ensemble: Optional[Union[bool, List[str]]] = None
    # Image for SDX vision backends. Raw base64 or a data-URI (data:image/...;base64,...).
    image: Optional[str] = None


def _decode_image_to_tempfile(b64: str) -> str:
    """Decode a base64 or data-URI image to a temp file; return the file path."""
    if b64.startswith("data:"):
        # data:image/jpeg;base64,/9j/...
        header, data = b64.split(",", 1)
        ext = header.split(";")[0].split("/")[-1] or "jpg"
        raw = base64.b64decode(data)
    else:
        ext = "jpg"
        raw = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix="sdx_img_")
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    return path


async def _extract_sdx_vision(
    backend, model: str, image_b64: str, user_text: str, session_id: str
) -> tuple[list[dict] | None, str | None]:
    """Extract image description with SDX vision sub-model.

    Returns ``(injection_messages, description)`` where injection_messages is
    the ``[SDX-Image]`` system message to insert into the messages list, and
    description is what gets persisted to SQLite. Returns ``(None, None)`` if
    the backend does not support vision.
    """
    if not image_b64 or not hasattr(backend, "extract_vision"):
        return None, None
    path = _decode_image_to_tempfile(image_b64)
    try:
        description = await backend.extract_vision(path, user_text, model=model)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    context.append("system", f"[SDX-Image]: {description}", session_id=session_id)
    inject = {"role": "system", "content": f"[SDX-Image]: {description}"}
    return inject, description


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

    # SDX vision: extract description before calling the text model.
    # The description is persisted to SQLite here (before the assistant message)
    # so future context reconstructions include it in the right order.
    inject, _desc = await _extract_sdx_vision(backend, model, req.image, req.message, session_id)
    if inject is not None:
        messages = messages[:-1] + [inject, messages[-1]]

    cfg = load_config()
    ensemble_models = ensemble.resolve_models(req.ensemble, cfg)
    votes = None
    if len(ensemble_models) >= 2:
        ecfg = cfg.get("ensemble", {})
        response, model, votes = await ensemble.run(
            backend, messages, ensemble_models,
            judge_model=ecfg.get("judge"),
            strategy=ecfg.get("strategy", "llm_judge"),
        )
    elif req.tools and tool_registry.all_tools():
        response = await tool_registry.run(backend, model, messages)
    else:
        response = await backend.chat(messages, model)

    context.append(
        "assistant", response,
        session_id=session_id,
        model=model,
        backend=type(backend).__name__,
    )

    result = {
        "response": response,
        "model": model,
        "backend": type(backend).__name__,
        "session_id": session_id,
        "context_length": context.count(session_id=session_id),
    }
    if votes is not None:
        result["ensemble_votes"] = votes
    return result


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
        nonlocal messages
        parts: list[str] = []
        try:
            # SDX vision: emit an "analyzing" event, extract the description,
            # persist it, then stream text-model tokens as normal.
            if req.image and hasattr(backend, "extract_vision"):
                yield f"data: {json.dumps({'analyzing': True, 'message': 'Analyzing image...'})}\n\n"
                inject, _desc = await _extract_sdx_vision(
                    backend, model, req.image, req.message, session_id
                )
                if inject is not None:
                    messages = messages[:-1] + [inject, messages[-1]]

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
