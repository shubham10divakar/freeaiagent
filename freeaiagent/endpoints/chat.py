from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from .. import context, router
from ..caller import resolve_session, CALLER_HEADER
from ..config import load as load_config

api = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    session_id: str = "default"


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
    cfg = load_config()
    max_messages = cfg.get("max_messages", 0)

    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages += context.as_llm_messages(session_id=session_id, max_messages=max_messages)
    messages.append({"role": "user", "content": req.message})

    context.append("user", req.message, session_id=session_id)

    try:
        backend, model = await router.resolve(
            override_model=req.model,
            override_backend=req.backend,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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
