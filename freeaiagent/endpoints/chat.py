from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from .. import context, router
from ..config import load as load_config

api = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = None
    model: Optional[str] = None     # override model for this message only
    backend: Optional[str] = None   # override backend for this message only


@api.post("/chat")
async def chat(req: ChatRequest):
    """
    Send a message. Conversation history is preserved across calls.
    Optionally override the model or backend for this single message.
    """
    cfg = load_config()
    max_messages = cfg.get("max_messages", 0)

    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages += context.as_llm_messages(max_messages=max_messages)
    messages.append({"role": "user", "content": req.message})

    context.append("user", req.message)

    try:
        backend, model = await router.resolve(
            override_model=req.model,
            override_backend=req.backend,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = await backend.chat(messages, model)

    context.append("assistant", response)

    return {
        "response": response,
        "model": model,
        "backend": type(backend).__name__,
        "context_length": context.count(),
    }
