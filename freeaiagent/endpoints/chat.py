from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from .. import context, router

api = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = None


@api.post("/chat")
async def chat(req: ChatRequest):
    """Send a message and get a response. Conversation history is preserved."""
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages += context.as_llm_messages()
    messages.append({"role": "user", "content": req.message})

    context.append("user", req.message)

    backend, model = await router.resolve()
    response = await backend.chat(messages, model)

    context.append("assistant", response)

    return {
        "response": response,
        "model": model,
        "context_length": context.count(),
    }
