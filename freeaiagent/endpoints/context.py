from fastapi import APIRouter
from .. import context

api = APIRouter(tags=["context"])


@api.get("/context")
async def get_context():
    """Return the full conversation history."""
    messages = context.all_messages()
    return {"messages": messages, "total": len(messages)}


@api.delete("/context")
async def clear_context():
    """Wipe the entire conversation history."""
    cleared = context.clear()
    return {"cleared": cleared, "message": f"Cleared {cleared} messages."}
