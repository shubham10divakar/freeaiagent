from fastapi import APIRouter, Query
from .. import context

api = APIRouter(tags=["context"])


@api.get("/context")
async def get_context(session: str = Query("default", description="Session ID.")):
    """Return conversation history for a session."""
    messages = context.all_messages(session_id=session)
    return {"messages": messages, "total": len(messages), "session_id": session}


@api.delete("/context")
async def clear_context(session: str = Query("default", description="Session ID.")):
    """Wipe conversation history for a session."""
    cleared = context.clear(session_id=session)
    return {"cleared": cleared, "message": f"Cleared {cleared} messages.", "session_id": session}
