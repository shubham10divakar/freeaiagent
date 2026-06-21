import uuid as _uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from .. import context

api = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


@api.get("/sessions")
async def list_sessions():
    """List all sessions ordered by most recent activity."""
    return {"sessions": context.all_sessions()}


@api.post("/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    """Create a session explicitly. A session is also auto-created on the first /chat message."""
    session_id = req.id or str(_uuid.uuid4())
    session = context.create_session(session_id, title=req.title or "")
    return session


@api.patch("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest):
    """Rename a session."""
    ok = context.rename_session(session_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"id": session_id, "title": req.title}


@api.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    ok = context.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"deleted": session_id}
