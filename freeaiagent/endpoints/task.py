from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from .. import router

api = APIRouter(tags=["task"])

_DEFAULT_SYSTEM = (
    "You are a precise task-execution assistant. "
    "Complete the given task exactly as described. Be concise."
)


class TaskRequest(BaseModel):
    task: str
    input: Optional[str] = None
    model: Optional[str] = None
    system: Optional[str] = None


@api.post("/task")
async def run_task(req: TaskRequest):
    """Run a one-shot task with no shared context. Context is not read or written."""
    content = f"{req.task}\n\n{req.input}" if req.input else req.task
    messages = [
        {"role": "system", "content": req.system or _DEFAULT_SYSTEM},
        {"role": "user", "content": content},
    ]
    try:
        backend, model = await router.resolve(override_model=req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    result = await backend.chat(messages, model)
    return {"result": result, "model": model}
