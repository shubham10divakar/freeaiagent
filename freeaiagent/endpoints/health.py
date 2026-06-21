from fastapi import APIRouter, Query
from typing import Optional
from .. import router as llm_router
from ..config import load

api = APIRouter(tags=["meta"])


@api.get("/health")
async def health():
    """Check server health and active backend."""
    config = load()
    try:
        backend, model = await llm_router.resolve()
        return {
            "status": "ok",
            "active_backend": config.get("default_backend"),
            "default_model": model,
        }
    except RuntimeError as e:
        return {"status": "degraded", "error": str(e)}


@api.get("/models")
async def models(backend: Optional[str] = Query(None, description="Backend name to list models for. Defaults to active backend.")):
    """List models available on the specified backend (or active backend if not specified)."""
    return {"models": await llm_router.available_models(backend_name=backend)}
