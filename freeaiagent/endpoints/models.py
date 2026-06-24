import asyncio
import json
import queue
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import catalog, installed as installed_mod, pull as pull_mod
from ..backends.llamafile import LlamafileBackend, LLAMAFILE_DIR, MODELS_DIR
from ..config import load as load_config

api = APIRouter(tags=["models"])

# Downloads are large and disk/network bound — allow only one at a time.
_download_lock = threading.Lock()


class PullRequest(BaseModel):
    model: Optional[str] = None   # catalog name, hf:<repo>/<file>, or URL; None = default
    force: bool = False


@api.get("/models/catalog")
async def models_catalog():
    """List the curated local-model catalog, each flagged installed or not.

    Lets the SDK render a downloadable-models picker without knowing where
    ``~/.freeaiagent/`` lives or how installed-ness is determined.
    """
    port = load_config().get("backends", {}).get("llamafile", {}).get("port", 8080)
    out = []
    for name, e in catalog.all_entries():
        backend = LlamafileBackend(port=port, model=name)
        out.append({
            "name": name,
            "display": e["display"],
            "kind": e.get("kind", "llamafile"),
            "size_gb": e["size_gb"],
            "min_ram_gb": e["min_ram_gb"],
            "tier": e["tier"],
            "description": e["description"],
            "installed": backend._installed(),
        })
    return {"models": out}


@api.get("/models/installed")
async def models_installed():
    """List local model files actually present on disk, with paths and sizes."""
    return {"models": installed_mod.installed_files()}


@api.delete("/models/installed/{name}")
async def delete_installed_model(name: str):
    """Delete a downloaded model file to free disk space.

    ``name`` is a catalog name (e.g. ``llama-3.2-3b``) or an on-disk filename
    (as listed by ``/models/installed``). The shared engine binary is never
    removed. Returns 404 if no matching model is installed, 400 for an unsafe
    name.
    """
    try:
        return installed_mod.delete(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@api.post("/pull/stream")
async def pull_stream(req: PullRequest):
    """Download a local model server-side, streaming live progress as SSE.

        data: {"type": "start",    "phase": "model", "label": "...", "total_mb": 4700}
        data: {"type": "progress", "phase": "model", "pct": 12, "downloaded_mb": 564, ...}
        data: {"type": "done",     "path": "~/.freeaiagent/models/...gguf"}
        data: [DONE]

    GGUF models emit an "engine" phase first (the one-time shared runtime).
    Returns 400 for an unknown model, 409 if another download is in progress.
    """
    cfg = load_config()
    port = cfg.get("backends", {}).get("llamafile", {}).get("port", 8080)
    target = req.model or cfg.get("default_model", catalog.DEFAULT_MODEL)

    # Resolve up front so a bad target is a clean 400, not a mid-stream error.
    try:
        pt = pull_mod.resolve_target(target, port=port)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not _download_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A download is already in progress.")

    q: "queue.Queue" = queue.Queue()
    _SENTINEL = object()

    emitter = pull_mod.ProgressEmitter(
        labels={"engine": "llamafile engine", "model": pt.label},
        emit=q.put,
    )

    def worker():
        try:
            path = pt.backend.download(force=req.force, on_chunk=emitter)
            q.put({"type": "done", "path": str(path)})
        except Exception as e:  # surface to the client as an error event
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(_SENTINEL)

    async def event_stream():
        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                ev = await asyncio.to_thread(q.get)
                if ev is _SENTINEL:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            _download_lock.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
