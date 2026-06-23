from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import load as load_config, set_value

api = APIRouter(tags=["config"])


class ConfigSetRequest(BaseModel):
    key: str    # dotted key, e.g. "default_backend" or "backends.groq.api_key"
    value: Any


@api.get("/config")
async def get_config():
    """Return the effective configuration as JSON.

    Lets the SDK read and manage config without knowing the config file path.
    Note: API keys are returned in full — this server is loopback-only.
    """
    return load_config()


@api.post("/config/set")
async def set_config(req: ConfigSetRequest):
    """Set a single dotted config key, mirroring `freeaiagent config set`."""
    set_value(req.key, req.value)
    return {"key": req.key, "value": req.value}
