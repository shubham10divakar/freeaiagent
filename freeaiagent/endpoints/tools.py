from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from .. import tools as registry

api = APIRouter(tags=["tools"])


class RegisterToolRequest(BaseModel):
    name: str
    description: str
    endpoint: str               # URL that receives a POST of the call arguments
    parameters: Optional[dict] = None  # OpenAI/JSON-Schema parameter object


@api.post("/tools/register", status_code=201)
async def register_tool(req: RegisterToolRequest):
    """Register an HTTP tool the model may call during a /chat with tools=true."""
    return registry.register(req.name, req.description, req.endpoint, req.parameters)


@api.get("/tools")
async def list_tools():
    """List all registered tools."""
    return {"tools": registry.all_tools()}


@api.delete("/tools/{name}")
async def delete_tool(name: str):
    """Unregister a tool."""
    if not registry.unregister(name):
        raise HTTPException(status_code=404, detail=f"Tool '{name}' is not registered.")
    return {"deleted": name}
