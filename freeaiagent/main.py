import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from .endpoints.chat import api as chat_api
from .endpoints.task import api as task_api
from .endpoints.context import api as context_api
from .endpoints.health import api as health_api
from .endpoints.sessions import api as sessions_api

app = FastAPI(
    title="freeaiagent",
    description=(
        "Local AI agent service. "
        "Persistent context, multi-model LLM backends, HTTP endpoints for any app to delegate tasks."
    ),
    version="1.0.0",
)

app.include_router(chat_api)
app.include_router(task_api)
app.include_router(context_api)
app.include_router(health_api)
app.include_router(sessions_api)

_UI_PATH = os.path.join(os.path.dirname(__file__), "ui", "index.html")


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    return FileResponse(_UI_PATH)
