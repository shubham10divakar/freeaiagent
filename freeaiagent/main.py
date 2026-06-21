from fastapi import FastAPI
from .endpoints.chat import api as chat_api
from .endpoints.task import api as task_api
from .endpoints.context import api as context_api
from .endpoints.health import api as health_api

app = FastAPI(
    title="freeaiagent",
    description=(
        "Local AI agent service. "
        "Persistent context, multi-model LLM backends, HTTP endpoints for any app to delegate tasks."
    ),
    version="0.1.0",
)

app.include_router(chat_api)
app.include_router(task_api)
app.include_router(context_api)
app.include_router(health_api)
