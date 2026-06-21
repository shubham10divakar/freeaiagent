import json
from typing import Optional

import httpx2 as httpx
import typer

from .config import load, set_value

app = typer.Typer(
    name="freeaiagent",
    help="Local AI agent service — chat, delegate tasks, manage context.",
    no_args_is_help=True,
)
context_app = typer.Typer(help="Manage conversation context.")
config_app = typer.Typer(help="Read and update configuration.")
app.add_typer(context_app, name="context")
app.add_typer(config_app, name="config")

_SETUP_GUIDE = """
No LLM backend available. Get a free key or run locally:

  Option 1 — Ollama  (local, no key, no internet)
    Download:  https://ollama.com
    Then:      ollama pull llama3.2:3b
               freeaiagent start

  Option 2 — Groq  (free API key, fastest inference)
    Sign up:   https://console.groq.com  (no credit card)
    Then:      freeaiagent config set backends.groq.api_key gsk_...
               freeaiagent config set default_backend groq
               freeaiagent config set default_model openai/gpt-oss-20b
               freeaiagent start

  Option 3 — Google Gemini  (free, 1500 requests/day)
    Get key:   https://aistudio.google.com/apikey
    Then:      freeaiagent config set backends.gemini.type openai_compat
               freeaiagent config set backends.gemini.base_url https://generativelanguage.googleapis.com/v1beta/openai
               freeaiagent config set backends.gemini.api_key AIza...
               freeaiagent config set default_backend gemini
               freeaiagent config set default_model gemini-2.0-flash
               freeaiagent start

  Option 4 — OpenRouter  (free models, 50+ providers)
    Sign up:   https://openrouter.ai  (free credits on signup)
    Then:      freeaiagent config set backends.openrouter.type openai_compat
               freeaiagent config set backends.openrouter.base_url https://openrouter.ai/api
               freeaiagent config set backends.openrouter.api_key sk-or-...
               freeaiagent config set default_backend openrouter
               freeaiagent config set default_model meta-llama/llama-3.1-8b-instruct:free
               freeaiagent start

Run 'freeaiagent keys' to see this guide anytime.
"""


def _base_url() -> str:
    return f"http://localhost:{load().get('port', 7731)}"


def _agent_post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    try:
        r = httpx.post(f"{_base_url()}{path}", json=payload, timeout=timeout)
        if r.status_code == 503:
            detail = r.json().get("detail", "No backend available.")
            typer.echo(f"Error:   {detail}", err=True)
            typer.echo(_SETUP_GUIDE)
            raise typer.Exit(1)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        typer.echo("Agent not running. Start it with: freeaiagent start", err=True)
        raise typer.Exit(1)


def _agent_get(path: str, timeout: float = 10.0) -> dict:
    try:
        r = httpx.get(f"{_base_url()}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        typer.echo("Agent not running. Start it with: freeaiagent start", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

@app.command()
def start(
    port: Optional[int] = typer.Option(None, help="Port to listen on (default: 7731)"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)"),
):
    """Start the freeaiagent server."""
    import uvicorn
    from .main import app as fastapi_app

    p = port or load().get("port", 7731)
    typer.echo(f"freeaiagent running at http://localhost:{p}")
    typer.echo(f"API docs:           http://localhost:{p}/docs")
    uvicorn.run(
        "freeaiagent.main:app",
        host="0.0.0.0",
        port=p,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send. Omit for interactive mode."),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="Override system prompt."),
):
    """Chat with the agent (context is preserved between calls)."""
    if message:
        _do_chat(message, system)
    else:
        typer.echo("freeaiagent interactive chat — Ctrl+C or type 'exit' to quit\n")
        while True:
            try:
                msg = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("\nBye.")
                break
            if not msg:
                continue
            if msg.lower() in ("exit", "quit", "q"):
                break
            _do_chat(msg, system)


def _do_chat(message: str, system: Optional[str]):
    payload: dict = {"message": message}
    if system:
        payload["system"] = system
    data = _agent_post("/chat", payload)
    typer.echo(f"Agent [{data['model']}]: {data['response']}\n")


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------

@app.command()
def task(
    description: str = typer.Argument(..., help="Task description."),
    input_text: Optional[str] = typer.Option(None, "--input", "-i", help="Additional input text."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model."),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="Override system prompt."),
):
    """Run a one-shot task — no context read or written."""
    payload: dict = {"task": description}
    if input_text:
        payload["input"] = input_text
    if model:
        payload["model"] = model
    if system:
        payload["system"] = system
    data = _agent_post("/task", payload)
    typer.echo(data["result"])


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status():
    """Show server health, active backend, and model."""
    data = _agent_get("/health")
    typer.echo(f"Status:  {data['status']}")
    if data["status"] == "ok":
        typer.echo(f"Backend: {data['active_backend']}")
        typer.echo(f"Model:   {data['default_model']}")
    else:
        typer.echo(f"Error:   {data.get('error', 'unknown')}", err=True)
        typer.echo(_SETUP_GUIDE)


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------

@app.command()
def keys():
    """Show where to get free API keys and how to configure each backend."""
    typer.echo(_SETUP_GUIDE)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@app.command()
def models():
    """List models available on the active backend."""
    data = _agent_get("/models")
    if not data["models"]:
        typer.echo("No models found on active backend.")
        return
    for m in data["models"]:
        typer.echo(f"  {m}")


# ---------------------------------------------------------------------------
# context subcommands
# ---------------------------------------------------------------------------

@context_app.command("show")
def context_show():
    """Print the full conversation history."""
    data = _agent_get("/context")
    if not data["messages"]:
        typer.echo("No context yet.")
        return
    for m in data["messages"]:
        label = "You  " if m["role"] == "user" else "Agent"
        typer.echo(f"[{label}] {m['content']}\n")
    typer.echo(f"Total: {data['total']} messages")


@context_app.command("clear")
def context_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Clear the entire conversation history."""
    if not yes:
        typer.confirm("Clear all context?", abort=True)
    try:
        r = httpx.delete(f"{_base_url()}/context", timeout=5.0)
        r.raise_for_status()
        typer.echo(r.json()["message"])
    except httpx.ConnectError:
        typer.echo("Agent not running.", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# config subcommands
# ---------------------------------------------------------------------------

@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted key, e.g. default_model or backends.groq.api_key"),
    value: str = typer.Argument(..., help="Value to set."),
):
    """Set a configuration value."""
    set_value(key, value)
    typer.echo(f"Set {key} = {value}")


@config_app.command("show")
def config_show():
    """Print the current configuration (API keys are shown in full — keep safe)."""
    typer.echo(json.dumps(load(), indent=2))
