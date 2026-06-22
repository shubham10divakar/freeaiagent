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

  Option 0 — Local model  (zero setup, no key, fully offline)
    Run:       freeaiagent pull        (one-time ~2.3 GB download)
               freeaiagent start

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

  (Cloud presets below are built in — just add a key, pick the backend + a model.)

  Option 3 — Google Gemini  (free, 1500 requests/day)
    Get key:   https://aistudio.google.com/apikey
    Then:      freeaiagent config set backends.gemini.api_key AIza...
               freeaiagent config set default_backend gemini
               freeaiagent config set default_model gemini-2.0-flash
               freeaiagent start

  Option 4 — OpenRouter  (free models, 50+ providers)
    Sign up:   https://openrouter.ai  (free credits on signup)
    Then:      freeaiagent config set backends.openrouter.api_key sk-or-...
               freeaiagent config set default_backend openrouter
               freeaiagent config set default_model meta-llama/llama-3.1-8b-instruct:free
               freeaiagent start

  Option 5 — Together AI  (free tier)
    Sign up:   https://api.together.xyz
    Then:      freeaiagent config set backends.together.api_key ...
               freeaiagent config set default_backend together
               freeaiagent config set default_model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
               freeaiagent start

  Option 6 — Cerebras  (free tier, very fast)
    Sign up:   https://cloud.cerebras.ai
    Then:      freeaiagent config set backends.cerebras.api_key csk-...
               freeaiagent config set default_backend cerebras
               freeaiagent config set default_model llama-3.3-70b
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

    cfg = load()
    p = port or cfg.get("port", 7731)
    typer.echo(f"freeaiagent running at http://localhost:{p}")
    typer.echo(f"Chat UI:            http://localhost:{p}/ui")
    typer.echo(f"API docs:           http://localhost:{p}/docs")
    typer.echo(f"Backend / model:    {cfg.get('default_backend')} / {cfg.get('default_model')}")
    typer.echo(f"Try it:             freeaiagent chat \"hello\"")
    typer.echo("")
    uvicorn.run(
        "freeaiagent.main:app",
        host="0.0.0.0",
        port=p,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

@app.command()
def pull(
    model: Optional[str] = typer.Argument(
        None,
        help="Catalog name (e.g. llama-3.2-3b) or a direct llamafile URL. "
        "Omit to pull the current default model.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if the model already exists."),
):
    """Download a local model for offline, no-key operation.

    Examples:
      freeaiagent pull                 # the current default model
      freeaiagent pull gemma-2-2b      # a catalog model by name
      freeaiagent pull https://.../model.llamafile   # any llamafile URL
    """
    import shutil
    from . import catalog
    from .backends.llamafile import LlamafileBackend, LLAMAFILE_DIR

    bcfg = load().get("backends", {}).get("llamafile", {})
    port = bcfg.get("port", 8080)
    target = model or load().get("default_model", catalog.DEFAULT_MODEL)

    if target.startswith(("http://", "https://")):
        backend = LlamafileBackend(port=port, download_url=target)
        label, size_gb, min_ram = target.rsplit("/", 1)[-1], None, None
    else:
        entry = catalog.get(target)
        if entry is None:
            typer.echo(
                f"Unknown model '{target}'.\n"
                f"See available models with: freeaiagent models --available\n"
                f"Or pass a direct llamafile URL.",
                err=True,
            )
            raise typer.Exit(1)
        backend = LlamafileBackend(port=port, model=target)
        label, size_gb, min_ram = entry["display"], entry["size_gb"], entry["min_ram_gb"]

    path = backend._bin()
    if path.exists() and not force:
        typer.echo(f"Already installed: {label}\n  {path}")
        typer.echo("Re-download with: freeaiagent pull --force")
        return

    # Disk-space sanity check (best-effort; download still allowed).
    if size_gb:
        LLAMAFILE_DIR.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(LLAMAFILE_DIR).free / (1024 ** 3)
        if free_gb < size_gb + 0.5:
            typer.echo(
                f"Warning: only {free_gb:.1f} GB free; this model needs ~{size_gb} GB.",
                err=True,
            )
        typer.echo(f"Downloading {label} (~{size_gb} GB, needs ~{min_ram} GB RAM to run):")
    else:
        typer.echo(f"Downloading {label}:")
    typer.echo(f"  {path}\n")

    try:
        backend.download(force=force)
    except Exception as e:
        typer.echo(f"\nDownload failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Ready. Use it with: freeaiagent config set default_model {target}"
               if model and not target.startswith("http")
               else "Ready. Start the agent with: freeaiagent start")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send. Omit for interactive mode."),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="Override system prompt."),
    session: str = typer.Option("default", "--session", help="Session ID for conversation context."),
):
    """Chat with the agent (context is preserved per session)."""
    if message:
        _do_chat(message, system, session)
    else:
        typer.echo(f"freeaiagent chat — session: {session} — Ctrl+C or 'exit' to quit\n")
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
            _do_chat(msg, system, session)


def _do_chat(message: str, system: Optional[str], session: str = "default"):
    payload: dict = {"message": message, "session_id": session}
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
# sessions
# ---------------------------------------------------------------------------

@app.command("sessions")
def list_sessions():
    """List all chat sessions."""
    data = _agent_get("/sessions")
    rows = data.get("sessions", [])
    if not rows:
        typer.echo("No sessions yet. Start one with: freeaiagent chat --session <name>")
        return
    for s in rows:
        sid = s["id"][:8] + "…" if len(s["id"]) > 8 else s["id"]
        typer.echo(f"  {sid}  {s['title']:<40}  {s['message_count']} msgs")


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
def models(
    available: bool = typer.Option(
        False, "--available", "-a",
        help="List downloadable catalog models instead of the active backend's models.",
    ),
):
    """List models — running on the active backend, or downloadable from the catalog."""
    if available:
        from . import catalog
        default = load().get("default_model", catalog.DEFAULT_MODEL)
        typer.echo("Downloadable local models  (freeaiagent pull <name>):\n")
        for name, e in catalog.all_entries():
            mark = "*" if name == default else " "
            run_via = "engine" if e.get("kind") == "gguf" else "fused"
            typer.echo(
                f" {mark} {name:<14} {e['size_gb']:>4.1f} GB  RAM>={e['min_ram_gb']:>2}GB  "
                f"[{e['tier']:<4}] {run_via:<6} {e['description']}"
            )
        typer.echo("\n  * = current default. 'engine' models also fetch a one-time ~305 MB runtime.")
        typer.echo("  Any other GGUF: freeaiagent pull <gguf-url>")
        return

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
def context_show(
    session: str = typer.Option("default", "--session", help="Session ID."),
):
    """Print the conversation history for a session."""
    data = _agent_get(f"/context?session={session}")
    if not data["messages"]:
        typer.echo(f"No context for session '{session}'.")
        return
    for m in data["messages"]:
        label = "You  " if m["role"] == "user" else "Agent"
        typer.echo(f"[{label}] {m['content']}\n")
    typer.echo(f"Total: {data['total']} messages  (session: {session})")


@context_app.command("clear")
def context_clear(
    session: str = typer.Option("default", "--session", help="Session ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Clear conversation history for a session."""
    if not yes:
        typer.confirm(f"Clear context for session '{session}'?", abort=True)
    try:
        r = httpx.delete(f"{_base_url()}/context", params={"session": session}, timeout=5.0)
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
