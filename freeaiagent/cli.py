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
service_app = typer.Typer(help="Inspect the installed system service.")
app.add_typer(context_app, name="context")
app.add_typer(config_app, name="config")
app.add_typer(service_app, name="service")

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
    open: bool = typer.Option(False, "--open", help="Open the Chat UI in your browser after the server starts."),
):
    """Start the freeaiagent server."""
    import threading
    import time
    import webbrowser
    import uvicorn
    from . import server as server_mod

    cfg = load()
    p = port or cfg.get("port", 7731)
    ui_url = f"http://localhost:{p}/ui"
    typer.echo(f"freeaiagent running at http://localhost:{p}")
    typer.echo(f"Chat UI:            {ui_url}")
    typer.echo(f"API docs:           http://localhost:{p}/docs")
    typer.echo(f"Backend / model:    {cfg.get('default_backend')} / {cfg.get('default_model')}")
    typer.echo(f"Try it:             freeaiagent chat \"hello\"")
    typer.echo("")

    if open:
        def _open_when_ready():
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    httpx.get(f"http://localhost:{p}/health", timeout=1.0)
                    webbrowser.open(ui_url)
                    return
                except Exception:
                    time.sleep(0.5)
        threading.Thread(target=_open_when_ready, daemon=True).start()

    # Publish the port so SDK clients can auto-discover us; clean up on exit.
    server_mod.write_lock(p)
    try:
        uvicorn.run(
            "freeaiagent.main:app",
            host="127.0.0.1",
            port=p,
            reload=reload,
        )
    finally:
        server_mod.remove_lock()


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

@app.command()
def pull(
    model: Optional[str] = typer.Argument(
        None,
        help="Catalog name (e.g. llama-3.2-3b), an hf:<repo>/<file.gguf> ref, or a "
        "direct URL. Omit to pull the current default model.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if the model already exists."),
):
    """Download a local model for offline, no-key operation.

    Examples:
      freeaiagent pull                 # the current default model
      freeaiagent pull gemma-2-2b      # a catalog model by name
      freeaiagent pull qwen2.5-7b      # a catalog GGUF (also fetches the engine)
      freeaiagent pull hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf
      freeaiagent pull https://.../model.gguf        # any GGUF / llamafile URL
    """
    import shutil
    from . import catalog, pull as pull_mod
    from .backends.llamafile import LLAMAFILE_DIR

    bcfg = load().get("backends", {}).get("llamafile", {})
    port = bcfg.get("port", 8080)
    target = model or load().get("default_model", catalog.DEFAULT_MODEL)

    try:
        pt = pull_mod.resolve_target(target, port=port)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    backend = pt.backend
    label, size_gb, min_ram = pt.label, pt.size_gb, pt.min_ram_gb

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

    is_catalog_name = bool(model) and not target.startswith(("http://", "https://", "hf:"))
    if not model:
        typer.echo("Ready. Start the agent with: freeaiagent start")
    elif is_catalog_name:
        typer.echo(f"Ready. Make it the default with:\n"
                   f"  freeaiagent config set default_model {target}")
    else:
        # arbitrary URL/hf model: select it by pinning the backend's download_url
        typer.echo(f"Ready. Use it with:\n"
                   f"  freeaiagent config set backends.llamafile.download_url {backend.download_url}\n"
                   f"  freeaiagent start")


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------

@app.command()
def rm(
    model: str = typer.Argument(
        ..., help="Catalog name (e.g. llama-3.2-3b) or an installed filename to delete."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Delete a downloaded model to free disk space.

    Works offline (operates on local files directly). The shared llamafile
    engine binary is never removed. List what's installed with:
      freeaiagent models   (or the SDK's agent.models.installed())
    """
    from . import installed

    path = installed.resolve_path(model)
    if path is None:
        typer.echo(f"No installed model named '{model}'.", err=True)
        files = installed.installed_files()
        if files:
            typer.echo("\nInstalled models:")
            for f in files:
                typer.echo(f"  {f['name']:<48} {f['size_mb']:>8.0f} MB")
        raise typer.Exit(1)

    size_mb = path.stat().st_size / (1024 * 1024)
    if not yes:
        typer.confirm(f"Delete {path.name} ({size_mb:.0f} MB)?", abort=True)
    result = installed.delete(model)
    typer.echo(f"Deleted {result['deleted']} — freed {result['freed_mb']:.0f} MB")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search term (lists GGUF repos), or a full repo id "
        "'owner/name' (lists that repo's GGUF files).",
    ),
    limit: int = typer.Option(20, help="Max repos to show for a term search."),
):
    """Search HuggingFace for GGUF models to pull.

    Examples:
      freeaiagent search qwen2.5                       # find GGUF repos
      freeaiagent search bartowski/Qwen2.5-7B-Instruct-GGUF   # list that repo's files
    """
    from . import hf

    if "/" in query:
        try:
            files = hf.list_gguf_files(query)
        except Exception as e:
            typer.echo(f"Could not list '{query}': {e}", err=True)
            raise typer.Exit(1)
        if not files:
            typer.echo("No .gguf files found in that repo.")
            return
        typer.echo(f"GGUF files in {query}:\n")
        for f in files:
            gb = f["size"] / (1024 ** 3)
            typer.echo(f"  {f['path']:<58} {gb:6.2f} GB")
        typer.echo(f"\nDownload one with:\n  freeaiagent pull hf:{query}/<filename>")
        return

    try:
        repos = hf.search_models(query, limit=limit)
    except Exception as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(1)
    if not repos:
        typer.echo(f"No GGUF repos found for '{query}'.")
        return
    typer.echo(f"GGUF repos matching '{query}' (most downloaded first):\n")
    for m in repos:
        typer.echo(f"  {m['id']:<58} dl={m['downloads']:>10,}  likes={m['likes']}")
    typer.echo("\nList a repo's files with:\n  freeaiagent search <owner/name>")


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


# ---------------------------------------------------------------------------
# install / uninstall / service
# ---------------------------------------------------------------------------

@app.command()
def install():
    """Install freeaiagent to start automatically at login (no admin needed).

    Linux uses a systemd user unit, macOS a launchd agent, Windows an HKCU
    Run entry. Afterwards the agent is always available — apps can use
    Client(auto_start=False).
    """
    from . import service
    try:
        service.install()
    except Exception as e:
        typer.echo(f"Install failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Installed. freeaiagent will start automatically on login.")
    typer.echo("Check status with: freeaiagent service status")


@app.command()
def uninstall():
    """Remove the auto-start service installed by `freeaiagent install`."""
    from . import service
    try:
        service.uninstall()
    except Exception as e:
        typer.echo(f"Uninstall failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Uninstalled. freeaiagent will no longer start automatically.")


@service_app.command("status")
def service_status():
    """Show whether the auto-start service is installed/running."""
    from . import service
    typer.echo(f"Service: {service.status()}")
