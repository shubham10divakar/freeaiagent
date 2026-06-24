"""Synchronous Python SDK for freeaiagent.

One import, zero HTTP boilerplate. Streaming and download methods return plain
iterators so callers never touch ``async``:

    from freeaiagent import Client

    agent = Client(name="magpie", auto_start=True)
    print(agent.chat("hello"))

    for token in agent.stream("write a haiku"):
        print(token, end="", flush=True)

    for p in agent.pull("qwen2.5-7b"):
        print(f"{p.pct:.0f}%")
"""
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx2 as httpx

from . import server as server_mod


# ── Exceptions ───────────────────────────────────────────────────────────────

class FreeAIAgentError(Exception):
    """Base class for all SDK errors."""


class ServerNotRunning(FreeAIAgentError):
    """The freeaiagent server is not reachable (start it, or pass auto_start=True)."""


class BackendUnavailable(FreeAIAgentError):
    """The server is up but no LLM backend is available (HTTP 503)."""


class DownloadInProgress(FreeAIAgentError):
    """Another model download is already running (HTTP 409)."""


# ── Pull progress ────────────────────────────────────────────────────────────

@dataclass
class PullProgress:
    type: str                       # "start" | "progress" | "done" | "error"
    phase: str = ""                 # "engine" | "model"
    label: str = ""
    pct: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    path: Optional[str] = None       # set on "done"
    error: Optional[str] = None      # set on "error"

    @classmethod
    def from_event(cls, ev: dict) -> "PullProgress":
        return cls(
            type=ev.get("type", ""),
            phase=ev.get("phase", ""),
            label=ev.get("label", ""),
            pct=float(ev.get("pct") or 0.0),
            downloaded_mb=float(ev.get("downloaded_mb") or 0.0),
            total_mb=float(ev.get("total_mb") or 0.0),
            speed_mbps=float(ev.get("speed_mbps") or 0.0),
            path=ev.get("path"),
            error=ev.get("message") or ev.get("error"),
        )


def _iter_sse(lines: Iterator[str]) -> Iterator[dict]:
    """Yield decoded JSON objects from SSE ``data:`` lines until ``[DONE]``."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


# ── Client ───────────────────────────────────────────────────────────────────

class Client:
    """A handle to a running freeaiagent server.

    Args:
        name: identifies the caller; sent as the ``X-Caller-ID`` header so this
            app gets its own auto-resolved session.
        session: default session id for chat calls.
        base_url: pin an explicit URL (skips lock-file port discovery).
        port: pin a port on localhost (skips discovery).
        auto_start: start the server in the background if it isn't running.
        timeout: default request timeout (seconds).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        session: str = "default",
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        auto_start: bool = False,
        timeout: float = 120.0,
        _http: Optional["httpx.Client"] = None,
    ):
        self.name = name
        self.session = session
        self.timeout = timeout
        self._explicit_base = base_url.rstrip("/") if base_url else None
        self._port = port
        self._http = _http              # injected client (tests / advanced use)
        self._proc: Optional[subprocess.Popen] = None

        self.models = _Models(self)
        self.sessions = _Sessions(self)
        self.context = _Context(self)
        self.config = _Config(self)
        self.tools = _Tools(self)

        if auto_start:
            self.start()

    # ── transport ────────────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        if self._explicit_base:
            return self._explicit_base
        port = self._port or server_mod.discover_port()
        return f"http://localhost:{port}"

    def _headers(self) -> Dict[str, str]:
        return {"X-Caller-ID": self.name} if self.name else {}

    def _url(self, path: str) -> str:
        # An injected client carries its own base_url; use the relative path.
        return path if self._http is not None else f"{self.base_url}{path}"

    def _request(self, method: str, path: str, *, params=None, json=None, timeout=None):
        try:
            if self._http is not None:
                # Injected transports (e.g. TestClient) carry their own timeout.
                r = self._http.request(method, path, params=params, json=json,
                                       headers=self._headers())
            else:
                r = httpx.request(method, self._url(path), params=params, json=json,
                                  headers=self._headers(), timeout=timeout or self.timeout)
        except httpx.ConnectError as e:
            raise ServerNotRunning(
                "freeaiagent server is not running. Start it with `freeaiagent start` "
                "or pass auto_start=True."
            ) from e
        self._raise_for_status(r)
        return r.json()

    def _stream_cm(self, method: str, path: str, *, json=None, timeout=None):
        if self._http is not None:
            return self._http.stream(method, path, json=json, headers=self._headers())
        return httpx.stream(method, self._url(path), json=json, headers=self._headers(),
                            timeout=timeout or self.timeout)

    @staticmethod
    def _raise_for_status(r) -> None:
        if r.status_code == 503:
            raise BackendUnavailable(_detail(r, "No backend available."))
        if r.status_code == 409:
            raise DownloadInProgress(_detail(r, "A download is already in progress."))
        r.raise_for_status()

    # ── chat / task ──────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        *,
        session: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        system: Optional[str] = None,
        tools: bool = False,
    ) -> str:
        """Send a message; conversation history is preserved per session."""
        payload: Dict[str, Any] = {"message": message, "session_id": session or self.session}
        if model:
            payload["model"] = model
        if backend:
            payload["backend"] = backend
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = True
        return self._request("POST", "/chat", json=payload)["response"]

    def stream(
        self,
        message: str,
        *,
        session: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream a reply token-by-token."""
        payload: Dict[str, Any] = {"message": message, "session_id": session or self.session}
        if model:
            payload["model"] = model
        if backend:
            payload["backend"] = backend
        if system:
            payload["system"] = system
        try:
            with self._stream_cm("POST", "/chat/stream", json=payload) as r:
                if r.status_code >= 400:
                    r.read()
                    self._raise_for_status(r)
                for obj in _iter_sse(r.iter_lines()):
                    if "error" in obj:
                        raise BackendUnavailable(obj["error"])
                    if "token" in obj:
                        yield obj["token"]
        except httpx.ConnectError as e:
            raise ServerNotRunning("freeaiagent server is not running.") from e

    def task(
        self,
        description: str,
        *,
        input: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str:
        """Run a one-shot task — no shared context is read or written."""
        payload: Dict[str, Any] = {"task": description}
        if input is not None:
            payload["input"] = input
        if model:
            payload["model"] = model
        if system:
            payload["system"] = system
        return self._request("POST", "/task", json=payload)["result"]

    # ── pull / discovery ─────────────────────────────────────────────────────

    def pull(
        self,
        model: Optional[str] = None,
        *,
        force: bool = False,
        on_progress: Optional[Callable[["PullProgress"], None]] = None,
    ) -> Iterator["PullProgress"]:
        """Download a model server-side, yielding live ``PullProgress`` events.

        Pass ``on_progress`` for a callback style; the iterator still yields the
        same events. Raises ``DownloadInProgress`` if another pull is running.
        """
        payload = {"model": model, "force": force}
        try:
            with self._stream_cm("POST", "/pull/stream", json=payload, timeout=None) as r:
                if r.status_code >= 400:
                    r.read()
                    self._raise_for_status(r)
                for obj in _iter_sse(r.iter_lines()):
                    p = PullProgress.from_event(obj)
                    if on_progress is not None:
                        on_progress(p)
                    yield p
        except httpx.ConnectError as e:
            raise ServerNotRunning("freeaiagent server is not running.") from e

    def search(self, query: str, *, limit: int = 20) -> list:
        """Search HuggingFace for GGUF models, or list a repo's GGUF files.

        A bare term lists repos; an ``owner/name`` lists that repo's files.
        """
        from . import hf
        if "/" in query:
            return hf.list_gguf_files(query)
        return hf.search_models(query, limit=limit)

    # ── health / lifecycle ───────────────────────────────────────────────────

    def health(self) -> dict:
        """Return the server's health payload."""
        return self._request("GET", "/health", timeout=10.0)

    def is_running(self) -> bool:
        """Fast reachability check — True if the server answers /health."""
        try:
            self.health()
            return True
        except (ServerNotRunning, httpx.HTTPError):
            return False

    def start(self, *, wait: float = 30.0) -> None:
        """Start the server in the background if it isn't already running."""
        if self.is_running():
            return
        port = self._port or server_mod.discover_port()
        cmd = [sys.executable, "-m", "freeaiagent", "start", "--port", str(port)]
        self._proc = subprocess.Popen(cmd)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if self.is_running():
                return
            time.sleep(0.5)
        raise ServerNotRunning(f"server did not become ready within {wait:.0f}s.")

    def stop(self) -> None:
        """Stop a server this client started (best-effort)."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def _detail(r, default: str) -> str:
    try:
        return r.json().get("detail", default)
    except Exception:
        return default


# ── namespaces ───────────────────────────────────────────────────────────────

class _Models:
    def __init__(self, c: "Client"):
        self._c = c

    def list(self, *, backend: Optional[str] = None) -> List[str]:
        """Models available on the active (or named) backend."""
        params = {"backend": backend} if backend else None
        return self._c._request("GET", "/models", params=params)["models"]

    def catalog(self) -> list:
        """Curated downloadable catalog, each flagged installed or not."""
        return self._c._request("GET", "/models/catalog")["models"]

    def installed(self) -> list:
        """Local model files on disk (name, path, size, kind)."""
        return self._c._request("GET", "/models/installed")["models"]

    def rm(self, name: str) -> dict:
        """Delete an installed model file to free disk space.

        ``name`` is a catalog name or an on-disk filename (see ``installed()``).
        Returns ``{"deleted", "path", "freed_mb"}``.
        """
        return self._c._request("DELETE", f"/models/installed/{name}")

    def active(self) -> Optional[str]:
        """The model the active backend would currently use."""
        return self._c.health().get("default_model")


class _Sessions:
    def __init__(self, c: "Client"):
        self._c = c

    def list(self) -> list:
        return self._c._request("GET", "/sessions")["sessions"]

    def create(self, id: Optional[str] = None, *, title: Optional[str] = None) -> dict:
        return self._c._request("POST", "/sessions", json={"id": id, "title": title})

    def rename(self, id: str, title: str) -> dict:
        return self._c._request("PATCH", f"/sessions/{id}", json={"title": title})

    def delete(self, id: str) -> dict:
        return self._c._request("DELETE", f"/sessions/{id}")


class _Context:
    def __init__(self, c: "Client"):
        self._c = c

    def get(self, *, session: Optional[str] = None) -> list:
        s = session or self._c.session
        return self._c._request("GET", "/context", params={"session": s})["messages"]

    def clear(self, *, session: Optional[str] = None) -> int:
        s = session or self._c.session
        return self._c._request("DELETE", "/context", params={"session": s})["cleared"]


class _Config:
    def __init__(self, c: "Client"):
        self._c = c

    def get(self) -> dict:
        return self._c._request("GET", "/config")

    def set(self, key: str, value: Any) -> dict:
        return self._c._request("POST", "/config/set", json={"key": key, "value": value})


class _Tools:
    def __init__(self, c: "Client"):
        self._c = c

    def register(self, name: str, *, description: str, endpoint: str,
                 parameters: Optional[dict] = None) -> dict:
        return self._c._request("POST", "/tools/register", json={
            "name": name, "description": description,
            "endpoint": endpoint, "parameters": parameters,
        })

    def list(self) -> list:
        return self._c._request("GET", "/tools")["tools"]

    def unregister(self, name: str) -> dict:
        return self._c._request("DELETE", f"/tools/{name}")
