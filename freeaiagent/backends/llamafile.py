import atexit
import platform
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Callable

import httpx2 as httpx
from typing import AsyncIterator

# on_chunk(done_bytes, total_bytes, phase) — phase is "engine" | "model".
ProgressCallback = Callable[[int, int, str], None]
from .base import BaseBackend
from ._sse import openai_sse_deltas
from .. import catalog

_BASE_DIR = Path.home() / ".freeaiagent"
LLAMAFILE_DIR = _BASE_DIR / "llamafile"   # fused .llamafile executables
MODELS_DIR = _BASE_DIR / "models"         # external .gguf weights (engine mode)
ENGINE_DIR = _BASE_DIR / "engine"         # the bare llamafile engine binary

# Bare llamafile engine (no weights). Runs any external GGUF via `-m`.
# Pinned for reproducibility; ~305 MB, downloaded once and reused for all GGUFs.
ENGINE_VERSION = "0.10.3"
ENGINE_URL = (
    f"https://github.com/mozilla-ai/llamafile/releases/download/"
    f"{ENGINE_VERSION}/llamafile-{ENGINE_VERSION}"
)

# Default local model is a catalog name (resolved to a URL via catalog.py).
# 3B (not 1B) because the fallback workload includes reasoning/Q&A, which 1B can't do.
DEFAULT_MODEL = catalog.DEFAULT_MODEL  # "llama-3.2-3b"
DEFAULT_URL = catalog.url_for(DEFAULT_MODEL)

_proc: Optional[subprocess.Popen] = None


def _stop() -> None:
    if _proc and _proc.poll() is None:
        _proc.terminate()


atexit.register(_stop)


class LlamafileBackend(BaseBackend):
    """
    Self-contained local LLM backend — no Ollama, no keys, no installs.
    Downloads a single executable (~1.4 GB) on first use and starts it as
    a local HTTP server on the configured port. Subsequent starts are instant.
    """

    def __init__(
        self,
        port: int = 8080,
        model: str = DEFAULT_MODEL,
        download_url: Optional[str] = None,
        auto_download: bool = False,
        auto_start: bool = True,
    ):
        self.port = port
        self.model = model
        # explicit download_url wins; otherwise resolve the model via the catalog;
        # fall back to the default URL for unknown names.
        self.download_url = download_url or catalog.url_for(model) or DEFAULT_URL
        self.auto_download = auto_download
        self.auto_start = auto_start

    @property
    def _api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _is_gguf(self) -> bool:
        """External GGUF weights (engine mode) vs. a fused .llamafile executable."""
        return self.download_url.split("?")[0].endswith(".gguf")

    def _bin(self) -> Path:
        """Path to the model artifact (a fused .llamafile, or a .gguf in engine mode)."""
        name = self.download_url.split("/")[-1].split("?")[0]
        if self._is_gguf():
            return MODELS_DIR / name
        # fused llamafile: needs a .exe extension to run on Windows
        if platform.system() == "Windows" and not name.endswith(".exe"):
            name += ".exe"
        return LLAMAFILE_DIR / name

    def _engine_path(self) -> Path:
        name = f"llamafile-{ENGINE_VERSION}"
        if platform.system() == "Windows":
            name += ".exe"
        return ENGINE_DIR / name

    def _installed(self) -> bool:
        """True when everything needed to run this model is present locally."""
        if not self._bin().exists():
            return False
        if self._is_gguf() and not self._engine_path().exists():
            return False
        return True

    def _running(self) -> bool:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/health", timeout=2
            ).close()
            return True
        except Exception:
            return False

    def download(self, force: bool = False, on_chunk: Optional[ProgressCallback] = None) -> Path:
        """Download what's needed to run this model, with a progress bar.

        For GGUF models this also fetches the shared llamafile engine binary
        (once). Returns the path to the model artifact. Idempotent.

        `on_chunk(done_bytes, total_bytes, phase)` is invoked per chunk when
        supplied; `phase` is "engine" for the shared runtime and "model" for the
        weights. When omitted, the CLI progress bar is printed to stdout instead
        (unchanged behaviour). This callback is what lets the SSE endpoint and
        the SDK subscribe to live download progress.
        """
        if self._is_gguf():
            engine = self._engine_path()
            if not engine.exists():
                if on_chunk is None:
                    print("First-time setup: downloading the llamafile engine (~305 MB, one-time).")
                self._download_file(ENGINE_URL, engine, make_exec=True, on_chunk=on_chunk, phase="engine")
        return self._download_file(
            self.download_url, self._bin(), force=force,
            make_exec=not self._is_gguf(), on_chunk=on_chunk, phase="model",
        )

    def _download_file(
        self,
        url: str,
        dest: Path,
        force: bool = False,
        make_exec: bool = False,
        on_chunk: Optional[ProgressCallback] = None,
        phase: str = "model",
    ) -> Path:
        if dest.exists() and not force:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / (dest.name + ".part")
        try:
            with urllib.request.urlopen(url) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    while chunk := resp.read(1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if on_chunk is not None:
                            on_chunk(done, total, phase)
                        elif total:
                            self._print_progress(done, total)
            tmp.rename(dest)
            if on_chunk is None:
                print("\n  Download complete.\n")
            if make_exec and platform.system() != "Windows":
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return dest

    @staticmethod
    def _print_progress(done: int, total: int) -> None:
        pct = done * 100 // total
        mb = done / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        bar_len = 30
        filled = pct * bar_len // 100
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:3d}%  {mb:6.1f} / {total_mb:.1f} MB", end="", flush=True)

    def _command(self) -> list:
        """Subprocess argv: run the engine with -m for GGUF, else the fused file."""
        common = [
            "--server",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--nobrowser",
            "-ngl", "9999",      # use GPU layers if available, CPU otherwise
        ]
        if self._is_gguf():
            return [str(self._engine_path()), "-m", str(self._bin())] + common
        return [str(self._bin())] + common

    def _start(self) -> None:
        global _proc
        if self._running():
            return
        if not self._installed():
            raise FileNotFoundError(f"local model not installed: {self._bin()}")
        url = f"http://127.0.0.1:{self.port}"
        engine_note = " (engine + GGUF)" if self._is_gguf() else ""
        print(f"\n[local model] starting: {self._bin().name}{engine_note}")
        print(f"[local model] server:   {url}  (OpenAI-compatible at {url}/v1)")
        print(f"[local model] loading model into memory...")
        _proc = subprocess.Popen(
            self._command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for i in range(60):
            if self._running():
                print(f"[local model] ready at {url} — send chats to the agent "
                      f"(e.g. `freeaiagent chat \"hi\"`), not directly here.\n")
                return
            time.sleep(1)
            if i > 0 and i % 10 == 9:
                print(f"[local model] still loading... ({i + 1}s)")
        raise RuntimeError("llamafile did not become ready within 60 seconds.")

    async def is_available(self) -> bool:
        import asyncio
        if self._running():
            return True
        if not self.auto_start:
            return False
        try:
            if not self._installed():
                if not self.auto_download:
                    return False  # model not installed — run `freeaiagent pull`
                await asyncio.to_thread(self.download)
            if self._installed():
                await asyncio.to_thread(self._start)
                return self._running()
        except Exception as e:
            print(f"\nLlamafile setup failed: {e}")
        return False

    async def chat(self, messages: List[Dict], model: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._api_base}/chat/completions",
                json={"model": model, "messages": messages},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def stream(self, messages: List[Dict], model: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._api_base}/chat/completions",
                json={"model": model, "messages": messages, "stream": True},
            ) as r:
                r.raise_for_status()
                async for delta in openai_sse_deltas(r):
                    yield delta

    async def chat_completion(self, messages: List[Dict], model: str, tools=None) -> Dict:
        payload: Dict = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{self._api_base}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]

    async def available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._api_base}/models")
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", [])]
                return ids if ids else [DEFAULT_MODEL]
        except Exception:
            return [DEFAULT_MODEL]
