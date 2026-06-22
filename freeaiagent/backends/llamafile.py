import atexit
import platform
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

import httpx2 as httpx
from typing import AsyncIterator
from .base import BaseBackend
from ._sse import openai_sse_deltas
from .. import catalog

LLAMAFILE_DIR = Path.home() / ".freeaiagent" / "llamafile"

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

    def _bin(self) -> Path:
        name = self.download_url.split("/")[-1].split("?")[0]
        if platform.system() == "Windows" and not name.endswith(".exe"):
            name += ".exe"
        return LLAMAFILE_DIR / name

    def _running(self) -> bool:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/health", timeout=2
            ).close()
            return True
        except Exception:
            return False

    def download(self, force: bool = False) -> Path:
        """Download the llamafile model, streaming a progress bar to stdout.

        Returns the path to the (already- or newly-) downloaded binary.
        Idempotent: a no-op if the file already exists unless force=True.
        """
        path = self._bin()
        if path.exists() and not force:
            return path
        LLAMAFILE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(self.download_url) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    while chunk := resp.read(1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            self._print_progress(done, total)
            tmp.rename(path)
            print("\n  Download complete.\n")
            if platform.system() != "Windows":
                path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def _print_progress(done: int, total: int) -> None:
        pct = done * 100 // total
        mb = done / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        bar_len = 30
        filled = pct * bar_len // 100
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:3d}%  {mb:6.1f} / {total_mb:.1f} MB", end="", flush=True)

    def _start(self) -> None:
        global _proc
        if self._running():
            return
        path = self._bin()
        if not path.exists():
            raise FileNotFoundError(f"llamafile binary not found: {path}")
        url = f"http://127.0.0.1:{self.port}"
        print(f"\n[local model] starting: {path.name}")
        print(f"[local model] server:   {url}  (OpenAI-compatible at {url}/v1)")
        print(f"[local model] loading model into memory...")
        _proc = subprocess.Popen(
            [
                str(path),
                "--server",
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--nobrowser",
                "-ngl", "9999",      # use GPU layers if available, CPU otherwise
            ],
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
            if not self._bin().exists():
                if not self.auto_download:
                    return False  # model not installed — run `freeaiagent pull`
                await asyncio.to_thread(self.download)
            if self._bin().exists():
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

    async def available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._api_base}/models")
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", [])]
                return ids if ids else [DEFAULT_MODEL]
        except Exception:
            return [DEFAULT_MODEL]
