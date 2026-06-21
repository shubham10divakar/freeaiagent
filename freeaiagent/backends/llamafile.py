import atexit
import platform
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

import httpx2 as httpx
from .base import BaseBackend

LLAMAFILE_DIR = Path.home() / ".freeaiagent" / "llamafile"

# Llama-3.2-1B-Instruct: ~1.4 GB, fast, good quality, self-contained executable
DEFAULT_URL = (
    "https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile"
    "/resolve/main/Llama-3.2-1B-Instruct.Q6_K.llamafile"
)
DEFAULT_MODEL = "Llama-3.2-1B-Instruct"

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
        download_url: str = DEFAULT_URL,
        auto_download: bool = True,
        auto_start: bool = True,
    ):
        self.port = port
        self.download_url = download_url
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

    def _download(self) -> None:
        path = self._bin()
        if path.exists():
            return
        LLAMAFILE_DIR.mkdir(parents=True, exist_ok=True)
        print(
            "\nFirst-time setup: downloading local AI model (~1.4 GB)."
            "\nThis happens once — subsequent starts are instant.\n"
        )
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
                            pct = done * 100 // total
                            mb = done // (1024 * 1024)
                            total_mb = total // (1024 * 1024)
                            print(f"\r  {pct:3d}%  {mb} / {total_mb} MB", end="", flush=True)
            tmp.rename(path)
            print("\n  Download complete.\n")
            if platform.system() != "Windows":
                path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _start(self) -> None:
        global _proc
        if self._running():
            return
        path = self._bin()
        if not path.exists():
            raise FileNotFoundError(f"llamafile binary not found: {path}")
        print(f"Starting local AI model on port {self.port}...")
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
                print("  Ready.\n")
                return
            time.sleep(1)
            if i > 0 and i % 10 == 9:
                print(f"  Still loading... ({i + 1}s)")
        raise RuntimeError("llamafile did not become ready within 60 seconds.")

    async def is_available(self) -> bool:
        import asyncio
        if self._running():
            return True
        if not self.auto_start:
            return False
        try:
            if self.auto_download and not self._bin().exists():
                await asyncio.to_thread(self._download)
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

    async def available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._api_base}/models")
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", [])]
                return ids if ids else [DEFAULT_MODEL]
        except Exception:
            return [DEFAULT_MODEL]
