"""In-process GGUF backend via ``llama-cpp-python``.

A pure-Python path that loads a GGUF model directly into this process — no
llamafile subprocess, no local HTTP hop. Optional: install with
``pip install freeaiagent[llama-cpp]`` (or ``pip install llama-cpp-python``).

The model is loaded lazily on first use and cached per file path, so repeated
chats reuse the same in-memory model. Reuses the GGUF weights that
``freeaiagent pull`` already downloads into ``~/.freeaiagent/models/``.
"""
import asyncio
import queue
import threading
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from .base import BaseBackend
from .. import catalog
from .llamafile import MODELS_DIR

# Cache loaded models by path — loading is expensive (seconds + RAM).
_INSTANCES: dict = {}


def _import_llama():
    """Return the ``Llama`` class, or None if llama-cpp-python isn't installed."""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        return None


class LlamaCppBackend(BaseBackend):
    def __init__(
        self,
        model: Optional[str] = None,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        **extra,
    ):
        self.model = model or catalog.DEFAULT_MODEL
        self._explicit_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.extra = extra

    def _resolve_path(self) -> Optional[Path]:
        """The GGUF file this backend would load (explicit path or catalog)."""
        if self._explicit_path:
            return Path(self._explicit_path)
        entry = catalog.get(self.model)
        if entry and entry.get("kind") == "gguf":
            name = entry["url"].split("/")[-1].split("?")[0]
            return MODELS_DIR / name
        return None

    def _available_path(self) -> Optional[Path]:
        p = self._resolve_path()
        return p if (p and p.exists()) else None

    async def is_available(self) -> bool:
        """True only when the package is installed *and* a GGUF file is present."""
        return _import_llama() is not None and self._available_path() is not None

    def _get_llama(self):
        Llama = _import_llama()
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install freeaiagent[llama-cpp]"
            )
        path = self._available_path()
        if path is None:
            raise FileNotFoundError(
                f"No GGUF model found for '{self.model}'. "
                f"Download one with: freeaiagent pull {self.model}"
            )
        key = str(path)
        if key not in _INSTANCES:
            _INSTANCES[key] = Llama(
                model_path=key,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                **self.extra,
            )
        return _INSTANCES[key]

    async def chat(self, messages: List[Dict], model: str) -> str:
        def _run() -> str:
            llm = self._get_llama()
            out = llm.create_chat_completion(messages=messages)
            return out["choices"][0]["message"]["content"]

        return await asyncio.to_thread(_run)

    async def stream(self, messages: List[Dict], model: str) -> AsyncIterator[str]:
        """Stream tokens by bridging llama-cpp's sync generator to async."""
        q: "queue.Queue" = queue.Queue()
        sentinel = object()

        def worker():
            try:
                llm = self._get_llama()
                for chunk in llm.create_chat_completion(messages=messages, stream=True):
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        q.put(token)
            except Exception as e:  # propagate to the consumer
                q.put(e)
            finally:
                q.put(sentinel)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await asyncio.to_thread(q.get)
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def available_models(self) -> List[str]:
        return [self.model] if self._available_path() else []
