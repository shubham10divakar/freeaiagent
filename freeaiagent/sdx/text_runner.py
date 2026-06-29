"""Text model runner for the SDX compound engine.

Wraps llama-cpp-python GGUF inference with async streaming via a thread +
queue bridge. The model is lazy-loaded on first use.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import AsyncIterator, Optional

_STOP_SEQUENCES = ["\nUser:", "[Current turn]", "\n[System]"]


class TextRunner:
    """Stream tokens from a GGUF model given a flat text context string."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm = None
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

    def load(self) -> None:
        """Load the GGUF text model into memory (blocking)."""
        with self._lock:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install it with: pip install freeaiagent[llama-cpp]"
                )
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )

    async def generate(self, context: str) -> AsyncIterator[str]:
        """Yield tokens from a flat text context string.

        Bridges the synchronous llama-cpp generator to an async generator via
        a queue. The model is loaded lazily on first call.
        """
        if self._llm is None:
            await asyncio.to_thread(self.load)

        self._stop_flag.clear()
        q: "queue.Queue[object]" = queue.Queue()
        _sentinel = object()
        loop = asyncio.get_event_loop()

        def _worker() -> None:
            try:
                stream = self._llm(
                    context,
                    max_tokens=1024,
                    stop=_STOP_SEQUENCES,
                    stream=True,
                )
                for chunk in stream:
                    if self._stop_flag.is_set():
                        break
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        loop.call_soon_threadsafe(q.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _sentinel)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await asyncio.to_thread(q.get)
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]

    def stop(self) -> None:
        self._stop_flag.set()

    def unload(self) -> None:
        with self._lock:
            self._llm = None

    def is_loaded(self) -> bool:
        return self._llm is not None
