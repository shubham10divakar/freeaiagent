"""SDXEngine — public orchestrator for the compound text+vision engine.

Owns VisionRunner and TextRunner. Exposes two high-level async methods:
- ``extract_vision`` — image → description string (runs vision model only)
- ``stream``         — messages + text → token iterator (runs text model only)

Vision extraction and text streaming are intentionally separate so the
endpoint can persist the description to SQLite between the two steps.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Optional

from .context_builder import ContextBuilder
from .text_runner import TextRunner
from .vision_runner import VisionRunner


class SDXEngine:
    """Compound text+vision engine (Smart Decision eXecution).

    Both sub-models are lazy-loaded on first use and cached for the lifetime
    of this engine instance. Unload releases native memory.
    """

    def __init__(
        self,
        text_model_path: str,
        vision_model_path: str,
        mmproj_path: Optional[str] = None,
        token_budget: int = 8192,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
    ) -> None:
        self._text = TextRunner(text_model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        self._vision = VisionRunner(
            vision_model_path, mmproj_path=mmproj_path, n_gpu_layers=n_gpu_layers
        )
        self._token_budget = token_budget

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def load(self, on_progress: Optional[Callable[[str, float], None]] = None) -> None:
        """Pre-load both sub-models. ``on_progress(phase, pct)`` is optional."""
        if on_progress:
            on_progress("text", 0.0)
        await asyncio.to_thread(self._text.load)
        if on_progress:
            on_progress("text", 100.0)
            on_progress("vision", 0.0)
        await asyncio.to_thread(self._vision.load)
        if on_progress:
            on_progress("vision", 100.0)

    def stop(self) -> None:
        self._text.stop()

    def unload(self) -> None:
        self._text.unload()
        self._vision.unload()

    def is_loaded(self) -> bool:
        return self._text.is_loaded()

    # ── Inference ────────────────────────────────────────────────────────────

    async def extract_vision(self, image_path: str, user_text: str) -> str:
        """Run vision extraction and return a description string.

        This is intentionally separate from ``stream`` so the caller can
        persist the description to SQLite before the text model starts.
        """
        return await self._vision.extract(image_path, user_text)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens for the current turn.

        ``messages`` is the full OpenAI-format message list (from SQLite),
        already including any ``[SDX-Image]:`` system messages for prior image
        turns and the current user message as the last element.
        """
        ctx = ContextBuilder(messages, token_budget=self._token_budget).build()
        async for token in self._text.generate(ctx):
            yield token
