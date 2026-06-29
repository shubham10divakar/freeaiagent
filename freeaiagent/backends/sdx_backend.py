"""SDX backend — wraps SDXEngine as a BaseBackend.

One SDXEngine instance is created per downloaded model tier and cached for
the server's lifetime. The backend is registered under type ``"sdx"`` in
``router._build_backends()``.

Vision extraction is exposed as ``extract_vision(image_path, user_text)``
rather than being hidden inside ``stream()`` so the chat endpoint can:
  1. Call extract_vision, receive the description string.
  2. Persist the description to SQLite *before* the assistant message.
  3. Inject the description into the messages list for ContextBuilder.
  4. Call stream() for normal text-model token generation.
"""
from __future__ import annotations

from typing import AsyncIterator, List, Dict, Optional

from .base import BaseBackend
from ..sdx.catalog import SDX_CATALOG, SDX_DIR, is_installed, model_paths
from ..sdx.engine import SDXEngine

# Cache SDXEngine instances by model id; loading is expensive.
_ENGINES: dict[str, SDXEngine] = {}


class SDXBackend(BaseBackend):

    def __init__(self, config: dict) -> None:
        self._config = config
        self._default_model = config.get("model", "sdx-standard")

    # ── Engine lifecycle ─────────────────────────────────────────────────────

    def _build_engine(self, model_id: str) -> SDXEngine:
        if model_id not in SDX_CATALOG:
            raise ValueError(
                f"Unknown SDX model '{model_id}'. "
                f"Available: {list(SDX_CATALOG)}"
            )
        if not is_installed(model_id):
            raise FileNotFoundError(
                f"SDX model '{model_id}' is not downloaded. "
                f"Run: freeaiagent pull {model_id}"
            )
        entry = SDX_CATALOG[model_id]
        paths = model_paths(model_id)
        return SDXEngine(
            text_model_path=paths["text"],
            vision_model_path=paths["vision"],
            mmproj_path=paths["mmproj"],
            token_budget=entry["token_budget"],
            n_ctx=entry["token_budget"],
            n_gpu_layers=self._config.get("n_gpu_layers", 0),
        )

    def _get_engine(self, model_id: str) -> SDXEngine:
        if model_id not in _ENGINES:
            _ENGINES[model_id] = self._build_engine(model_id)
        return _ENGINES[model_id]

    # ── Vision hook (called by the chat endpoint before streaming) ───────────

    async def extract_vision(self, image_path: str, user_text: str, model: Optional[str] = None) -> str:
        """Extract an image description with the vision sub-model."""
        engine = self._get_engine(model or self._default_model)
        return await engine.extract_vision(image_path, user_text)

    # ── BaseBackend interface ────────────────────────────────────────────────

    async def stream(self, messages: List[Dict], model: str, **kwargs) -> AsyncIterator[str]:
        engine = self._get_engine(model or self._default_model)
        async for token in engine.stream(messages):
            yield token

    async def chat(self, messages: List[Dict], model: str) -> str:
        return "".join([t async for t in self.stream(messages, model)])

    async def available_models(self) -> List[str]:
        return [m for m in SDX_CATALOG if is_installed(m)]

    async def is_available(self) -> bool:
        return bool(await self.available_models())
