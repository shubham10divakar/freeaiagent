"""Vision model runner for the SDX compound engine.

Wraps llama-cpp-python multimodal inference to extract image descriptions.
Lazy-loaded: the model is only brought into memory on first use.
"""
from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import Optional

_EXTRACT_PROMPT = (
    "If the user asked a question about the image, answer it concisely. "
    "Otherwise, describe the image in detail: objects, text, layout, colors, quantities. "
    'User\'s message: "{user_text}"'
)


class VisionRunner:
    """Wraps a multimodal llama-cpp-python model for image description extraction.

    Supports two model families:
    - moondream2 (no mmproj): pass ``mmproj_path=None``
    - LLaVA 1.5/1.6 (requires mmproj): pass ``mmproj_path="path/to/mmproj.gguf"``
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_gpu_layers: int = 0,
    ) -> None:
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._n_gpu_layers = n_gpu_layers
        self._llm = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the vision model into memory (blocking, safe to call from thread pool)."""
        with self._lock:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import (
                    MoondreamChatHandler,
                    Llava16ChatHandler,
                )
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install it with: pip install freeaiagent[llama-cpp]"
                )
            if self._mmproj_path:
                handler = Llava16ChatHandler(
                    clip_model_path=self._mmproj_path, verbose=False
                )
                n_ctx = 4096
            else:
                handler = MoondreamChatHandler(verbose=False)
                n_ctx = 2048
            self._llm = Llama(
                model_path=self._model_path,
                chat_handler=handler,
                n_ctx=n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )

    async def extract(self, image_path: str, user_text: str) -> str:
        """Return an image description or direct answer to ``user_text``.

        Loads the model lazily on first call. Runs in a thread pool so the
        event loop is not blocked during the ~2–8 s inference.
        """
        if self._llm is None:
            await asyncio.to_thread(self.load)

        img_bytes = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lstrip(".").lower() or "jpeg"
        b64 = base64.b64encode(img_bytes).decode()
        data_uri = f"data:image/{ext};base64,{b64}"

        prompt = _EXTRACT_PROMPT.format(user_text=user_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        def _run() -> str:
            result = self._llm.create_chat_completion(messages=messages, max_tokens=512)
            return result["choices"][0]["message"]["content"].strip()

        return await asyncio.to_thread(_run)

    def unload(self) -> None:
        with self._lock:
            self._llm = None

    def is_loaded(self) -> bool:
        return self._llm is not None
