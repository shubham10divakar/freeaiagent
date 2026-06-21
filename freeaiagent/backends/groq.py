import httpx2 as httpx
from typing import List, Dict
from .base import BaseBackend

# Fallback used when API key is absent or /v1/models call fails.
# Excludes audio (whisper/orpheus), TTS, and content-moderation (safeguard/guard) models.
# Keep this list updated as Groq's lineup evolves.
GROQ_FALLBACK_MODELS = [
    # Production — current (June 2026)
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound",
    "groq/compound-mini",
    # Production — deprecated Aug 16 2026, still active
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    # Preview — deprecated Jul 17 2026, still active
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    # Preview — active
    "qwen/qwen3.6-27b",
]

# Fragments that identify non-chat models (audio, TTS, moderation)
_EXCLUDE = ("whisper", "tts", "orpheus", "safeguard", "guard")


def _is_chat_model(model_id: str) -> bool:
    lower = model_id.lower()
    return not any(frag in lower for frag in _EXCLUDE)


class GroqBackend(BaseBackend):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    async def chat(self, messages: List[Dict], model: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def available_models(self) -> List[str]:
        """Fetch live model list from Groq API. Falls back to GROQ_FALLBACK_MODELS on error."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                r.raise_for_status()
                all_ids = [m["id"] for m in r.json().get("data", [])]
                return sorted(m for m in all_ids if _is_chat_model(m))
        except Exception:
            return GROQ_FALLBACK_MODELS
