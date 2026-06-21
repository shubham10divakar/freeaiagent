import httpx2 as httpx
from typing import List, Dict
from .base import BaseBackend

# Free-tier models available on Groq as of mid-2026
GROQ_FREE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


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
        return GROQ_FREE_MODELS
