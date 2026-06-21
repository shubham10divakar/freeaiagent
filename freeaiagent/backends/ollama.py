import httpx2 as httpx
from typing import List, Dict
from .base import BaseBackend


class OllamaBackend(BaseBackend):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def chat(self, messages: List[Dict], model: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def available_models(self) -> List[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
