import httpx2 as httpx
from typing import List, Dict
from .base import BaseBackend


class OpenAICompatibleBackend(BaseBackend):
    """
    Generic backend for any server that speaks the OpenAI /v1/chat/completions protocol.

    Covers: llamafile, LM Studio, LocalAI, Jan, llama.cpp server, GPT4All server.
    Config example:
        {
          "type": "openai_compat",
          "base_url": "http://localhost:1234",
          "api_key": "not-needed",      # optional, sent as Bearer token
          "models": ["mistral-7b"]      # optional fallback if /v1/models is unavailable
        }
    """

    def __init__(self, base_url: str, api_key: str = "not-needed", model_list: List[str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model_list = model_list or []

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                )
                return r.status_code == 200
        except Exception:
            return False

    async def chat(self, messages: List[Dict], model: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json={"model": model, "messages": messages},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                )
                if r.status_code == 200:
                    return [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
        return self._model_list
