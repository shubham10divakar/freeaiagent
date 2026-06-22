from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict


class BaseBackend(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict], model: str) -> str: ...

    @abstractmethod
    async def available_models(self) -> List[str]: ...

    @abstractmethod
    async def is_available(self) -> bool: ...

    async def stream(self, messages: List[Dict], model: str) -> AsyncIterator[str]:
        """Yield response text in chunks.

        Default implementation falls back to a single non-streamed chunk, so a
        backend without native streaming still works through the streaming API.
        Backends that support token streaming override this.
        """
        yield await self.chat(messages, model)

    async def chat_completion(self, messages: List[Dict], model: str, tools=None) -> Dict:
        """Return the raw assistant message dict (may contain `tool_calls`).

        Default ignores `tools` and returns a plain answer, so backends/models
        without tool support degrade gracefully. OpenAI-protocol backends
        override this to pass `tools` and surface tool calls.
        """
        return {"role": "assistant", "content": await self.chat(messages, model)}
