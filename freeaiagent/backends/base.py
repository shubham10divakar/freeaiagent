from abc import ABC, abstractmethod
from typing import List, Dict


class BaseBackend(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict], model: str) -> str: ...

    @abstractmethod
    async def available_models(self) -> List[str]: ...

    @abstractmethod
    async def is_available(self) -> bool: ...
