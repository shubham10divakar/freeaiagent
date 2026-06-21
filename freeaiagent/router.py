from typing import List, Tuple
from .config import load
from .backends.base import BaseBackend
from .backends.ollama import OllamaBackend
from .backends.groq import GroqBackend


def _build_backends(config: dict) -> dict[str, BaseBackend]:
    backends: dict[str, BaseBackend] = {}
    bcfg = config.get("backends", {})
    if "ollama" in bcfg:
        backends["ollama"] = OllamaBackend(
            bcfg["ollama"].get("base_url", "http://localhost:11434")
        )
    if "groq" in bcfg and bcfg["groq"].get("api_key"):
        backends["groq"] = GroqBackend(bcfg["groq"]["api_key"])
    return backends


async def resolve(override_model: str | None = None) -> Tuple[BaseBackend, str]:
    """Return (backend, model) for the best available backend, with fallback."""
    config = load()
    backends = _build_backends(config)
    default_backend = config.get("default_backend", "ollama")
    model = override_model or config.get("default_model", "llama3.2:3b")
    fallback_order: list[str] = config.get("fallback_order", ["ollama", "groq"])

    # try default first, then the rest of fallback_order
    ordered = [default_backend] + [b for b in fallback_order if b != default_backend]

    for name in ordered:
        backend = backends.get(name)
        if backend and await backend.is_available():
            return backend, model

    tried = ", ".join(ordered)
    raise RuntimeError(
        f"No backend available (tried: {tried}). "
        "Start Ollama or set a Groq API key with: "
        "freeaiagent config set backends.groq.api_key <key>"
    )


async def available_models() -> List[str]:
    config = load()
    backends = _build_backends(config)
    name = config.get("default_backend", "ollama")
    backend = backends.get(name)
    if backend and await backend.is_available():
        return await backend.available_models()
    return []
