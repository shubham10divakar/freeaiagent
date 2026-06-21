from typing import List, Tuple
from .config import load
from .backends.base import BaseBackend
from .backends.ollama import OllamaBackend
from .backends.groq import GroqBackend
from .backends.openai_compat import OpenAICompatibleBackend


def _build_backends(config: dict) -> dict[str, BaseBackend]:
    backends: dict[str, BaseBackend] = {}
    for name, bcfg in config.get("backends", {}).items():
        btype = bcfg.get("type", name)  # default type = key name for ollama/groq

        if btype == "ollama":
            backends[name] = OllamaBackend(
                bcfg.get("base_url", "http://localhost:11434")
            )
        elif btype == "groq":
            if bcfg.get("api_key"):
                backends[name] = GroqBackend(bcfg["api_key"])
        elif btype == "openai_compat":
            if bcfg.get("base_url"):
                backends[name] = OpenAICompatibleBackend(
                    base_url=bcfg["base_url"],
                    api_key=bcfg.get("api_key", "not-needed"),
                    model_list=bcfg.get("models", []),
                )

    return backends


async def resolve(
    override_model: str | None = None,
    override_backend: str | None = None,
) -> Tuple[BaseBackend, str]:
    """Return (backend, model), respecting per-call overrides then config defaults."""
    config = load()
    backends = _build_backends(config)
    model = override_model or config.get("default_model", "llama3.2:3b")

    # per-call backend override — fail fast if it's not available
    if override_backend:
        backend = backends.get(override_backend)
        if backend is None:
            raise RuntimeError(
                f"Backend '{override_backend}' is not configured. "
                f"Available: {list(backends)}"
            )
        if not await backend.is_available():
            raise RuntimeError(
                f"Backend '{override_backend}' is configured but not reachable."
            )
        return backend, model

    # normal fallback chain
    default_backend = config.get("default_backend", "ollama")
    fallback_order: list[str] = config.get("fallback_order", ["ollama", "groq"])
    ordered = [default_backend] + [b for b in fallback_order if b != default_backend]

    for name in ordered:
        backend = backends.get(name)
        if backend and await backend.is_available():
            return backend, model

    tried = ", ".join(ordered)
    raise RuntimeError(
        f"No backend available (tried: {tried}). "
        "Start Ollama, set a Groq API key, or configure an openai_compat backend."
    )


async def available_models(backend_name: str | None = None) -> List[str]:
    config = load()
    backends = _build_backends(config)
    name = backend_name or config.get("default_backend", "ollama")
    backend = backends.get(name)
    if backend and await backend.is_available():
        return await backend.available_models()
    return []
