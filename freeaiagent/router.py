from typing import List, Tuple
from .config import load
from .backends.base import BaseBackend
from .backends.llamafile import LlamafileBackend
from . import catalog
from .backends.ollama import OllamaBackend
from .backends.groq import GroqBackend
from .backends.openai_compat import OpenAICompatibleBackend


def _build_backends(config: dict) -> dict[str, BaseBackend]:
    backends: dict[str, BaseBackend] = {}
    for name, bcfg in config.get("backends", {}).items():
        btype = bcfg.get("type", name)  # default type = key name for ollama/groq

        if btype == "llamafile":
            backends[name] = LlamafileBackend(
                port=bcfg.get("port", 8080),
                model=config.get("default_model", catalog.DEFAULT_MODEL),
                download_url=bcfg.get("download_url"),
                auto_download=bcfg.get("auto_download", False),
                auto_start=bcfg.get("auto_start", True),
            )
        elif btype == "ollama":
            backends[name] = OllamaBackend(
                bcfg.get("base_url", "http://localhost:11434")
            )
        elif btype == "groq":
            if bcfg.get("api_key"):
                backends[name] = GroqBackend(bcfg["api_key"])
        elif btype == "openai_compat":
            # Only build if there's somewhere to talk to AND, for hosted providers,
            # a key. Presets ship with an empty api_key and stay inert until set.
            if bcfg.get("base_url") and bcfg.get("api_key", "not-needed") != "":
                backends[name] = OpenAICompatibleBackend(
                    base_url=bcfg["base_url"],
                    api_key=bcfg.get("api_key", "not-needed"),
                    model_list=bcfg.get("models", []),
                    api_prefix=bcfg.get("api_prefix", "/v1"),
                )

    return backends


def _max_messages(config: dict, backend_name: str, override: int | None) -> int:
    """Effective context window: per-call override → backend-level → global.

    A backend-level ``max_messages`` lets an 8k-context model keep a short
    window while a 128k model keeps a long one, under one config.
    """
    if override is not None:
        return override
    bcfg = config.get("backends", {}).get(backend_name, {})
    if isinstance(bcfg, dict) and "max_messages" in bcfg:
        return bcfg["max_messages"]
    return config.get("max_messages", 0)


async def resolve(
    override_model: str | None = None,
    override_backend: str | None = None,
    override_max_messages: int | None = None,
) -> Tuple[BaseBackend, str, int]:
    """Return ``(backend, model, max_messages)``.

    Respects per-call overrides, then config defaults. ``max_messages`` is the
    effective context window for the chosen backend (see ``_max_messages``).
    """
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
        return backend, model, _max_messages(config, override_backend, override_max_messages)

    # normal fallback chain
    default_backend = config.get("default_backend", "ollama")
    fallback_order: list[str] = config.get("fallback_order", ["ollama", "groq"])
    ordered = [default_backend] + [b for b in fallback_order if b != default_backend]

    for name in ordered:
        backend = backends.get(name)
        if backend and await backend.is_available():
            return backend, model, _max_messages(config, name, override_max_messages)

    tried = ", ".join(ordered)
    raise RuntimeError(
        f"No backend available (tried: {tried}). "
        "Run `freeaiagent keys` for setup options."
    )


async def available_models(backend_name: str | None = None) -> List[str]:
    config = load()
    backends = _build_backends(config)
    name = backend_name or config.get("default_backend", "ollama")
    backend = backends.get(name)
    if backend and await backend.is_available():
        return await backend.available_models()
    return []
