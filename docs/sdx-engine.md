# SDX Engine — Smart Decision eXecution for Desktop

> **Status:** Design v1.0 — to be implemented as Phase 9.

---

## What it is

**SDX** (Smart Decision eXecution) is a compound inference engine that presents a single logical "model" to the caller while internally routing between two specialist sub-models — a text model and a vision model — depending on whether the user attaches an image.

The user downloads one bundle and uses it like any other model. The orchestration is invisible.

```
SDX-Standard bundle (~4.7 GB)
  ├── Text model  — Llama-3.2-3B-Instruct Q4_K_M (GGUF, ~2.0 GB)
  └── Vision model — llava-phi-3-mini Q4_K_M (GGUF ~2.4 GB + mmproj ~0.3 GB)
```

When an image is attached, the vision model silently extracts a description; that description is injected as text into the conversation, then the text model generates the reply with full context. Without an image, the text model handles everything directly.

---

## The problem it solves

Today's freeaiagent has a standalone vision model as a catalog entry — the user must explicitly switch to it to handle images and loses their chat context when they do. There is no unified path for mixed text + image conversations in a single session.

SDX makes vision transparent: one model entry, one download, images just work inside any chat session.

---

## Model tiers

Desktop has far more RAM and disk than mobile — SDX ships five tiers from ultra-light to flagship.

| ID | Display | Text model | Vision model | Bundle size | Min RAM |
|---|---|---|---|---|---|
| `sdx-nano` | SDX Nano | Qwen2.5-0.5B Q4_K_M | moondream2 Q4 | ~2.1 GB | 4 GB |
| `sdx-mini` | SDX Mini | Llama-3.2-1B Q4_K_M | moondream2 Q4 | ~2.5 GB | 4 GB |
| `sdx-standard` | SDX Standard | Llama-3.2-3B Q4_K_M | llava-phi-3-mini Q4 + mmproj | ~4.7 GB | 8 GB |
| `sdx-plus` | SDX Plus | Qwen2.5-7B Q4_K_M | llava-v1.6-mistral-7b Q4 + mmproj | ~9.4 GB | 16 GB |
| `sdx-max` | SDX Max | Qwen2.5-14B Q4_K_M | llava-v1.6-mistral-7b Q4 + mmproj | ~13.4 GB | 24 GB |

**Vision model notes:**
- `moondream2` — single GGUF, no separate mmproj needed. Small, fast, solid for basic image understanding. Uses `MoondreamChatHandler` in llama-cpp-python.
- `llava-phi-3-mini` — Phi-3-mini based LLaVA, stronger vision with a small footprint. Uses `Llava15ChatHandler` with a separate mmproj GGUF.
- `llava-v1.6-mistral-7b` — LLaVA 1.6 on Mistral 7B, highest quality vision at this weight class. Uses `Llava16ChatHandler` with a separate mmproj GGUF.

**Context windows — token budget for the flat text context sent to the text model:**

| Tier | Token budget | n_ctx for text model |
|---|---|---|
| Nano | 4 096 | 4 096 |
| Mini | 8 192 | 8 192 |
| Standard | 8 192 | 8 192 |
| Plus | 16 384 | 16 384 |
| Max | 32 768 | 32 768 |

---

## Design principles

1. **Fully standalone codebase.** SDX lives in `freeaiagent/sdx/` with zero imports from the rest of the package (no `router`, `context`, `catalog`, `config` imports). The only coupling to the rest of the app is the thin `SDXBackend` wrapper in `freeaiagent/backends/sdx_backend.py`.

2. **Context as text.** The ContextBuilder flattens conversation history into a single flat text block (system prompt + turns). No reliance on multi-turn chat-template handling. Both models always receive the same plain format.

3. **Images become text permanently.** After the vision model processes an image, the extracted description is persisted as a special system message in SQLite. The raw image data is never stored long-term. Future turns see `[Image: description]` in the context.

4. **Vision model is a preprocessor, not the responder.** The vision model's only job is extracting a description from the image. The text model generates all responses the user sees. Response quality is determined by the stronger text model; latency is predictable.

5. **Bundled combo, one download.** Text model + vision model + optional mmproj are declared as a single catalog bundle. The download flow fetches all files sequentially with combined progress events under a single `freeaiagent pull sdx-standard` command.

6. **Stateless-per-call reconstruction.** The SDX context is rebuilt on every `/chat` call from the SQLite message history passed in as `messages: list[dict]` by the router. No per-session engine state is cached between calls. This is simpler, survives server restarts, and is fully correct.

---

## File structure

```
freeaiagent/sdx/
  ├── __init__.py          — exports SDXEngine
  ├── types.py             — SDXMessage, SDXHistory, SDXBundle dataclasses
  ├── context_builder.py   — flat text context builder + token budget management
  ├── vision_runner.py     — llama-cpp-python multimodal wrapper (image → description)
  ├── text_runner.py       — llama-cpp-python GGUF streaming wrapper (context → tokens)
  └── engine.py            — SDXEngine orchestrator (public API)

freeaiagent/backends/sdx_backend.py   — BaseBackend wrapper; bridges router → SDXEngine
```

No file in `freeaiagent/sdx/` imports from outside `freeaiagent/sdx/` except:
- `llama_cpp` (the llama-cpp-python package, for both runners)
- Python stdlib only

---

## Public API — `SDXEngine`

```python
class SDXEngine:
    def __init__(
        self,
        text_model_path: str,
        vision_model_path: str,
        mmproj_path: str | None = None,
        token_budget: int = 8192,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
    ): ...

    # Load both models into memory. Calls on_progress(phase, pct) if provided.
    async def load(self, on_progress=None) -> None: ...

    # Send one turn. messages = full SQLite history (OpenAI format).
    # image_path = temp file path owned by the caller (caller deletes after).
    # Yields tokens from the text model as they stream.
    async def send(
        self,
        messages: list[dict],
        text: str,
        image_path: str | None = None,
    ) -> AsyncIterator[str]: ...

    # Interrupt in-flight generation.
    def stop(self) -> None: ...

    # Free native memory. Call when switching away from SDX.
    def unload(self) -> None: ...

    def is_loaded(self) -> bool: ...
```

`messages` is the full conversation history in OpenAI format (from SQLite), including any `[SDX-Image]` annotation system messages stored from prior turns. The ContextBuilder rebuilds the flat text from this list on each call.

---

## Turn flow

```
SDXBackend.stream(messages, model, image_path=None)
        │
        ├─ text = messages[-1]["content"]   (current user turn)
        │
        ├─ image_path present?
        │   │
        │   ├─ vision_runner.extract(image_path, text)
        │   │      → llama_cpp LLaVA/moondream: generate description
        │   │      → returns description string  (takes ~2–8 s depending on model)
        │   │      → caller/UI shows "Analyzing image…" during this gap
        │   │
        │   └─ inject description into messages before current turn:
        │          messages.insert(-1, {
        │              "role": "system",
        │              "content": f"[SDX-Image]: {description}"
        │          })
        │          (endpoint persists this to SQLite after the call)
        │
        ├─ context_str = ContextBuilder(messages, token_budget).build()
        │
        └─ text_runner.generate(context_str)
               → llama_cpp.Llama streams tokens
               → yields token-by-token to the SSE response
               → endpoint stores full response to SQLite on completion
```

---

## Context format

The text model receives the entire conversation as a single flat string. This sidesteps multi-turn chat-template edge cases and gives SDX full control over what fits in the context window.

```
[System]
You are FreeAIAgent, a helpful and concise assistant running fully on the user's device.
Answer directly. Use Markdown when helpful.

[Conversation so far]
User: Can you help me understand this chart?
[Image: A bar chart showing monthly revenue Jan–Dec 2024. January ~$12k,
peaks August ~$45k, drops sharply in Q4 to ~$18k by December.]
Assistant: The chart shows strong seasonal growth peaking mid-year...

User: What's the biggest single-month drop?
Assistant: August to September — from ~$45k down to ~$31k, about 31%.

[Current turn]
User: Why might that happen seasonally?
Assistant:
```

---

## ContextBuilder — token budget management

```python
class ContextBuilder:
    SYSTEM_PROMPT = (
        "You are FreeAIAgent, a helpful and concise assistant running fully "
        "on the user's device.\nAnswer directly. Use Markdown when helpful."
    )
    SDX_IMAGE_PREFIX = "[SDX-Image]: "

    def __init__(
        self,
        messages: list[dict],
        token_budget: int,
        min_turns_keep: int = 2,
    ):
        # messages[-1] is always the current user turn
        # messages[:-1] is the prior history
        ...

    def build(self) -> str:
        # 1. Render SYSTEM block (always included; charged against budget)
        # 2. Walk history oldest-to-newest:
        #      - system messages starting with SDX_IMAGE_PREFIX:
        #          render as "[Image: <description>]" on its own line
        #      - user messages: render as "User: <content>"
        #      - assistant messages: render as "Assistant: <content>"
        # 3. If total estimated tokens > token_budget - 512 (response headroom):
        #      drop oldest complete turn pairs (user + assistant) until it fits
        #      always keep at least min_turns_keep recent complete pairs
        #      current user turn is never dropped
        # 4. Append "[Current turn]\nUser: <current_message>\nAssistant:"
        # 5. Return the full string
        ...

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4   # fast approximation, no tokenizer needed
```

**Budget rules:**
- `token_budget` matches the text model's `n_ctx` (minus 512 for the response)
- If the full history doesn't fit, drop the oldest **complete turn pair** (user + assistant message) at a time, never partial pairs
- The system prompt and the current user turn are **never dropped**
- A minimum of `min_turns_keep` recent complete pairs is always kept even when over budget — the model always has some short-term memory
- Image descriptions from dropped turns are also dropped (the description itself was the representation, there is no separately kept image)

---

## VisionRunner

```python
class VisionRunner:
    """Wraps llama-cpp-python for multimodal image description extraction."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str | None = None,
        n_gpu_layers: int = 0,
    ):
        self._llm = None
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._n_gpu_layers = n_gpu_layers

    def load(self) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import (
            MoondreamChatHandler, Llava15ChatHandler, Llava16ChatHandler,
        )
        if self._mmproj_path:
            handler = Llava16ChatHandler(clip_model_path=self._mmproj_path)
            n_ctx = 4096
        else:
            handler = MoondreamChatHandler()
            n_ctx = 2048
        self._llm = Llama(
            model_path=self._model_path,
            chat_handler=handler,
            n_ctx=n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

    async def extract(self, image_path: str, user_text: str) -> str:
        """Return image description (or direct answer if user_text is a question)."""
        import asyncio, base64
        from pathlib import Path

        ext = Path(image_path).suffix.lstrip(".") or "jpeg"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        data_uri = f"data:image/{ext};base64,{b64}"

        prompt = (
            "If the user asked a question about the image, answer it concisely. "
            "Otherwise, describe the image in detail: objects, text, layout, "
            f'colors, quantities.\nUser\'s message: "{user_text}"'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        def _run():
            result = self._llm.create_chat_completion(messages=messages, max_tokens=512)
            return result["choices"][0]["message"]["content"]

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    def unload(self) -> None:
        self._llm = None
```

---

## TextRunner

```python
class TextRunner:
    """Wraps llama-cpp-python for streaming GGUF text generation."""

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = 0):
        self._llm = None
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._stop_flag = threading.Event()

    def load(self) -> None:
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

    async def generate(self, context: str) -> AsyncIterator[str]:
        """Stream tokens from the flat text context string."""
        import asyncio
        self._stop_flag.clear()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            stream = self._llm(
                context,
                max_tokens=1024,
                stop=["\nUser:", "[Current turn]"],
                stream=True,
            )
            for chunk in stream:
                if self._stop_flag.is_set():
                    break
                token = chunk["choices"][0]["text"]
                loop.call_soon_threadsafe(queue.put_nowait, token)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_run, daemon=True).start()
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    def stop(self) -> None:
        self._stop_flag.set()

    def unload(self) -> None:
        self._llm = None
```

---

## HTTP API — image input

The `/chat` and `/chat/stream` endpoints gain an optional `image` field:

```json
POST /chat
{
  "message": "What does this chart show?",
  "session_id": "work",
  "backend": "sdx",
  "model": "sdx-standard",
  "image": "<base64-encoded-image-data>"
}
```

**Endpoint lifecycle for image field:**
1. Decode the base64 string to a temp file (`tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)`)
2. Pass the temp file path into the SDXBackend alongside the messages
3. After the response is fully streamed, delete the temp file (`os.unlink`)
4. After the call, persist the description to SQLite as `{"role": "system", "content": "[SDX-Image]: <description>"}` immediately before the assistant message

Any non-SDX backend receiving an `image` field ignores it gracefully.

---

## SDX Backend wrapper

```python
class SDXBackend(BaseBackend):
    """Thin wrapper bridging SDXEngine into the BaseBackend interface."""

    def __init__(self, config: dict):
        self._config = config
        self._engines: dict[str, SDXEngine] = {}  # keyed by model id

    async def stream(
        self,
        messages: list[dict],
        model: str,
        image_path: str | None = None,
    ) -> AsyncIterator[str]:
        engine = await self._get_or_load(model)
        text = messages[-1]["content"]
        async for token in engine.send(messages, text, image_path):
            yield token

    async def chat(self, messages: list[dict], model: str) -> str:
        return "".join([t async for t in self.stream(messages, model)])

    async def available_models(self) -> list[str]:
        return [m for m in SDX_CATALOG if self._is_downloaded(m)]

    async def is_available(self) -> bool:
        return True

    async def _get_or_load(self, model: str) -> SDXEngine:
        if model not in self._engines:
            self._engines[model] = self._build_engine(model)
            await self._engines[model].load()
        return self._engines[model]

    def _build_engine(self, model: str) -> SDXEngine:
        bundle = SDX_CATALOG[model]
        paths = self._resolve_paths(model)
        return SDXEngine(
            text_model_path=paths["text"],
            vision_model_path=paths["vision"],
            mmproj_path=paths.get("mmproj"),
            token_budget=bundle["token_budget"],
            n_ctx=bundle["token_budget"],
            n_gpu_layers=self._config.get("n_gpu_layers", 0),
        )
```

---

## Catalog entries

```python
SDX_CATALOG: dict[str, dict] = {
    "sdx-nano": {
        "display": "SDX Nano",
        "tagline": "Text + vision on any PC — ultra-light bundle (2.1 GB)",
        "kind": "sdx",
        "tier": "nano",
        "min_ram_gb": 4,
        "token_budget": 4096,
        "files": {
            "text": {
                "url": "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
                "size_gb": 0.4,
                "sha256": None,
            },
            "vision": {
                "url": "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-q4_k_m.gguf",
                "size_gb": 1.7,
                "sha256": None,
                "mmproj": None,
            },
        },
    },
    "sdx-mini": {
        "display": "SDX Mini",
        "tagline": "Text + vision — small and sharp (2.5 GB)",
        "kind": "sdx",
        "tier": "mini",
        "min_ram_gb": 4,
        "token_budget": 8192,
        "files": {
            "text": {
                "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "size_gb": 0.8,
                "sha256": None,
            },
            "vision": {
                "url": "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-q4_k_m.gguf",
                "size_gb": 1.7,
                "sha256": None,
                "mmproj": None,
            },
        },
    },
    "sdx-standard": {
        "display": "SDX Standard",
        "tagline": "The balanced all-rounder — chat, code, and images (4.7 GB)",
        "kind": "sdx",
        "tier": "standard",
        "min_ram_gb": 8,
        "token_budget": 8192,
        "files": {
            "text": {
                "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "size_gb": 2.0,
                "sha256": None,
            },
            "vision": {
                "url": "https://huggingface.co/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-int4.gguf",
                "size_gb": 2.4,
                "sha256": None,
                "mmproj": {
                    "url": "https://huggingface.co/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-mmproj-f16.gguf",
                    "size_gb": 0.3,
                    "sha256": None,
                },
            },
        },
    },
    "sdx-plus": {
        "display": "SDX Plus",
        "tagline": "High-quality answers and strong vision (9.4 GB)",
        "kind": "sdx",
        "tier": "plus",
        "min_ram_gb": 16,
        "token_budget": 16384,
        "files": {
            "text": {
                "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "size_gb": 4.7,
                "sha256": None,
            },
            "vision": {
                "url": "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-1.6-mistral-7b.Q4_K_M.gguf",
                "size_gb": 4.1,
                "sha256": None,
                "mmproj": {
                    "url": "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
                    "size_gb": 0.6,
                    "sha256": None,
                },
            },
        },
    },
    "sdx-max": {
        "display": "SDX Max",
        "tagline": "Full-power text and vision — flagship machines only (13.4 GB)",
        "kind": "sdx",
        "tier": "max",
        "min_ram_gb": 24,
        "token_budget": 32768,
        "files": {
            "text": {
                "url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                "size_gb": 8.7,
                "sha256": None,
            },
            "vision": {
                "url": "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-1.6-mistral-7b.Q4_K_M.gguf",
                "size_gb": 4.1,
                "sha256": None,
                "mmproj": {
                    "url": "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
                    "size_gb": 0.6,
                    "sha256": None,
                },
            },
        },
    },
}
```

> **Note:** All `sha256` values are `None` in the design doc and must be populated at implementation time by downloading the files and computing their hashes. All URLs should be verified at implementation time — GGUF repos on HuggingFace occasionally reorganize files.

---

## Integration with existing system — minimal touch

### `router.py`

Add `sdx` as a valid `btype` in `_build_backends()`:

```python
elif btype == "sdx":
    from freeaiagent.backends.sdx_backend import SDXBackend
    backends[name] = SDXBackend(cfg)
```

### `endpoints/chat.py`

Add `image: str | None = None` to `ChatRequest`. Before calling the router:

```python
image_path = None
if request.image:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(base64.b64decode(request.image))
    tmp.close()
    image_path = tmp.name
```

After the response, in a `finally` block:

```python
if image_path:
    os.unlink(image_path)
```

Pass `image_path` as an extra kwarg into the backend `.stream()` / `.chat()` call. Non-SDX backends accept and ignore it.

After the full response is stored to SQLite, if `description` was returned by the SDX backend (via a side-channel), persist it to the session as:

```python
await ctx.append(session_id, "system", f"[SDX-Image]: {description}")
```

This is inserted immediately before the assistant response row so ContextBuilder sees it in the right order on the next turn.

### `config.py`

Add SDX section to the default config schema:

```python
"sdx": {
    "type": "sdx",
    "model": "sdx-standard",
    "n_gpu_layers": 0,
    "auto_unload_vision": False,
}
```

- `n_gpu_layers`: GPU offload layers applied to both sub-models (default 0 = CPU-only)
- `auto_unload_vision`: if True, unload the vision model from memory immediately after each image extraction (saves ~0.6–1.5 GB at cost of ~3–5 s reload on the next image turn; useful on 8 GB RAM machines)

---

## Download design — multi-file bundle

For SDX catalog entries, `freeaiagent pull sdx-standard` must download 2–3 files. The existing `/pull/stream` SSE phase field is extended with new phase values:

```
data: {"type": "start",    "phase": "text_model",   "label": "Llama-3.2-3B (text)",          "total_mb": 2000}
data: {"type": "progress", "phase": "text_model",   "pct": 42, "downloaded_mb": 840, ...}
data: {"type": "start",    "phase": "vision_model",  "label": "llava-phi-3-mini (vision)",    "total_mb": 2400}
data: {"type": "progress", "phase": "vision_model",  "pct": 71, ...}
data: {"type": "start",    "phase": "mmproj",        "label": "Vision projector",              "total_mb": 300}
data: {"type": "progress", "phase": "mmproj",        "pct": 100, ...}
data: {"type": "done",     "path": "~/.freeaiagent/sdx/sdx-standard/"}
data: [DONE]
```

Files are stored under:

```
~/.freeaiagent/
└── sdx/
    ├── sdx-nano/
    │   ├── text.gguf            (Qwen2.5-0.5B)
    │   └── vision.gguf          (moondream2)
    ├── sdx-mini/
    │   ├── text.gguf            (Llama-3.2-1B)
    │   └── vision.gguf          (moondream2)
    ├── sdx-standard/
    │   ├── text.gguf            (Llama-3.2-3B)
    │   ├── vision.gguf          (llava-phi-3-mini)
    │   └── mmproj.gguf          (vision projector)
    ├── sdx-plus/
    │   ├── text.gguf            (Qwen2.5-7B)
    │   ├── vision.gguf          (llava-v1.6-mistral-7b)
    │   └── mmproj.gguf
    └── sdx-max/
        ├── text.gguf            (Qwen2.5-14B)
        ├── vision.gguf          (llava-v1.6-mistral-7b)
        └── mmproj.gguf
```

All files downloaded with SHA256 verification (matching the existing download integrity system). Partial downloads resume from byte offset. If any file fails, the whole bundle is left in partial state and retry re-starts from the first incomplete file.

---

## Web UI changes (`ui/index.html`)

1. **Image attach button** — paperclip icon (📎) next to the text input, wired to `<input type="file" accept="image/*" capture="environment">`. Hidden by default; shown when active model is an SDX kind.
2. **Image preview** — small inline thumbnail above the send bar when an image is selected. Click × to clear.
3. **"Analyzing image…" status** — shown between the user message bubble appearing and the text model's first token arriving (the vision extraction gap of ~2–8 s).
4. **Image thumbnail in chat bubble** — user message bubbles that include an image show a small thumbnail (rendered from the original data URI) alongside the text.
5. **SDX model badge** — in the model selector dropdown, SDX models show a `📷 Vision` chip to indicate image support.
6. **Graceful degradation** — if no image is attached, SDX behaves exactly like any text model and the image UI stays hidden.

The image is read with `FileReader.readAsDataURL()`, the `data:image/...;base64,` prefix is stripped, and the raw base64 is sent as the `image` field in the JSON body.

---

## What SDX does NOT do

- Does not implement multi-model fan-out (that's `ensemble.py`)
- Does not share conversation state with other backends — switching away from SDX and back rebuilds from SQLite (full history is preserved, ContextBuilder is reconstructed fresh)
- Does not apply the existing `summarize` context strategy — ContextBuilder's drop logic is the SDX context management layer
- Does not support tool calling — text models in the lower SDX tiers are too small for reliable structured tool calls; use a dedicated backend for that
- Does not regenerate on vision turns — image extraction is non-deterministic; regenerate is only supported for text-only turns
- Does not implement `BaseBackend.chat_completion()` — tool protocol not supported

---

## Build order

Each step is independently testable before the next. Steps 1–6 have zero dependency on the rest of the package and can be unit-tested in isolation.

1. `freeaiagent/sdx/types.py` — `SDXMessage`, `SDXHistory`, `SDXBundle` dataclasses
2. `freeaiagent/sdx/context_builder.py` — flat text builder + token budget + drop logic
3. `freeaiagent/sdx/vision_runner.py` — llama-cpp-python multimodal wrapper
4. `freeaiagent/sdx/text_runner.py` — llama-cpp-python streaming wrapper
5. `freeaiagent/sdx/engine.py` — `SDXEngine` orchestrator + public API
6. `freeaiagent/sdx/__init__.py` — `from freeaiagent.sdx import SDXEngine`
7. `freeaiagent/backends/sdx_backend.py` — `SDXBackend(BaseBackend)` wrapper
8. `catalog.py` — merge `SDX_CATALOG` into the main catalog dict (5 tier entries)
9. `config.py` — add SDX backend config schema and default values
10. `router.py` — add `sdx` to `_build_backends()` dispatch
11. `endpoints/chat.py` — `image` field, temp file lifecycle, description persistence
12. `pull.py` — multi-file bundle download for SDX kind entries
13. `ui/index.html` — image attach, preview, thumbnail, SDX badge, analyzing state
14. `tests/test_sdx_context_builder.py` — context builder unit tests (pure Python, fast)
15. `tests/test_sdx_engine.py` — engine integration tests (require llama-cpp-python)
16. `tests/test_sdx_api.py` — endpoint tests with mocked SDX backend
