# freeaiagent — Roadmap & Design Plans

## Status

| Phase | Feature | Status |
|---|---|---|
| 1 | Core server, Ollama + Groq backends, global context, CLI | **Done** |
| 1 | OpenAI-compatible backend (LM Studio, llamafile, LocalAI, Jan) | **Done** |
| 1 | Per-call model + backend override on `/chat` and `/task` | **Done** |
| 1 | Sliding window context (`max_messages`) | **Done** |
| 2 | Named sessions | Planned |
| 3 | Auto caller detection | Planned |
| 4 | More backends, streaming, tool use, web UI, PyPI publish | Planned |

---

## Context Handling

This is the most nuanced part of the system. Full strategy below.

### Current — Sliding Window (Phase 1)

Every `/chat` call retrieves the full conversation history from SQLite and sends
it to the LLM. To prevent exceeding model context limits, `max_messages` caps
how many past messages are included.

```json
{ "max_messages": 20 }
```

Set via config:
```bash
freeaiagent config set max_messages 20   # keep last 20 messages
freeaiagent config set max_messages 0    # unlimited (default)
```

**How it works:** `context.as_llm_messages(max_messages=N)` slices the tail of
the message list. The full history is always stored in SQLite — only the window
sent to the LLM is trimmed.

**Trade-off:** Simple and fast. Loses old context when history grows beyond the
window. Good enough for most single-session use.

**When it breaks:** Long-running research sessions where early context matters
(e.g. "as we discussed earlier, my constraint is X"). Summarization solves this.

---

### Phase 2 Context — Per-Backend Message Limits

Different models have wildly different context windows (Gemma 8k vs Llama 128k).
A single global `max_messages` doesn't fit all backends.

**Planned config:**
```json
{
  "max_messages": 0,
  "backends": {
    "ollama":   { "base_url": "...", "max_messages": 50 },
    "lmstudio": { "type": "openai_compat", "base_url": "...", "max_messages": 20 },
    "groq":     { "api_key": "...", "max_messages": 100 }
  }
}
```

Resolution order: per-call override → backend-level limit → global `max_messages`.

**Implementation:** `router.resolve()` returns `(backend, model, effective_max_messages)`.
`chat.py` uses the resolved limit instead of loading from config directly.

---

### Phase 3 Context — Summarization

When history grows long, summarize the oldest messages into a single compressed
"memory" block rather than discarding them.

**How it works:**
1. When `context.count() > summarize_threshold` (e.g. 40 messages)
2. Take the oldest `summarize_batch` messages (e.g. 30)
3. Send them to the LLM with prompt: *"Summarize this conversation so far in
   bullet points, preserving key facts, decisions, and context."*
4. Replace the 30 messages with one `{"role": "system", "content": "[Summary] ..."}` entry
5. Continue with the remaining 10 messages + the summary

**Config:**
```json
{
  "context_strategy": "summarize",
  "summarize_threshold": 40,
  "summarize_batch": 30,
  "summarize_model": "llama3.2:3b"
}
```

**Trade-off:** Preserves semantic content. Costs one extra LLM call per
summarization event. Summary quality depends on the model.

**When it breaks:** If the summary model is too weak it loses nuance. Also
doesn't help if the user needs exact prior wording (quotes, code snippets).

---

### Phase 4 Context — RAG (Retrieval-Augmented Generation)

Instead of sending a window of recent messages, embed all messages and retrieve
the K most semantically relevant ones for the current query.

**How it works:**
1. On every `context.append()`, embed the message and store the vector
2. On `/chat`, embed the incoming message
3. Retrieve top-K messages by cosine similarity
4. Send those K messages as context (not chronological, semantic)

**Config:**
```json
{
  "context_strategy": "rag",
  "rag_top_k": 10,
  "embedding_model": "nomic-embed-text",
  "embedding_backend": "ollama"
}
```

**Trade-off:** Best quality for very long histories. Heavy — requires an
embedding model running alongside the chat model. Adds latency per call.
Out-of-order messages can confuse some models.

**Implementation notes:**
- SQLite + sqlite-vec or chromadb for vector storage
- Ollama has embedding support (`ollama pull nomic-embed-text`)
- Falls back to sliding window if embedding backend is unavailable

---

## Session Handling

### Phase 2 — Named Sessions

Each caller passes a `session_id`. Different apps maintain separate context threads.

**API change:**
```json
POST /chat
{ "message": "...", "session_id": "magpie-project" }

POST /task
{ "task": "...", "session_id": "my-script" }

GET  /context?session=magpie-project
DELETE /context?session=magpie-project
GET  /sessions
```

**DB change:** Add `session_id TEXT NOT NULL DEFAULT 'default'` column to `messages`.
All existing calls use `session_id = "default"` — fully backwards-compatible.

**CLI:**
```bash
freeaiagent chat --session magpie
freeaiagent context show --session magpie
freeaiagent context clear --session magpie
```

---

### Phase 3 — Auto Caller Detection

Sessions created automatically — zero config for multi-app use.

**Resolution order:**
1. `session_id` field in request body (explicit, wins)
2. `X-Caller-ID` request header (app sets it once, forget it)
3. Caller port fingerprint `app-{port}` (last resort, best-effort)
4. Falls back to `"default"` session

**Example:**
```python
# Magpie sets this header on every request — never thinks about sessions again
headers = {"X-Caller-ID": "magpie", "Content-Type": "application/json"}
```

**Trade-off:** Convenient but port-based detection is unreliable (ephemeral
ports change per process). Header-based is solid. Document both.

---

## Backend Roadmap

### Phase 4 Backends

| Backend | Status | Notes |
|---|---|---|
| Ollama | Done | Own protocol + openai-compat |
| Groq | Done | Free tier, fast |
| OpenAI-compatible | Done | Covers LM Studio, llamafile, LocalAI, Jan, llama.cpp |
| llamafile | Planned | Dedicated backend: auto-start the .exe if not running |
| Together AI | Planned | Free tier available |
| OpenRouter | Planned | Aggregates 100+ models, free tier |
| Cerebras | Planned | Fast inference, free tier |
| Gemini | Planned | Free tier (1500 req/day) |

### llamafile Dedicated Backend (Phase 4)

Unlike other backends, llamafile is a single file the user drops in a folder.
The dedicated backend would auto-start it as a subprocess if not already running.

```json
{
  "backends": {
    "llamafile": {
      "type": "llamafile",
      "path": "~/Magpie/llamafile/llama3.2-3b.llamafile",
      "port": 8080,
      "auto_start": true
    }
  }
}
```

This closes the Magpie use case completely: no Ollama install, no service, no
UAC. User drops one file, freeaiagent handles the rest.

---

## Phase 4 Features

### Streaming (`/chat/stream`)

SSE endpoint for streaming responses token-by-token.

```python
GET /chat/stream?message=hello
# streams: data: {"token": "Hello"}\n\n
#          data: {"token": " there"}\n\n
#          data: [DONE]\n\n
```

All backends that support streaming will use it. Ollama streams natively.
Groq supports SSE. OpenAI-compat backends vary.

### Tool Use / Function Calling

Register tools the agent can call during a chat:

```bash
POST /tools/register
{ "name": "read_file", "description": "...", "endpoint": "http://localhost:8000/read" }
```

Agent decides when to call tools mid-conversation. Results injected as
`{"role": "tool", ...}` messages before the final response.

### Web UI

Minimal UI served at `http://localhost:7731` — just a chat box, context
viewer, and model switcher. No framework, plain HTML + JS. Optional, off
by default.

```bash
freeaiagent start --ui
```

### PyPI Publish

```bash
python -m build
twine upload dist/*
```

Pre-publish checklist:
- [ ] Smoke tests pass against Ollama
- [ ] README reviewed
- [ ] Version bumped in `__init__.py` + `pyproject.toml` + `setup.py`
- [ ] `CHANGELOG.md` written
- [ ] GitHub release tagged

---

## Architecture Decisions Log

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Build system | setuptools + pyproject.toml | hatchling | Wider tooling compat, `setup.py` shim support |
| HTTP client | httpx2 | httpx | Starlette TestClient deprecation, forward-compat |
| Context store | SQLite (stdlib) | Redis, JSON file | Zero deps, persistent, queryable |
| Server | FastAPI + uvicorn | Flask, aiohttp | Async, auto-docs, Pydantic validation |
| CLI | Typer | Click, argparse | Subcommands, help text, minimal boilerplate |
| Context default | Global (Phase 1) | Sessions from start | Simplest path; sessions are Phase 2 |
| Context windowing | Sliding window | Token counting | No tokenizer needed, simple, good enough |
