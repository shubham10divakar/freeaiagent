# freeaiagent — Roadmap & Design Plans

## Status

| Phase | Feature | Status |
|---|---|---|
| 1 | Core server, Ollama + Groq backends, global context, CLI | **Done** |
| 1 | OpenAI-compatible backend (LM Studio, llamafile, LocalAI, Jan) | **Done** |
| 1 | Per-call model + backend override on `/chat` and `/task` | **Done** |
| 1 | Sliding window context (`max_messages`) | **Done** |
| 2 | Named sessions (`session_id`, `/sessions` CRUD) | **Done** |
| 2 | Chat web UI (ChatGPT-style, served at `/ui`) | **Done** |
| 4 | llamafile dedicated backend (auto-download + auto-start, `freeaiagent pull`) | **Done** |
| 3 | Auto caller detection (`X-Caller-ID`) | **Done** |
| 4 | Cloud backend presets (Together, OpenRouter, Cerebras, Gemini) | **Done** |
| 4 | Streaming (`/chat/stream` SSE) | **Done** |
| 4 | Tool use / function calling (`/tools`, `tools=true`) | **Done** |
| 4 | Multi-model local backend (GGUF catalog + engine mode + `search`/`pull hf:`) — see [MULTI_MODEL_DESIGN.md](MULTI_MODEL_DESIGN.md) | **Done** |
| 4 | PyPI publish | Planned |

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

### Phase 2 — Named Sessions — **Done**

Each caller passes a `session_id`. Different apps and UI chats maintain separate context threads.
(Shipped — endpoints, `sessions` table, and CLI below are implemented.)

**API changes:**
```
POST /chat                        — add session_id field (defaults to "default")
POST /task                        — add session_id field
GET  /context?session=id          — messages for one session
DELETE /context?session=id        — clear messages for one session

GET  /sessions                    — list all sessions with metadata
POST /sessions                    — create named session explicitly
PATCH /sessions/{id}              — rename a session
DELETE /sessions/{id}             — delete session + all its messages
```

**Session metadata (new `sessions` table):**

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PRIMARY KEY | UUID or user-supplied slug |
| `title` | TEXT | Auto-set from first 60 chars of first message; user can rename |
| `model` | TEXT | Last model used in this session |
| `backend` | TEXT | Last backend used |
| `created_at` | TEXT | ISO timestamp |
| `last_updated` | TEXT | ISO timestamp, updated on every message |
| `message_count` | INT | Denormalised count for fast listing |

**`messages` table change:** Add `session_id TEXT NOT NULL DEFAULT 'default'`.
All existing calls without a `session_id` field continue to use `"default"` — fully backwards-compatible.

**`GET /sessions` response shape:**
```json
{
  "sessions": [
    {
      "id": "abc123",
      "title": "Explain the difference between...",
      "model": "openai/gpt-oss-20b",
      "backend": "groq",
      "message_count": 14,
      "last_updated": "2026-06-21T10:45:00Z"
    }
  ]
}
```

**CLI:**
```bash
freeaiagent chat --session magpie
freeaiagent context show --session magpie
freeaiagent context clear --session magpie
freeaiagent sessions                        # list all sessions
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
| llamafile | **Done** | Dedicated zero-install backend: auto-download + auto-start a self-contained model |
| Together AI | Planned | Free tier available |
| OpenRouter | Planned | Aggregates 100+ models, free tier |
| Cerebras | Planned | Fast inference, free tier |
| Gemini | Planned | Free tier (1500 req/day) |

### llamafile Dedicated Backend — **Done** (download-based, not path-based)

> **Shipped, but differently than originally specced.** The original plan assumed
> the user manually *drops a `.llamafile` in a folder* (`path`). The implemented
> design is **download-based**: freeaiagent fetches a self-contained model on
> demand and auto-starts it — no manual file handling.

llamafile is a single self-contained executable (llama.cpp engine + GGUF weights
fused via Cosmopolitan libc). The backend downloads it once and runs it as a
local subprocess, exposing an OpenAI-compatible server.

**Actual config (as implemented):**
```json
{
  "backends": {
    "llamafile": {
      "type": "llamafile",
      "port": 8080,
      "auto_download": false
    }
  }
}
```

**Workflow:**
```bash
freeaiagent pull            # one-time ~2.3 GB download (Llama-3.2-3B), with progress bar
freeaiagent start           # auto-starts the model on first /chat
```

- `auto_download` defaults to **false** — the download is explicit via
  `freeaiagent pull`, not a silent first-run fetch. Set it `true` to fetch on
  first request instead.
- `auto_start` (default true): if the binary exists but isn't running, the
  backend launches it and logs where it's serving.
- Default model is **Llama-3.2-3B-Instruct Q4_K_M** — chosen over 1B because the
  fallback workload includes reasoning/Q&A.

This closes the Magpie use case completely: no Ollama install, no service, no
UAC. `pip install` + `freeaiagent pull` and freeaiagent handles the rest.

**Next step — variety of models:** the single hardcoded model evolves into a
user-chosen GGUF catalog (low→high end) plus later live HuggingFace search.
Full design in [MULTI_MODEL_DESIGN.md](MULTI_MODEL_DESIGN.md).

---

## Ensemble Inference (Future)

Send the same query to multiple models in parallel and pick the best response before returning it to the caller.

### Motivation

Individual models have blind spots — a small fast model may hallucinate where a larger one wouldn't, and vice versa. Running the same prompt across a family (e.g. llama-3.1-8b, llama-3.3-70b, llama-4-scout) and selecting the best output gives higher reliability without switching backends.

### How it would work

1. Caller opts in via request field: `"ensemble": true` (or `"ensemble": ["model-a", "model-b"]`)
2. Router fans out the prompt to all models in parallel (`asyncio.gather`)
3. A **judge step** picks the winner — options ranked by preference:
   - **LLM-as-judge**: send all responses to a small fast model with prompt: *"Which of these answers is most accurate and complete? Reply with just the number."*
   - **Longest non-repetitive**: heuristic fallback when no judge model is available
   - **Majority vote**: for factual/short answers, pick the most common response
4. Return the winning response; include `ensemble_votes` metadata in the response body so callers can inspect what each model said

### Config

```json
{
  "ensemble": {
    "enabled": false,
    "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"],
    "judge": "openai/gpt-oss-20b",
    "strategy": "llm_judge"
  }
}
```

### Trade-offs

| Pro | Con |
|---|---|
| Higher answer quality | 2–3× latency (parallel, not serial) |
| Catches per-model blind spots | 2–3× token usage / API quota |
| Works across same backend or mixed | Judge model itself can be wrong |
| Transparent — caller sees all outputs | Not useful for simple factual tasks |

### When it matters

Best for high-stakes one-shot tasks (code generation, summarization, analysis) where getting the answer right matters more than speed. Less useful for back-and-forth chat where context continuity matters more than any single response.

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

### Chat Web UI — **Done**

A simple ChatGPT-style interface served at `http://localhost:7731/ui`.
No npm, no bundler, no framework — one HTML file (`freeaiagent/ui/index.html`)
with inline CSS + JS, served via FastAPI's `FileResponse`. Built on Phase 2
(Named Sessions).

**Start:**
```bash
freeaiagent start          # then open http://localhost:7731/ui in a browser
```

> **Not yet implemented:** the `freeaiagent start --ui` convenience flag
> (auto-open browser). For now navigate to `/ui` manually. The spec below is the
> as-built design.

**Layout:**

```
┌─────────────────────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────────────────────────┐ │
│  │  + New Chat  │  │  Model: [openai/gpt-oss-20b ▾]   │ │
│  ├──────────────┤  ├──────────────────────────────────┤ │
│  │ Chat 1       │  │                                  │ │
│  │ Chat 2       │  │   [assistant bubble]             │ │
│  │ Chat 3  ···  │  │         [user bubble]            │ │
│  │              │  │   [assistant bubble]             │ │
│  │              │  │                                  │ │
│  │              │  ├──────────────────────────────────┤ │
│  │              │  │  [  Type a message...    ] [Send]│ │
│  └──────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Features (keep it simple):**
- Left sidebar: list of sessions ordered by `last_updated` desc; click to switch
- "New Chat" button: creates a new session, clears the right pane
- Session title: auto-set from first user message (first 60 chars); click to rename inline
- Model dropdown: populated by `GET /models`; stored per session, changeable mid-chat
- Chat area: scrollable bubbles (user right, assistant left); timestamps on hover
- Input: textarea (Enter to send, Shift+Enter for newline); disabled while waiting
- Streaming: if backend supports it, tokens appear as they arrive (SSE); falls back to single response

**What it does NOT do (keep scope tight):**
- No auth, no multi-user
- No file upload, no image support
- No markdown rendering (plain text only in v1; can add later)
- No export / share
- No theme switcher

**Implementation approach:**
- `freeaiagent/ui/index.html` — single file, self-contained
- FastAPI serves it: `@app.get("/ui") → FileResponse("freeaiagent/ui/index.html")`
- All API calls are `fetch()` to the same origin (`localhost:7731`)
- Session state held in `localStorage` (active session ID only); truth lives in SQLite
- No cookies, no WebSockets (SSE or polling for streaming)

**Session flow:**
1. On load: `GET /sessions` → populate sidebar
2. Click session: `GET /context?session=id` → render messages
3. Send message: `POST /chat` with `{message, session_id, model}` → append bubble
4. New chat: generate UUID client-side, use as `session_id` on first POST (session auto-created on first message)
5. Rename: click title → inline edit → `PATCH /sessions/{id}`
6. Delete: hover session → trash icon → `DELETE /sessions/{id}`

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
