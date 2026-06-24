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
| 4 | PyPI publish (v1.0.0 → v1.1.0) | **Done** |
| 5 | Python SDK (`freeaiagent.Client`) — full CLI parity + live pull progress | **Done** |
| 5 | `/pull/stream` SSE endpoint — server-side download with live progress events | **Done** |
| 5 | `/models/catalog`, `/models/installed`, `/config` HTTP endpoints | **Done** |
| 5 | Port lock file (`~/.freeaiagent/server.json`) — auto-discovery for SDK | **Done** |
| 5 | OpenAI-compatible `/v1/chat/completions` proxy endpoint | **Done** |
| 5 | System service install (`freeaiagent install` / `freeaiagent uninstall`) | **Done** |
| 5 | PyPI publish (v1.2.0) | Built — pending upload |

> **v1.2.0** is built and validated (`dist/`, `twine check` passed) but not yet
> uploaded to PyPI or tagged. 210 tests pass. Phases 1–5 complete.

---

## Phase 6 — What's Next

Phases 1–5 are done. Candidates for the next cycle, roughly in priority order:

| # | Item | Why now | Effort |
|---|---|---|---|
| 1 | **Publish v1.2.0** — `twine upload`, tag `v1.2.0`, GitHub release | The SDK is the whole point of Phase 5; nothing ships value until it's on PyPI | XS |
| 2 | **Wire Magpie onto the SDK** — replace its raw HTTP calls with `Client(name="magpie", auto_start=True)` | The origin use case; proves the SDK end-to-end and surfaces real gaps | S |
| 3 | **`freeaiagent rm <model>`** + reuse `/models/installed` to free disk — **Done** (`installed.py` shared helper; CLI `rm`, `DELETE /models/installed/{name}`, SDK `models.rm()`) | Pulls accumulate multi-GB files with no delete path today | S |
| 4 | **Download integrity** — SHA256 verify after `pull`, resume `.part` files — **Done** (HTTP Range resume; partials kept on network error; opt-in `sha256` catalog key verified post-download) | Big downloads over flaky links silently corrupt; only `.part` rename guards now | M |
| 5 | **Per-backend context limits** (Phase 2 Context below) — **Done** (`router.resolve` returns `(backend, model, max_messages)`; order: per-call `max_messages` → `backends.<name>.max_messages` → global) | Single global `max_messages` doesn't fit 8k vs 128k models | M |
| 6 | **Summarization context strategy** (Phase 3 Context below) — **Done** (`summarize.py`; `context_strategy=summarize` folds oldest `summarize_batch` into a system memory block past `summarize_threshold`; summary reuses min id to stay at head) | Long research sessions lose early context under the sliding window | M |
| 7 | **Ensemble inference** (design below) — **Done** (`ensemble.py`; `/chat` `"ensemble"` field or config; parallel fan-out + llm_judge/longest/majority; `ensemble_votes` in response; failed models dropped) | Higher answer quality for high-stakes one-shot tasks | L |
| 8 | **In-process engine option** (`llama-cpp-python`) | Avoids the subprocess/llamafile hop for users who want a pure-Python path | L |

Design details for the context strategies and ensemble inference are in the
sections below; the rest are scoped enough to pick up directly.

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
| Ollama | **Done** | Own protocol + openai-compat |
| Groq | **Done** | Free tier, fast |
| OpenAI-compatible | **Done** | Covers LM Studio, llamafile, LocalAI, Jan, llama.cpp |
| llamafile | **Done** | Dedicated zero-install backend: auto-download + auto-start a self-contained model |
| Together AI | **Done** | Built-in openai_compat preset; set api_key to activate |
| OpenRouter | **Done** | Built-in openai_compat preset; set api_key to activate |
| Cerebras | **Done** | Built-in openai_compat preset; set api_key to activate |
| Gemini | **Done** | Built-in openai_compat preset with api_prefix; set api_key to activate |

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

### PyPI Publish — **Done**

v1.0.0 shipped. v1.1.0 shipped with model catalog, engine/weights split,
streaming, tool use, cloud presets, caller detection, HF discovery, Chat UI.

---

## Phase 5 — Python SDK & Seamless App Integration

**Goal:** any app (Magpie, scripts, other projects) calls freeaiagent from Python
with one import and zero HTTP boilerplate. Live download progress so UIs can
show a real progress bar during `pull`.

---

### The core problems

**Problem 1 — Download progress is print-only today.**

`_download_file()` calls `_print_progress()` which hardcodes `print()` to stdout.
No way for outside code to subscribe to progress. Must be refactored to a callback
before anything else can build on it.

```
today:   _download_file()  →  _print_progress()  →  stdout
target:  _download_file(on_chunk=fn)  →  fn(done_bytes, total_bytes, phase)
                                     ↗  CLI: print callback (unchanged UX)
                                     ↗  SSE endpoint: queue.put callback
                                     ↗  SDK: yields PullProgress objects
```

**Problem 2 — Server must own the download (not the SDK).**

Magpie shouldn't know or care where `~/.freeaiagent/models/` is. The server
manages files. Magpie calls `agent.pull("qwen2.5-7b")` and watches a live stream.
Correct architecture: `/pull/stream` SSE endpoint; SDK subscribes.

**Problem 3 — SDK surface must feel like a library, not a CLI port.**

Every CLI command gets a Python equivalent, grouped naturally.

---

### Build order (each step unblocks the next)

```
Step 1  _download_file callback refactor      → unblocks step 2
Step 2  /pull/stream SSE endpoint             → unblocks Client.pull()
Step 3  /models/catalog + /models/installed + /config HTTP endpoints
Step 4  freeaiagent/client.py full SDK
Step 5  Port lock file (~/.freeaiagent/server.json)
Step 6  /v1/chat/completions OpenAI-compat proxy
Step 7  freeaiagent install (system service)
```

---

### Step 1 — `_download_file` callback refactor

Add `on_chunk` and `phase` params. CLI path passes `None` — behaviour and UX
unchanged. New SSE endpoint passes a `queue.put` callback.

```python
def _download_file(self, url, dest, *, force=False, make_exec=False,
                   on_chunk=None, phase="model") -> Path:
    ...
    while chunk := resp.read(1024 * 1024):
        f.write(chunk)
        done += len(chunk)
        if on_chunk:
            on_chunk(done, total, phase)
        else:
            self._print_progress(done, total)   # unchanged CLI fallback
```

---

### Step 2 — `/pull/stream` SSE endpoint

Server runs the download in a thread; a queue feeds the SSE stream. One active
download at a time (return 409 if another is running).

```
POST /pull/stream
Body: {"model": "qwen2.5-7b", "force": false}

data: {"type": "start",    "phase": "engine", "total_mb": 305,  "label": "llamafile engine"}
data: {"type": "progress", "phase": "engine", "pct": 42,  "downloaded_mb": 128, "total_mb": 305, "speed_mbps": 5.2}
data: {"type": "start",    "phase": "model",  "total_mb": 4700, "label": "qwen2.5-7b"}
data: {"type": "progress", "phase": "model",  "pct": 12,  "downloaded_mb": 564, "total_mb": 4700, "speed_mbps": 8.1}
data: {"type": "done",     "path": "~/.freeaiagent/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
data: [DONE]
```

Error event if download fails:
```
data: {"type": "error", "message": "Connection reset by peer"}
data: [DONE]
```

---

### Step 3 — New management endpoints

| Endpoint | Purpose |
|---|---|
| `GET /models/catalog` | Return catalog with `installed: true/false` per entry |
| `GET /models/installed` | List locally downloaded model files with paths + sizes |
| `GET /config` | Return `~/.freeaiagent/config.json` as JSON |
| `POST /config/set` | Body: `{"key": "default_backend", "value": "groq"}` |

These are currently CLI-only (read from disk in the CLI process). Moving them
behind HTTP lets the SDK manage config without knowing the config file path.

---

### Step 4 — `freeaiagent/client.py`

Single file, synchronous public API. SSE and streaming methods return iterators
so callers never need `async`.

**`PullProgress` dataclass:**

```python
@dataclass
class PullProgress:
    type: str           # "start" | "progress" | "done" | "error"
    phase: str          # "engine" | "model"
    label: str          # "llamafile engine" | "qwen2.5-7b"
    pct: float          # 0–100
    downloaded_mb: float
    total_mb: float
    speed_mbps: float
    path: str | None    # set on "done"
    error: str | None   # set on "error"
```

**`Client` class:**

```python
from freeaiagent import Client

agent = Client(
    name="magpie",        # auto sets X-Caller-ID on every request
    auto_start=True,      # starts freeaiagent server if not running
    session="magpie",     # default session for chat calls
)

# ── Chat ──────────────────────────────────────────────────────────────────
response = agent.chat("hello")
response = agent.chat("hello", session="work", model="qwen2.5-7b", tools=True)

for token in agent.stream("write a haiku"):
    print(token, end="", flush=True)

result = agent.task("extract all TODOs", input=code_text)

# ── Download with live progress ───────────────────────────────────────────
for p in agent.pull("qwen2.5-7b"):
    if p.type == "progress":
        print(f"[{p.phase}] {p.pct:.0f}%  {p.downloaded_mb:.0f}/{p.total_mb:.0f} MB")
    elif p.type == "done":
        print(f"Saved to {p.path}")

# callback style for simple callers:
agent.pull("qwen2.5-7b", on_progress=lambda p: print(f"{p.pct:.0f}%"))

# ── Discovery ─────────────────────────────────────────────────────────────
agent.search("qwen2.5")                               # list HF repos
agent.search("bartowski/Qwen2.5-7B-Instruct-GGUF")   # list files in repo

# ── Models ────────────────────────────────────────────────────────────────
agent.models.catalog()       # list with installed=True/False
agent.models.installed()     # locally downloaded files + paths + sizes
agent.models.active()        # currently loaded model name

# ── Sessions ──────────────────────────────────────────────────────────────
agent.sessions.list()
agent.sessions.create("work", title="Work project")
agent.sessions.rename("work", "Work Project v2")
agent.sessions.delete("work")

# ── Context ───────────────────────────────────────────────────────────────
agent.context.get(session="work")    # list of {role, content, timestamp}
agent.context.clear(session="work")

# ── Config ────────────────────────────────────────────────────────────────
agent.config.get()
agent.config.set("default_backend", "groq")
agent.config.set("backends.groq.api_key", "gsk_...")

# ── Tools ─────────────────────────────────────────────────────────────────
agent.tools.register("get_weather",
    description="Get weather for a city",
    endpoint="http://localhost:9000/weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)
agent.tools.list()
agent.tools.unregister("get_weather")

# ── Health / lifecycle ────────────────────────────────────────────────────
agent.health()        # {"status": "ok", "backend": "llamafile", "model": "llama-3.2-3b"}
agent.is_running()    # bool — fast health check
agent.start()         # start server subprocess, wait until ready
agent.stop()          # stop it
```

`auto_start=True` behaviour:
1. Call `is_running()` — if True, do nothing
2. Read `~/.freeaiagent/server.json` for port (step 5)
3. If not running, spawn `freeaiagent start` as a subprocess
4. Poll `/health` until ready (max 30 s) then proceed

---

### Step 5 — Port lock file

Server writes `~/.freeaiagent/server.json` on start, removes it on clean exit:

```json
{"port": 7731, "pid": 12345, "started_at": "2026-06-22T10:00:00Z"}
```

`Client()` reads it instead of hardcoding 7731. Apps survive port changes with
zero config on their side. If the file exists but the PID is dead, Client treats
it as not running and re-starts.

---

### Step 6 — `/v1/chat/completions` OpenAI-compat proxy

freeaiagent speaks the OpenAI wire protocol on the outside, routes internally to
whatever backend is active. Any app already using the OpenAI SDK, LangChain,
or LlamaIndex points `base_url` at freeaiagent — zero code change on their side.

```python
from openai import OpenAI
llm = OpenAI(base_url="http://localhost:7731/v1", api_key="none")
llm.chat.completions.create(model="qwen2.5-7b", messages=[...])
```

Also enables LangChain:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:7731/v1", api_key="none")
```

Endpoints to implement: `POST /v1/chat/completions`, `GET /v1/models`.
freeaiagent handles context, sessions, fallback transparently.

---

### Step 7 — System service (`freeaiagent install`)

```bash
freeaiagent install      # Windows: SC service or NSSM; Linux: systemd unit; Mac: launchd plist
freeaiagent uninstall
freeaiagent service status
```

After install, freeaiagent starts on boot and is always available. Apps use
`Client(auto_start=False)` and assume it's up — like a local database. The
`auto_start=True` path becomes a fast health check rather than a subprocess spawn.

---

### Magpie integration example (target state)

```python
# In Magpie — no LLM logic, no model management, no context handling
from freeaiagent import Client

agent = Client(name="magpie", auto_start=True)

def ask_llm(prompt: str) -> str:
    return agent.chat(prompt)

def summarize(text: str) -> str:
    return agent.task("summarize this concisely", input=text)

def download_model(name: str, on_progress):
    for p in agent.pull(name):
        on_progress(p)   # Magpie UI renders a real progress bar
```

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
