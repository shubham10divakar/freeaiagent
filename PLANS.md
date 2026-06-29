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
| 6 | `freeaiagent rm` — delete installed models | **Done** |
| 6 | Download integrity — resume + SHA256 verify | **Done** |
| 6 | Per-backend context limits | **Done** |
| 6 | Summarization context strategy | **Done** |
| 6 | Ensemble inference (fan-out + judge) | **Done** |
| 6 | In-process GGUF backend (`llama-cpp-python`) | **Done** |
| 6 | PyPI publish (v1.3.0) | Built — pending upload |
| 7 | Desktop app (Tauri/Electron + `/ui`) — public launch | Planned |
| 7 | Web inference (WebLLM / WebGPU) | Planned |
| 7 | Cloud layer (accounts, sync, premium boost) | Planned |
| 7 | Mobile native (React Native + llama.rn) | Planned |
| 8 | **ModelX-1.0 backend** — Engine X-1.0, compound vision+language cascade | Superseded by Phase 9 |
| 9 | **SDX Engine** — Smart Decision eXecution; 5-tier compound text+vision engine (Nano→Max); standalone `freeaiagent/sdx/` module; transparent image handling | **Done** |

> **v1.3.0** is built and on `main` but not yet uploaded to PyPI or tagged. 277 tests pass. Phases 1–6 complete.
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
| 8 | **In-process engine option** (`llama-cpp-python`) — **Done** (`backends/llama_cpp.py`; loads GGUF in-process, cached per path; reuses pulled weights; optional extra `freeaiagent[llama-cpp]`; type `llama_cpp` in config) | Avoids the subprocess/llamafile hop for users who want a pure-Python path | L |

Design details for the context strategies and ensemble inference are in the
sections below; the rest are scoped enough to pick up directly.

---

## Phase 7 — Going public ("on-device ChatGPT")

The library is the foundation; the goal now is a **public, ChatGPT-like product**
users interact with freely via **desktop, mobile, and web** — **local-first**
(models run on the *user's* device, so inference is free and private) with an
**optional cloud-premium** tier for weak devices / bigger models.

**Origin & framing.** freeaiagent exists because cloud-API token/cost limits
pushed toward running models on the user's own phone/PC. So the product doubles
down on that: privacy + free + offline is the pitch — **not** "smarter than
ChatGPT." Honest capability ceiling: phones ~1–3B, PCs 7–14B.

### Core architecture principle

**freeaiagent defines the API contract; each platform is an interchangeable
engine behind it.** Write the app UI once against the chat / OpenAI-compatible
API; it doesn't care where inference runs. "Local + cloud" is just the existing
fallback chain (device-local default, hosted opt-in).

| Client | Engine behind the same contract |
|---|---|
| Desktop (Win/Mac/Linux) | freeaiagent (Python) embedded, model on disk |
| Mobile (iOS/Android) | native llama.cpp / MLC-LLM (Python can't run on-device); small models |
| Web (browser) | WebLLM / WebGPU, zero-install, desktop browsers |
| Cloud (premium) | hosted freeaiagent → cloud model / GPU |

> iOS forbids downloading + running executables → **llamafile won't work on
> mobile; inference must be compiled into the app.** Distribution (packaging onto
> devices) is the new hard problem, replacing GPU capacity.

### Sequenced roadmap (each phase ships and earns the next)

| # | Phase | Scope | Effort |
|---|---|---|---|
| 7.1 | **Desktop app + web UI** | Wrap freeaiagent in Tauri/Electron, bundle runtime, reuse `/ui`, auto-update. Purely local. **The public launch.** | M |
| 7.2 | **Web (WebLLM)** | Zero-install in-browser inference for desktop users; same API contract | M |
| 7.3 | **Cloud layer** | Accounts, encrypted cross-device sync, premium "cloud boost." Where auth + Postgres + tenant isolation + moderation + GPU land — **scoped to paying users only** | L |
| 7.4 | **Mobile native** | Embedded llama.cpp/MLC on-device (small models) + cloud fallback for weak phones. Biggest lift; after traction | XL |

Realistically 4 client platforms + a backend — multi-quarter, likely multi-person.
**Desktop + web (7.1–7.2) is the real launch**; cloud and mobile are heavier
follow-ons. Do not ship all four at once.

### Security — two tracks (see also the dedicated review)

**On-device (7.1 / 7.2 / 7.4):**
- Bind **`127.0.0.1`** (today `cli.py` binds `0.0.0.0` → LAN-exposed) with an opt-in `--host`.
- Ship a **signed + SHA256-checksummed catalog only**; no arbitrary-URL `pull` in the shipped UI (a fused llamafile is downloaded + executed = code execution on the user's machine).
- **Strip admin endpoints** (`/config`, `/config/set`, `/pull/stream`, `DELETE /models/*`, `/tools/register`) from any exposed surface.
- Wins: no central chat store (no honeypot / minimal GDPR for chat), no LLM API keys to leak.

**Hosted cloud-premium (7.3) — full hardening, premium slice only:**
- Real **AuthN** (user accounts / short-lived JWT); freeaiagent trusts only the gateway. `X-Caller-ID` is routing, not security.
- **Tenant isolation**: every session/context query scoped to the authenticated `user_id` (today any caller can read any `session_id`).
- **Postgres** context store (opt-in, behind a storage abstraction; SQLite stays the local default + WAL for light concurrency).
- TLS at the gateway; freeaiagent + inference on a **private network**.
- **Rate limits + token quotas**, prompt-size / `max_tokens` caps, request timeouts.
- **Content moderation** (input/output) + usage policy + abuse logging — required for a public app.
- **SSRF guard** on tools; config file `chmod 600`.

### Inference for the cloud tier
Do **not** use the built-in llamafile backend for serving many users (single
subprocess, effectively serial). Run a dedicated **vLLM / TGI** server
(continuous batching) and point freeaiagent's `openai_compat` backend at it;
freeaiagent does orchestration, the engine does GPU batching.

### Monetization
Free = fully local. **Premium** = cloud boost (bigger models) + encrypted sync +
priority. Cloud cost is bounded to paying users, so the economics stay clean.

---

### Phase 7.1 — Desktop App (decided approach)

**Decided:** `pip install freeaiagent` + `freeaiagent start --open` is the public launch path. No Tauri, no Electron, no signing, no sidecar. The browser is the window.

```bash
pip install freeaiagent
freeaiagent start --open
# → starts server on 127.0.0.1:7731
# → opens http://127.0.0.1:7731/ui in the user's default browser automatically
```

The `/ui` already exists and works. `--open` is a one-line CLI addition. This is already a full "desktop app experience" for any user who has Python.

#### Why this wins over a Tauri/Electron wrapper

| | `--open` flag | Tauri wrapper |
|---|---|---|
| Effort | XS (one CLI flag) | M–L (Rust, PyInstaller, sidecar, CI) |
| Code signing | Not needed | Required for SmartScreen / Gatekeeper |
| User requirement | `pip install` | Download + run installer |
| Target audience | Developers, power users | General consumers |
| Shipping time | Days | Weeks (signing alone is 1–2 weeks) |
| Signing problem | Doesn't exist | Hard — MOTW, SmartScreen, notarization |

The Tauri wrapper only adds: dedicated app icon, no browser chrome, appears in taskbar as its own app. That's a polish layer, not a functional one. Ship `--open` first.

#### Why there's no signing problem here

Files built locally or installed via `pip` have no **Mark of the Web** (MOTW) — the NTFS tag Windows uses to flag "this came from the internet." SmartScreen only triggers on MOTW files. A `pip install` puts Python files and scripts on disk with no MOTW — no warning, no block.

#### Implementation — `freeaiagent start --open`

One addition to `cli.py`:

```python
import webbrowser

@app.command()
def start(open: bool = typer.Option(False, "--open", help="Open /ui in browser after start")):
    # ... existing start logic ...
    # after server is confirmed healthy:
    if open:
        webbrowser.open("http://127.0.0.1:7731/ui")
```

`webbrowser.open()` is stdlib — zero new dependencies. Works on Windows, Mac, Linux.

#### What also needs doing (same security notes as before)

- Bind `127.0.0.1` not `0.0.0.0` in `cli.py` — no LAN exposure
- These are one-line changes, not blockers for shipping `--open`

#### Tauri wrapper — deferred, not cancelled

The Tauri/Electron wrapper design is fully documented above and remains valid for a future consumer push (Phase 7.3+) when the audience is no longer developers. At that point signing cost is justified and the audience expects a downloadable installer. Until then, `--open` covers the launch.

#### User experience with `--open`

```
pip install freeaiagent       # one-time setup
freeaiagent pull              # download a model (~2–4 GB, one time)
freeaiagent start --open      # server starts, browser opens to /ui automatically
```

Browser opens to a full ChatGPT-style interface. Chat, switch sessions, pull models — all in the browser tab. Close the terminal → server stops. Reopen anytime with `freeaiagent start --open`.

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

### Context passing in ensemble

Yes — context is fully passed. `ensemble.run()` receives `messages: List[dict]`, which is the full conversation history already loaded from SQLite (trimmed to the effective window by `context.as_llm_messages(max_messages=N)`). Every model in the fan-out receives the same `messages` list via `backend.chat(messages, model)`. No model in the ensemble sees a context-free prompt.

---

### Planned — ModelX-1.0 / Engine X-1.0 (Compound Multi-Modal Model)

**Concept:** ModelX-1.0 is a single logical model backed by **Engine X-1.0**, which internally contains **two specialist sub-models** and routes between them automatically — the caller treats it as one model.

```
Engine X-1.0
├── Sub-model A — Vision / OCR  (activates when image detected in input)
└── Sub-model B — Language / Text  (always runs; receives final prompt)
```

**Flow when an image is present:**

```
user message: [image attachment] + "what does this say?"
        │
        ▼
Engine X-1.0 detects image in message content
        │
        ├─► Sub-model A (OCR/vision): extract text from image
        │         └── returns: extracted_text
        │
        └─► Sub-model B (LLM): receives full context messages
                              + extracted_text injected as a system turn
                              → produces final response
```

**Flow without an image (text-only):**

```
user message: "summarise what we discussed"
        │
        ▼
Engine X-1.0 detects no image → skips Sub-model A entirely
        │
        └─► Sub-model B (LLM): receives full context messages as-is
```

**Context handling:** the existing sliding-window / summarization context is passed through unchanged. Sub-model A's extracted text is injected as an extra `{"role": "system", "content": "[Image text]: ..."}` message *before* Sub-model B's call, so it becomes part of the conversation that subsequent turns can reference.

**Relation to ensemble:** this is a **cascade** rather than a fan-out. Ensemble fans the *same prompt* to *N models in parallel* and picks the best; ModelX-1.0 routes the prompt *through* two models *in sequence* where one conditionally pre-processes the input. The router will register it as a single named backend (`type: modelx`) — callers just specify `model: modelx-1.0`.

**Planned config:**

```json
{
  "backends": {
    "modelx": {
      "type": "modelx",
      "engine": "X-1.0",
      "vision_model": "sub-model-a",
      "language_model": "sub-model-b"
    }
  }
}
```

**Implementation notes (future):**
- Add `ModelXBackend` in `freeaiagent/backends/modelx.py` implementing `BaseBackend`
- Image detection: check `messages[-1]["content"]` for `image_url` / base64 content parts (OpenAI multimodal message format)
- Extracted text injected as a system message, not replacing the user message, so the original image reference stays in history
- Router registers `modelx` as a valid `btype` in `router._build_backends()`

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

---

## Phase 9 — SDX Engine (Smart Decision eXecution)

Full design: [`docs/sdx-engine.md`](docs/sdx-engine.md)

**What it is:** A compound inference engine that presents one logical "model" to the caller while internally routing between a text sub-model and a vision sub-model based on whether the user attaches an image. The orchestration is invisible — the user downloads one bundle, chats, and attaches images freely.

**Why now (after Phase 8 is superseded):** The original Phase 8 ModelX sketch is a correct concept but underspecified. SDX is the full design: standalone codebase, 5 tiers, explicit context budget management, clear integration points, and a detailed build order.

### SDX model tiers

| ID | Display | Text | Vision | Bundle | Min RAM |
|---|---|---|---|---|---|
| `sdx-nano` | SDX Nano | Qwen2.5-0.5B Q4_K_M | moondream2 Q4 | ~2.1 GB | 4 GB |
| `sdx-mini` | SDX Mini | Llama-3.2-1B Q4_K_M | moondream2 Q4 | ~2.5 GB | 4 GB |
| `sdx-standard` | SDX Standard | Llama-3.2-3B Q4_K_M | llava-phi-3-mini Q4 + mmproj | ~4.7 GB | 8 GB |
| `sdx-plus` | SDX Plus | Qwen2.5-7B Q4_K_M | llava-v1.6-mistral-7b Q4 + mmproj | ~9.4 GB | 16 GB |
| `sdx-max` | SDX Max | Qwen2.5-14B Q4_K_M | llava-v1.6-mistral-7b Q4 + mmproj | ~13.4 GB | 24 GB |

### Key design decisions (locked)

- **Fully standalone codebase** in `freeaiagent/sdx/` — zero imports from the rest of the package except stdlib and `llama_cpp`
- **llama-cpp-python** for both sub-models (reuses the existing `llama_cpp` backend infrastructure): `MoondreamChatHandler` for moondream2 tiers, `Llava16ChatHandler` for LLaVA tiers
- **Images become text permanently** — vision model extracts a description; description stored in SQLite as `[SDX-Image]: <desc>` system message; raw image never persisted long-term
- **Stateless-per-call reconstruction** — ContextBuilder is rebuilt from the SQLite `messages` list on every `/chat` call; no per-session engine state cached between calls
- **Token budget per tier** (4k → 32k) with drop logic: oldest complete turn pairs dropped when over budget; current turn and system prompt never dropped; minimum 2 pairs always kept
- **One bundle, one download** — `freeaiagent pull sdx-standard` fetches text.gguf + vision.gguf + mmproj.gguf sequentially with phases `text_model` / `vision_model` / `mmproj` in SSE events
- **Thin integration** — only 7 existing files touched: `catalog.py`, `config.py`, `router.py`, `endpoints/chat.py`, `pull.py`, `ui/index.html`; all SDX logic lives in `freeaiagent/sdx/`

### What SDX does NOT do

- No tool calling (text models too small in lower tiers)
- No ensemble fan-out (separate concern)
- No regenerate on vision turns
- Does not apply the `summarize` context strategy (owns its own budget/drop logic)

### Build order (16 steps)

Steps 1–6 are pure SDX module (no package coupling; fully unit-testable in isolation):

```
1.  freeaiagent/sdx/types.py
2.  freeaiagent/sdx/context_builder.py
3.  freeaiagent/sdx/vision_runner.py
4.  freeaiagent/sdx/text_runner.py
5.  freeaiagent/sdx/engine.py
6.  freeaiagent/sdx/__init__.py
7.  freeaiagent/backends/sdx_backend.py
8.  catalog.py          — add 5 SDX bundle entries
9.  config.py           — add SDX config schema
10. router.py           — type: "sdx" dispatch
11. endpoints/chat.py   — image field + temp file lifecycle
12. pull.py             — multi-file bundle download
13. ui/index.html       — image attach, preview, thumbnail, SDX badge
14. tests/test_sdx_context_builder.py
15. tests/test_sdx_engine.py
16. tests/test_sdx_api.py
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
