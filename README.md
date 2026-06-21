# freeaiagent

[![PyPI Downloads](https://img.shields.io/pypi/dm/freeaiagent?style=flat-square&logo=pypi&label=Downloads&color=blue)](https://pypi.org/project/freeaiagent/)
[![Total Downloads](https://pepy.tech/badge/freeaiagent)](https://pepy.tech/project/freeaiagent)
[![PyPI Version](https://img.shields.io/pypi/v/freeaiagent?style=flat-square&logo=pypi&color=green)](https://pypi.org/project/freeaiagent/)
[![Python](https://img.shields.io/pypi/pyversions/freeaiagent?style=flat-square&logo=python)](https://pypi.org/project/freeaiagent/)
[![License](https://img.shields.io/github/license/shubham10divakar/freeaiagent?style=flat-square)](https://github.com/shubham10divakar/freeaiagent/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/shubham10divakar/freeaiagent?style=flat-square&logo=github)](https://github.com/shubham10divakar/freeaiagent)

A local AI agent service you `pip install` once and call from anywhere.

Runs as a persistent HTTP server on `localhost:7731`. Stores conversation history in SQLite. Any app — script, CLI tool, personal project — can delegate tasks to it with a single HTTP call, no LLM code required on the caller's side.

Built on free LLM backends: **Ollama** (local, no API key) and **Groq** (free-tier cloud).

---

## Why

Embedding LLM logic directly into every app that needs it means duplicating prompt management, context handling, model configuration, and install detection across projects. When something changes — a new model, a different provider, a context bug — you fix it in every app separately.

`freeaiagent` is the single place that owns all of that. Your apps just call an endpoint.

---

## Install

```bash
pip install freeaiagent          # Ollama only
pip install "freeaiagent[groq]"  # + Groq support
```

Requires Python 3.10+.

---

## Free API Keys

No budget needed. Every option below is free.

### Ollama — local, no key, no internet
Runs entirely on your machine. Best for privacy.

1. Download from **[ollama.com](https://ollama.com)**
2. Pull a model: `ollama pull llama3.2:3b`
3. `freeaiagent start` — done.

### Groq — fastest free cloud inference
No credit card. ~1000 requests/day free.

1. Sign up at **[console.groq.com](https://console.groq.com)** → API Keys → Create
2. Wire it in:
```bash
freeaiagent config set backends.groq.api_key gsk_...
freeaiagent config set default_backend groq
freeaiagent config set default_model openai/gpt-oss-20b
```

### Google Gemini — 1500 free requests/day
Works via the existing OpenAI-compatible backend (Google provides an OpenAI-compatible endpoint).

1. Get a key at **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)** — free, no billing setup
2. Wire it in:
```bash
freeaiagent config set backends.gemini.type openai_compat
freeaiagent config set backends.gemini.base_url https://generativelanguage.googleapis.com/v1beta/openai
freeaiagent config set backends.gemini.api_key AIza...
freeaiagent config set default_backend gemini
freeaiagent config set default_model gemini-2.0-flash
```

Free Gemini models: `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`

### OpenRouter — 50+ providers, free models available
Free credits on signup. Many models have a `:free` variant.

1. Sign up at **[openrouter.ai](https://openrouter.ai)** → Keys → Create
2. Wire it in:
```bash
freeaiagent config set backends.openrouter.type openai_compat
freeaiagent config set backends.openrouter.base_url https://openrouter.ai/api
freeaiagent config set backends.openrouter.api_key sk-or-...
freeaiagent config set default_backend openrouter
freeaiagent config set default_model meta-llama/llama-3.1-8b-instruct:free
```

Some free OpenRouter models: `meta-llama/llama-3.1-8b-instruct:free`, `google/gemma-2-9b-it:free`, `mistralai/mistral-7b-instruct:free`

### LM Studio / Jan / LocalAI — local GUI apps, no key
These run models locally and expose an OpenAI-compatible server.

| App | Download | Default port |
|---|---|---|
| LM Studio | [lmstudio.ai](https://lmstudio.ai) | 1234 |
| Jan | [jan.ai](https://jan.ai) | 1337 |
| LocalAI | [localai.io](https://localai.io) | 8080 |

```bash
# Example: LM Studio running on default port
freeaiagent config set backends.lmstudio.type openai_compat
freeaiagent config set backends.lmstudio.base_url http://localhost:1234
freeaiagent config set default_backend lmstudio
```

> **No keys?** Run `freeaiagent keys` anytime to see this setup guide.

---

## Quick start

**1. Start the server**
```bash
freeaiagent start
# Running at http://localhost:7731
# API docs at http://localhost:7731/docs
```

**2. Chat with it**
```bash
freeaiagent chat
# You: what is the capital of France?
# Agent [llama3.2:3b]: Paris.
```

**3. Call it from any app**
```python
import urllib.request, json

req = urllib.request.Request(
    "http://localhost:7731/chat",
    data=json.dumps({"message": "summarize this for me: ..."}).encode(),
    headers={"Content-Type": "application/json"},
)
response = json.loads(urllib.request.urlopen(req).read())["response"]
```

No pip dependencies needed in the calling app. Pure stdlib.

---

## Chat UI

A browser-based chat interface — similar to ChatGPT, no signup, runs entirely on your machine.

**Open it:**
```bash
freeaiagent start   # then visit http://localhost:7731/ui
```

```
┌──────────────┬──────────────────────────────────────────────┐
│ + New Chat   │  Model: [openai/gpt-oss-20b ▾]               │
├──────────────┤──────────────────────────────────────────────┤
│ Chat 1       │                                              │
│ Chat 2       │    [assistant response]                      │
│ Chat 3  ···  │             [your message]                   │
│              │    [assistant response]                      │
│              ├──────────────────────────────────────────────┤
│              │  [ Type a message…                  ] [↑]   │
└──────────────┴──────────────────────────────────────────────┘
```

- **Sessions sidebar** — all your chats, ordered by most recent. Click to switch.
- **New Chat** — starts a fresh session instantly, no naming required.
- **Model picker** — dropdown populated live from your configured backends. Change mid-chat.
- **Rename** — double-click any session title to edit it inline.
- **Delete** — hover a session and click `×`.
- **Keyboard** — Enter to send, Shift+Enter for a newline.

Sessions and history are stored in `~/.freeaiagent/context.db` and persist across restarts.

---

## CLI

```bash
freeaiagent start                          # start server (default port 7731)
freeaiagent start --port 8080              # custom port
freeaiagent start --reload                 # dev mode: auto-reload on code change

freeaiagent chat                           # interactive chat (default session)
freeaiagent chat "quick one-liner"         # single message, then exit
freeaiagent chat --session work            # chat in a named session
freeaiagent chat --session work "hi"       # single message to named session

freeaiagent sessions                       # list all sessions

freeaiagent task "explain this" \
  --input "$(cat file.py)"                 # one-shot task, no context read/written
freeaiagent task "translate to French" \
  --input "Hello world" \
  --model mistral:7b                       # override model for this task

freeaiagent status                         # health check + active backend/model
freeaiagent models                         # list models on active backend
freeaiagent keys                           # show where to get free API keys

freeaiagent context show                   # print conversation history (default session)
freeaiagent context show --session work    # print history for a named session
freeaiagent context clear                  # wipe default session
freeaiagent context clear --session work   # wipe a named session

freeaiagent config show                    # print current config
freeaiagent config set default_model mistral:7b
freeaiagent config set default_backend groq
freeaiagent config set backends.groq.api_key gsk_...
```

---

## HTTP API

All endpoints accept and return JSON.

### `POST /chat`
Send a message. Conversation history is read and updated automatically per session.

```bash
curl -X POST http://localhost:7731/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what did I just ask you?", "session_id": "work"}'
```

```json
{
  "response": "You asked me what you just asked me.",
  "model": "llama3.2:3b",
  "session_id": "work",
  "context_length": 4
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | required |
| `session_id` | string | session to read/write context for (default: `"default"`) |
| `system` | string | optional system prompt override for this message |
| `model` | string | optional model override for this message |
| `backend` | string | optional backend override for this message |

---

### `POST /task`
One-shot task. No context is read or written — clean slate every time.

```bash
curl -X POST http://localhost:7731/task \
  -H "Content-Type: application/json" \
  -d '{"task": "list all TODO comments", "input": "..."}'
```

```json
{
  "result": "Line 42: TODO fix this\nLine 87: TODO add tests",
  "model": "llama3.2:3b"
}
```

| Field | Type | Description |
|---|---|---|
| `task` | string | required — the instruction |
| `input` | string | optional — content to work on |
| `model` | string | optional — override model for this call |
| `system` | string | optional — override system prompt |

---

### `GET /context?session=default`
Returns the conversation history for a session.

```json
{
  "messages": [
    {"role": "user", "content": "hello", "timestamp": "2026-06-21T10:00:00+00:00"},
    {"role": "assistant", "content": "hi there", "timestamp": "2026-06-21T10:00:01+00:00"}
  ],
  "total": 2,
  "session_id": "default"
}
```

### `DELETE /context?session=default`
Clears conversation history for a session.

```json
{"cleared": 4, "message": "Cleared 4 messages.", "session_id": "default"}
```

### `GET /sessions`
Lists all sessions ordered by most recent activity.

```json
{
  "sessions": [
    {
      "id": "work",
      "title": "Explain the difference between...",
      "model": "openai/gpt-oss-20b",
      "message_count": 14,
      "last_updated": "2026-06-21T10:45:00Z"
    }
  ]
}
```

### `POST /sessions`
Create a session explicitly (also auto-created on first `/chat` message).

```json
{"id": "my-project", "title": "My Project"}
```

### `PATCH /sessions/{id}`
Rename a session: `{"title": "New name"}`

### `DELETE /sessions/{id}`
Delete a session and all its messages.

### `GET /health`
```json
{"status": "ok", "active_backend": "ollama", "default_model": "llama3.2:3b"}
```
Returns `"status": "degraded"` with an `"error"` field if no backend is reachable.

### `GET /models`
```json
{"models": ["llama3.2:3b", "mistral:7b", "phi3:mini"]}
```

---

## Configuration

Config lives at `~/.freeaiagent/config.json` and is created on first run.

```json
{
  "default_backend": "ollama",
  "default_model": "llama3.2:3b",
  "port": 7731,
  "backends": {
    "ollama": {
      "base_url": "http://localhost:11434"
    },
    "groq": {
      "api_key": ""
    }
  },
  "fallback_order": ["ollama", "groq"]
}
```

Edit directly or use `freeaiagent config set <key> <value>` with dotted keys:

```bash
freeaiagent config set port 8080
freeaiagent config set backends.ollama.base_url http://192.168.1.10:11434
freeaiagent config set backends.groq.api_key gsk_abc123
freeaiagent config set default_backend groq
freeaiagent config set default_model llama-3.1-8b-instant
```

---

## Backends

### Ollama (default)
Runs locally. No API key. No data leaves your machine.

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:3b
freeaiagent start
```

Any model available on your Ollama instance works. Switch with:
```bash
freeaiagent config set default_model mistral:7b
```

### Groq (free-tier cloud)
Fast inference. Free API key at [console.groq.com](https://console.groq.com).

```bash
freeaiagent config set backends.groq.api_key gsk_...
freeaiagent config set default_backend groq
freeaiagent config set default_model llama-3.1-8b-instant
```

Models are fetched live from the Groq API when your key is set — the list below is the fallback.

| Model | Context | Notes |
|---|---|---|
| `openai/gpt-oss-20b` | 131k | Current production — replaces llama-3.1-8b |
| `openai/gpt-oss-120b` | 131k | Current production — most capable |
| `groq/compound` | 131k | Agentic system with web search + code execution |
| `groq/compound-mini` | 131k | Lighter compound system |
| `llama-3.1-8b-instant` | 131k | Deprecated Aug 16 2026 |
| `llama-3.3-70b-versatile` | 131k | Deprecated Aug 16 2026 |
| `qwen/qwen3.6-27b` | 131k | Preview |
| `qwen/qwen3-32b` | 131k | Preview — deprecated Jul 17 2026 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 131k | Preview — deprecated Jul 17 2026 |

### Automatic fallback
If the default backend is unreachable, `freeaiagent` tries the next one in `fallback_order` automatically. No configuration needed for this to work — just have both set up.

---

## Using from another app

The agent runs as a separate process. Your app calls it over HTTP — no LLM dependencies, no model management, no context handling in your code.

**Python (stdlib only):**
```python
import urllib.request, json

def ask(message):
    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        "http://localhost:7731/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())["response"]

def run_task(task, input_text=None):
    body = json.dumps({"task": task, "input": input_text}).encode()
    req = urllib.request.Request(
        "http://localhost:7731/task",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())["result"]
```

**Shell:**
```bash
curl -s -X POST http://localhost:7731/task \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"summarize\", \"input\": \"$(cat notes.txt)\"}" \
  | python -c "import sys,json; print(json.load(sys.stdin)['result'])"
```

**JavaScript / Node:**
```js
const res = await fetch("http://localhost:7731/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "hello" }),
});
const { response } = await res.json();
```

---

## Context storage

Conversation history is stored in `~/.freeaiagent/context.db` (SQLite). It persists across server restarts.

Each session has its own history. Clear a session any time:

```bash
freeaiagent context clear                  # clear the default session
freeaiagent context clear --session work   # clear a named session
# or via HTTP:
curl -X DELETE "http://localhost:7731/context?session=work"
curl -X DELETE "http://localhost:7731/sessions/work"  # also deletes session record
```

---

## Roadmap

**Done**
- Ollama, Groq, OpenAI-compatible backends (LM Studio, Gemini, OpenRouter, Jan, LocalAI, llamafile)
- Per-call model and backend overrides
- Sliding window context (`max_messages` config)
- Session-based context windows (`session_id` on all endpoints)
- Chat web UI at `localhost:7731/ui`

**Phase 3 — Auto caller detection**
Sessions created automatically per caller using `X-Caller-ID` header or caller port. Zero config for multi-app setups.

**Phase 4 — More features**
- llamafile dedicated backend (auto-start the `.exe`, no Ollama install)
- Streaming responses (`/chat/stream` SSE)
- Ensemble inference (fan out same query to multiple models, pick the best output)
- Tool use / function calling

---

## Development

```bash
git clone <repo>
cd freeaiagent
pip install -e ".[dev,groq]"

pytest                              # unit + integration (no LLM needed)
pytest tests/smoke/ -m smoke -v    # smoke tests (requires Ollama running)
```

---

## License

MIT
