# freeaiagent

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

## CLI

```bash
freeaiagent start                          # start server (default port 7731)
freeaiagent start --port 8080              # custom port
freeaiagent start --reload                 # dev mode: auto-reload on code change

freeaiagent chat                           # interactive chat (context preserved)
freeaiagent chat "quick one-liner"         # single message, then exit

freeaiagent task "explain this" \
  --input "$(cat file.py)"                 # one-shot task, no context read/written
freeaiagent task "translate to French" \
  --input "Hello world" \
  --model mistral:7b                       # override model for this task

freeaiagent status                         # health check + active backend/model
freeaiagent models                         # list models on active backend

freeaiagent context show                   # print conversation history
freeaiagent context clear                  # wipe conversation history

freeaiagent config show                    # print current config
freeaiagent config set default_model mistral:7b
freeaiagent config set default_backend groq
freeaiagent config set backends.groq.api_key gsk_...
```

---

## HTTP API

All endpoints accept and return JSON.

### `POST /chat`
Send a message. Conversation history is read and updated automatically.

```bash
curl -X POST http://localhost:7731/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what did I just ask you?"}'
```

```json
{
  "response": "You asked me what you just asked me.",
  "model": "llama3.2:3b",
  "context_length": 4
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | required |
| `system` | string | optional system prompt override |

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

### `GET /context`
Returns the full conversation history.

```json
{
  "messages": [
    {"role": "user", "content": "hello", "timestamp": "2026-06-21T10:00:00+00:00"},
    {"role": "assistant", "content": "hi there", "timestamp": "2026-06-21T10:00:01+00:00"}
  ],
  "total": 2
}
```

### `DELETE /context`
Clears all conversation history.

```json
{"cleared": 4, "message": "Cleared 4 messages."}
```

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

Free-tier models available on Groq:

| Model | Context |
|---|---|
| `llama-3.1-8b-instant` | 128k |
| `llama-3.3-70b-versatile` | 128k |
| `mixtral-8x7b-32768` | 32k |
| `gemma2-9b-it` | 8k |

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

Conversation history is stored in `~/.freeaiagent/context.db` (SQLite). It persists across server restarts. Clear it any time:

```bash
freeaiagent context clear
# or
curl -X DELETE http://localhost:7731/context
```

---

## Roadmap

**Phase 2 — Named sessions**
Multiple apps keep separate conversation histories via a `session_id` field.
```json
{"message": "hello", "session_id": "my-project"}
```

**Phase 3 — Auto caller detection**
Sessions created automatically per caller using `X-Caller-ID` header or caller port. Zero config for multi-app setups.

**Phase 4 — More backends and features**
- llamafile (single-file model, no Ollama install needed)
- Together AI, OpenRouter
- Streaming responses (`/chat/stream` SSE)
- Tool use / function calling
- Minimal web UI at `localhost:7731`

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
