# freeaiagent — Multi-Model Local Backend Design

**Status:** Proposed (design only — not yet implemented)
**Goal:** Replace the single hardcoded llamafile model with a *catalog* of free
local models that the user chooses and downloads on demand — low-end to
high-end — without requiring any external install or compiler.

---

## 1. Problem

Today the local backend hardcodes one fused llamafile (Llama-3.2-3B). That means:

- Users are stuck with whatever model we ship. No choice of size/quality.
- We're limited to the handful of models Mozilla pre-fused into `.llamafile`s.
- Bigger/smarter models (Qwen, Mistral, Phi) aren't reachable.

We want: **any GGUF model on HuggingFace, picked and downloaded by the user.**

---

## 2. Mental model — engine vs. weights

```
GGUF file (weights)  +  inference engine (runs the math)  =  working LLM
```

- A **GGUF** is just quantized weights. Not runnable alone.
- A **llamafile** = llama.cpp engine + a GGUF fused into one executable.
- The llamafile *engine binary alone* can load **any external GGUF**:
  `llamafile -m model.gguf --server --host 127.0.0.1 --port 8080`

**Key decision:** decouple engine from weights. Download the engine once
(~30 MB, rarely changes); download each model's GGUF on demand. This unlocks the
entire HuggingFace GGUF ecosystem while staying zero-install.

---

## 3. Two engines (both supported)

| | **llamafile** (default) | **llama-cpp-python** (opt-in) |
|---|---|---|
| How it runs | subprocess: engine binary + `-m gguf` | in-process Python C-extension |
| Install cost | one ~30 MB portable binary download | `pip install llama-cpp-python` |
| Compiler needed | **never** (Cosmopolitan binary) | only if no prebuilt wheel |
| Python 3.14 risk | none | wheels may lag → source build |
| Windows 4 GB cap | N/A for external GGUF | N/A |
| Subprocess mgmt | yes (already have it) | none |

**Default = llamafile + external GGUF.** Safest zero-friction path; reuses
existing subprocess code. **`llama-cpp-python` = optional backend** for users who
prefer in-process and have a working wheel/toolchain. Both load the same GGUF
catalog files, so models are shared between them.

---

## 4. Model catalog

A curated, versioned list of downloadable models. Ships in-code (a Python dict or
bundled `catalog.json`), overridable by the user.

### Schema (per entry)

```json
{
  "name": "qwen2.5-7b",
  "display": "Qwen2.5 7B Instruct",
  "url": "https://huggingface.co/.../Qwen2.5-7B-Instruct-Q4_K_M.gguf",
  "size_gb": 4.7,
  "min_ram_gb": 8,
  "context": 32768,
  "tier": "high",
  "description": "Strong reasoning and Q&A. Recommended when you need quality."
}
```

### Starter catalog (known-working GGUFs, Q4_K_M unless noted)

Ship a broader curated set now — all from well-maintained, high-download repos
(official, `bartowski`, `unsloth`) that are known to run cleanly in the
llamafile/llama.cpp engine. Live Hub search (§12) comes later; this is the
"it just works" front door.

| name | params | size | min RAM | tier | good for |
|---|---|---|---|---|---|
| `llama-3.2-1b`   | 1B   | 0.8 GB | 2 GB  | low  | fast classify / extract / tag |
| `qwen2.5-3b`     | 3B   | 2.0 GB | 4 GB  | low  | small but sharper than 1B |
| `llama-3.2-3b`   | 3B   | 2.3 GB | 4 GB  | mid  | balanced (current default) |
| `phi-3.5-mini`   | 3.8B | 2.4 GB | 4 GB  | mid  | strong reasoning per byte |
| `gemma-2-2b`     | 2B   | 1.7 GB | 4 GB  | mid  | concise, good summaries |
| `mistral-7b`     | 7B   | 4.4 GB | 8 GB  | high | general-purpose workhorse |
| `llama-3.1-8b`   | 8B   | 4.9 GB | 8 GB  | high | broad capability, 128k ctx |
| `qwen2.5-7b`     | 7B   | 4.7 GB | 8 GB  | high | reasoning, Q&A, summaries |
| `gemma-2-9b`     | 9B   | 5.8 GB | 12 GB | high | high-quality mid-size |
| `qwen2.5-14b`    | 14B  | 9.0 GB | 16 GB | max  | strongest local option |

> Sizes are approximate (Q4_K_M). Final catalog pins exact repo + filename +
> SHA256 per entry. Catalog is data, not code — adding a model = one entry.
> Users can also add their own via `freeaiagent config` (a custom entry pointing
> at any GGUF URL) or pull directly with `pull hf:<repo>/<file>` (§12).

---

## 5. CLI surface

```bash
# Discover
freeaiagent models --available     # show catalog: name, size, RAM, tier, desc
freeaiagent models                 # show locally installed GGUFs

# Acquire (with progress bar)
freeaiagent pull qwen2.5-7b        # download a catalog model by name
freeaiagent pull <url>             # download an arbitrary GGUF URL
freeaiagent rm llama-3.2-1b        # delete a downloaded GGUF to reclaim disk

# Select
freeaiagent config set default_model qwen2.5-7b
```

- `pull` with no argument → downloads the current `default_model`.
- `pull` warns if `size_gb` exceeds detected free disk or `min_ram_gb` exceeds
  system RAM (download still allowed, just flagged).

---

## 6. Config schema changes

```json
{
  "default_backend": "llamafile",
  "default_model": "llama-3.2-3b",
  "backends": {
    "llamafile": {
      "type": "llamafile",
      "engine": "llamafile",          // "llamafile" | "llama_cpp"
      "port": 8080,
      "auto_download": false,
      "gpu_layers": 9999              // -ngl; 0 = CPU only
    }
  },
  "catalog_overrides": {              // optional user-added models
    "my-model": { "url": "...", "size_gb": 5.0, "min_ram_gb": 8 }
  }
}
```

- `default_model` becomes a **catalog name**, not a raw filename.
- `engine` selects llamafile (subprocess) vs llama_cpp (in-process).

---

## 7. Storage layout

```
~/.freeaiagent/
  config.json
  engine/
    llamafile(.exe)                  # the ~30 MB engine binary (once)
  models/
    llama-3.2-3b.Q4_K_M.gguf
    qwen2.5-7b.Q4_K_M.gguf
```

Engine downloaded once and reused for every model. Models live side by side;
switching default_model is instant if already downloaded.

---

## 8. Backend changes

### `LlamafileBackend` → `LocalBackend` (or keep name, change internals)

- **Resolve model:** `default_model`/override → catalog lookup → GGUF path in
  `~/.freeaiagent/models/`.
- **Ensure engine:** download engine binary once if missing.
- **Ensure model:** if GGUF missing → error pointing to `freeaiagent pull <name>`
  (or download if `auto_download`).
- **Start:** `engine -m <gguf> --server --host 127.0.0.1 --port <p> -ngl <n>`.
- **Switching models:** restart the subprocess with a different `-m`. (Phase 2:
  keep a small LRU of running engines, or just restart on model change.)

### New `LlamaCppBackend` (opt-in)

- `from llama_cpp import Llama` (import guarded; only if installed).
- Loads the GGUF in-process; exposes same `chat()` / `available_models()`.
- Selected via `engine: "llama_cpp"`. If `llama-cpp-python` isn't installed,
  `is_available()` returns False with a hint to `pip install llama-cpp-python`.

Both implement the existing `BaseBackend` interface, so the router is unchanged.

---

## 9. Download module (shared)

Extract the current progress-bar download into a reusable `downloader.py`:

```python
def download(url, dest, *, label="", on_progress=None) -> Path
```

Used by engine download, catalog model download, and `pull`. Streams to a
`.part` temp file, renames on success, shows the ASCII progress bar.

---

## 10. Backwards compatibility

- Old config with `default_model: "Llama-3.2-3B-Instruct"` → map legacy names to
  catalog names on load (alias table), or warn + suggest `config set`.
- Existing fused `.llamafile` in `~/.freeaiagent/llamafile/` → detect and either
  keep working (legacy path) or prompt to migrate to engine+GGUF.
- `auto_download` semantics unchanged (default off; explicit `pull` preferred).

---

## 11. Risks & open questions

- **RAM/disk reality:** a 14B model needs ~16 GB RAM. Catalog must surface this;
  `pull` should warn. Bad UX if a user pulls 9 GB then it won't load.
- **CPU speed:** larger models on CPU are slow (a few tok/s). Set expectations in
  `models --available` output (tier + "CPU slow" note for high/max tiers).
- **Engine binary source/version:** pin a specific llamafile release URL +
  checksum for reproducibility. Decide update policy.
- **GGUF chat templates:** different model families use different prompt
  templates. The engine's `--server` + OpenAI `/chat/completions` endpoint
  handles most via the GGUF's embedded template — verify per catalog model.
- **Checksums:** add SHA256 per catalog entry to verify downloads (corruption /
  truncation safety).
- **llama-cpp-python on Python 3.14:** confirm wheel availability before
  recommending; otherwise document the CPU-wheel/toolchain requirement.

---

## 12. Live HuggingFace discovery (LATER — not in the first cut)

The curated catalog (§4) is the front door and ships first. *Later*, add a
power-user escape hatch: search the whole Hub and pull any GGUF. Confirmed
feasible with plain `httpx` — no extra dependency, no API key for public repos.

### Endpoints

| Purpose | Request | Returns |
|---|---|---|
| Search repos | `GET /api/models?search=<q>&filter=gguf&sort=downloads&limit=20` | `id`, `downloads`, `likes`, `tags`, `library_name` |
| List files | `GET /api/models/<repo>/tree/main` | per file: `path` (filename), `size` (bytes) |
| Download | `https://huggingface.co/<repo>/resolve/main/<file>` | the GGUF (same pattern as `pull`) |

### CLI

```bash
freeaiagent search qwen2.5                       # ranked repo list (by downloads)
freeaiagent search bartowski/Qwen2.5-7B-...-GGUF  # quant files + real sizes
freeaiagent pull hf:<repo>/<file>                 # download any GGUF directly
```

### Caveats (the API can't solve these)

- **Quality varies** — random repos may ship broken quants or wrong chat
  templates. Sort by `downloads`/`likes`; the curated catalog stays the safe path.
- **RAM not in the API** — estimate from quant + param count and warn (see §11).
- **Gated models** (some official Llama/Gemma) 401 without an HF token — emit a
  clear "this model is gated; set an HF token or pick an open mirror" error.
  Community re-quants (`bartowski`, `unsloth`) are generally ungated.
- **Chat templates** — usually embedded in the GGUF and read by the engine's
  server mode; verify per model.

---

## 13. Suggested phasing

1. ✅ **Catalog + downloader + `pull <name>` / `models --available`** — done.
2. ✅ **Engine/weights split** — done: bare llamafile engine (0.10.3) runs
   external GGUF via `-m`; storage split into `engine/` + `models/`; GGUF
   catalog entries (qwen2.5-7b, llama-3.1-8b, qwen2.5-14b).
3. ⏳ **`llama-cpp-python` opt-in backend** — not done (still deferred; the
   engine path covers the need without a compiled dependency).
4. ◐ **Polish** — disk warning + legacy migration done; checksums, custom
   catalog entries, and a `rm` command still TODO.
5. ✅ **Live HuggingFace discovery (§12)** — done: `freeaiagent search <term|repo>`
   and `freeaiagent pull hf:<repo>/<file>`.

---

## Relationship to existing roadmap

This expands PLANS.md Phase 4 ("more backends … llamafile auto-start"). It does
not change context handling, sessions, or the HTTP API — only how the local
backend acquires and runs models. The `/chat` and `/task` contracts are
unchanged; `model` overrides simply accept catalog names.
