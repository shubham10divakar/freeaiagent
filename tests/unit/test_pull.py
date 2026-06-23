import pytest

from freeaiagent import pull as pull_mod


# ── resolve_target ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resolve_catalog_name():
    pt = pull_mod.resolve_target("llama-3.2-3b")
    assert pt.is_catalog_name is True
    assert pt.label == "Llama 3.2 3B Instruct"
    assert pt.size_gb == 2.3
    assert pt.min_ram_gb == 4
    assert pt.backend.model == "llama-3.2-3b"


@pytest.mark.unit
def test_resolve_hf_ref_builds_resolve_url():
    pt = pull_mod.resolve_target(
        "hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )
    assert pt.is_catalog_name is False
    assert pt.label == "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    assert pt.backend.download_url == (
        "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )


@pytest.mark.unit
def test_resolve_invalid_hf_ref_raises():
    with pytest.raises(ValueError, match="Invalid reference"):
        pull_mod.resolve_target("hf:owner/name")


@pytest.mark.unit
def test_resolve_url_target():
    pt = pull_mod.resolve_target("https://example.com/path/model.gguf", port=9001)
    assert pt.is_catalog_name is False
    assert pt.label == "model.gguf"
    assert pt.backend.download_url == "https://example.com/path/model.gguf"
    assert pt.backend.port == 9001


@pytest.mark.unit
def test_resolve_unknown_catalog_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        pull_mod.resolve_target("does-not-exist")


# ── ProgressEmitter ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_emitter_emits_start_then_progress():
    events = []
    clock = iter([0.0, 0.0])  # start ts, then progress ts
    em = pull_mod.ProgressEmitter(
        labels={"model": "my-model"}, emit=events.append,
        min_interval=0.0, _clock=lambda: next(clock),
    )
    em(50, 100, "model")
    assert events[0] == {"type": "start", "phase": "model", "label": "my-model", "total_mb": 0.0}
    assert events[1]["type"] == "progress"
    assert events[1]["pct"] == 50.0
    assert events[1]["phase"] == "model"


@pytest.mark.unit
def test_emitter_throttles_but_always_emits_final():
    events = []
    ticks = [0.0, 0.0, 0.05, 1.0]  # start, p1, throttled p2 (final emits anyway)
    em = pull_mod.ProgressEmitter(
        labels={"model": "m"}, emit=events.append,
        min_interval=0.25, _clock=lambda: ticks.pop(0),
    )
    em(10, 100, "model")        # start + progress
    em(20, 100, "model")        # within min_interval, not final -> throttled
    em(100, 100, "model")       # final -> always emits
    types = [e["type"] for e in events]
    assert types == ["start", "progress", "progress"]
    assert events[-1]["pct"] == 100.0


@pytest.mark.unit
def test_emitter_new_phase_emits_new_start():
    events = []
    em = pull_mod.ProgressEmitter(
        labels={"engine": "eng", "model": "mod"}, emit=events.append,
        min_interval=0.0, _clock=lambda: 0.0,
    )
    em(10, 100, "engine")
    em(10, 100, "model")
    starts = [e for e in events if e["type"] == "start"]
    assert [s["phase"] for s in starts] == ["engine", "model"]
    assert [s["label"] for s in starts] == ["eng", "mod"]
