import pytest

from freeaiagent import ensemble


class FakeBackend:
    """Returns a canned response per model; an Exception value is raised."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    async def chat(self, messages, model):
        self.calls.append((model, messages))
        val = self.mapping.get(model)
        if isinstance(val, Exception):
            raise val
        return val


MSGS = [{"role": "user", "content": "q"}]


# ── resolve_models ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resolve_models_explicit_list():
    assert ensemble.resolve_models(["a", "b"], {}) == ["a", "b"]


@pytest.mark.unit
def test_resolve_models_true_uses_config():
    cfg = {"ensemble": {"models": ["x", "y"]}}
    assert ensemble.resolve_models(True, cfg) == ["x", "y"]


@pytest.mark.unit
def test_resolve_models_none_enabled_uses_config():
    cfg = {"ensemble": {"enabled": True, "models": ["x", "y"]}}
    assert ensemble.resolve_models(None, cfg) == ["x", "y"]


@pytest.mark.unit
def test_resolve_models_none_disabled_is_empty():
    cfg = {"ensemble": {"enabled": False, "models": ["x", "y"]}}
    assert ensemble.resolve_models(None, cfg) == []


@pytest.mark.unit
def test_resolve_models_false_is_empty():
    assert ensemble.resolve_models(False, {"ensemble": {"models": ["x"]}}) == []


# ── run / strategies ─────────────────────────────────────────────────────────

@pytest.mark.unit
async def test_longest_strategy_picks_richest():
    backend = FakeBackend({"a": "short", "b": "a considerably longer and richer answer"})
    winner, model, votes = await ensemble.run(backend, MSGS, ["a", "b"], strategy="longest")
    assert model == "b"
    assert winner.startswith("a considerably")
    assert len(votes) == 2


@pytest.mark.unit
async def test_majority_strategy_picks_most_common():
    backend = FakeBackend({"a": "yes", "b": "Yes", "c": "no"})
    winner, model, votes = await ensemble.run(backend, MSGS, ["a", "b", "c"], strategy="majority")
    assert winner.lower() == "yes"


@pytest.mark.unit
async def test_llm_judge_selects_by_number():
    backend = FakeBackend({"a": "ans a", "b": "ans b", "judge": "2"})
    winner, model, votes = await ensemble.run(
        backend, MSGS, ["a", "b"], strategy="llm_judge", judge_model="judge"
    )
    assert model == "b"


@pytest.mark.unit
async def test_llm_judge_falls_back_to_longest_on_error():
    backend = FakeBackend({"a": "x", "b": "the longer answer", "judge": ValueError("nope")})
    winner, model, votes = await ensemble.run(
        backend, MSGS, ["a", "b"], strategy="llm_judge", judge_model="judge"
    )
    assert model == "b"  # judge failed -> longest heuristic


@pytest.mark.unit
async def test_failed_models_recorded_and_dropped():
    backend = FakeBackend({"a": RuntimeError("boom"), "b": "ok"})
    winner, model, votes = await ensemble.run(backend, MSGS, ["a", "b"], strategy="longest")
    assert model == "b"
    errored = [v for v in votes if v.get("error")]
    assert len(errored) == 1 and errored[0]["model"] == "a"


@pytest.mark.unit
async def test_all_failed_raises():
    backend = FakeBackend({"a": RuntimeError(), "b": RuntimeError()})
    with pytest.raises(RuntimeError, match="all models failed"):
        await ensemble.run(backend, MSGS, ["a", "b"])


@pytest.mark.unit
async def test_single_valid_skips_judge():
    backend = FakeBackend({"a": "only one"})
    winner, model, votes = await ensemble.run(backend, MSGS, ["a"], strategy="llm_judge")
    assert model == "a" and winner == "only one"
