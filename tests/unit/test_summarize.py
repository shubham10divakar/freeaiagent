import pytest
from unittest.mock import AsyncMock

from freeaiagent import context, summarize
from freeaiagent.summarize import SUMMARY_PREFIX


@pytest.fixture
def seeded(isolated_db):
    """A session with 6 alternating messages."""
    for i in range(3):
        context.append("user", f"u{i}", session_id="s")
        context.append("assistant", f"a{i}", session_id="s")
    return "s"


# ── context helpers ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_oldest_messages_returns_in_order(seeded):
    rows = context.oldest_messages("s", 3)
    assert [r["content"] for r in rows] == ["u0", "a0", "u1"]
    assert all("id" in r for r in rows)


@pytest.mark.unit
def test_replace_with_summary_keeps_summary_first(seeded):
    rows = context.oldest_messages("s", 4)
    ids = [r["id"] for r in rows]
    context.replace_with_summary("s", ids, "MEMORY")

    msgs = context.all_messages(session_id="s")
    # 6 - 4 replaced + 1 summary = 3 messages; summary sorts first
    assert [m["content"] for m in msgs] == ["MEMORY", "u2", "a2"]
    assert msgs[0]["role"] == "system"
    assert context.count(session_id="s") == 3


# ── maybe_summarize ──────────────────────────────────────────────────────────

@pytest.mark.unit
async def test_no_summarize_when_under_threshold(seeded):
    backend = AsyncMock()
    did = await summarize.maybe_summarize(backend, "m", "s", threshold=10, batch=4)
    assert did is False
    backend.chat.assert_not_called()


@pytest.mark.unit
async def test_no_summarize_when_threshold_zero(seeded):
    backend = AsyncMock()
    did = await summarize.maybe_summarize(backend, "m", "s", threshold=0, batch=4)
    assert did is False


@pytest.mark.unit
async def test_summarize_folds_oldest_into_memory(seeded):
    backend = AsyncMock()
    backend.chat.return_value = "- u talked about things"

    did = await summarize.maybe_summarize(backend, "chat-model", "s", threshold=4, batch=4)
    assert did is True
    backend.chat.assert_awaited_once()

    msgs = context.all_messages(session_id="s")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith(SUMMARY_PREFIX)
    # 6 - 4 + 1 = 3 remain, recent messages preserved after the summary
    assert [m["content"] for m in msgs[1:]] == ["u2", "a2"]


@pytest.mark.unit
async def test_summarize_uses_summarize_model_when_given(seeded):
    backend = AsyncMock()
    backend.chat.return_value = "summary"
    await summarize.maybe_summarize(
        backend, "chat-model", "s", threshold=4, batch=4, summarize_model="tiny-model"
    )
    assert backend.chat.await_args[0][1] == "tiny-model"
