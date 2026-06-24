"""Summarization context strategy.

When a session's history grows past a threshold, the oldest batch of messages
is compressed into a single ``system`` "memory" entry instead of being dropped
by the sliding window — so long sessions keep their early context (decisions,
constraints, names) at the cost of one extra LLM call per summarization event.

Triggered from the chat endpoints when ``context_strategy == "summarize"``.
"""
from typing import Optional

from . import context

SUMMARY_PREFIX = "[Summary of earlier conversation]"

_SUMMARIZE_SYSTEM = (
    "Summarize the conversation so far in concise bullet points. Preserve key "
    "facts, decisions, names, numbers, and any stated constraints or preferences. "
    "Output only the summary — no preamble."
)


async def maybe_summarize(
    backend,
    model: str,
    session_id: str,
    *,
    threshold: int,
    batch: int,
    summarize_model: Optional[str] = None,
) -> bool:
    """Compress the oldest ``batch`` messages into one summary if over threshold.

    Returns True when a summarization happened. No-op (False) when summarization
    is disabled (``threshold <= 0``) or the history is still within the window.
    Uses ``summarize_model`` if given, else the active chat ``model``.
    """
    if threshold <= 0:
        return False
    total = context.count(session_id=session_id)
    if total <= threshold:
        return False

    n = min(batch, total)
    if n <= 0:
        return False

    oldest = context.oldest_messages(session_id, n)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in oldest)
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": convo},
    ]
    summary = await backend.chat(messages, summarize_model or model)

    ids = [m["id"] for m in oldest]
    context.replace_with_summary(session_id, ids, f"{SUMMARY_PREFIX}\n{summary}")
    return True
