"""Flat-text context builder for the SDX compound engine.

Converts the SQLite message history (OpenAI-format dicts) into a single
flat text string that both the text model and vision model understand.
Manages a token budget: oldest complete turn pairs are dropped when the
history would overflow the model's context window.
"""
from __future__ import annotations

SDX_IMAGE_PREFIX = "[SDX-Image]: "

SYSTEM_PROMPT = (
    "You are FreeAIAgent, a helpful and concise assistant running fully on the user's device.\n"
    "Answer directly. Use Markdown when helpful."
)


class ContextBuilder:
    """Build a flat text context string from an OpenAI-format message list.

    ``messages`` is the full list including the current user turn as the last
    element. History is everything before that last element.

    Budget rules:
    - System prompt and current turn are never dropped.
    - Oldest complete turn pairs are dropped first when over budget.
    - At least ``min_turns_keep`` recent pairs are always kept.
    """

    def __init__(
        self,
        messages: list[dict],
        token_budget: int = 8192,
        min_turns_keep: int = 2,
    ) -> None:
        if messages:
            self._history = messages[:-1]
            self._current_text = messages[-1].get("content", "")
        else:
            self._history = []
            self._current_text = ""
        self._token_budget = token_budget
        self._min_turns_keep = min_turns_keep

    # ── Token estimation ────────────────────────────────────────────────────

    def _est(self, text: str) -> int:
        """Fast token estimate: 1 token ≈ 4 chars. Never returns 0."""
        return max(1, len(text) // 4)

    # ── Message rendering ────────────────────────────────────────────────────

    def _render(self, msg: dict) -> str | None:
        """Render one message to a display line; None → skip."""
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            if content.startswith(SDX_IMAGE_PREFIX):
                return f"[Image: {content[len(SDX_IMAGE_PREFIX):]}]"
            return None  # non-SDX system messages are not relayed
        if role == "user":
            return f"User: {content}"
        if role == "assistant":
            return f"Assistant: {content}"
        return None

    # ── Pair grouping ────────────────────────────────────────────────────────

    def _group_pairs(self, msgs: list[dict]) -> list[list[dict]]:
        """Group messages into exchange pairs that end with an assistant turn.

        System (image annotation) messages attach to the user turn they
        precede. Any trailing messages without a closing assistant turn form
        an orphan pair and are still included.
        """
        pairs: list[list[dict]] = []
        current: list[dict] = []
        for msg in msgs:
            current.append(msg)
            if msg.get("role") == "assistant":
                pairs.append(current)
                current = []
        if current:
            pairs.append(current)
        return pairs

    def _pair_tokens(self, pair: list[dict]) -> int:
        lines = [self._render(m) for m in pair]
        return self._est("\n".join(ln for ln in lines if ln is not None))

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> str:
        system_block = f"[System]\n{SYSTEM_PROMPT}\n"
        current_block = f"\n[Current turn]\nUser: {self._current_text}\nAssistant:"

        budget = (
            self._token_budget
            - self._est(system_block)
            - self._est(current_block)
            - 512  # headroom for the response
        )

        pairs = self._group_pairs(self._history)
        selected = self._select_pairs(pairs, budget)

        hist_lines: list[str] = []
        for pair in selected:
            for msg in pair:
                line = self._render(msg)
                if line is not None:
                    hist_lines.append(line)

        history_block = ""
        if hist_lines:
            history_block = "\n[Conversation so far]\n" + "\n".join(hist_lines) + "\n"

        return system_block + history_block + current_block

    def _select_pairs(self, pairs: list[list[dict]], budget: int) -> list[list[dict]]:
        if not pairs:
            return []

        keep_n = min(self._min_turns_keep, len(pairs))
        must_keep = pairs[-keep_n:]
        optional = pairs[: len(pairs) - keep_n]

        must_tokens = sum(self._pair_tokens(p) for p in must_keep)
        remaining = budget - must_tokens

        added: list[list[dict]] = []
        for pair in reversed(optional):
            cost = self._pair_tokens(pair)
            if cost <= remaining:
                added.insert(0, pair)
                remaining -= cost
            else:
                break  # oldest pairs dropped first; stop once we can't fit

        return added + must_keep
