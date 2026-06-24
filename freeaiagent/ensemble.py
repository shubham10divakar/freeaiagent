"""Ensemble inference.

Send the same prompt to several models in parallel, then pick the best reply
with a judge step. Catches per-model blind spots at the cost of N× tokens and
~max-of-N latency (the calls run concurrently). Best for high-stakes one-shot
answers; less useful for chatty back-and-forth.

All ensemble models run through the *active* backend, so they must be models
that backend can serve (e.g. several Groq models). The judge strategies, in
order of preference: ``llm_judge`` (ask a small model to pick), ``longest``
(longest, repetition-penalised answer), ``majority`` (most common answer).
"""
import asyncio
import re
from typing import List, Optional, Tuple

Vote = dict
Candidate = Tuple[str, str]  # (model, response)


async def run(
    backend,
    messages: List[dict],
    models: List[str],
    *,
    judge_model: Optional[str] = None,
    strategy: str = "llm_judge",
) -> Tuple[str, str, List[Vote]]:
    """Fan ``messages`` out to ``models``; return ``(winner, model, votes)``.

    ``votes`` lists every model's response (or its error). Models that error are
    dropped from judging. Raises ``RuntimeError`` only if *every* model failed.
    """
    results = await asyncio.gather(
        *(backend.chat(messages, m) for m in models), return_exceptions=True
    )
    votes: List[Vote] = []
    valid: List[Candidate] = []
    for model, res in zip(models, results):
        if isinstance(res, Exception):
            votes.append({"model": model, "response": None, "error": str(res)})
        else:
            votes.append({"model": model, "response": res})
            valid.append((model, res))

    if not valid:
        raise RuntimeError("ensemble: all models failed")
    if len(valid) == 1:
        return valid[0][1], valid[0][0], votes

    winner_model, winner = await _judge(backend, valid, strategy, judge_model)
    return winner, winner_model, votes


async def _judge(backend, valid: List[Candidate], strategy: str,
                 judge_model: Optional[str]) -> Candidate:
    if strategy == "majority":
        return _majority(valid)
    if strategy == "longest":
        return _longest(valid)
    # default: LLM-as-judge, with a heuristic fallback if the judge call fails
    try:
        return await _llm_judge(backend, valid, judge_model)
    except Exception:
        return _longest(valid)


def _quality_score(text: str) -> float:
    """Length, penalised for repetition (unique-word ratio). Higher is better."""
    words = text.split()
    if not words:
        return 0.0
    ratio = len(set(w.lower() for w in words)) / len(words)
    return len(text) * ratio


def _longest(valid: List[Candidate]) -> Candidate:
    return max(valid, key=lambda c: _quality_score(c[1]))


def _majority(valid: List[Candidate]) -> Candidate:
    counts: dict = {}
    for model, resp in valid:
        key = " ".join(resp.lower().split())
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts, key=counts.get)
    # Return the first candidate whose normalised text is the most common one.
    for model, resp in valid:
        if " ".join(resp.lower().split()) == best_key:
            return model, resp
    return valid[0]


async def _llm_judge(backend, valid: List[Candidate],
                     judge_model: Optional[str]) -> Candidate:
    listing = "\n\n".join(f"[{i + 1}] {resp}" for i, (_m, resp) in enumerate(valid))
    prompt = [
        {"role": "system",
         "content": "You are a strict judge of answer quality. Choose the single "
                    "most accurate and complete answer."},
        {"role": "user",
         "content": f"{len(valid)} candidate answers follow. Reply with ONLY the "
                    f"number of the best one.\n\n{listing}"},
    ]
    raw = await backend.chat(prompt, judge_model or valid[0][0])
    match = re.search(r"\d+", raw or "")
    idx = (int(match.group()) - 1) if match else 0
    if not (0 <= idx < len(valid)):
        idx = 0
    return valid[idx]


def resolve_models(req_ensemble, config: dict) -> List[str]:
    """Resolve the request's ``ensemble`` field + config into a model list.

    - a non-empty list  → those models
    - ``True``          → config ``ensemble.models``
    - ``None`` + config ``ensemble.enabled`` → config ``ensemble.models``
    - otherwise         → ``[]`` (no ensemble)
    """
    ecfg = config.get("ensemble", {}) or {}
    if isinstance(req_ensemble, list):
        return req_ensemble
    if req_ensemble is True:
        return ecfg.get("models", [])
    if req_ensemble is None and ecfg.get("enabled"):
        return ecfg.get("models", [])
    return []
