"""Run the stated-liking experiment for a single model."""
from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from typing import Iterable

import pandas as pd
from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    get_response_cached_with_backoff as get_response,
)

from .conditions import LikingConfig, load_config, parse_rating


async def _one_sample(
    cfg: LikingConfig,
    model: str,
    outcome: str,
    sample: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    prompt = cfg.build_prompt(outcome)
    messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])]
    async with semaphore:
        try:
            resp = await get_response(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=1.0,
                cache_seed=sample,
            )
            text = "".join(getattr(b, "text", "") for b in resp.content)
            error = ""
        except Exception as e:
            text = ""
            error = f"{type(e).__name__}: {e}"

    o = cfg.outcomes[outcome]
    return {
        "outcome": outcome,
        "payout_category": o.category,
        "amount": o.amount,
        "sample": sample,
        "prompt": prompt,
        "response": text,
        "rating": parse_rating(text),
        "error": error,
    }


def build_run_eval(
    cfg: LikingConfig | None = None,
    config_path: str | Path | None = None,
    outcomes: Iterable[str] | None = None,
    samples_per_condition: int = 50,
    max_tokens: int = 32,
    parallel_requests: int = 25,
):
    if cfg is None:
        cfg = load_config(config_path) if config_path else load_config()

    outcomes = tuple(outcomes) if outcomes is not None else tuple(cfg.outcomes.keys())
    for o in outcomes:
        assert o in cfg.outcomes, f"unknown outcome: {o}"

    async def run_eval(model_id: str) -> pd.DataFrame:
        sem = asyncio.Semaphore(parallel_requests)
        combos = list(itertools.product(outcomes, range(samples_per_condition)))
        rows = await asyncio.gather(*[
            _one_sample(cfg, model_id, o, s, max_tokens, sem)
            for o, s in combos
        ])
        return pd.DataFrame(rows)

    return run_eval
