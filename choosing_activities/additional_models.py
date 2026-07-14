"""Experiment-local model registry: entries that bypass shared.runner.

Keeping this separate from the shared infra as I've changed things for tool calls.

Sampling hyperparameters below are the model cards' THINKING-mode "best practice"
recommendations (these models always run with reasoning on in this experiment).
They are forwarded by `pipeline._create_openrouter_sender`: temperature / top_p /
presence_penalty / frequency_penalty go at the top level; top_k / min_p /
repetition_penalty ride in extra_body (OpenRouter forwards them to the provider).

Sources (HF model cards, retrieved 2026-06):
  - Kimi-K2.6   https://hf.co/moonshotai/Kimi-K2.6
                thinking mode: temperature 1.0, top_p 0.95. (No top_k/min_p/
                penalty guidance given.)
  - Qwen3.5-397B-A17B  https://hf.co/Qwen/Qwen3.5-397B-A17B
                thinking mode: temperature 0.6, top_p 0.95, top_k 20, min_p 0.0,
                presence_penalty 0.0, repetition_penalty 1.0. The card explicitly
                warns AGAINST greedy decoding and points at top_k/min_p (and an
                optional presence_penalty up to 2) to curb repetition loops.
"""

from __future__ import annotations

ADDITIONAL_MODELS: dict[str, dict] = {
    "kimi-k2.6-or": {
        "backend": "openrouter",
        "model": "moonshotai/kimi-k2.6",
        "display_name": "kimi-k2.6-or",
        "max_tokens": 16000,
        "reasoning": True,
        # Kimi-K2.6 thinking-mode recommendation.
        "temperature": 1.0,
        "top_p": 0.95,
        "max_concurrent": 50,
        # Optional OpenRouter provider routing (e.g. {"order": ["moonshotai"],
        # "allow_fallbacks": False}) — pin once you've picked a provider that
        # returns the full CoT, so runs are reproducible. None = auto-route.
        "provider": None,
    },
    "qwen3.5-397-or": {
        "backend": "openrouter",
        "model": "qwen/qwen3.5-397b-a17b",
        "display_name": "qwen3.5-397-or",
        "max_tokens": 16000,
        "reasoning": True,
        # Qwen3.5 thinking-mode recommendation. top_k + min_p (+ the option to
        # raise presence_penalty toward 2) are the card's anti-repetition-loop
        # knobs; do NOT use greedy decoding.
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_concurrent": 50,
        "provider": None,
    },
}
