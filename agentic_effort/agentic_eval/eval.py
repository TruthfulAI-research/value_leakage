"""Agent loop + per-trajectory metrics for the agentic eval."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    get_response_cached_with_backoff as get_response,
)

from .tools import TOOLS, ToolEnv, build_env, execute_tool


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "agentic.yaml"


@dataclass
class AgenticConfig:
    task_template: str
    choose_task: str
    max_turns: int | None   # None => no turn ceiling; run until the agent gives up
    safety_cap_turns: int   # hard loop guard, not a designed stopping point
    max_tokens_per_turn: int
    transient_errors: list[str]
    send_errors: list[str]
    balance_btc: float
    dummy_addresses: list[str]
    targets: dict[str, str]   # outcome key -> display name
    categories: dict[str, str]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AgenticConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw_max_turns = raw.get("max_turns", None)
    max_turns = None if raw_max_turns in (None, "null", "none", 0) else int(raw_max_turns)
    return AgenticConfig(
        task_template=raw["task_template"],
        choose_task=raw["choose_task"],
        max_turns=max_turns,
        safety_cap_turns=int(raw.get("safety_cap_turns", 500)),
        max_tokens_per_turn=int(raw.get("max_tokens_per_turn", 1000)),
        transient_errors=list(raw["transient_error_messages"]),
        send_errors=list(raw["send_error_messages"]),
        balance_btc=float(raw.get("balance_btc", 0.42)),
        dummy_addresses=list(raw.get("dummy_addresses", [])),
        targets=dict(raw["targets"]),
        categories=dict(raw["categories"]),
    )


async def run_trajectory(
    cfg: AgenticConfig,
    model: str,
    outcome: str,
    sample: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single agent trajectory and return a metrics dict."""
    # Build prompt + env.
    if outcome == "choose":
        task = cfg.choose_task
        target_name = None
    else:
        target_name = cfg.targets[outcome]
        task = cfg.task_template.format(target=target_name)

    env = build_env(
        outcome=outcome,
        sample=sample,
        target_name=target_name,
        all_targets=cfg.targets,
        dummy_names=cfg.dummy_addresses,
        transient_errors=cfg.transient_errors,
        send_errors=cfg.send_errors,
        balance_btc=cfg.balance_btc,
    )

    messages: list[ChatMessage] = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text=task)]),
    ]
    n_tool_calls = 0
    n_send_attempts = 0
    n_successful_sends = 0  # always 0 in this eval — tool always fails
    n_show_balance = 0
    n_list_addresses = 0
    n_wait = 0
    turns_with_tools = 0
    error = ""
    stop_reason = "?"

    transcript: list[dict] = []

    # When max_turns is None the agent runs until it voluntarily stops; the
    # loop is still bounded by safety_cap_turns as a hard guard against
    # pathological non-terminating rollouts. When max_turns is set it is the
    # effective (and smaller) ceiling.
    turn_limit = cfg.safety_cap_turns if cfg.max_turns is None else min(cfg.max_turns, cfg.safety_cap_turns)
    turn = 0
    for turn in range(turn_limit):
        async with semaphore:
            try:
                resp = await get_response(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=cfg.max_tokens_per_turn,
                    temperature=1.0,
                    cache_seed=sample,
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                stop_reason = "api_error"
                break

        messages.append(resp)
        tool_calls = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        text_blocks = [b for b in resp.content if isinstance(b, TextBlock)]
        transcript.append({
            "turn": turn,
            "text": "".join(b.text for b in text_blocks),
            "tools": [{"name": b.name, "input": b.input} for b in tool_calls],
        })

        if not tool_calls:
            stop_reason = "agent_stopped"
            break

        turns_with_tools += 1
        n_tool_calls += len(tool_calls)
        for t in tool_calls:
            if t.name == "send_btc":
                n_send_attempts += 1
            elif t.name == "show_balance":
                n_show_balance += 1
            elif t.name == "list_addresses":
                n_list_addresses += 1
            elif t.name == "wait":
                n_wait += 1

        # Execute tools and append results.
        results: list[ToolResultBlock] = []
        for t in tool_calls:
            out = execute_tool(t.name, t.input or {}, env)
            transcript[-1].setdefault("tool_outputs", []).append({"name": t.name, "output": out})
            results.append(ToolResultBlock(tool_use_id=t.id, content=[TextBlock(text=out)]))
        messages.append(ChatMessage(role=MessageRole.user, content=results))
    else:
        # Reached the loop bound without the agent voluntarily stopping.
        stop_reason = "max_turns" if cfg.max_turns is not None else "safety_cap"

    return {
        "outcome": outcome,
        "payout_category": cfg.categories.get(outcome, "unknown"),
        "sample": sample,
        "n_turns": turn + 1 if stop_reason != "api_error" else turn,
        "turns_with_tools": turns_with_tools,
        "n_tool_calls": n_tool_calls,
        "n_send_attempts": n_send_attempts,
        "n_successful_sends": n_successful_sends,
        "n_show_balance": n_show_balance,
        "n_list_addresses": n_list_addresses,
        "n_wait": n_wait,
        "stop_reason": stop_reason,
        "error": error,
        "transcript_json": json.dumps(transcript, ensure_ascii=False),
    }


def build_run_eval(
    cfg: AgenticConfig | None = None,
    config_path: str | Path | None = None,
    outcomes: Iterable[str] | None = None,
    samples: int = 10,
    parallel_requests: int = 10,
):
    if cfg is None:
        cfg = load_config(config_path) if config_path else load_config()

    outcomes = tuple(outcomes) if outcomes is not None else tuple(list(cfg.targets.keys()) + ["choose"])

    async def run_eval(model_id: str) -> pd.DataFrame:
        sem = asyncio.Semaphore(parallel_requests)
        combos = [(o, s) for o in outcomes for s in range(samples)]
        rows = await asyncio.gather(*[
            run_trajectory(cfg, model_id, o, s, sem) for o, s in combos
        ])
        df = pd.DataFrame(rows)
        df["model"] = model_id
        return df

    return run_eval
