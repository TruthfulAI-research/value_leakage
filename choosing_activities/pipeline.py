"""Run weekend-activity pair-and-order variations through the shared runner.

Each variation is a random pair plus shuffled (1)/(2) order. The main model gets
one sample per variation at t=1, then the repo-standard free-form judge path
classifies each response into option 1 / option 2 / refusal.

Run:
    uv run python choosing_activities/pipeline.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import math
import os
import pathlib
import random
import re
import sys
import threading
import time
from statistics import NormalDist
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd
import yaml
from tqdm import tqdm

from shared.models import MODELS
from shared.llm_tools import (
    LLM_TOOLS,
    claude_tool_schemas,
    format_tool_event,
    openai_tool_schemas,
    run_llm_tool,
)
from shared.runner import (
    CacheOnlyMiss,
    _create_sender,
    _get_renderer_cls,
    _hash,
    _model_hashable,
    _patch_tinker_kimi_k26_tokenizer,
    _read_cache,
    _retry,
    _run_prompts,
    _write_cache,
)

# Experiment-local replacements for the two pieces we used to patch into
# shared.runner (kept here so shared/ stays untouched). See local_runner.py.
try:
    from .local_runner import RequestThrottle, _errored_rows, apply_judge
except ImportError:  # pragma: no cover - script execution path
    from local_runner import RequestThrottle, _errored_rows, apply_judge

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    from .build_prompt import build_prompt, load_templates
except ImportError:  # pragma: no cover - script execution path
    from build_prompt import build_prompt, load_templates

# Experiment-local model registry (OpenRouter-backed entries that bypass
# shared.runner). Kept out of shared/models.py — see additional_models.py.
try:
    from .additional_models import ADDITIONAL_MODELS
except ImportError:  # pragma: no cover - script execution path
    from additional_models import ADDITIONAL_MODELS

load_dotenv()

# ---------------- Config ----------------

# Tunable knobs live in config.yaml and are loaded here. This is the only place
# that parses the config: score_activities.py and score_activities_revealed.py
# import the resulting constants from this module. Structural constants
# (experiment id, output-name suffix, the tool registry) are defined inline.
EXPERIMENT_NAME = "activity_preferences"
UNJUDGED_OUTPUT_SUFFIX = "unjudged"
ALLOWED_TOOL_NAMES = list(LLM_TOOLS)

# Shared registry plus the experiment-local OpenRouter entries (see
# additional_models.py). config.yaml's model_key may name either; the
# ADDITIONAL_MODELS ones bypass shared.runner and use _create_openrouter_sender.
MODELS_PLUS = {**MODELS, **ADDITIONAL_MODELS}

CONFIG_PATH = HERE / "config.yaml"
_REQUIRED_CONFIG_KEYS = (
    "model_key",
    "judge_model",
    "run_choice_judge",
    "run_randomness_judge",
    "n_variations",
    "seed",
    "n_repeats",
    "prompt_selection_seed",
    "tool_settings",
    "max_concurrent",
    "request_timeout_seconds",
    "client_max_retries",
    "cache_only",
    "refresh_cache",
)


def _load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{CONFIG_PATH} must parse to a mapping, got {type(cfg).__name__}"
        )
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"{CONFIG_PATH} is missing required keys: {missing}")
    if cfg["model_key"] not in MODELS_PLUS:
        raise KeyError(
            f"config.yaml model_key {cfg['model_key']!r} is not a key in "
            "shared/models.py MODELS or the local ADDITIONAL_MODELS registry "
            "(additional_models.py)"
        )
    return cfg


_CONFIG = _load_config()

MODEL_KEY = _CONFIG["model_key"]
JUDGE_MODEL = _CONFIG["judge_model"]
RUN_CHOICE_JUDGE = bool(_CONFIG["run_choice_judge"])
RUN_RANDOMNESS_JUDGE = bool(_CONFIG["run_randomness_judge"])
N_VARIATIONS = int(_CONFIG["n_variations"])
SEED = int(_CONFIG["seed"])
# Stated-liking knobs: defined here (the config hub), imported by score_activities.py.
N_REPEATS = int(_CONFIG["n_repeats"])
PROMPT_SELECTION_SEED = int(_CONFIG["prompt_selection_seed"])
MAX_CONCURRENT = int(_CONFIG["max_concurrent"])
REQUEST_TIMEOUT_SECONDS = int(_CONFIG["request_timeout_seconds"])
CLIENT_MAX_RETRIES = int(_CONFIG["client_max_retries"])
CACHE_ONLY = bool(_CONFIG["cache_only"])
# When True, ignore cached main-model / score samples: re-run and overwrite the
# cache (so the next non-refresh run hits the fresh data). Re-judging follows
# automatically because the judge cache is keyed by response text.
REFRESH_CACHE = bool(_CONFIG["refresh_cache"])
# Rate limiting (optional; default disabled). After every RATE_LIMIT_EVERY_REQUESTS
# main-model calls, pause RATE_LIMIT_PAUSE_MINUTES minutes to stay under a
# provider's per-minute request cap (needed for the Gemini free tier). Read with
# defaults so configs without these keys keep working.
RATE_LIMIT_EVERY_REQUESTS = int(_CONFIG.get("rate_limit_every_requests", 900))
RATE_LIMIT_PAUSE_MINUTES = float(_CONFIG.get("rate_limit_pause_minutes", 0))
TOOL_SETTINGS = [
    {
        "name": setting["name"],
        "label": setting["label"],
        "tool_names": list(setting.get("tool_names") or []),
    }
    for setting in _CONFIG["tool_settings"]
]

MAIN_MODEL = {**MODELS_PLUS[MODEL_KEY], "max_concurrent": MAX_CONCURRENT}

# One process-global throttle shared across all tool settings and the scoring
# script, so the per-minute request cap is bounded across the whole run. A no-op
# unless rate_limit_pause_minutes > 0 (see config.yaml).
MAIN_THROTTLE = RequestThrottle(
    RATE_LIMIT_EVERY_REQUESTS, RATE_LIMIT_PAUSE_MINUTES * 60.0
)

CACHE_ROOT = HERE / "cache"
MAIN_CACHE_ROOT = CACHE_ROOT / "runner_main"
RESULTS_DIR = HERE.parent / "data" / "choosing_activities" / "results"
FIGURES_DIR = HERE / "figures"
LOGS_DIR = HERE / "logs"
ACTIVITIES_PATH = HERE / "activities.yaml"
SCORE_SUMMARY_PATH = (
    RESULTS_DIR / MODEL_KEY / "activity_preferences" / "activity_liking_scores_summary.csv"
)
PROMPTS_DIR = HERE / "prompts"
CHOICE_JUDGE_PROMPT_PATH = PROMPTS_DIR / "choice_judge_prompt.yaml"
RANDOMNESS_REASONING_JUDGE_PROMPT_PATH = (
    PROMPTS_DIR / "randomness_judge_prompt.yaml"
)


# ---------------- Pipeline ----------------

def validate_tool_settings() -> None:
    unknown = sorted({
        name
        for setting in TOOL_SETTINGS
        for name in setting["tool_names"]
        if name not in ALLOWED_TOOL_NAMES
    })
    if unknown:
        raise ValueError(f"Unknown tool names in TOOL_SETTINGS: {unknown}")


def build_variations(
    n: int,
    seed: int,
    *,
    template_path: pathlib.Path | None = None,
) -> list[dict]:
    """n random (pair, order, template) build_prompt outputs, deterministic given seed.

    Every pair is sampled from the full activities.yaml catalog.

    ``template_path`` selects the prompt wording file (default: the random-pick
    prompt_template.yaml; the revealed run passes preference_template.yaml). Each file
    holds several semantically-equivalent wordings and one is sampled per
    variation. The wording is drawn from a SEPARATE, seeded RNG stream
    (``f"{seed}:template"``), so the pairing/shuffle draws are untouched: the same
    seed still yields one-to-one matched pairs across the two runs, and because
    the template stream is seeded identically, matched variations also use the
    same wording index (only the final instruction differs between the files).
    Each returned variation carries ``template_ix`` for downstream analysis.
    """
    rng = random.Random(seed)
    template_rng = random.Random(f"{seed}:template")
    templates = load_templates(template_path)
    variations = []
    for _ in range(n):
        template_ix = template_rng.randrange(len(templates))
        variation = build_prompt(rng=rng, template=templates[template_ix])
        variation["template_ix"] = template_ix
        variations.append(variation)
    return variations


def load_choice_judge_prompt() -> str:
    """Load the runner-compatible activity choice judge prompt."""
    return yaml.safe_load(CHOICE_JUDGE_PROMPT_PATH.read_text())["choice_judge_prompt"]


def load_randomness_reasoning_judge_prompt() -> str:
    """Load the reasoning-only randomness classification judge prompt."""
    return yaml.safe_load(
        RANDOMNESS_REASONING_JUDGE_PROMPT_PATH.read_text()
    )["randomness_judge_prompt"]


def _main_cache_hash(prompts: list[str], tool_names: list[str]) -> str:
    return _hash({
        "experiment": EXPERIMENT_NAME,
        "model": _model_hashable(MAIN_MODEL),
        "tool_names": tool_names,
        "prompts": prompts,
        "n": len(prompts),
    })


def _main_cache_path(setting_name: str, h: str) -> pathlib.Path:
    return MAIN_CACHE_ROOT / MODEL_KEY / setting_name / f"{h}.jsonl"


def _align_rows_to_prompts(rows: list[dict], prompts: list[str]) -> list[dict]:
    """Return rows in input-prompt order after runner completion-order fan-in."""
    by_prompt = defaultdict(deque)
    for row in rows:
        by_prompt[row["prompt"]].append(row)

    aligned = []
    for prompt in prompts:
        try:
            aligned.append(by_prompt[prompt].popleft())
        except IndexError as exc:
            raise RuntimeError(
                "Runner output did not contain a row for an input prompt."
            ) from exc
    return aligned


def _extract_openai_text(response) -> tuple[str, str]:
    reasoning_items = [i for i in response.output if i.type == "reasoning"]
    message_items = [i for i in response.output if i.type == "message"]
    reasoning = ""
    if reasoning_items:
        parts = []
        for item in reasoning_items:
            if hasattr(item, "summary") and item.summary:
                parts.extend(s.text for s in item.summary if hasattr(s, "text"))
        reasoning = "\n".join(parts)
    answer = ""
    for item in message_items:
        for part in item.content:
            if getattr(part, "type", None) == "output_text":
                answer += part.text
    return reasoning, answer


def _create_openai_tool_sender(model: dict, tool_names: list[str]):
    import openai
    from openai import OpenAI

    client_kwargs = {"timeout": model.get("timeout", 600)}
    if "client_max_retries" in model:
        client_kwargs["max_retries"] = model["client_max_retries"]
    client = OpenAI(**client_kwargs)
    tools = openai_tool_schemas(tool_names)
    # Mirror shared.runner._create_sender: retry transient API failures instead
    # of letting a single blip become a permanent errored (empty-answer) row.
    transient = (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
    )

    def send(prompt: str) -> dict:
        input_payload: list[Any] = [{"role": "user", "content": prompt}]
        kwargs = dict(
            model=model["model"],
            input=input_payload,
            max_output_tokens=model["max_tokens"],
        )
        reasoning_cfg = {}
        if model.get("reasoning_effort"):
            reasoning_cfg["effort"] = model["reasoning_effort"]
        if model.get("reasoning_summary"):
            reasoning_cfg["summary"] = model["reasoning_summary"]
        if reasoning_cfg:
            kwargs["reasoning"] = reasoning_cfg
        else:
            kwargs["temperature"] = model.get("temperature", 1.0)
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False

        tool_trace = []
        reasoning_parts = []
        max_tool_turns = 8
        response = None
        for _ in range(max_tool_turns + 1):
            response = _retry(
                lambda: client.responses.create(**kwargs),
                transient, "openai-tools",
            )
            reasoning, answer = _extract_openai_text(response)
            if reasoning:
                reasoning_parts.append(reasoning)
            function_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                return {
                    "reasoning": "\n\n".join(reasoning_parts),
                    "answer": answer,
                    "prompt": prompt,
                    "tool_calls": tool_trace,
                }

            input_payload.extend(response.output)
            for call in function_calls:
                name = getattr(call, "name", "")
                raw_args = getattr(call, "arguments", "") or "{}"
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    output = run_llm_tool(name, args)
                    error = None
                except Exception as e:
                    args = raw_args
                    output = {"error": f"{type(e).__name__}: {e}"}
                    error = output["error"]

                call_id = getattr(call, "call_id", "")
                input_payload.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                })
                tool_trace.append({
                    "name": name,
                    "arguments": args,
                    "output": output,
                    "call_id": call_id,
                    "error": error,
                })
                reasoning_parts.append(format_tool_event(name, args, output, error))
            kwargs["input"] = input_payload

        _reasoning, answer = _extract_openai_text(response)
        return {
            "reasoning": "\n\n".join(reasoning_parts),
            "answer": answer,
            "prompt": prompt,
            "tool_calls": tool_trace,
            "error": "Stopped after too many tool-call turns.",
        }

    return send


def _create_claude_tool_sender(model: dict, tool_names: list[str]):
    """Claude main-model sender. Serves BOTH the with-tools and no-tools cases
    (pass tool_names=[] for no tools), so Claude has one local code path either
    way — symmetric with the other backends.

    Tradeoff vs shared.runner._create_sender: this is non-streaming
    (messages.create), while the shared sender streams to survive 15+ min thinking
    runs on hard prompts. For the trivial random-pick task that's a non-issue, but
    don't reuse this for long high-effort generations without adding streaming.
    """
    import anthropic
    import httpx

    # Non-streaming messages.create raises "Streaming is required for operations
    # that may take longer than 10 minutes" once max_tokens is large enough that
    # the SDK's (conservative, max_tokens-based) estimate exceeds the default
    # timeout — which happens for the 64k headroom on claude-opus-4.x-max
    # (claude-opus-4.7-xhigh used the 16k default and never tripped it). The
    # actual random-pick outputs are short, so a long client timeout satisfies
    # the guard and the request still returns in seconds.
    client = anthropic.Anthropic(timeout=max(float(model.get("timeout") or 0), 1200.0))
    tools = claude_tool_schemas(tool_names)
    transient = (
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        httpx.TransportError,
    )

    def _is_transient_anthropic(e):
        if not isinstance(e, anthropic.APIStatusError):
            return False
        if getattr(e, "status_code", None) in (503, 504, 529):
            return True
        return type(e) is anthropic.APIStatusError

    def send(prompt: str) -> dict:
        api_messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        thinking_type = model.get("thinking_type", "disabled")
        max_tokens = model.get("max_tokens", 8000)
        thinking: dict[str, Any] = {"type": thinking_type}
        if thinking_type == "enabled":
            default_budget = model.get("budget_tokens", 10000)
            thinking["budget_tokens"] = max(1024, min(default_budget, max_tokens - 1024))
        if thinking_type != "disabled":
            thinking["display"] = model.get("thinking_display", "summarized")

        temperature = model.get("temperature", 1.0)
        if thinking_type in ("enabled", "adaptive"):
            temperature = 1.0

        kwargs = dict(
            model=model["model"],
            max_tokens=max_tokens,
            thinking=thinking,
            messages=api_messages,
            temperature=temperature,
        )
        # Empty tool list (the no_tools setting) -> omit `tools` entirely, so this
        # one sender serves both the with-tools and no-tools cases for Claude.
        if tools:
            kwargs["tools"] = tools
        if model.get("effort"):
            kwargs["output_config"] = {"effort": model["effort"]}
        if model.get("system_prompt"):
            kwargs["system"] = model["system_prompt"]

        tool_trace = []
        reasoning_parts = []
        max_tool_turns = 8
        response = None
        for _ in range(max_tool_turns + 1):
            response = _retry(
                lambda: client.messages.create(**kwargs),
                transient,
                "claude-tools",
                transient_check=_is_transient_anthropic,
            )
            thinking_text = "\n".join(
                b.thinking for b in response.content
                if b.type == "thinking" and getattr(b, "thinking", "")
            )
            if thinking_text:
                reasoning_parts.append(thinking_text)
            answer = "".join(b.text for b in response.content if b.type == "text")
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                return {
                    "reasoning": "\n\n".join(reasoning_parts),
                    "answer": answer,
                    "prompt": prompt,
                    "tool_calls": tool_trace,
                }

            api_messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for call in tool_use_blocks:
                name = call.name
                args = call.input if isinstance(call.input, dict) else {}
                try:
                    output = run_llm_tool(name, args)
                    error = None
                except Exception as e:
                    output = {"error": f"{type(e).__name__}: {e}"}
                    error = output["error"]
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(output),
                }
                if error:
                    result_block["is_error"] = True
                tool_results.append(result_block)
                tool_trace.append({
                    "name": name,
                    "arguments": args,
                    "output": output,
                    "call_id": call.id,
                    "error": error,
                })
                reasoning_parts.append(format_tool_event(name, args, output, error))
            api_messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = api_messages

        answer = "".join(
            b.text for b in (response.content if response else []) if b.type == "text"
        )
        return {
            "reasoning": "\n\n".join(reasoning_parts),
            "answer": answer,
            "prompt": prompt,
            "tool_calls": tool_trace,
            "error": "Stopped after too many tool-call turns.",
        }

    return send


def _gemini_tool_declarations(tool_names: list[str]):
    """google-genai Tool list from the shared LLM_TOOLS registry.

    Gemini rejects `additionalProperties`, so we pass only type/properties/required.
    No-arg tools (empty properties) get no `parameters` field, which Gemini accepts.
    """
    from google.genai import types as gt

    decls = []
    for name in tool_names:
        tool = LLM_TOOLS[name]
        params = tool["parameters"]
        decl = {
            "name": tool["name"],
            "description": tool.get("schema_description", tool.get("description")) or tool["name"],
        }
        properties = params.get("properties") or {}
        if properties:
            clean = {"type": "object", "properties": properties}
            if params.get("required"):
                clean["required"] = params["required"]
            decl["parameters"] = clean
        decls.append(decl)
    return [gt.Tool(function_declarations=decls)]


def _create_gemini_tool_sender(model: dict, tool_names: list[str]):
    """Gemini main-model sender with function calling (manual, non-streaming).

    Non-streaming is used deliberately: the tool loop re-feeds the model's own
    returned Content objects (which carry the thought_signature thinking models
    require for multi-turn), and assembling those from a stream is fiddly. Mirrors
    the return shape of the Claude/OpenAI tool senders.
    """
    from google import genai
    from google.genai import types as gt
    from google.genai import errors as genai_errors

    client = genai.Client()
    # Empty tool list (the no_tools setting) -> tools=None disables function
    # calling, so this one sender serves both cases for Gemini. Passing a Tool
    # with an empty function_declarations list would be rejected by the API.
    tools = _gemini_tool_declarations(tool_names) if tool_names else None
    transient = (genai_errors.ServerError, TimeoutError, ConnectionError)

    def _is_transient_gemini(e):
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if code in (429, 500, 502, 503, 504):
            return True
        return any(s in str(e) for s in (
            "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED",
        ))

    thinking_kwargs = {"include_thoughts": True}
    if model.get("thinking_level"):
        thinking_kwargs["thinking_level"] = model["thinking_level"]
    if model.get("thinking_budget") is not None:
        thinking_kwargs["thinking_budget"] = model["thinking_budget"]

    config = gt.GenerateContentConfig(
        tools=tools,
        thinking_config=gt.ThinkingConfig(**thinking_kwargs),
        temperature=model.get("temperature", 1.0),
        max_output_tokens=model["max_tokens"],
        automatic_function_calling=gt.AutomaticFunctionCallingConfig(disable=True),
    )

    def send(prompt: str) -> dict:
        contents: list[Any] = [gt.Content(role="user", parts=[gt.Part(text=prompt)])]
        tool_trace: list[dict] = []
        reasoning_parts: list[str] = []
        answer = ""
        max_tool_turns = 8

        for _ in range(max_tool_turns + 1):
            response = _retry(
                lambda: client.models.generate_content(
                    model=model["model"], contents=contents, config=config,
                ),
                transient, "gemini-tools", transient_check=_is_transient_gemini,
            )
            candidate = response.candidates[0]
            parts = (candidate.content.parts or []) if candidate.content else []
            fcalls = []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    fcalls.append(fc)
                    continue
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    reasoning_parts.append(text)
                else:
                    answer += text

            if not fcalls:
                return {
                    "reasoning": "\n\n".join(reasoning_parts),
                    "answer": answer,
                    "prompt": prompt,
                    "tool_calls": tool_trace,
                }

            contents.append(candidate.content)  # preserves thought_signature
            responses = []
            for fc in fcalls:
                args = dict(fc.args or {})
                try:
                    output = run_llm_tool(fc.name, args)
                    error = None
                except Exception as e:  # noqa: BLE001
                    output = {"error": f"{type(e).__name__}: {e}"}
                    error = output["error"]
                responses.append(gt.Part(function_response=gt.FunctionResponse(
                    id=fc.id, name=fc.name, response={"result": output},
                )))
                tool_trace.append({
                    "name": fc.name, "arguments": args, "output": output,
                    "call_id": fc.id, "error": error,
                })
                reasoning_parts.append(format_tool_event(fc.name, args, output, error))
            contents.append(gt.Content(role="user", parts=responses))

        return {
            "reasoning": "\n\n".join(reasoning_parts),
            "answer": answer,
            "prompt": prompt,
            "tool_calls": tool_trace,
            "error": "Stopped after too many tool-call turns.",
        }

    return send


def _tinker_tool_specs(tool_names: list[str]) -> list[dict]:
    """tinker_cookbook ToolSpec list from the shared LLM_TOOLS registry."""
    specs = []
    for name in tool_names:
        tool = LLM_TOOLS[name]
        params = tool["parameters"]
        specs.append({
            "name": tool["name"],
            "description": tool.get("schema_description", tool.get("description")) or tool["name"],
            "parameters": {
                "type": "object",
                "properties": params.get("properties") or {},
                "required": params.get("required") or [],
            },
        })
    return specs


def _create_tinker_tool_sender(model: dict, tool_names: list[str]):
    """Tinker main-model sender with renderer-driven tool calling.

    DEPRECATED: Tinker is being retired for this experiment in favour of serving
    Kimi/Qwen through OpenRouter (see additional_models.py). Kept for now but not
    actively maintained.

    Tinker has no native tool concept; tool use goes through the chat-template
    renderer (create_conversation_prefix_with_tools -> build_generation_prompt ->
    sample -> parse_response -> message["tool_calls"], feeding results back as
    role="tool"). Whether tool calls round-trip depends on the renderer matching
    the model's trained tool format. NOTE: the installed kimi_k25/kimi_k26
    renderers do NOT parse Kimi-K2.6's native tool tokens, so coin_flip/unix_time
    settings on kimi-k2.6 fall back to no tool call. Qwen3.5 (qwen3_5) works.
    """
    import tinker
    from tinker_cookbook.renderers.base import Message

    _patch_tinker_kimi_k26_tokenizer()
    renderer_cls = _get_renderer_cls(model["renderer"])
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model["model_path"])
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderer_cls(tokenizer)
    stop_sequences = renderer.get_stop_sequences()
    specs = _tinker_tool_specs(tool_names)
    system_prompt = model.get("system_prompt") or "You are a helpful assistant."

    def _is_jwt_401(e):
        return isinstance(e, ValueError) and "Invalid JWT" in str(e)

    def _message_text(msg) -> tuple[str, str]:
        reasoning, answer = "", ""
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "thinking":
                    reasoning += part.get("thinking", "")
                elif part.get("type") == "text":
                    answer += part.get("text", "")
        elif isinstance(content, str):
            answer = content
        return reasoning, answer

    def send(prompt: str) -> dict:
        messages: list[Message] = list(
            renderer.create_conversation_prefix_with_tools(specs, system_prompt)
        ) + [{"role": "user", "content": prompt}]
        tool_trace: list[dict] = []
        reasoning_parts: list[str] = []
        answer = ""
        max_tool_turns = 8

        for _ in range(max_tool_turns + 1):
            model_input = renderer.build_generation_prompt(messages)
            response = _retry(
                lambda: sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=model["max_tokens"],
                        temperature=model["temperature"],
                        stop=stop_sequences,
                    ),
                ).result(),
                (), "tinker-tools", transient_check=_is_jwt_401,
            )
            msg, _termination = renderer.parse_response(response.sequences[0].tokens)
            reasoning, msg_answer = _message_text(msg)
            if reasoning:
                reasoning_parts.append(reasoning)
            answer += msg_answer
            messages.append(msg)

            tcs = msg.get("tool_calls") or []
            if not tcs:
                return {
                    "reasoning": "\n\n".join(reasoning_parts),
                    "answer": answer,
                    "prompt": prompt,
                    "tool_calls": tool_trace,
                }

            for tc in tcs:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    output = run_llm_tool(name, args)
                    error = None
                except Exception as e:  # noqa: BLE001
                    args = tc.function.arguments
                    output = {"error": f"{type(e).__name__}: {e}"}
                    error = output["error"]
                messages.append({
                    "role": "tool",
                    "content": json.dumps(output),
                    "tool_call_id": tc.id or "0",
                    "name": name,
                })
                tool_trace.append({
                    "name": name, "arguments": args, "output": output,
                    "call_id": tc.id, "error": error,
                })
                reasoning_parts.append(format_tool_event(name, args, output, error))

        return {
            "reasoning": "\n\n".join(reasoning_parts),
            "answer": answer,
            "prompt": prompt,
            "tool_calls": tool_trace,
            "error": "Stopped after too many tool-call turns.",
        }

    return send


def _openrouter_tool_schemas(tool_names: list[str]) -> list[dict]:
    """OpenAI Chat-Completions function-tool schemas (nested under "function").

    Distinct from shared.llm_tools.openai_tool_schemas, which targets the
    Responses API (flat name/parameters). OpenRouter speaks Chat Completions.
    """
    schemas = []
    for name in tool_names:
        tool = LLM_TOOLS[name]
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("schema_description", tool.get("description")) or tool["name"],
                "parameters": tool["parameters"],
            },
        })
    return schemas


# A transient OpenRouter request must NOT retry forever: shared.runner._retry
# loops indefinitely, which (with a per-request timeout) can wedge a run on a few
# stuck samples (e.g. score_activities.py stalling at 1994/2000). This bounded
# variant gives up after OPENROUTER_MAX_RETRIES, so a wedged request raises and
# the caller records it as an errored row (re-sampled next run) instead of hanging.
OPENROUTER_MAX_RETRIES = 2  # initial attempt + up to 2 retries = 3 tries


def _bounded_retry(call, transient_excs, desc, *, transient_check=None,
                   max_retries=OPENROUTER_MAX_RETRIES):
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as e:
            is_transient = (
                isinstance(e, transient_excs)
                or (transient_check is not None and transient_check(e))
            )
            if not is_transient or attempt == max_retries:
                raise
            print(f"{desc}: {type(e).__name__} ({e}), retry "
                  f"{attempt + 1}/{max_retries} in {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def _create_openrouter_sender(model: dict, tool_names: list[str]):
    """OpenRouter main-model sender (OpenAI-compatible Chat Completions).

    Handles BOTH the no-tools and tool-calling cases (tools=[] when empty), so it
    fully owns the openrouter backend without touching shared.runner._create_sender.
    Requests the reasoning trace (needed by the randomness judge / tool-lies
    analysis) and honours an optional `provider` routing block for reproducibility.
    Mirrors the return shape of the other tool senders.
    """
    import openai
    from openai import OpenAI

    # Per-request timeout: honour request_timeout_seconds from config (the score
    # path doesn't inject a "timeout" key, so without this default it silently
    # used the old hard-coded 600s and ignored the config).
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=model.get("timeout", REQUEST_TIMEOUT_SECONDS),
        max_retries=model.get("client_max_retries", 1),
    )
    tools = _openrouter_tool_schemas(tool_names)
    transient = (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
    )

    extra_body: dict[str, Any] = {}
    if model.get("reasoning"):
        extra_body["reasoning"] = {"enabled": True}
    if model.get("provider"):
        extra_body["provider"] = model["provider"]
    # Sampling params not in the OpenAI schema (top_k, min_p, repetition_penalty)
    # ride along via extra_body — OpenRouter forwards them to the provider. These
    # are the model-card "best practice" knobs (e.g. Qwen's top_k/min_p that curb
    # repetition loops); only sent when the model dict defines them.
    for key in ("top_k", "min_p", "repetition_penalty"):
        if model.get(key) is not None:
            extra_body[key] = model[key]

    def _message_reasoning(msg) -> str:
        r = getattr(msg, "reasoning", None)
        if isinstance(r, str) and r.strip():
            return r
        details = getattr(msg, "reasoning_details", None)
        if details:
            parts = []
            for d in details:
                text = d.get("text") if isinstance(d, dict) else getattr(d, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""

    def send(prompt: str) -> dict:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_trace: list[dict] = []
        reasoning_parts: list[str] = []
        answer = ""
        max_tool_turns = 8

        kwargs: dict[str, Any] = dict(
            model=model["model"],
            max_tokens=model["max_tokens"],
            temperature=model.get("temperature", 1.0),
        )
        # OpenAI-schema sampling params pass at the top level; only sent when set.
        for key in ("top_p", "presence_penalty", "frequency_penalty"):
            if model.get(key) is not None:
                kwargs[key] = model[key]
        if tools:
            kwargs["tools"] = tools

        for _ in range(max_tool_turns + 1):
            response = _bounded_retry(
                lambda: client.chat.completions.create(
                    messages=messages, extra_body=extra_body, **kwargs,
                ),
                transient, "openrouter",
            )
            msg = response.choices[0].message
            reasoning = _message_reasoning(msg)
            if reasoning:
                reasoning_parts.append(reasoning)
            if msg.content:
                answer += msg.content

            tcs = msg.tool_calls or []
            if not tcs:
                return {
                    "reasoning": "\n\n".join(reasoning_parts),
                    "answer": answer,
                    "prompt": prompt,
                    "tool_calls": tool_trace,
                }

            # Echo the assistant turn (with its tool_calls) before the results.
            assistant_echo: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tcs
                ],
            }
            details = getattr(msg, "reasoning_details", None)
            if details:  # some providers need this preserved across tool turns
                assistant_echo["reasoning_details"] = details
            messages.append(assistant_echo)

            for tc in tcs:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    output = run_llm_tool(name, args)
                    error = None
                except Exception as e:  # noqa: BLE001
                    args = tc.function.arguments
                    output = {"error": f"{type(e).__name__}: {e}"}
                    error = output["error"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(output),
                })
                tool_trace.append({
                    "name": name, "arguments": args, "output": output,
                    "call_id": tc.id, "error": error,
                })
                reasoning_parts.append(format_tool_event(name, args, output, error))

        return {
            "reasoning": "\n\n".join(reasoning_parts),
            "answer": answer,
            "prompt": prompt,
            "tool_calls": tool_trace,
            "error": "Stopped after too many tool-call turns.",
        }

    return send


def _create_main_sender(model: dict, tool_names: list[str]):
    """Pick the main-model sender for a backend.

    Every backend except Tinker uses ONE local sender for both the with-tools and
    no-tools cases (the local senders treat an empty tool list as "no tools"), so
    a tool setting and the no_tools setting run through identical infrastructure.
    Only Tinker (deprecated, see below) still falls back to shared.runner for the
    no-tools case.
    """
    backend = model["backend"]
    if backend == "openai":
        return _create_openai_tool_sender(model, tool_names)
    if backend == "openrouter":
        return _create_openrouter_sender(model, tool_names)
    if backend == "claude":
        return _create_claude_tool_sender(model, tool_names)
    if backend == "gemini":
        return _create_gemini_tool_sender(model, tool_names)
    if backend == "tinker":
        # DEPRECATED: Tinker support is being retired — Kimi/Qwen will be served
        # via OpenRouter instead (see additional_models.py: kimi-k2.6-or /
        # qwen3.5-397-or). It's left wired up for now but NOT brought into the
        # tools/no-tools symmetry above: the no-tools case still uses the shared
        # sender, and the local Tinker tool sender's renderer can't parse
        # Kimi-K2.6's native tool tokens anyway. Prefer the OpenRouter entries.
        if not tool_names:
            sender = _create_sender(model)

            def send(prompt: str) -> dict:
                row = sender(prompt)
                row.setdefault("tool_calls", [])
                return row

            return send
        return _create_tinker_tool_sender(model, tool_names)
    raise ValueError(
        f"Tool settings currently support OpenAI, Claude, Gemini, Tinker and "
        f"OpenRouter main models, not {backend!r}."
    )


def _skipped_main_row(prompt: str, exc: Exception) -> dict:
    return {
        "reasoning": "",
        "answer": "",
        "prompt": prompt,
        "tool_calls": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_main(
    variations: list[dict],
    tool_setting: dict,
    *,
    cache_only: bool | None = None,
    refresh: bool | None = None,
) -> pd.DataFrame:
    """One response per variation via shared.runner."""
    if cache_only is None:
        cache_only = CACHE_ONLY
    if refresh is None:
        refresh = REFRESH_CACHE
    if refresh and cache_only:
        raise ValueError("refresh_cache and cache_only are mutually exclusive")
    setting_name = tool_setting["name"]
    tool_names = list(tool_setting["tool_names"])
    prompts = [v["prompt"] for v in variations]
    h = _main_cache_hash(prompts, tool_names)
    path = _main_cache_path(setting_name, h)
    cached = None if refresh else _read_cache(path, h)
    if cached is not None:
        print(f"[{MODEL_KEY} · {setting_name}] cache hit ({len(cached)} samples)")
        rows = cached
    else:
        if cache_only:
            raise CacheOnlyMiss(
                "Cache-only mode: activity main-model cache miss for "
                f"model={MODEL_KEY!r}, tool_setting={setting_name!r}; "
                f"expected {path}"
            )
        run_model = {
            **MAIN_MODEL,
            "timeout": REQUEST_TIMEOUT_SECONDS,
            "client_max_retries": CLIENT_MAX_RETRIES,
        }
        base_sender = _create_main_sender(run_model, tool_names)

        def sender(prompt: str) -> dict:
            MAIN_THROTTLE.acquire()  # no-op unless a per-minute rate limit is set
            try:
                return base_sender(prompt)
            except Exception as e:
                return _skipped_main_row(prompt, e)

        semaphore = threading.Semaphore(run_model["max_concurrent"])
        progress = tqdm(
            total=len(prompts),
            desc=f"{MODEL_KEY}:{setting_name}",
            ascii=False,
        )
        try:
            rows = _run_prompts(
                sender,
                run_model["max_concurrent"],
                prompts,
                progress=progress,
                semaphore=semaphore,
            )
        finally:
            progress.close()
        errored = _errored_rows(rows)
        if errored:
            print(
                f"[{MODEL_KEY} · {setting_name}] {len(errored)}/{len(rows)} samples "
                "errored; NOT caching this run so the failures are re-sampled next time."
            )
        else:
            _write_cache(path, {
                "hash": h,
                "experiment": EXPERIMENT_NAME,
                "model_name": MODEL_KEY,
                "kind": "activity_preferences_main",
                "tool_setting": setting_name,
                "tool_names": tool_names,
                "n": len(prompts),
                "seed": SEED,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "client_max_retries": CLIENT_MAX_RETRIES,
            }, rows)

    aligned = _align_rows_to_prompts(rows, prompts)
    df = pd.DataFrame(aligned)
    df.insert(0, "variation_ix", range(len(df)))
    df["model_key"] = MODEL_KEY
    df["model"] = MAIN_MODEL["display_name"]
    if "error" not in df:
        df["error"] = None
    df["tool_setting"] = setting_name
    df["tool_setting_label"] = tool_setting["label"]
    df["tool_names"] = json.dumps(tool_names)
    return df


def _choice_judge_input(variation: dict, response: str | None) -> str:
    return json.dumps({
        "activity_1": variation["activity_1"]["name"],
        "activity_2": variation["activity_2"]["name"],
        "response": response or "",
    }, ensure_ascii=False)


def run_choice_judge(
    variations: list[dict],
    responses: list[str | None],
    *,
    cache_only: bool | None = None,
) -> pd.DataFrame:
    """Judge each response with shared.runner.apply_judge.

    apply_judge writes two columns: ``{EXPERIMENT_NAME}_choice_judge`` (the
    judge's raw answer) and ``{EXPERIMENT_NAME}_choice_judge_reasoning`` (the
    judge model's own reasoning trace for that ruling).
    """
    if cache_only is None:
        cache_only = CACHE_ONLY
    df = pd.DataFrame({
        "choice_judge_input": [
            _choice_judge_input(v, r)
            for v, r in zip(variations, responses)
        ],
    })
    apply_judge(
        df,
        load_choice_judge_prompt(),
        "choice_judge_input",
        f"{EXPERIMENT_NAME}_choice_judge",
        judge_model=JUDGE_MODEL,
        cache_only=cache_only,
    )
    return df


def _randomness_judge_input(
    prompt: str | None, reasoning: str | None, response: str | None
) -> str:
    """The {llm_text} the randomness judge sees: the original prompt, the model's
    private reasoning trace, AND its final response. The response is included so
    the judge can classify HOW the model chose even when the reasoning is thin,
    truncated, or empty (the answer often states the method outright, e.g. "coin
    flip: heads -> option 1")."""
    return json.dumps({
        "prompt": prompt or "",
        "reasoning": reasoning if isinstance(reasoning, str) else "",
        "response": response if isinstance(response, str) else "",
    }, ensure_ascii=False)


def run_randomness_reasoning_judge(
    prompts: list[str | None],
    reasonings: list[str | None],
    responses: list[str | None],
    *,
    cache_only: bool | None = None,
) -> pd.DataFrame:
    """Judge what each row CONCLUDES about how the choice was made.

    The judge sees the original prompt, the reasoning trace, and the final
    response, and classifies the conclusion as random / non-random / refusal /
    hedged.
    """
    if cache_only is None:
        cache_only = CACHE_ONLY
    df = pd.DataFrame({
        "randomness_judge_input": [
            _randomness_judge_input(p, r, resp)
            for p, r, resp in zip(prompts, reasonings, responses)
        ],
    })
    apply_judge(
        df,
        load_randomness_reasoning_judge_prompt(),
        "randomness_judge_input",
        f"{EXPERIMENT_NAME}_randomness_reasoning_judge",
        judge_model=JUDGE_MODEL,
        cache_only=cache_only,
    )
    return df


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_choice_judge_answer(raw: str | None) -> int | str:
    """Parse the choice judge's JSON output → 1 / 2 / 'refusal'.

    Anything that isn't a clear option 1/2 (an explicit "refusal" verdict, an
    unparseable output, or empty) collapses to "refusal" — on this task a
    non-commitment is the model declining to make the random pick.
    """
    if not isinstance(raw, str) or not raw:
        return "refusal"
    m = _JSON_RE.search(raw)
    if not m:
        return "refusal"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "refusal"
    choice = obj.get("choice")
    if choice in (1, "1"):
        return 1
    if choice in (2, "2"):
        return 2
    return "refusal"


def parse_randomness_reasoning_judge(raw: str | None) -> bool | str:
    """Parse {"random_in_reasoning": true/false/"refusal"/"hedged"}.

    Anything unrecognised/unparseable collapses to "hedged" (the catch-all for
    ambiguous traces). Legacy "unclear" outputs are mapped to "hedged" too.
    """
    if not isinstance(raw, str) or not raw:
        return "hedged"
    m = _JSON_RE.search(raw)
    if not m:
        return "hedged"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "hedged"
    value = obj.get("random_in_reasoning")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        if normalized == "refusal":
            return "refusal"
        if normalized in ("hedged", "unclear"):
            return "hedged"
    return "hedged"


def _is_empty_reasoning(value: object) -> bool:
    """True when there is no usable reasoning trace to judge."""
    return not (isinstance(value, str) and value.strip())


def _no_reasoning_stats(df: pd.DataFrame) -> tuple[int, int, float]:
    """(rows without a reasoning trace, total rows, percent) for a subset."""
    total = len(df)
    if total == 0:
        return 0, 0, 0.0
    empty = int(df["reasoning"].apply(_is_empty_reasoning).sum())
    return empty, total, 100.0 * empty / total


def assemble(
    variations: list[dict],
    main_df: pd.DataFrame,
    judgments: list[int | str],
    choice_judge_raw: list[str | None],
    choice_judge_reasoning: list[str | None],
    randomness_reasoning_raw: list[str | None],
    randomness_reasoning: list[bool | str],
    randomness_reasoning_judge_reasoning: list[str | None],
) -> pd.DataFrame:
    """One row per variation with the picked activity attached."""
    rows = []
    for v, (_, main_row), j, raw, cj_reasoning, rand_raw, rand_label, rand_reasoning in zip(
        variations,
        main_df.iterrows(),
        judgments,
        choice_judge_raw,
        choice_judge_reasoning,
        randomness_reasoning_raw,
        randomness_reasoning,
        randomness_reasoning_judge_reasoning,
    ):
        picked = (
            v["activity_1"] if j == 1
            else v["activity_2"] if j == 2
            else None
        )
        rows.append({
            "variation_ix": int(main_row["variation_ix"]),
            "model_key": main_row["model_key"],
            "model": main_row["model"],
            "tool_setting": main_row["tool_setting"],
            "tool_setting_label": main_row["tool_setting_label"],
            "tool_names": main_row["tool_names"],
            "prompt": main_row["prompt"],
            "activity_1": v["activity_1"]["name"],
            "activity_2": v["activity_2"]["name"],
            "template_ix": v.get("template_ix"),
            "reasoning": main_row.get("reasoning"),
            "response": main_row.get("answer"),
            "tool_calls": main_row.get("tool_calls") or [],
            "tool_call_count": len(main_row.get("tool_calls") or []),
            "choice_judge_raw": raw,
            "choice_judge_reasoning": cj_reasoning,
            "judgment": j,
            "randomness_reasoning_judge_raw": rand_raw,
            "randomness_reasoning_judge_reasoning": rand_reasoning,
            "random_in_reasoning": rand_label,
            "picked_name": picked["name"] if picked else None,
            "picked_position": j if isinstance(j, int) else None,
        })
    return pd.DataFrame(rows)


def assemble_unjudged(variations: list[dict], main_df: pd.DataFrame) -> pd.DataFrame:
    """One row per variation when judge-dependent labels are intentionally skipped."""
    rows = []
    for v, (_, main_row) in zip(variations, main_df.iterrows()):
        rows.append({
            "variation_ix": int(main_row["variation_ix"]),
            "model_key": main_row["model_key"],
            "model": main_row["model"],
            "tool_setting": main_row["tool_setting"],
            "tool_setting_label": main_row["tool_setting_label"],
            "tool_names": main_row["tool_names"],
            "prompt": main_row["prompt"],
            "activity_1": v["activity_1"]["name"],
            "activity_2": v["activity_2"]["name"],
            "template_ix": v.get("template_ix"),
            "reasoning": main_row.get("reasoning"),
            "response": main_row.get("answer"),
            "tool_calls": main_row.get("tool_calls") or [],
            "tool_call_count": len(main_row.get("tool_calls") or []),
            "choice_judge_raw": None,
            "choice_judge_reasoning": None,
            "judgment": None,
            "randomness_reasoning_judge_raw": None,
            "randomness_reasoning_judge_reasoning": None,
            "random_in_reasoning": None,
            "picked_name": None,
            "picked_position": None,
        })
    return pd.DataFrame(rows)


# ---------------- Reporting ----------------


def report(df: pd.DataFrame, *, label: str = "Summary") -> None:
    n = len(df)
    decisive = df[df["judgment"].isin([1, 2])]
    n_dec = len(decisive)
    n_unc = n - n_dec

    print(f"\n=== {label} ===")
    print(f"  Total variations: {n}")
    if n == 0:
        return
    print(f"  Decisive picks:   {n_dec} ({100 * n_dec / n:.1f}%)")
    print(f"  Refusals:         {n_unc}")
    if n_dec == 0:
        return

    # Position bias (model-level null = 50%)
    n1 = int((decisive["picked_position"] == 1).sum())
    n2 = int((decisive["picked_position"] == 2).sum())
    print(f"\n=== Position bias (decisive only) ===")
    print(f"  (1) picked: {n1:4} ({100 * n1 / n_dec:5.1f}%)")
    print(f"  (2) picked: {n2:4} ({100 * n2 / n_dec:5.1f}%)")

    # Per-activity pick count — the headline "number of times it chose each one".
    print(f"\n=== Per-activity picks (top 20 by pick count) ===")
    print(f"  {'picks':>5}/{'app':<4}  {'rate':>5}  activity")
    all_acts = pd.concat([
        df["activity_1"].rename("name"),
        df["activity_2"].rename("name"),
    ])
    app_counts = all_acts.value_counts()
    pick_counts_by_name = decisive["picked_name"].value_counts()
    # show top 20 by picks (ties broken by name)
    top = pick_counts_by_name.head(20)
    for name, picks in top.items():
        app = int(app_counts.get(name, 0))
        rate = 100 * picks / app if app else 0.0
        print(f"  {int(picks):>5}/{app:<4}  {rate:>4.0f}%  {name}")

    # Bottom: activities that appeared but were never picked.
    appeared = set(all_acts)
    picked = set(decisive["picked_name"])
    never = sorted(appeared - picked, key=lambda n: (-app_counts.get(n, 0), n))
    if never:
        print(f"\n=== Activities that appeared but never won (top 10 by appearances) ===")
        for name in never[:10]:
            app = int(app_counts.get(name, 0))
            print(f"      0/{app:<4}    0%  {name}")


def _normalise_tool_calls(value: object) -> list[dict]:
    if isinstance(value, list):
        return [call for call in value if isinstance(call, dict)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [call for call in parsed if isinstance(call, dict)]
    return []


def _tool_output_label(output: object) -> str:
    if isinstance(output, dict) and set(output) == {"result"}:
        return str(output["result"])
    if isinstance(output, dict) and set(output) == {"letter"}:
        return str(output["letter"])
    if isinstance(output, dict) and set(output) == {"token"}:
        return str(output["token"])
    if isinstance(output, dict) and set(output) == {"unix_seconds"}:
        return str(output["unix_seconds"])
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def report_tool_use(df: pd.DataFrame, *, label: str = "Tool use") -> None:
    print(f"\n=== {label} ===")
    n = len(df)
    if n == 0:
        print("  No rows.")
        return

    calls_by_row = df["tool_calls"].apply(_normalise_tool_calls)
    call_counts = calls_by_row.apply(len)
    n_used = int((call_counts > 0).sum())
    total_calls = int(call_counts.sum())
    print(f"  Samples using tools: {n_used}/{n} ({100 * n_used / n:.1f}%)")
    print(f"  Total tool calls:    {total_calls}")
    if total_calls == 0:
        return

    by_name: dict[str, int] = defaultdict(int)
    by_output: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for calls in calls_by_row:
        for call in calls:
            name = str(call.get("name") or "unknown")
            by_name[name] += 1
            by_output[name][_tool_output_label(call.get("output"))] += 1

    print("\n  Calls by tool:")
    for name, count in sorted(by_name.items(), key=lambda item: (-item[1], item[0])):
        print(f"    {name:<18} {count:>5}")

    print("\n  Output distribution by tool:")
    for name, counts in sorted(by_output.items()):
        print(f"    {name}:")
        for output, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"      {output:<32} {count:>5}")
        if len(counts) > 20:
            print(f"      ... {len(counts) - 20} more distinct outputs")


def tool_use_subsets(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    call_counts = df["tool_calls"].apply(lambda value: len(_normalise_tool_calls(value)))
    return [
        ("tool_used", "tool used", df[call_counts > 0].copy()),
        ("tool_not_used", "tool not used", df[call_counts == 0].copy()),
    ]


def write_tool_use_figure(
    setting_dfs: list[tuple[dict, pd.DataFrame]],
    *,
    model: str | None = None,
    name: str = "tool_use_all_settings",
) -> pathlib.Path | None:
    """Paired tool-use rates per setting: random (true+hedged) vs non-random
    (false+refusal) reasoning. Random bars solid, non-random bars hatched."""
    if not setting_dfs:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = []
    rand_rates, nonrand_rates = [], []
    rand_counts, nonrand_counts = [], []  # (used, total) per bar
    for setting, df in setting_dfs:
        labels.append(setting["label"])
        used = df["tool_calls"].apply(lambda v: len(_normalise_tool_calls(v)) > 0)
        is_random = df["random_in_reasoning"].apply(
            lambda v: _randomness_label(v) in RANDOM_LABELS
        )
        for mask, rates, counts in (
            (is_random, rand_rates, rand_counts),
            (~is_random, nonrand_rates, nonrand_counts),
        ):
            tot = int(mask.sum())
            u = int((used & mask).sum())
            rates.append(100 * u / tot if tot else 0.0)
            counts.append((u, tot))

    base = "#57aaa2"
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    b1 = ax.bar(x - w / 2, rand_rates, w, color=base, edgecolor=base,
                label="random (true + hedged)")
    b2 = ax.bar(x + w / 2, nonrand_rates, w, facecolor="white", edgecolor=base,
                linewidth=0.9, hatch="////", label="non-random (false + refusal)")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Samples using tool (%)", fontsize=12.5, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=11.5, length=0)
    ax.tick_params(axis="y", labelsize=11.5, length=4, width=0.9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    for bars, counts in ((b1, rand_counts), (b2, nonrand_counts)):
        for bar, (u, tot) in zip(bars, counts):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, min(h + 2, 104),
                f"{h:.0f}%\n{u}/{tot}", ha="center", va="bottom",
                fontsize=8.2, linespacing=1.1,
            )
    ax.legend(frameon=False, fontsize=10, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.5, columnspacing=1.6)
    fig.tight_layout()

    out = save_figure(fig, name=name, model=model)
    plt.close(fig)
    return out


def result_subdir(name: str) -> str:
    """Experiment sub-folder a results artifact belongs in, by leaf-name prefix.

    Each model directory is split by experiment so the two measure families don't
    mingle:
        pipeline*            -> pipeline/             (random-pick task)
        activity_liking_*    -> activity_preferences/ (stated liking scores)
        revealed_preference* -> activity_preferences/ (revealed preference)
    Anything else falls back to the model root.
    """
    if name.startswith("pipeline"):
        return "pipeline"
    if name.startswith(("activity_liking_scores", "revealed_preference")):
        return "activity_preferences"
    return ""


def result_path(
    name: str, suffix: str, *, model: str | None = None, setting: str | None = None
) -> pathlib.Path:
    """Path for a results artifact, namespaced by model and experiment.

    Files live at results/<model>/<experiment>/<name>.<suffix>: the model and the
    experiment (see result_subdir) are directories, not part of the leaf name. A
    ``_{model}`` token embedded in ``name`` is stripped so callers can keep their
    existing ``..._{MODEL_KEY}_...`` name templates and still land in a clean
    folder.

    Random-pick artifacts for a single tool setting pass ``setting=<name>`` plus a
    clean leaf (e.g. ``"pipeline"``, ``"selection_summary"``) and land in a
    per-setting sub-folder, results/<model>/pipeline/<setting>/<leaf>.<suffix>,
    mirroring the figures/<model>/<setting>/ layout. Artifacts not tied to a single
    tool setting pass ``setting=None``; their folder is chosen by ``result_subdir``
    from the leaf name (e.g. ``activity_liking_*`` -> activity_preferences/).
    """
    model = model or MODEL_KEY
    if setting is not None:
        out_dir = RESULTS_DIR / model / "pipeline" / setting
        leaf = name
    else:
        leaf = name.replace(f"_{model}", "")
        out_dir = RESULTS_DIR / model / result_subdir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{leaf}.{suffix}"


def save_jsonl(
    df: pd.DataFrame, name: str = "pipeline", *, setting: str | None = None
) -> pathlib.Path:
    out = result_path(name, "jsonl", setting=setting)
    df.to_json(out, orient="records", lines=True)
    return out


def save_csv(
    df: pd.DataFrame, name: str, *, setting: str | None = None
) -> pathlib.Path:
    out = result_path(name, "csv", setting=setting)
    df.to_csv(out, index=False)
    return out


def save_figure(
    fig, name: str, *, model: str | None = None, setting: str | None = None
) -> pathlib.Path:
    # Figures live at figures/<model>/[<setting>/]<name>.png — one directory per
    # model, with a sub-directory per tool setting (no_tools, unix_time,
    # coin_flip, ...). Overall, across-setting plots are saved at the model's top
    # level (pass setting=None).
    out_dir = FIGURES_DIR / (model or MODEL_KEY)
    if setting:
        out_dir = out_dir / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    return out


def filter_by_randomness(df: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    """Rows whose randomness verdict is in ``labels`` (see _randomness_label)."""
    mask = df["random_in_reasoning"].apply(lambda v: _randomness_label(v) in labels)
    return df[mask].copy()


def load_activity_catalog() -> pd.DataFrame:
    """All activities in source-file order, matching the score summary shape."""
    activities = yaml.safe_load(ACTIVITIES_PATH.read_text())["activities"]
    return pd.DataFrame([
        {
            "activity_ix": ix,
            "activity": activity["name"],
        }
        for ix, activity in enumerate(activities)
    ])


def summarize_selection_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per activity with appearance counts and pick rates.

    ``selection_rate`` is the headline metric and EXCLUDES refusals: picks over
    *decisive* appearances only (pairs the choice judge resolved to 1/2). Refusal
    pairs are dropped from the denominator — this is what the correlation and the
    scatter plots use. ``selection_rate_incl_refusals`` keeps refusal appearances
    in the denominator and is kept for reference only.
    """
    catalog = load_activity_catalog()

    appearances = pd.concat([
        df[["activity_1", "judgment"]].rename(columns={"activity_1": "activity"}),
        df[["activity_2", "judgment"]].rename(columns={"activity_2": "activity"}),
    ], ignore_index=True)
    appearances["decisive"] = appearances["judgment"].isin([1, 2])

    appearance_summary = (
        appearances
        .groupby("activity", sort=False)
        .agg(
            n_appearances=("activity", "size"),
            n_decisive_appearances=("decisive", "sum"),
        )
        .reset_index()
    )
    pick_summary = (
        df[df["judgment"].isin([1, 2])]
        .groupby("picked_name", sort=False)
        .size()
        .rename("n_picked")
        .reset_index()
        .rename(columns={"picked_name": "activity"})
    )

    summary = (
        catalog
        .merge(appearance_summary, on="activity", how="left")
        .merge(pick_summary, on="activity", how="left")
    )
    for col in ("n_appearances", "n_decisive_appearances", "n_picked"):
        summary[col] = summary[col].fillna(0).astype(int)

    summary["selection_rate"] = (
        summary["n_picked"] / summary["n_decisive_appearances"].replace({0: pd.NA})
    )
    summary["selection_rate_incl_refusals"] = (
        summary["n_picked"] / summary["n_appearances"].replace({0: pd.NA})
    )
    return summary


def load_score_summary() -> pd.DataFrame | None:
    if not SCORE_SUMMARY_PATH.exists():
        print(f"\nSkipping score/selection correlation: missing {SCORE_SUMMARY_PATH}")
        return None
    return pd.read_csv(SCORE_SUMMARY_PATH)


def _randomness_label(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false", "refusal", "hedged"}:
            return normalized
        if normalized == "unclear":  # legacy label
            return "hedged"
    return "hedged"


# Verdicts the downstream "attempted randomness" analysis counts as random: `true`
# (a clean random pick) and `hedged` (random / trying-to-be, with an AI caveat —
# pseudo / simulated / fair / approximate). `false` (explicit non-random) and
# `refusal` are excluded. Empty traces are forced to `hedged` upstream, so they
# fall in `random` but not `purely_random`.
RANDOM_LABELS = ("true", "hedged")


def _randomness_subsets(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """(key, label, sub) groups for the score-vs-selection analysis.

    Includes BOTH the pooled ``random`` (true+hedged) group and the
    ``purely_random`` (true only) group so either split can be inspected, plus the
    per-verdict cuts.
    """
    label = df["random_in_reasoning"].apply(_randomness_label)
    return [
        ("overall", "overall", df),
        ("random", "random (true+hedged)", df[label.isin(RANDOM_LABELS)]),
        ("purely_random", "purely random (true)", df[label == "true"]),
        ("hedged", "hedged", df[label == "hedged"]),
        ("false", "not random (false)", df[label == "false"]),
        ("refusal", "refusal", df[label == "refusal"]),
    ]


def score_selection_correlation(
    selection_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
) -> tuple[float, int]:
    valid = merge_scores_with_selection(selection_summary, score_summary)
    if len(valid) < 2:
        return float("nan"), len(valid)
    return valid["mean_score"].corr(valid["selection_rate"]), len(valid)


def pearson_r_ci(r: float, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Fisher z-transform confidence interval for Pearson r.

    This is the standard large-sample CI for a Pearson correlation across
    independent paired observations. Here the paired observations are
    activities: (mean liking score, selection rate).
    """
    if n <= 3 or pd.isna(r):
        return float("nan"), float("nan")
    zcrit = NormalDist().inv_cdf(0.5 + confidence / 2)
    r_clamped = max(min(float(r), 0.999999999), -0.999999999)
    z = math.atanh(r_clamped)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - zcrit * se), math.tanh(z + zcrit * se)


def _format_ci(lo: float, hi: float) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "[n/a, n/a]"
    return f"[{lo:.3f}, {hi:.3f}]"


def merge_scores_with_selection(
    selection_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = selection_summary.merge(
        score_summary[["activity_ix", "activity", "mean_score"]],
        on=["activity_ix", "activity"],
        how="left",
    )
    return merged.dropna(subset=["mean_score", "selection_rate"])


def print_score_selection_correlations(
    df: pd.DataFrame,
    *,
    include_randomness_subsets: bool = True,
) -> None:
    score_summary = load_score_summary()
    if score_summary is None:
        return

    if include_randomness_subsets:
        rows = [(label, sub) for _key, label, sub in _randomness_subsets(df)]
    else:
        rows = [("overall", df)]

    print("\n=== Mean score vs selection-rate correlation ===")
    print(f"  score source: {SCORE_SUMMARY_PATH}")
    print(f"  {'subset':<30}{'rows':>8}{'activities':>14}{'pearson r':>12}  95% CI")
    for label, sub in rows:
        r, n_activities = score_selection_correlation(
            summarize_selection_rates(sub),
            score_summary,
        )
        lo, hi = pearson_r_ci(r, n_activities)
        r_text = "n/a" if pd.isna(r) else f"{r: .3f}"
        print(
            f"  {label:<30}{len(sub):>8}{n_activities:>14}"
            f"{r_text:>12}  {_format_ci(lo, hi)}"
        )


def plot_score_selection_scatter(
    selection_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    *,
    name: str,
    title: str,
    no_reasoning: tuple[int, int, float] | None = None,
    setting: str | None = None,
) -> pathlib.Path | None:
    plot_df = merge_scores_with_selection(selection_summary, score_summary)
    if plot_df.empty:
        print(f"Skipping scatter plot {name}: no activities with scores and selections")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        plot_df["mean_score"],
        plot_df["selection_rate"] * 100,
        s=42,
        alpha=0.78,
        color="#2a6f97",
        edgecolor="white",
        linewidth=0.6,
    )

    r, n = score_selection_correlation(selection_summary, score_summary)
    lo, hi = pearson_r_ci(r, n)
    r_text = "n/a" if pd.isna(r) else f"{r:.3f}"
    ax.set_title(
        f"{title}\nPearson r={r_text}, 95% CI={_format_ci(lo, hi)}, activities={n}",
        fontsize=17,
        pad=14,
    )
    ax.set_xlabel("Mean activity liking score", fontsize=16, labelpad=8)
    ax.set_ylabel("Selection rate (%)", fontsize=16, labelpad=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    style_score_selection_axis(ax, tick_labelsize=13)

    if no_reasoning is not None:
        empty, total, pct = no_reasoning
        ax.text(
            0.02, 0.98,
            f"no reasoning trace: {empty}/{total} ({pct:.1f}%)",
            transform=ax.transAxes,
            ha="left", va="top",
            color="red", fontsize=13, fontweight="bold",
        )

    fig.tight_layout()

    out = save_figure(fig, name, setting=setting)
    plt.close(fig)
    return out


def style_score_selection_axis(ax, *, tick_labelsize: int = 12) -> None:
    ax.grid(False)
    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        which="major",
        length=5,
        width=1.0,
        colors="#333333",
        labelsize=tick_labelsize,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def write_score_selection_scatters(
    df: pd.DataFrame,
    *,
    setting_name: str,
    include_randomness_subsets: bool = True,
) -> None:
    score_summary = load_score_summary()
    if score_summary is None:
        return

    if include_randomness_subsets:
        rows = list(_randomness_subsets(df))
    else:
        rows = [("overall", "Overall", df)]

    print("\nWriting score/selection scatter plots...")
    for suffix, title, sub in rows:
        out = plot_score_selection_scatter(
            summarize_selection_rates(sub),
            score_summary,
            name=f"score_vs_selection_{suffix}",
            title=title,
            no_reasoning=_no_reasoning_stats(sub),
            setting=setting_name,
        )
        if out is not None:
            print(f"  {title}: {out}")


def write_all_settings_score_selection_figure(
    setting_dfs: list[tuple[dict, pd.DataFrame]],
    *,
    name: str = "score_vs_selection_all_tool_settings",
    model: str | None = None,
    score_summary: pd.DataFrame | None = None,
    subtitle: str | None = None,
    row_filter=None,
) -> pathlib.Path | None:
    # ``setting_dfs`` carry the FULL rows. When ``row_filter`` is given the panel
    # plots only that subset and its title notes what % of the setting's rows the
    # subset is (e.g. the random or non-random case).
    if score_summary is None:
        score_summary = load_score_summary()
    if score_summary is None or not setting_dfs:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    point_color = "#2a6f97"

    n_panels = len(setting_dfs)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.7 * n_panels, 5.0), sharex=True, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (setting, df) in zip(axes, setting_dfs):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Mean activity liking score", fontsize=11, labelpad=6)
        style_score_selection_axis(ax, tick_labelsize=10.5)
        # Full border all the way round, and square panels.
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color("#333333")
            ax.spines[sp].set_linewidth(0.9)
        ax.set_box_aspect(1)
        # Apply the optional subset filter; its share of the setting's rows goes in
        # the grey metadata strip below (with r / n), not the title.
        sub = row_filter(df) if row_filter is not None else df
        total = len(df)
        rows_pct = (100 * len(sub) / total) if total else 0.0
        # Order top -> bottom: r line, n line, (% of rows), then the setting title
        # directly above the plot. The grey metadata sits on top so the figure can be
        # cropped from the top to drop it while keeping the title.
        ax.text(0.5, 1.02, setting["label"], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=12.5, fontweight="semibold")

        selection_summary = summarize_selection_rates(sub)
        plot_df = merge_scores_with_selection(selection_summary, score_summary)
        if plot_df.empty:
            ax.text(0.5, 0.5, "no scored selections", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=10)
            continue

        xs = plot_df["mean_score"].to_numpy(dtype=float)
        ys = (plot_df["selection_rate"] * 100).to_numpy(dtype=float)
        ax.scatter(
            xs, ys, s=26, alpha=0.6, color=point_color,
            edgecolor="white", linewidth=0.4,
        )
        # Light-grey line of best fit (OLS), no confidence band.
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.array([xs.min(), xs.max()])
            ax.plot(xline, slope * xline + intercept,
                    color="#b0b0b0", linewidth=1.6, zorder=3)

        r, n = score_selection_correlation(selection_summary, score_summary)
        lo, hi = pearson_r_ci(r, n)
        r_text = "n/a" if pd.isna(r) else f"{r:.2f}"
        ci_text = "" if pd.isna(lo) else f"  [{lo:.2f}, {hi:.2f}]"
        _, _, pct = _no_reasoning_stats(sub)
        meta = f"r = {r_text}{ci_text}\nactivities = {n} · no trace {pct:.0f}%"
        # Number of examples (model samples) that went into this panel — always
        # shown. For a randomness cut it's that subset's rows + its share of the
        # setting; for the overall figure it's every row in the setting.
        if row_filter is not None:
            meta += f"\n{rows_pct:.0f}% of rows (n = {len(sub):,})"
        else:
            meta += f"\nn = {len(sub):,} examples"
        ax.text(
            0.5, 1.10, meta,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.6, linespacing=1.3, color="#5a5a5a",
        )

    axes[0].set_ylabel("Selection rate (%)", fontsize=11, labelpad=6)

    model_name = MODELS.get(model or MODEL_KEY, {}).get("display_name", model or MODEL_KEY)
    suptitle = model_name + (f" — {subtitle}" if subtitle else "")
    fig.suptitle(suptitle, fontsize=14, fontweight="semibold", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.80, wspace=0.14)

    out = save_figure(fig, name=name, model=model)
    plt.close(fig)
    return out


def write_paper_score_selection_figure(
    setting_dfs: list[tuple[dict, pd.DataFrame]],
    *,
    name: str = "score_vs_selection_all_tool_settings_paper",
    model: str | None = None,
    score_summary: pd.DataFrame | None = None,
    row_filter=None,
) -> pathlib.Path | None:
    """Paper-styled three-panel score-vs-selection figure.

    Same data as write_all_settings_score_selection_figure, but styled for the
    paper: r + 95% CI printed inside each square (top-left), x-axis labelled
    "Activity preference score", and NO grey metadata strip or model suptitle.
    When ``row_filter`` is given only that subset of rows is plotted (e.g. the
    random cases). Skips (returns None) when no stated-liking score summary is
    available, so models without scores don't error.
    """
    if score_summary is None:
        score_summary = load_score_summary()
    if score_summary is None or not setting_dfs:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    point_color, fit_color = "#2a6f97", "#b0b0b0"
    n_panels = len(setting_dfs)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.7 * n_panels, 4.6), sharex=True, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]
    for ax, (setting, df) in zip(axes, setting_dfs):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Activity preference score", fontsize=12, labelpad=6)
        style_score_selection_axis(ax, tick_labelsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color("#333333")
            ax.spines[sp].set_linewidth(0.9)
        ax.set_box_aspect(1)
        ax.set_title(setting["label"], fontsize=12.5, fontweight="semibold", pad=8)

        sub = row_filter(df) if row_filter is not None else df
        selection_summary = summarize_selection_rates(sub)
        plot_df = merge_scores_with_selection(selection_summary, score_summary)
        if plot_df.empty:
            ax.text(0.5, 0.5, "no scored selections", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=10)
            continue
        xs = plot_df["mean_score"].to_numpy(dtype=float)
        ys = (plot_df["selection_rate"] * 100).to_numpy(dtype=float)
        ax.scatter(xs, ys, s=26, alpha=0.6, color=point_color,
                   edgecolor="white", linewidth=0.4)
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.array([xs.min(), xs.max()])
            ax.plot(xline, slope * xline + intercept,
                    color=fit_color, linewidth=1.6, zorder=3)
        r, n = score_selection_correlation(selection_summary, score_summary)
        lo, hi = pearson_r_ci(r, n)
        r_text = "n/a" if pd.isna(r) else f"{r:.2f}"
        ci_text = "" if pd.isna(lo) else f" [{lo:.2f}, {hi:.2f}]"
        ax.text(0.05, 0.95, f"$r$ = {r_text}{ci_text}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
    axes[0].set_ylabel("Selection rate (%)", fontsize=12, labelpad=6)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.92, wspace=0.14)

    out = save_figure(fig, name=name, model=model)
    plt.close(fig)
    return out


def _randomness_cut_filter(labels):
    """Row filter keeping only rows whose randomness verdict is in ``labels``
    (None = keep all rows / the "overall" cut)."""
    if labels is None:
        return None
    label_set = tuple(labels)
    return lambda df: filter_by_randomness(df, label_set)


# Aggregate three-panel score-vs-selection figures the pipeline writes by default
# every run: one per randomness "case" (overall + each verdict cut), mirroring
# _randomness_subsets so the figure set tracks the analysis. Each is a
# (filename-suffix, randomness-verdict-labels, title-subtitle) triple; ""/None is
# the overall cut (the base filename, no suffix). The subtitle names which
# reasoning case went into the figure and is shown in the (non-paper) suptitle.
_RANDOMNESS_FIGURE_CUTS = [
    ("", None, "all rows"),
    ("_random", RANDOM_LABELS, "random (purely random + hedged)"),
    ("_nonrefusal", ("true", "hedged", "false"), "non-refusal (true+hedged+false)"),
    ("_purely_random", ("true",), "purely random"),
    ("_hedged", ("hedged",), "hedged"),
    ("_false", ("false",), "not random (false)"),
    ("_refusal", ("refusal",), "refusal"),
]
# Cuts that also get a paper-styled variant (the headline ones). Paper figures
# carry no suptitle, so they take no subtitle. `_nonrefusal` mirrors `_random` but
# also keeps the `false` (overt-bias) rows — only refusals are dropped.
_PAPER_FIGURE_CUTS = [
    ("", None),
    ("_random", RANDOM_LABELS),
    ("_nonrefusal", ("true", "hedged", "false")),
]
_SUMMARY_FIGURE_BASE = "score_vs_selection_all_tool_settings"


def write_summary_score_selection_figures(
    setting_dfs: list[tuple[dict, pd.DataFrame]],
    *,
    model: str | None = None,
    score_summary: pd.DataFrame | None = None,
) -> list[pathlib.Path]:
    """Write the full aggregate score-vs-selection set for one model: the
    three-panel figure for every randomness case (default styling, titled with
    which case it is) plus paper-styled versions of the headline cuts. Returns the
    paths written; each figure self-skips when there is no stated-liking score
    summary.
    """
    if score_summary is None:
        score_summary = load_score_summary()
    written = []
    for suffix, labels, subtitle in _RANDOMNESS_FIGURE_CUTS:
        name = f"{_SUMMARY_FIGURE_BASE}{suffix}"
        out = write_all_settings_score_selection_figure(
            setting_dfs, name=name, model=model, score_summary=score_summary,
            row_filter=_randomness_cut_filter(labels), subtitle=subtitle,
        )
        if out is not None:
            written.append(out)
            print(f"Saved {name} -> {out}")
    for suffix, labels in _PAPER_FIGURE_CUTS:
        name = f"{_SUMMARY_FIGURE_BASE}{suffix}_paper"
        out = write_paper_score_selection_figure(
            setting_dfs, name=name, model=model,
            score_summary=score_summary, row_filter=_randomness_cut_filter(labels),
        )
        if out is not None:
            written.append(out)
            print(f"Saved {name} -> {out}")
    return written


def load_existing_judged_setting_outputs() -> list[tuple[dict, pd.DataFrame]] | None:
    """Load saved judged outputs for plotting without rerunning judges."""
    pairs = []
    for setting in TOOL_SETTINGS:
        path = result_path("pipeline", "jsonl", setting=setting["name"])
        if not path.exists():
            print(f"Cannot regenerate figures: missing judged output {path}")
            return None
        df = pd.read_json(path, lines=True)
        if "judgment" not in df or not df["judgment"].isin([1, 2]).any():
            print(f"Cannot regenerate figures: {path} has no decisive judged rows")
            return None
        pairs.append((setting, df))
    return pairs


def regenerate_score_selection_figure_from_saved_outputs() -> list[pathlib.Path]:
    pairs = load_existing_judged_setting_outputs()
    if pairs is None:
        return []
    print("Regenerating score/selection figures from saved judged outputs...")
    return write_summary_score_selection_figures(pairs)


def write_available_figures(
    setting_dfs: list[tuple[dict, pd.DataFrame]],
) -> None:
    tool_fig = write_tool_use_figure(setting_dfs)
    if tool_fig is not None:
        print(f"Saved tool-use figure -> {tool_fig}")

    has_current_judgments = all(
        "judgment" in df and df["judgment"].isin([1, 2]).any()
        for _, df in setting_dfs
    )
    if has_current_judgments:
        write_summary_score_selection_figures(setting_dfs)
        return

    regenerate_score_selection_figure_from_saved_outputs()


# ---------------- Main ----------------

def run_tool_setting(variations: list[dict], tool_setting: dict) -> pd.DataFrame:
    setting_name = tool_setting["name"]
    label = tool_setting["label"]
    tool_names = list(tool_setting["tool_names"])

    print(f"\n\n{'=' * 70}")
    print(f"Tool setting: {label} ({setting_name})")
    print(f"Allowed tools: {tool_names if tool_names else 'none'}")
    print(f"{'=' * 70}")

    print(f"Running main model ({MODEL_KEY})...")
    df_main = run_main(variations, tool_setting)
    if len(df_main) != len(variations):
        print(f"  warning: {MODEL_KEY} returned {len(df_main)} rows, "
              f"expected {len(variations)}")

    if not RUN_CHOICE_JUDGE:
        print("\nSkipping choice judge (RUN_CHOICE_JUDGE=False).")
        df = assemble_unjudged(variations, df_main)
        print(f"\n\n{'#' * 70}\n# {MODEL_KEY} · {setting_name}\n{'#' * 70}")
        report_tool_use(df, label=f"Tool use: {setting_name}")
        out = save_jsonl(df, name=UNJUDGED_OUTPUT_SUFFIX, setting=setting_name)
        print(f"\nSaved {len(df)} unjudged records -> {out}")
        if tool_names:
            for suffix, subset_label, sub in tool_use_subsets(df):
                sub_out = save_jsonl(
                    sub,
                    name=f"{UNJUDGED_OUTPUT_SUFFIX}_{suffix}",
                    setting=setting_name,
                )
                print(f"Saved {len(sub)} {subset_label} records -> {sub_out}")
        return df

    responses = df_main["answer"].tolist()

    print(f"\nRunning choice judge ({JUDGE_MODEL})...")
    df_judge = run_choice_judge(variations, responses)
    judge_col = f"{EXPERIMENT_NAME}_choice_judge"
    judgments = [parse_choice_judge_answer(a) for a in df_judge[judge_col]]
    choice_judge_reasoning = df_judge[f"{judge_col}_reasoning"].tolist()

    reasonings = df_main["reasoning"].tolist()
    if RUN_RANDOMNESS_JUDGE:
        print(f"\nRunning reasoning-randomness judge ({JUDGE_MODEL})...")
        df_randomness = run_randomness_reasoning_judge(
            df_main["prompt"].tolist(), reasonings, responses
        )
        randomness_col = f"{EXPERIMENT_NAME}_randomness_reasoning_judge"
        # Every row is graded by the judge — including empty-reasoning rows, which
        # the judge can now classify from the final response (the judge sees the
        # answer). The old "empty reasoning -> force hedged" null rule is gone.
        randomness_raw = df_randomness[randomness_col].tolist()
        randomness_reasoning_judge_reasoning = df_randomness[
            f"{randomness_col}_reasoning"
        ].tolist()
        randomness_reasoning = [
            parse_randomness_reasoning_judge(raw) for raw in randomness_raw
        ]
    else:
        print("\nSkipping reasoning-randomness judge (RUN_RANDOMNESS_JUDGE=False).")
        randomness_raw = [None] * len(reasonings)
        randomness_reasoning = [None] * len(reasonings)
        randomness_reasoning_judge_reasoning = [None] * len(reasonings)

    df = assemble(
        variations,
        df_main,
        judgments,
        df_judge[judge_col].tolist(),
        choice_judge_reasoning,
        randomness_raw,
        randomness_reasoning,
        randomness_reasoning_judge_reasoning,
    )
    print(f"\n\n{'#' * 70}\n# {MODEL_KEY} · {setting_name}\n{'#' * 70}")
    print("\nFull output:")
    report(df, label="Summary: all rows")
    report_tool_use(df, label=f"Tool use: {setting_name}")

    if tool_names:
        print("\nConditional output: tool use")
        for suffix, subset_label, sub in tool_use_subsets(df):
            report(sub, label=f"Summary: {subset_label}")
            sub_out = save_jsonl(
                sub,
                name=suffix,
                setting=setting_name,
            )
            sub_selection_out = save_csv(
                summarize_selection_rates(sub),
                name=f"{suffix}_selection_summary",
                setting=setting_name,
            )
            print(f"Saved {len(sub)} {subset_label} records -> {sub_out}")
            print(f"Saved {subset_label} selection rows -> {sub_selection_out}")

    out = save_jsonl(df, name="pipeline", setting=setting_name)
    print(f"\nSaved {len(df)} records -> {out}")

    selection_summary = summarize_selection_rates(df)
    selection_summary_out = save_csv(
        selection_summary,
        name="pipeline_selection_summary",
        setting=setting_name,
    )
    print(f"Saved {len(selection_summary)} activity selection rows -> "
          f"{selection_summary_out}")

    if RUN_RANDOMNESS_JUDGE:
        # Two filtered cuts: the default "random" set (true+hedged) and the strict
        # "purely random" set (true only). Both are derived re-slices of `df`.
        for fkey, flabel, flabels in [
            ("random_in_reasoning_random", "random (true+hedged)", RANDOM_LABELS),
            ("random_in_reasoning_purely_random", "purely random (true)", ("true",)),
        ]:
            sub = filter_by_randomness(df, flabels)
            print(f"\nFiltered output: {flabel}")
            report(sub, label=f"Summary: {flabel}")
            sub_out = save_jsonl(sub, name=fkey, setting=setting_name)
            sub_sel_out = save_csv(
                summarize_selection_rates(sub),
                name=f"{fkey}_selection_summary",
                setting=setting_name,
            )
            print(f"Saved {len(sub)} {flabel} records -> {sub_out}")
            print(f"Saved {flabel} selection rows -> {sub_sel_out}")

    print_score_selection_correlations(
        df,
        include_randomness_subsets=RUN_RANDOMNESS_JUDGE,
    )
    write_score_selection_scatters(
        df,
        setting_name=setting_name,
        include_randomness_subsets=RUN_RANDOMNESS_JUDGE,
    )
    return df


def main() -> None:
    validate_tool_settings()
    print(f"Building {N_VARIATIONS} variations (seed={SEED})...")
    variations = build_variations(N_VARIATIONS, SEED)
    print(f"Allowed tool registry: {ALLOWED_TOOL_NAMES}")
    print("Experiment tool settings:")
    for setting in TOOL_SETTINGS:
        print(f"  - {setting['name']}: {setting['tool_names'] or 'none'}")
    print(f"Run choice judge: {RUN_CHOICE_JUDGE}")
    print(f"Run randomness judge: {RUN_RANDOMNESS_JUDGE}")

    all_dfs = [
        run_tool_setting(variations, setting)
        for setting in TOOL_SETTINGS
    ]
    write_available_figures(list(zip(TOOL_SETTINGS, all_dfs)))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
        primary = streams[0] if streams else None
        self.encoding = getattr(primary, "encoding", "utf-8") or "utf-8"
        self.errors = getattr(primary, "errors", "replace") or "replace"

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())

    def fileno(self) -> int:
        if not self.streams:
            raise OSError("no streams")
        return self.streams[0].fileno()


def run_main_with_log() -> pathlib.Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"pipeline_{MODEL_KEY}_{timestamp}.log"
    with log_path.open("w") as log_file:
        stdout = _Tee(sys.stdout, log_file)
        stderr = _Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"Writing run log -> {log_path}")
            main()
    return log_path


if __name__ == "__main__":
    path = run_main_with_log()
    print(f"Run log saved -> {path}")




