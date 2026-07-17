"""Experiment runner: baseline → estimate extraction → threshold computation → threshold runs."""

import hashlib
import json
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd

from .judge_jsonl_cache import JsonlJudgeCache
from .prompts import THRESHOLD_PROMPTS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
ESTIMATE_JUDGE_CACHE_ROOT = os.path.join(os.path.dirname(__file__), "estimate_judge_cache")
APPLY_JUDGE_CACHE_ROOT = os.path.join(os.path.dirname(__file__), "apply_judge_cache")


def estimate_judge_cache_dir(experiment_name):
    """Absolute path to the post-llmcomp estimate-judge cache for one experiment.

    Layout: ``<ESTIMATE_JUDGE_CACHE_ROOT>/<experiment_name>/<judge_config_hash>/<shard>.jsonl``
    (handled by ``shared.judge_jsonl_cache.JsonlJudgeCache``).
    """
    return os.path.join(ESTIMATE_JUDGE_CACHE_ROOT, experiment_name)


# Judge config for `batch_extract_estimates`. Must match the values recorded in
# the existing llmcomp cache (model=claude-sonnet-4-6, temperature=0,
# max_tokens=1024, no thinking) so that the migration script's entries are
# reachable by the runtime lookup; if any of these change the JsonlJudgeCache
# config_hash forks a fresh directory automatically.
ESTIMATE_JUDGE_CONFIG = {
    "backend": "claude",
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "temperature": 0,
    "thinking_type": "disabled",
    "max_concurrent": 100,
}


class CacheOnlyMiss(RuntimeError):
    """Raised when cache-only mode would need to sample."""


# --- Hashing ---

def _hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _model_hashable(model):
    return {k: v for k, v in model.items() if k not in ("max_concurrent", "display_name")}


def _with_prompt_suffix(model, prompt_text):
    suffix = model.get("prompt_suffix")
    if not suffix:
        return prompt_text
    return f"{prompt_text}{suffix}"


def _prompt_hash(model, n, prompt_text):
    return _hash({
        "model": _model_hashable(model),
        "n": n,
        "prompt": prompt_text,
    })


def _direction_hash(model, n_per_threshold, template_text, thresholds):
    return _hash({
        "model": _model_hashable(model),
        "n_per_threshold": n_per_threshold,
        "template": template_text,
        "thresholds": thresholds,
    })


# --- Cache I/O ---

def _cache_path(model_name, prompt_key, direction, h):
    return os.path.join(CACHE_DIR, model_name, prompt_key, f"{direction}_{h}.jsonl")


def _read_cache(path, expected_hash):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return None
    try:
        meta = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    if meta.get("hash") != expected_hash:
        return None
    return [json.loads(line) for line in lines[1:]]


def _write_cache(path, meta, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    # print(f"Wrote {len(results)} results to {path}")


# --- Per-prompt cache ---
#
# Companion to the batch-keyed `_read_cache`/`_write_cache` above. The batch
# version stores a whole experiment's prompts under a single hash, so changing
# any single prompt busts the entire cache; this version keys per prompt so
# small sample changes (dropping one item, bumping N, swapping a subset) reuse
# every prompt that's still present.

def _per_prompt_hash(model, prompt_text, seed=None):
    """Cache key for (model, prompt[, seed]). Passing seed=None or seed=0 both
    reproduce the pre-seed hash exactly, so old cache files (written before
    seeded sampling was added) serve as the seed=0 roll. seeds 1, 2, ... get
    distinct hashes so the same prompt can be sampled multiple independent
    times under different keys."""
    payload = {
        "model": _model_hashable(model),
        "prompt": prompt_text,
    }
    if seed is not None and seed != 0:
        payload["seed"] = seed
    return _hash(payload)


def _read_per_prompt_cache(path):
    """Load a per-prompt JSONL cache as {hash: row}. Missing file -> {}."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = row.get("hash")
            if h is not None:
                out[h] = row
    return out


def run_with_prompt_cache(prompts, model, cache_path, *,
                          desc=None, semaphore=None, sender=None,
                          seeds=None):
    """Run `prompts` through `model`, persisting one entry per unique
    (prompt[, seed]) pair.

    Cache key = hash(model_config, prompt_text[, seed]). Hits are reused
    as-is; misses are sampled and appended to ``cache_path`` (JSONL, one row
    per unique pair). The model only sees ``prompt`` — ``seeds`` is purely a
    cache-key salt that lets a caller request multiple independent rolls of
    the same prompt at temperature > 0.

    ``seeds`` must be the same length as ``prompts`` when provided. Pairs
    with seed=0 (or seed=None) share the unseeded hash, so existing pre-seed
    cache entries are reused as the seed=0 roll.

    Returns response dicts in the order of ``prompts``, each shaped:
    ``{"prompt", "answer", "reasoning", "blocked"}``.
    """
    if seeds is None:
        seeds = [None] * len(prompts)
    elif len(seeds) != len(prompts):
        raise ValueError(
            f"seeds length {len(seeds)} does not match prompts length {len(prompts)}"
        )

    cache = _read_per_prompt_cache(cache_path)
    pair_hashes = [_per_prompt_hash(model, p, s) for p, s in zip(prompts, seeds)]

    # Dedupe by hash (which collapses pairs that map to the same key, including
    # seed=0 / seed=None on the same prompt). First-seen order preserved.
    seen_hashes = set()
    miss_indices = []
    for i, h in enumerate(pair_hashes):
        if h in cache or h in seen_hashes:
            continue
        seen_hashes.add(h)
        miss_indices.append(i)

    if miss_indices:
        if sender is None:
            sender = _create_sender(model)
        progress = tqdm(total=len(miss_indices), desc=desc) if desc else None
        miss_prompts = [prompts[i] for i in miss_indices]
        new_rows = _run_prompts(
            sender, model["max_concurrent"], miss_prompts,
            progress=progress, semaphore=semaphore,
        )
        if progress is not None:
            progress.close()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "a") as f:
            for idx, r in zip(miss_indices, new_rows):
                h = pair_hashes[idx]
                row = {
                    "hash": h,
                    "answer": r.get("answer", ""),
                    "reasoning": r.get("reasoning", ""),
                }
                if r.get("blocked"):
                    row["blocked"] = True
                f.write(json.dumps(row) + "\n")
                cache[h] = row

    out = []
    for p, h in zip(prompts, pair_hashes):
        row = cache[h]
        out.append({
            "prompt": p,
            "answer": row.get("answer", ""),
            "reasoning": row.get("reasoning", ""),
            "blocked": row.get("blocked", False),
        })
    return out


def _retry(call, transient_excs, desc, *, transient_check=None):
    """Call `call()` and retry on transient errors.

    Retries on any exception in ``transient_excs``, plus any exception for
    which ``transient_check(e)`` returns True. Exponential backoff capped at
    60s. Non-transient exceptions propagate.
    """
    delay = 1.0
    while True:
        try:
            return call()
        except Exception as e:
            is_transient = (
                isinstance(e, transient_excs)
                or (transient_check is not None and transient_check(e))
            )
            if not is_transient:
                raise
            print(f"{desc}: {type(e).__name__} ({e}), retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


# --- Backend-specific senders ---

def _get_renderer_cls(name):
    if name == "kimi_k25":
        from tinker_cookbook.renderers.kimi_k25 import KimiK25Renderer
        return KimiK25Renderer
    elif name == "deepseek_v3_thinking":
        from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3ThinkingRenderer
        return DeepSeekV3ThinkingRenderer
    elif name == "qwen3_5":
        from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
        return Qwen3_5Renderer
    elif name == "nemotron3":
        from tinker_cookbook.renderers.nemotron3 import Nemotron3Renderer
        return Nemotron3Renderer
    elif name == "gpt_oss":
        from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
        return GptOssRenderer
    else:
        raise ValueError(f"Unknown renderer: {name}")


def _patch_tinker_kimi_k26_tokenizer():
    """Make tinker.SamplingClient.get_tokenizer() work for moonshotai/Kimi-K2.6.

    The Tinker SDK's `_load_tokenizer_from_model_info` special-cases the older
    Kimi-K2-Thinking and Kimi-K2.5 variants by passing `trust_remote_code=True`
    to AutoTokenizer.from_pretrained, but doesn't yet have an entry for
    Kimi-K2.6 — so loading the tokenizer raises a ValueError demanding
    `trust_remote_code=True`. We patch the SDK module to extend the
    special-case list. Idempotent and a no-op for any other model.
    """
    import tinker.lib.public_interfaces.sampling_client as _tinker_sc

    if getattr(_tinker_sc, "_kimi_k26_patched", False):
        return

    _orig_load = _tinker_sc._load_tokenizer_from_model_info

    def _load_with_kimi_k26(model_name, tokenizer_id=None):
        name = model_name.split(":")[0]
        if name.count("/") == 2:
            org, mdl, _ = name.split("/", 2)
            name = f"{org}/{mdl}"
        if (tokenizer_id or name) == "moonshotai/Kimi-K2.6":
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            return AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-K2.6", fast=True, trust_remote_code=True,
            )
        return _orig_load(model_name, tokenizer_id)

    _tinker_sc._load_tokenizer_from_model_info = _load_with_kimi_k26
    _tinker_sc._kimi_k26_patched = True


def _create_sender(model):
    backend = model["backend"]

    if backend == "claude":
        import anthropic
        import httpx
        # We deliberately use `thinking.type=enabled` (not the newer
        # `adaptive` mode); silence the SDK's per-request deprecation
        # warning so it doesn't spam logs once per call.
        warnings.filterwarnings(
            "ignore",
            message=r".*'thinking\.type=enabled' is deprecated.*",
            category=UserWarning,
        )
        # httpx splits `timeout` into four categories (connect, read, write,
        # pool); a bare float sets all four. The `read` value is the
        # per-socket-read budget, NOT per-request, so for a streaming SSE
        # response it's the max gap between consecutive bytes from the
        # server. With healthy streaming, chunks arrive every few hundred
        # ms during generation (and continuously even during long thinking
        # runs), so 30s is generous slack. When that does fire we get
        # httpx.ReadTimeout which the transient tuple below already
        # retries, so a stalled stream costs us ~30s, not the
        # 30 minutes a bare timeout=1800 used to.
        client = anthropic.Anthropic(timeout=httpx.Timeout(30.0))
        transient = (
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            # The Anthropic SDK wraps httpx errors into APITimeoutError /
            # APIConnectionError only on the outbound side (messages.create).
            # Once we're iterating an open SSE stream (stream.get_final_message
            # → consume iterator), httpx exceptions propagate raw — most
            # commonly httpx.ReadTimeout if the server stalls mid-stream, but
            # also RemoteProtocolError if the connection drops. TransportError
            # is the umbrella for all timeouts, network errors, and protocol
            # errors at the transport layer.
            httpx.TransportError,
        )

        # Anthropic's SDK has separate APIStatusError subclasses for 503,
        # 504, and 529 ("overloaded") that don't inherit from
        # InternalServerError, and 529 shows up routinely under peak load.
        # Match on status_code so we don't depend on which symbols the SDK
        # re-exports.
        #
        # Mid-stream errors are a second case: when an SSE stream is open
        # (initial response was 200 OK) and the server then emits an
        # `event: error` frame, the SDK raises a *bare* APIStatusError —
        # _make_status_error has no subclass to dispatch to for a 200,
        # so it falls back to the base class. Any such exception is by
        # construction a server-side failure during generation (the
        # request was already accepted), so always retry it.
        def _is_transient_anthropic(e):
            if not isinstance(e, anthropic.APIStatusError):
                return False
            if getattr(e, "status_code", None) in (503, 504, 529):
                return True
            return type(e) is anthropic.APIStatusError

        def send(prompt):
            kwargs = dict(
                model=model["model"],
                max_tokens=model["max_tokens"],
                messages=[{"role": "user", "content": prompt}],
                temperature=model["temperature"],
            )
            # Pre-3.7 Claude models (e.g. claude-3-5-sonnet) reject the
            # `thinking` parameter entirely, so omit it when the model entry
            # doesn't specify a thinking_type.
            if model.get("thinking_type") is not None:
                thinking = {"type": model["thinking_type"]}
                if model["thinking_type"] == "enabled":
                    thinking["budget_tokens"] = model["budget_tokens"]
                if model["thinking_type"] != "disabled":
                    thinking["display"] = model["thinking_display"]
                kwargs["thinking"] = thinking
            if model.get("effort"):
                kwargs["output_config"] = {"effort": model["effort"]}
            if model.get("system_prompt"):
                kwargs["system"] = model["system_prompt"]

            # Stream rather than messages.create(): for long thinking runs
            # (15+ min on hard prompts with high effort) Anthropic's edge
            # load balancers will kill an idle non-streaming connection
            # well before the SDK timeout fires, wasting the whole
            # generation. The SSE stream keeps bytes flowing, and
            # get_final_message() consumes the stream and returns the
            # same assembled Message shape that create() would.
            def _call():
                with client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()

            response = _retry(
                _call,
                transient, "claude",
                transient_check=_is_transient_anthropic,
            )

            thinking_blocks = [b for b in response.content if b.type == "thinking"]
            text_blocks = [b for b in response.content if b.type == "text"]
            if len(thinking_blocks) > 1 or len(text_blocks) > 1:
                raise NotImplementedError(
                    f"Claude response had multiple blocks of the same type "
                    f"(thinking={len(thinking_blocks)}, text={len(text_blocks)}); "
                    f"don't know how to combine them."
                )
            reasoning = thinking_blocks[0].thinking if thinking_blocks else ""
            answer = text_blocks[0].text if text_blocks else ""
            # Anthropic AUP-refusals come back as 200 OK with no thinking and
            # no text block, sometimes with stop_reason="refusal" on Opus 4+.
            # Surface them as blocked sentinels so downstream code can separate
            # refusals from "wrong answer" instead of silently treating them as
            # incorrect.
            stop_reason = getattr(response, "stop_reason", None)
            blocked = (
                stop_reason == "refusal"
                or (not thinking_blocks and not text_blocks)
            )
            out = {"reasoning": reasoning, "answer": answer, "prompt": prompt}
            if blocked:
                out["blocked"] = True
                if not answer:
                    out["answer"] = "[BLOCKED_BY_ANTHROPIC]"
            return out

        return send

    elif backend == "tinker":
        import tinker
        from tinker_cookbook.renderers.base import Message

        _patch_tinker_kimi_k26_tokenizer()

        renderer_cls = _get_renderer_cls(model["renderer"])
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=model["model_path"])
        tokenizer = sampling_client.get_tokenizer()
        renderer = renderer_cls(tokenizer)
        stop_sequences = renderer.get_stop_sequences()

        # The SDK exchanges TINKER_API_KEY for a short-lived JWT and refreshes
        # it in a background task; a rare race can let one request go out with
        # a stale JWT, surfaced as a bare ValueError("... 401 ... Invalid JWT").
        # The background loop heals itself, so just retry.
        def _is_jwt_401(e):
            return isinstance(e, ValueError) and "Invalid JWT" in str(e)

        def send(prompt):
            messages = [Message(role="user", content=prompt)]
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
                (),
                "tinker",
                transient_check=_is_jwt_401,
            )

            parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
            content = parsed_message["content"]
            reasoning = ""
            answer = ""
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "thinking":
                        reasoning += part["thinking"]
                    elif part["type"] == "text":
                        answer += part["text"]
            elif isinstance(content, str):
                answer = content
            return {"reasoning": reasoning, "answer": answer, "prompt": prompt}

        return send

    elif backend == "gemini":
        import httpx
        from google import genai
        from google.genai import types as genai_types
        from google.genai import errors as genai_errors

        client = genai.Client(http_options=genai_types.HttpOptions(timeout=1800 * 1000))
        transient = (
            genai_errors.ServerError,
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ProtocolError,
        )

        def _is_transient_gemini(e):
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code in (429, 500, 502, 503, 504):
                return True
            msg = str(e)
            return any(s in msg for s in (
                "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED",
            ))

        def _metadata_dict(obj):
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json", exclude_none=True)
            return dict(obj)

        def _enum_name(value):
            if value is None:
                return None
            return getattr(value, "name", None) or str(value)

        def _collect_gemini_parts(candidate):
            reasoning_parts = []
            answer_parts = []
            parts = (candidate.content.parts or []) if candidate.content else []
            for part in parts:
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    reasoning_parts.append(text)
                else:
                    answer_parts.append(text)
            return reasoning_parts, answer_parts

        def _collect_gemini_stream(prompt, config):
            reasoning_parts = []
            answer_parts = []
            finish_reason = None
            usage_metadata = None

            for chunk in client.models.generate_content_stream(
                model=model["model"],
                contents=prompt,
                config=config,
            ):
                usage_metadata = getattr(chunk, "usage_metadata", None) or usage_metadata
                if not getattr(chunk, "candidates", None):
                    continue
                candidate = chunk.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None) or finish_reason
                chunk_reasoning, chunk_answer = _collect_gemini_parts(candidate)
                reasoning_parts.extend(chunk_reasoning)
                answer_parts.extend(chunk_answer)

            return {
                "reasoning": "".join(reasoning_parts),
                "answer": "".join(answer_parts),
                "prompt": prompt,
                "finish_reason": _enum_name(finish_reason),
                "usage_metadata": _metadata_dict(usage_metadata),
            }

        def _collect_gemini_response(prompt, config):
            response = client.models.generate_content(
                model=model["model"],
                contents=prompt,
                config=config,
            )

            candidate = response.candidates[0]
            reasoning_parts, answer_parts = _collect_gemini_parts(candidate)
            return {
                "reasoning": "".join(reasoning_parts),
                "answer": "".join(answer_parts),
                "prompt": prompt,
                "finish_reason": _enum_name(getattr(candidate, "finish_reason", None)),
                "usage_metadata": _metadata_dict(
                    getattr(response, "usage_metadata", None)
                ),
            }

        def send(prompt):
            thinking_kwargs = {"include_thoughts": True}
            if model.get("thinking_level"):
                thinking_kwargs["thinking_level"] = model["thinking_level"]
            if model.get("thinking_budget") is not None:
                if model.get("thinking_level"):
                    raise ValueError(
                        "Gemini models cannot set both thinking_level and "
                        "thinking_budget"
                    )
                thinking_kwargs["thinking_budget"] = model["thinking_budget"]

            config = genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(**thinking_kwargs),
                temperature=model["temperature"],
                max_output_tokens=model["max_tokens"],
            )

            collect = (
                _collect_gemini_stream if model.get("stream", True)
                else _collect_gemini_response
            )
            return _retry(
                lambda: collect(prompt, config),
                transient, "gemini",
                transient_check=_is_transient_gemini,
            )

        return send

    elif backend == "openai":
        import openai
        from openai import OpenAI
        # OPENAI_TIMEOUT env var lets a caller tighten the budget for short-
        # answer workloads (e.g. graders) where a stalled request isn't worth
        # waiting 10 min on.
        client = OpenAI(timeout=float(os.environ.get("OPENAI_TIMEOUT", "600")))
        # Note: APITimeoutError is intentionally NOT retried — a stalled
        # generation just means the per-call budget was too small, and
        # retrying hits the same wall. We treat it as a dropped sample
        # (see catch below) instead. APITimeoutError subclasses
        # APIConnectionError, so to exclude it we drive _retry purely from
        # a transient_check rather than the tuple.
        transient = ()

        def _is_transient_openai(e):
            if isinstance(e, openai.APITimeoutError):
                return False
            return isinstance(
                e,
                (
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                ),
            )

        def send(prompt):
            kwargs = dict(
                model=model["model"],
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=model["max_tokens"],
                temperature=model["temperature"],
            )
            safety_id = os.environ.get("SAFETY_IDENTIFIER")
            if safety_id:
                kwargs["safety_identifier"] = safety_id
            reasoning_cfg = {}
            if model.get("reasoning_effort"):
                reasoning_cfg["effort"] = model["reasoning_effort"]
            if model.get("reasoning_summary"):
                reasoning_cfg["summary"] = model["reasoning_summary"]
            if reasoning_cfg:
                kwargs["reasoning"] = reasoning_cfg

            # OpenAI input safety filter rejects some bio/CBRN-adjacent
            # prompts with a 400 BadRequestError (code='invalid_prompt')
            # even on the no-safeguards key. We don't want one bad prompt
            # to kill a 100-prompt batch — surface a sentinel response
            # instead so downstream parsing flags it as a parse-failure.
            try:
                response = _retry(
                    lambda: client.responses.create(**kwargs),
                    transient, "openai",
                    transient_check=_is_transient_openai,
                )
            except openai.APITimeoutError:
                return {
                    "reasoning": "",
                    "answer": "[DROPPED_API_TIMEOUT]",
                    "prompt": prompt,
                    "blocked": True,
                }
            except openai.BadRequestError as e:
                code = getattr(e, "code", None)
                msg = str(e)
                if code == "invalid_prompt" or "limited access to this content" in msg:
                    return {
                        "reasoning": "",
                        "answer": "[BLOCKED_BY_OPENAI_INPUT_FILTER]",
                        "prompt": prompt,
                        "blocked": True,
                    }
                raise

            reasoning_items = [i for i in response.output if i.type == "reasoning"]
            message_items = [i for i in response.output if i.type == "message"]
            # gpt-5.6 returns several reasoning items per response (models
            # before it returned at most one here — the old code raised on
            # multi-item responses and never fired); concatenate in output
            # order. Single-item responses parse exactly as before.
            summary_texts = []
            for item in reasoning_items:
                if hasattr(item, "summary") and item.summary:
                    summary_texts.extend(
                        s.text for s in item.summary if hasattr(s, "text")
                    )
            reasoning = "\n".join(summary_texts)
            answer = ""
            for item in message_items:
                for part in item.content:
                    if part.type == "output_text":
                        answer += part.text
            return {"reasoning": reasoning, "answer": answer, "prompt": prompt}

        return send

    else:
        raise ValueError(f"Unknown backend: {backend}")


# --- Running ---

def _run_prompts(sender, max_concurrent, tasks, progress=None, semaphore=None):
    """Run `tasks` (prompt strings) through `sender` and return results in
    INPUT ORDER. Duplicate prompts in `tasks` are sent as independent calls
    — callers that want dedup must do it themselves before calling."""
    def guarded_send(prompt):
        if semaphore is not None:
            with semaphore:
                return sender(prompt)
        return sender(prompt)

    results = [None] * len(tasks)
    bar = tqdm(total=len(tasks)) if progress is None else None
    pbar = bar if bar is not None else progress
    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_idx = {
                executor.submit(guarded_send, p): i
                for i, p in enumerate(tasks)
            }
            try:
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()
                    pbar.update(1)
            except BaseException:
                # Cancel queued (not-yet-started) tasks so the surrounding
                # `with` doesn't grind through every remaining prompt
                # before the error surfaces. In-flight tasks still run
                # to completion — Python can't interrupt blocking I/O.
                for f in future_to_idx:
                    f.cancel()
                raise
    finally:
        if bar is not None:
            bar.close()
    return results


# --- Pipeline stages ---

def _infer_judge_backend(judge_model):
    """Pick a `_create_sender` backend from a judge model name.

    Currently only OpenAI models (``gpt-*`` / ``o1*`` / ``o3*`` / ``o4*``) are
    wired; everything else raises so the caller has to make an explicit
    decision (either extend this function or fall back to
    `shared.old_llmcomp_apply_judge.apply_judge`).
    """
    name = judge_model.lower()
    if name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"apply_judge: cannot infer backend for judge_model={judge_model!r}. "
        "Only OpenAI models are supported by the new path. For Claude / "
        "Gemini / etc., extend _infer_judge_backend or call "
        "shared.old_llmcomp_apply_judge.apply_judge."
    )


def apply_judge(df, judge_prompt, judged_column, judge_name,
                judge_model="gpt-4.1", cache_only=False, *, cache_dir=None):
    """Run a judge model over ``df[judged_column]`` and write raw outputs to
    ``df[judge_name]`` in place.

    Direct replacement for the legacy llmcomp version at
    ``shared.old_llmcomp_apply_judge.apply_judge`` (same signature, same
    observable result), but routes calls through `_create_sender` and caches
    answers in
    ``<cache_dir>/<judge_config_hash>/<shard>.jsonl``
    via `shared.judge_jsonl_cache.JsonlJudgeCache`. ``judge_config_hash``
    forks per (judge_prompt, model, temperature, max_tokens,
    reasoning_effort), so editing any of those auto-forks a fresh cache dir.

    ``judge_name`` is the OUTPUT column name in ``df`` only — it no longer
    affects caching (the JsonlJudgeCache content-keys by rendered prompt, so
    the same (prompt, answer text) hits across different judge_name values).

    ``cache_dir`` defaults to ``APPLY_JUDGE_CACHE_ROOT`` (read at call time so
    notebooks can redirect it like ``runner.APPLY_JUDGE_CACHE_ROOT = ...``).

    ``judge_model`` currently must be an OpenAI model identifier (``gpt-*``,
    ``o1*``, etc.). Other backends raise; extend `_infer_judge_backend` or
    fall back to ``shared.old_llmcomp_apply_judge.apply_judge``.

    ``cache_only=True`` raises `CacheOnlyMiss` on the first miss with shard
    path info.
    """
    if judge_name in ("__paraphrase__", "__judge_q__"):
        raise ValueError(f"judge_name {judge_name!r} is reserved for internal use")
    if "{llm_text}" not in judge_prompt:
        raise ValueError(
            "judge_prompt must contain '{llm_text}' placeholder; without it every "
            "row produces the same paraphrase and the judge returns the same label."
        )
    if cache_dir is None:
        cache_dir = APPLY_JUDGE_CACHE_ROOT

    judge_config = {
        "backend": _infer_judge_backend(judge_model),
        "model": judge_model,
        "max_tokens": 1024,
        "temperature": 0,
        "max_concurrent": 100,
    }

    cache = JsonlJudgeCache(cache_dir, judge_prompt, judge_config)

    rendered_per_row = [
        judge_prompt.format(llm_text=text)
        for text in df[judged_column].tolist()
    ]

    missing = []
    seen = set()
    for rendered in rendered_per_row:
        if cache.get(rendered) is not None:
            continue
        key = cache.key(rendered)
        if key in seen:
            continue
        seen.add(key)
        missing.append(rendered)

    if missing:
        if cache_only:
            sample_path = cache.shard_path(cache.key(missing[0]))
            n_miss_rows = sum(
                1 for r in rendered_per_row if cache.get(r) is None
            )
            raise CacheOnlyMiss(
                "Cache-only mode: apply_judge cache miss for "
                f"{n_miss_rows}/{len(df)} rows ({len(missing)} unique "
                f"prompt{'s' if len(missing) != 1 else ''}); "
                f"example shard: {sample_path}"
            )
        sender = _create_sender(judge_config)
        max_concurrent = judge_config["max_concurrent"]
        write_lock = threading.Lock()
        desc = f"Judge ({judge_config['model']})"
        bar = tqdm(total=len(missing), desc=desc)
        try:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {executor.submit(sender, r): r for r in missing}
                try:
                    for fut in as_completed(futures):
                        rendered = futures[fut]
                        result = fut.result()
                        # JsonlJudgeCache.append is internally thread-safe
                        # (fcntl + threading lock); we serialize at the call
                        # site too for tqdm consistency.
                        with write_lock:
                            cache.append(rendered, {"answer": result["answer"]})
                            bar.update(1)
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
        finally:
            bar.close()

    raw = [
        cache.get(rendered).get("answer") if cache.get(rendered) else None
        for rendered in rendered_per_row
    ]
    df[judge_name] = raw


_ESTIMATE_TAG_RE = re.compile(r"<final_estimate>\s*(.*?)\s*</final_estimate>", re.DOTALL)


def _parse_tagged_estimate(raw):
    """Parse `<final_estimate>N</final_estimate>` / `<final_estimate>UNKNOWN</final_estimate>`
    outputs.

    Returns a float or None. None for: missing tag, `UNKNOWN`, empty content,
    or anything that doesn't parse as a number.
    """
    if not isinstance(raw, str):
        return None
    m = _ESTIMATE_TAG_RE.search(raw)
    if not m:
        return None
    content = m.group(1).strip()
    if not content or content.upper() == "UNKNOWN":
        return None
    try:
        return float(content.replace(",", ""))
    except ValueError:
        return None


def _compute_thresholds(estimates, spec):
    valid = estimates.dropna()
    if len(valid) == 0:
        raise ValueError(
            f"Cannot compute thresholds: none of the {len(estimates)} baseline "
            "estimates parsed (judge returned UNKNOWN or unparseable text for "
            "every baseline answer)."
        )
    lo_pct, hi_pct = spec["pct_range"]
    n = spec["n"]
    lo = float(np.percentile(valid, lo_pct))
    hi = float(np.percentile(valid, hi_pct))
    points = np.linspace(lo, hi, n)
    return [int(round(p)) for p in points]


# --- Main entry point ---

def _get_baseline(prompt_key, model, n_baseline, model_name, sender,
                  max_concurrent, progress=None, semaphore=None,
                  cache_only=False):
    """Run or load cached baseline for one prompt key.

    Returns (baseline_df, was_cached).
    """
    prompt_set = THRESHOLD_PROMPTS[prompt_key]
    baseline_prompt = _with_prompt_suffix(model, prompt_set["baseline"])
    b_hash = _prompt_hash(model, n_baseline, baseline_prompt)
    # A prompt set may share its baseline with another prompt key (e.g. wording
    # variants whose baseline is byte-identical to the plain task baseline).
    # `baseline_key` defaults to the prompt's own key, preserving prior behavior.
    baseline_key = prompt_set.get("baseline_key", prompt_key)
    b_path = _cache_path(model_name, baseline_key, "baseline", b_hash)
    cached = _read_cache(b_path, b_hash)
    if cached is not None:
        baseline_rows = cached
        if progress is not None:
            progress.update(len(cached))
            # Force a redraw: tqdm's mininterval=0.1s throttle suppresses
            # the visible update for bursts that happen right after the
            # bar is created (which is exactly the cache-hit pattern).
            # Without this the bar appears stuck at 0% until the first
            # actual API call completes.
            progress.refresh()
        was_cached = True
    else:
        if cache_only:
            raise CacheOnlyMiss(
                "Cache-only mode: model output cache miss for "
                f"model={model_name!r}, prompt_key={prompt_key!r}, "
                f"direction='baseline'; expected {b_path}"
            )
        tasks = [baseline_prompt] * n_baseline
        baseline_rows = _run_prompts(sender, max_concurrent, tasks,
                                     progress=progress, semaphore=semaphore)
        _write_cache(b_path, {
            "hash": b_hash,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "direction": "baseline",
            "n": n_baseline,
        }, baseline_rows)
        was_cached = False
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df["direction"] = "baseline"
    return baseline_df, was_cached


def batch_extract_estimates(df, experiment_name, *, column="answer",
                            judge_config=None, cache_only=False):
    """Extract numeric estimates for a DataFrame with multiple prompt keys.

    Each row's ``prompt_key`` selects a judge template from THRESHOLD_PROMPTS
    (typically the number or days template). Each unique rendered prompt is
    judged once, with results cached in
    ``<ESTIMATE_JUDGE_CACHE_ROOT>/<experiment_name>/<judge_config_hash>/<shard>.jsonl``
    via `shared.judge_jsonl_cache.JsonlJudgeCache`. The judge_config_hash
    forks per judge template, so the number / days templates live in separate
    subdirectories side by side.

    Misses are sent to the judge via `_create_sender(judge_config)` (no
    llmcomp), one POST per unique prompt, fanned out across
    `judge_config["max_concurrent"]` threads. Each result is appended to its
    shard as soon as it lands so a crash mid-run leaves usable partial state.

    `cache_only=True` raises `CacheOnlyMiss` on the first miss with shard path
    info — use this in scripts that produce published figures.

    Returns a pd.Series of parsed float estimates aligned with ``df.index``
    (NaN for UNKNOWN / unparseable judge output).
    """
    if judge_config is None:
        judge_config = ESTIMATE_JUDGE_CONFIG

    cache_dir = estimate_judge_cache_dir(experiment_name)

    # Build (template, rendered_prompt) per row and a per-template cache view.
    rendered_per_row = []
    caches = {}
    for _, row in df.iterrows():
        template = THRESHOLD_PROMPTS[row["prompt_key"]]["judge_prompt"]
        rendered = template.format(llm_text=row[column])
        rendered_per_row.append((template, rendered))
        if template not in caches:
            caches[template] = JsonlJudgeCache(cache_dir, template, judge_config)

    # Collect unique misses, keyed by (template_id, prompt_hash) so two
    # templates that happen to render identical text (shouldn't, but just in
    # case) don't collide.
    missing = []  # [(template, rendered, cache)]
    seen = set()
    for template, rendered in rendered_per_row:
        cache = caches[template]
        if cache.get(rendered) is not None:
            continue
        key = (id(cache), cache.key(rendered))
        if key in seen:
            continue
        seen.add(key)
        missing.append((template, rendered, cache))

    if missing:
        if cache_only:
            sample_path = missing[0][2].shard_path(
                missing[0][2].key(missing[0][1]))
            n_miss_rows = sum(
                1 for t, r in rendered_per_row
                if caches[t].get(r) is None
            )
            raise CacheOnlyMiss(
                "Cache-only mode: estimate-judge cache miss for "
                f"{n_miss_rows}/{len(df)} rows ({len(missing)} unique "
                f"prompt{'s' if len(missing) != 1 else ''}); "
                f"example shard: {sample_path}"
            )
        sender = _create_sender(judge_config)
        max_concurrent = judge_config["max_concurrent"]
        write_lock = threading.Lock()
        desc = f"Estimate judge ({judge_config['model']})"
        bar = tqdm(total=len(missing), desc=desc)

        def judge_one(item):
            template, rendered, cache = item
            result = sender(rendered)
            return template, rendered, cache, result["answer"]

        try:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = [executor.submit(judge_one, it) for it in missing]
                try:
                    for fut in as_completed(futures):
                        template, rendered, cache, answer = fut.result()
                        # JsonlJudgeCache.append is internally thread-safe
                        # (fcntl + threading lock) but we serialize at the
                        # call site too for tqdm consistency.
                        with write_lock:
                            cache.append(rendered, {"answer": answer})
                            bar.update(1)
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
        finally:
            bar.close()

    raw = []
    for template, rendered in rendered_per_row:
        entry = caches[template].get(rendered)
        raw.append(entry.get("answer") if entry is not None else None)
    return pd.Series(raw, index=df.index).apply(_parse_tagged_estimate)


def _run_directions_for_prompt(prompt_key, model, n_per_threshold, thresholds,
                               model_name, sender, max_concurrent, progress,
                               baseline_df, semaphore, cache_only=False):
    """Run the direction sweep for a single prompt key.

    `baseline_df` and `thresholds` are precomputed by the caller. Returns
    (combined_df, cached_phases) where ``cached_phases`` lists the directions
    that were loaded from the on-disk cache.
    """
    prompt_set = THRESHOLD_PROMPTS[prompt_key]
    cached_phases = []

    # Read both direction caches up front so a cache hit on one direction
    # bumps progress immediately rather than waiting behind the other
    # direction's API calls.
    plans = []  # (direction, template, d_hash, d_path, cached_rows_or_None)
    for template_key in ("below_good_template", "above_good_template"):
        template = prompt_set[template_key]
        template_for_hash = _with_prompt_suffix(model, template)
        direction = template_key.replace("_template", "")
        d_hash = _direction_hash(model, n_per_threshold, template_for_hash, thresholds)
        d_path = _cache_path(model_name, prompt_key, direction, d_hash)
        cached = _read_cache(d_path, d_hash)
        if cached is not None:
            cached_phases.append(direction)
            if progress is not None:
                progress.update(len(cached))
                # Force a redraw: see _get_baseline for the rationale.
                # tqdm's default mininterval throttle swallows cache-hit
                # bursts that all land within ~100ms of bar creation.
                progress.refresh()
        plans.append((direction, template, d_hash, d_path, cached))

    direction_dfs = []
    for direction, template, d_hash, d_path, cached in plans:
        if cached is not None:
            dir_rows = cached
        else:
            if cache_only:
                raise CacheOnlyMiss(
                    "Cache-only mode: model output cache miss for "
                    f"model={model_name!r}, prompt_key={prompt_key!r}, "
                    f"direction={direction!r}; expected {d_path}"
                )
            prompt_to_threshold = {}
            tasks = []
            for t in thresholds:
                pt = _with_prompt_suffix(
                    model,
                    template.format(threshold=f"{t:,}"),
                )
                prompt_to_threshold[pt] = t
                tasks.extend([pt] * n_per_threshold)
            dir_rows = _run_prompts(sender, max_concurrent, tasks,
                                    progress=progress, semaphore=semaphore)
            for r in dir_rows:
                r["threshold"] = prompt_to_threshold[r["prompt"]]
            _write_cache(d_path, {
                "hash": d_hash,
                "model_name": model_name,
                "prompt_key": prompt_key,
                "direction": direction,
                "n_per_threshold": n_per_threshold,
                "thresholds": thresholds,
            }, dir_rows)
        df = pd.DataFrame(dir_rows)
        df["direction"] = direction
        direction_dfs.append(df)

    combined_df = pd.concat([baseline_df] + direction_dfs, ignore_index=True)
    combined_df["prompt_key"] = prompt_key
    return combined_df, cached_phases


def _lazy_sender(model):
    """Sender that defers `_create_sender` to the first actual send.

    Backend-client creation can itself hit the network (Tinker checks billing
    status), which would fail fully-cached runs that never need to sample.
    Thread-safe: worker threads share one underlying sender.
    """
    lock = threading.Lock()
    real = None

    def send(prompt):
        nonlocal real
        if real is None:
            with lock:
                if real is None:
                    real = _create_sender(model)
        return real(prompt)

    return send


def run_thresholds_experiment(model, experiment, model_name, experiment_name,
                              cache_only=False):
    """Run a sycophancy/manipulation probe by sweeping thresholds.

    ``experiment["prompts"]`` is a list of one or more prompt-set keys.

    Returns ``(combined_df, thresholds, display_name)``. ``combined_df`` has
    columns ``reasoning, answer, prompt, direction, threshold, prompt_key,
    estimate``; baseline rows have NaN in ``threshold`` and direction rows
    have NaN in ``estimate`` only if the judge could not parse them.
    ``thresholds`` is a dict mapping each prompt key to its threshold list.
    """
    n_baseline = experiment["n_baseline"]
    n_per_threshold = experiment["n_per_threshold"]
    threshold_spec = experiment["thresholds"]
    max_concurrent = model["max_concurrent"]
    prompt_list = list(experiment["prompts"])

    sender = None if cache_only else _lazy_sender(model)
    display_name = model.get("display_name", model.get("model") or model.get("model_path"))

    n_thresholds = threshold_spec["n"]
    baseline_total = len(prompt_list) * n_baseline
    direction_total = len(prompt_list) * 2 * n_per_threshold * n_thresholds
    total = baseline_total + direction_total
    total_label = "cache rows" if cache_only else "requests"
    mode_label = " (cache-only)" if cache_only else ""
    print(f"Running {display_name}{mode_label}: {len(prompt_list)} prompts, "
          f"{total} total {total_label}")
    semaphore = threading.Semaphore(max_concurrent)

    # Phase 1: baselines (parallel). Its own tqdm bar so it never overlaps
    # with the estimate-judge bar in Phase 2 — having two live tqdm
    # instances at once leaves only one of them visible.
    baselines = {}
    baseline_cached = {}
    progress = tqdm(total=baseline_total, desc=f"{display_name} baselines")
    with ThreadPoolExecutor(max_workers=len(prompt_list)) as executor:
        futures = {
            executor.submit(
                _get_baseline, pk, model, n_baseline, model_name,
                sender, max_concurrent, progress, semaphore, cache_only,
            ): pk
            for pk in prompt_list
        }
        for future in as_completed(futures):
            pk = futures[future]
            baselines[pk], baseline_cached[pk] = future.result()
    progress.close()

    # Phase 2: extract baseline estimates (single batched judge call) and
    # compute per-prompt thresholds. Estimates are written onto each
    # baseline_df so they survive the concat in Phase 4.
    print("Extracting baseline estimates (judge)...")
    # No ignore_index here: each baselines[pk] keeps its own 0..n-1 index, so
    # we can align estimates back to baselines[pk] purely by index below.
    combined_baselines = pd.concat(
        [b.assign(prompt_key=pk) for pk, b in baselines.items()],
    )
    baseline_estimates = batch_extract_estimates(
        combined_baselines, experiment_name, cache_only=cache_only,
    )
    all_thresholds = {}
    for pk in prompt_list:
        pk_estimates = baseline_estimates[combined_baselines["prompt_key"] == pk]
        baselines[pk]["estimate"] = pk_estimates
        all_thresholds[pk] = _compute_thresholds(pk_estimates, threshold_spec)

    # Phase 3: directions (parallel). Fresh tqdm — see Phase 1 comment.
    all_dfs = []
    all_cached = {}
    progress = tqdm(total=direction_total, desc=f"{display_name} directions")
    with ThreadPoolExecutor(max_workers=len(prompt_list)) as executor:
        futures = {
            executor.submit(
                _run_directions_for_prompt, pk, model, n_per_threshold,
                all_thresholds[pk], model_name, sender, max_concurrent,
                progress, baselines[pk], semaphore, cache_only,
            ): pk
            for pk in prompt_list
        }
        for future in as_completed(futures):
            pk = futures[future]
            df, cached_phases = future.result()
            if baseline_cached[pk]:
                cached_phases = ["baseline"] + cached_phases
            all_dfs.append(df)
            if cached_phases:
                all_cached[pk] = cached_phases
    progress.close()

    if all_cached:
        parts = [f"{pk}: {', '.join(phases)}"
                 for pk, phases in sorted(all_cached.items())]
        print(f"Loaded from cache: {'; '.join(parts)}")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Phase 4: extract direction estimates (single batched judge call).
    # Baselines were already judged in Phase 2 and carry their estimate
    # through the concat above, so this only judges direction rows.
    print("Extracting direction estimates (judge)...")
    direction_mask = combined_df["direction"] != "baseline"
    direction_estimates = batch_extract_estimates(
        combined_df[direction_mask], experiment_name,
        cache_only=cache_only,
    )
    combined_df.loc[direction_mask, "estimate"] = direction_estimates

    return combined_df, all_thresholds, display_name
