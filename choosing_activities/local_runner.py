"""Experiment-local runner helpers.

These two pieces used to live as local patches on top of ``shared.runner``; they
are kept here instead so ``shared/`` stays untouched by the activity-preferences
work. Everything they rely on (``_create_sender``, ``_infer_judge_backend``,
``CacheOnlyMiss``, ``JsonlJudgeCache``, ``MODELS``) is imported read-only from the
shared package; nothing in ``shared/`` is modified. The judge cache root is
defined locally (``APPLY_JUDGE_CACHE_ROOT`` below) and points at the repo's
``final_data/apply_judge_cache/activity_preferences/`` so judge outputs live with
the other released caches under ``final_data/`` rather than inside ``shared/``
(``shared.runner.APPLY_JUDGE_CACHE_ROOT`` still defaults to ``shared/`` for other
callers; only this experiment is redirected).

Contents:
  - ``_errored_rows``: filter sender rows that recorded an ``error`` so callers
    can skip caching a batch that contains failures (otherwise a transient API
    blip on a flaky backend gets baked into the jsonl cache forever).
  - ``apply_judge``: a drop-in for ``shared.runner.apply_judge`` with two
    additions this experiment needs — (1) it resolves a ``judge_model`` that is
    registered in ``shared.models.MODELS`` to that model's *full* config (so a
    reasoning judge keeps its real ``max_tokens``/reasoning settings instead of
    the legacy hard-coded ``max_tokens=1024`` that truncated traces into spurious
    refusals), and (2) it persists the judge's own reasoning trace and exposes it
    as a parallel ``{judge_name}_reasoning`` column for auditing verdicts.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from shared.judge_jsonl_cache import JsonlJudgeCache
from shared.models import MODELS
from shared.runner import (
    CacheOnlyMiss,
    _create_sender,
    _infer_judge_backend,
    _retry,
)

# Judge outputs cache under final_data/ (alongside the other released caches),
# not inside shared/. Defined here -- not imported from shared.runner, whose
# default still points at shared/apply_judge_cache for other callers -- so the
# activity-preferences pipeline writes to the released location while shared/
# stays untouched. __file__ = harry/activity_preferences/local_runner.py, so
# parents[2] is the repo root.
APPLY_JUDGE_CACHE_ROOT = str(
    Path(__file__).resolve().parents[1]
    / "data" / "final_data" / "apply_judge_cache" / "activity_preferences"
)


class RequestThrottle:
    """Thread-safe gate that pauses all workers after every ``every`` calls.

    For providers with a low request cap (e.g. the Gemini free tier ~1000
    requests/min): call ``acquire()`` once just before each main-model API call,
    and after every ``every`` acquisitions the throttle makes the issuing worker —
    and any others that reach the gate during the window — sleep until
    ``pause_seconds`` have elapsed. With a thread pool this effectively caps
    throughput at roughly ``every`` requests per ``pause_seconds``.

    Disabled (a true no-op, zero overhead) when ``pause_seconds <= 0`` or
    ``every <= 0`` — which is the default for every backend except the
    rate-limited Gemini models. Cache hits never call ``acquire()`` (the senders
    are only built on a cache miss), so cached reruns are never throttled.

    Counting is process-global per instance, so one shared instance correctly
    bounds the rate across all tool settings / scripts in a single run.
    """

    def __init__(self, every: int, pause_seconds: float):
        self.every = int(every)
        self.pause_seconds = float(pause_seconds)
        self.enabled = self.every > 0 and self.pause_seconds > 0
        self._lock = threading.Lock()
        self._count = 0
        self._resume_at = 0.0

    def acquire(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._count += 1
            if self._count % self.every == 0:
                self._resume_at = time.monotonic() + self.pause_seconds
                tqdm.write(
                    f"[rate limit] {self._count} requests issued; pausing "
                    f"{self.pause_seconds / 60:.0f} min to stay under the provider cap."
                )
            resume_at = self._resume_at
        wait = resume_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def _errored_rows(results):
    """Rows whose sender recorded an ``error`` (transient API failure that
    survived retries, a non-transient error, max-tool-turns, etc.).

    Callers use this to avoid caching a batch that contains failures: a cached
    error is silly — every later run would be served the same dead row instead
    of re-sampling it. Skip the cache write while any row is errored; once a run
    completes clean it caches normally.
    """
    return [r for r in results if r.get("error")]


def _create_judge_sender(judge_config):
    """Judge sender that tolerates multi-item OpenAI Responses output.

    ``shared.runner._create_sender``'s OpenAI branch raises ``NotImplementedError``
    when a response carries more than one ``reasoning`` or ``message`` item — the
    Responses API can segment/interleave output as reasoning, message, reasoning,
    message, and that one judge call then kills the whole batch via ``fut.result()``.

    For the OpenAI backend we use a local copy of that sender, identical except it
    COMBINES the items in order (join every reasoning summary, concatenate every
    message's output_text) instead of bailing — the same thing ``pipeline.py``'s
    ``_extract_openai_text`` already does for the main model. Every other backend
    delegates to the shared sender unchanged, so ``shared/`` stays untouched.
    """
    if judge_config.get("backend") != "openai":
        return _create_sender(judge_config)

    import openai
    from openai import OpenAI

    model = judge_config
    client = OpenAI(timeout=float(os.environ.get("OPENAI_TIMEOUT", "600")))
    # Mirror shared: APITimeoutError is NOT transient (subclasses
    # APIConnectionError), so drive _retry from transient_check only.
    transient = ()

    def _is_transient_openai(e):
        if isinstance(e, openai.APITimeoutError):
            return False
        return isinstance(
            e,
            (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError),
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

        try:
            response = _retry(
                lambda: client.responses.create(**kwargs),
                transient, "openai-judge",
                transient_check=_is_transient_openai,
            )
        except openai.APITimeoutError:
            return {"reasoning": "", "answer": "[DROPPED_API_TIMEOUT]",
                    "prompt": prompt, "blocked": True}
        except openai.BadRequestError as e:
            code = getattr(e, "code", None)
            msg = str(e)
            if code == "invalid_prompt" or "limited access to this content" in msg:
                return {"reasoning": "", "answer": "[BLOCKED_BY_OPENAI_INPUT_FILTER]",
                        "prompt": prompt, "blocked": True}
            raise

        # Combine ALL items of each type, in output order (the only change vs the
        # shared sender, which raises on >1 of either).
        reasoning_parts = []
        for item in (i for i in response.output if i.type == "reasoning"):
            if getattr(item, "summary", None):
                reasoning_parts.extend(s.text for s in item.summary if hasattr(s, "text"))
        answer = ""
        for item in (i for i in response.output if i.type == "message"):
            for part in item.content:
                if getattr(part, "type", None) == "output_text":
                    answer += part.text
        return {"reasoning": "\n".join(reasoning_parts), "answer": answer, "prompt": prompt}

    return send


def apply_judge(df, judge_prompt, judged_column, judge_name,
                judge_model="gpt-4.1", cache_only=False, *, cache_dir=None):
    """Run a judge model over ``df[judged_column]`` and write raw outputs to
    ``df[judge_name]`` in place.

    Same observable contract as ``shared.runner.apply_judge`` (routes through
    ``_create_sender``, caches in
    ``<cache_dir>/<judge_config_hash>/<shard>.jsonl`` via
    ``shared.judge_jsonl_cache.JsonlJudgeCache``; ``judge_config_hash`` forks per
    (judge_prompt, model, temperature, max_tokens, reasoning_effort) so editing
    any of those auto-forks a fresh cache dir), with two experiment-local
    additions:

    1. ``judge_model`` is resolved against ``shared.models.MODELS`` when
       registered there (inheriting that model's backend, max_tokens, reasoning
       settings, …); an unregistered string falls back to a minimal OpenAI config
       and must be an OpenAI identifier (``gpt-*``, ``o1*``, …) or
       ``_infer_judge_backend`` raises. This stops a reasoning judge from being
       truncated by the legacy hard-coded ``max_tokens=1024``.
    2. The judge's own reasoning trace is persisted alongside its answer and
       surfaced as a parallel ``df[f"{judge_name}_reasoning"]`` column. Additive:
       cache entries written without it simply read back as empty.

    ``judge_name`` is the OUTPUT column name in ``df`` only — it does not affect
    caching (the cache content-keys by rendered prompt). ``cache_dir`` defaults to
    ``APPLY_JUDGE_CACHE_ROOT``. ``cache_only=True`` raises ``CacheOnlyMiss`` on
    the first miss with shard path info.
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

    # Judge sampling config is the model's own definition from shared.models.MODELS
    # verbatim — same backend / max_tokens / temperature / reasoning settings as
    # everywhere else. Only max_concurrent is added, an operational thread-pool
    # knob for the judge that isn't part of a model definition. Model strings not
    # registered in MODELS fall back to the previous minimal OpenAI config, so
    # ad-hoc callers are unaffected.
    if judge_model in MODELS:
        judge_config = {**MODELS[judge_model], "max_concurrent": 100}
    else:
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
        sender = _create_judge_sender(judge_config)
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
                            cache.append(rendered, {
                                "answer": result["answer"],
                                # Persist the judge's own reasoning trace too, so
                                # callers can review *why* a judge ruled as it did
                                # (the judge is itself a reasoning model). Additive:
                                # pre-existing cache entries simply lack this key.
                                "reasoning": result.get("reasoning", ""),
                            })
                            bar.update(1)
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
        finally:
            bar.close()

    cached_per_row = [cache.get(rendered) for rendered in rendered_per_row]
    df[judge_name] = [c.get("answer") if c else None for c in cached_per_row]
    # Parallel column with the judge's reasoning trace (empty for older cache
    # entries written before reasoning was captured). Additive — existing callers
    # that only read df[judge_name] are unaffected.
    df[f"{judge_name}_reasoning"] = [
        (c.get("reasoning") if c else None) for c in cached_per_row
    ]
