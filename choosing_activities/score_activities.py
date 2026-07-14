"""Score every activity with the model being evaluated in pipeline.py.

Each activity from activities.yaml is sent to the same shared-runner model
configuration used by pipeline.py. The model returns a 0-100 liking score,
which is parsed and saved to
results/<MODEL_KEY>/activity_preferences/activity_liking_scores.csv.

To add wording diversity (and a more robust per-activity mean), each sample
draws one of several semantically-similar prompts from
prompts/activity_liking_prompt.yaml. The draw is balanced across the N repeats
and seeded, so every prompt variant is represented roughly equally and the run
is deterministic. With a single prompt in the YAML this collapses to the old
behaviour (the same prompt repeated N times).

Run:
    uv run python choosing_activities/score_activities.py
"""

from __future__ import annotations

from collections import defaultdict, deque
import argparse
import json
import pathlib
import random
import re
import sys
import threading

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

load_dotenv(REPO_ROOT / ".env")

from shared.runner import (  # noqa: E402
    CacheOnlyMiss,
    _create_sender,
    _hash,
    _model_hashable,
    _read_cache,
    _retry,
    _run_prompts,
    _write_cache,
)

# Experiment-local helper (kept out of shared/ — see local_runner.py).
try:
    from .local_runner import _errored_rows
except ImportError:  # pragma: no cover - script execution path
    from local_runner import _errored_rows

# pipeline.py loads config.yaml; the knobs (and shared paths/helpers) come from it.
try:
    from .pipeline import (
        CACHE_ONLY,
        CACHE_ROOT,
        CLIENT_MAX_RETRIES,
        MAIN_MODEL,
        MAIN_THROTTLE,
        MODEL_KEY,
        N_REPEATS,
        PROMPT_SELECTION_SEED,
        REFRESH_CACHE,
        REQUEST_TIMEOUT_SECONDS,
        _create_openrouter_sender,
        result_path,
    )
except ImportError:  # pragma: no cover - script execution path
    from pipeline import (
        CACHE_ONLY,
        CACHE_ROOT,
        CLIENT_MAX_RETRIES,
        MAIN_MODEL,
        MAIN_THROTTLE,
        MODEL_KEY,
        N_REPEATS,
        PROMPT_SELECTION_SEED,
        REFRESH_CACHE,
        REQUEST_TIMEOUT_SECONDS,
        _create_openrouter_sender,
        result_path,
    )


EXPERIMENT_NAME = "activity_liking_scores"
ACTIVITIES_PATH = HERE / "activities.yaml"
PROMPT_PATH = HERE / "prompts" / "activity_liking_prompt.yaml"
SCORE_CACHE_ROOT = CACHE_ROOT / "runner_activity_scores"

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_BARE_SCORE_RE = re.compile(r"(?<!\d)(100|[1-9]?\d)(?!\d)")


def load_activities(path: pathlib.Path = ACTIVITIES_PATH) -> list[dict]:
    """Load activities with a stable row index."""
    activities = yaml.safe_load(path.read_text())["activities"]
    return [
        {
            "activity_ix": i,
            "activity": activity["name"],
        }
        for i, activity in enumerate(activities)
    ]


def load_score_prompts(path: pathlib.Path = PROMPT_PATH) -> list[str]:
    """Load the list of semantically-similar score prompts.

    Accepts `score_prompts` (a list) or, for backward compatibility, a single
    `score_prompt` string. Every prompt must contain the `{activity}`
    placeholder so render_prompt can fill it.
    """
    data = yaml.safe_load(path.read_text())
    prompts = data.get("score_prompts")
    if prompts is None:
        single = data.get("score_prompt")
        if single is None:
            raise KeyError(
                f"{path} must define 'score_prompts' (list) or 'score_prompt'"
            )
        prompts = [single]
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    for i, prompt in enumerate(prompts):
        if "{activity}" not in prompt:
            raise ValueError(
                f"score_prompts[{i}] in {path} is missing the '{{activity}}' "
                "placeholder"
            )
    return prompts


def render_prompt(activity: dict, prompt_template: str) -> str:
    return prompt_template.format(activity=activity["activity"])


def _balanced_prompt_indices(
    n_repeats: int, n_prompts: int, rng: random.Random
) -> list[int]:
    """Assign a prompt index to each of n_repeats samples, balanced + shuffled.

    Each prompt is used either floor(n_repeats / n_prompts) or one more time, so
    coverage is as even as possible (e.g. 10 repeats over 5 prompts -> 2 each;
    20 -> 4 each). The shuffle randomizes which prompts get the extra sample when
    n_repeats is not a multiple of n_prompts, and the order within the activity.
    """
    reps = -(-n_repeats // n_prompts)  # ceil division
    pool = list(range(n_prompts)) * reps
    rng.shuffle(pool)
    return pool[:n_repeats]


def build_score_tasks(
    activities: list[dict],
    prompt_templates: list[str],
    *,
    n_repeats: int = N_REPEATS,
    seed: int = PROMPT_SELECTION_SEED,
) -> list[dict]:
    """Return one task per (activity, repeat) with a randomly-assigned prompt.

    For each activity, the n_repeats samples are spread across `prompt_templates`
    via a balanced, seeded shuffle (see _balanced_prompt_indices). The per-task
    `prompt_ix` records which variant was used so wording effects can be
    analysed downstream.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if not prompt_templates:
        raise ValueError("prompt_templates must contain at least one prompt")
    tasks = []
    n_prompts = len(prompt_templates)
    for activity in activities:
        # Seed per activity so an activity's assignment is stable regardless of
        # catalog position of the others.
        rng = random.Random(f"{seed}:{activity['activity_ix']}")
        prompt_ixs = _balanced_prompt_indices(n_repeats, n_prompts, rng)
        for repeat_ix, prompt_ix in enumerate(prompt_ixs):
            tasks.append({
                **activity,
                "repeat_ix": repeat_ix,
                "prompt_ix": prompt_ix,
                "prompt": render_prompt(activity, prompt_templates[prompt_ix]),
            })
    return tasks


def _score_cache_hash(
    prompts: list[str],
    prompt_templates: list[str],
    *,
    n_repeats: int,
    seed: int,
) -> str:
    return _hash({
        "experiment": EXPERIMENT_NAME,
        "model": _model_hashable(MAIN_MODEL),
        "prompt_templates": list(prompt_templates),
        "n_prompts": len(prompt_templates),
        "prompts": prompts,
        "n": len(prompts),
        "n_repeats": n_repeats,
        "seed": seed,
    })


def _score_cache_path(h: str) -> pathlib.Path:
    return SCORE_CACHE_ROOT / MODEL_KEY / f"{h}.jsonl"


def _align_rows_to_prompts(rows: list[dict], prompts: list[str]) -> list[dict]:
    """Return runner rows in input-prompt order after completion-order fan-in."""
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


def _create_activity_score_sender(model: dict):
    """Create the activity-score sender, with local timeout handling for OpenAI."""
    if model["backend"] == "openrouter":
        # Stated-liking scoring is tool-free; reuse the pipeline's OpenRouter
        # sender with no tools (kept harry-local, shared.runner untouched).
        return _create_openrouter_sender(model, [])
    if model["backend"] != "openai":
        return _create_sender(model)

    import openai
    from openai import OpenAI

    client = OpenAI(
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=CLIENT_MAX_RETRIES,
    )
    # Mirror shared.runner._create_sender: retry transient API failures instead
    # of letting a single blip become a permanent errored (empty-answer) row.
    transient = (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
    )

    def send(prompt: str) -> dict:
        kwargs = dict(
            model=model["model"],
            input=[{"role": "user", "content": prompt}],
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

        response = _retry(
            lambda: client.responses.create(**kwargs),
            transient, "openai-score",
        )
        reasoning, answer = _extract_openai_text(response)
        return {
            "reasoning": reasoning,
            "answer": answer,
            "prompt": prompt,
        }

    return send


def run_activity_scores(
    activities: list[dict],
    *,
    prompt_templates: list[str],
    n_repeats: int = N_REPEATS,
    seed: int = PROMPT_SELECTION_SEED,
    cache_only: bool = CACHE_ONLY,
    refresh: bool = REFRESH_CACHE,
) -> pd.DataFrame:
    """Run or load the evaluated model's activity-liking scores."""
    if refresh and cache_only:
        raise ValueError("refresh_cache and cache_only are mutually exclusive")
    tasks = build_score_tasks(
        activities,
        prompt_templates,
        n_repeats=n_repeats,
        seed=seed,
    )
    prompts = [task["prompt"] for task in tasks]
    h = _score_cache_hash(
        prompts, prompt_templates, n_repeats=n_repeats, seed=seed
    )
    path = _score_cache_path(h)
    cached = None if refresh else _read_cache(path, h)
    if cached is not None:
        print(f"[{MODEL_KEY}] score cache hit ({len(cached)} samples)")
        rows = cached
    else:
        if cache_only:
            raise CacheOnlyMiss(
                "Cache-only mode: activity-score cache miss for "
                f"model={MODEL_KEY!r}; expected {path}"
            )
        score_model = {
            **MAIN_MODEL,
        }
        base_sender = _create_activity_score_sender(score_model)

        def sender(prompt: str) -> dict:
            MAIN_THROTTLE.acquire()  # no-op unless a per-minute rate limit is set
            try:
                return base_sender(prompt)
            except Exception as e:
                return {
                    "reasoning": "",
                    "answer": "",
                    "prompt": prompt,
                    "error": f"{type(e).__name__}: {e}",
                }

        semaphore = threading.Semaphore(score_model["max_concurrent"])
        progress = tqdm(
            total=len(prompts),
            desc=f"{MODEL_KEY} activity scores",
            ascii=False,
        )
        try:
            rows = _run_prompts(
                sender,
                score_model["max_concurrent"],
                prompts,
                progress=progress,
                semaphore=semaphore,
            )
        finally:
            progress.close()
        errored = _errored_rows(rows)
        if errored:
            print(
                f"[{MODEL_KEY}] {len(errored)}/{len(rows)} samples errored; "
                "NOT caching this run so the failures are re-sampled next time."
            )
        else:
            _write_cache(path, {
                "hash": h,
                "experiment": EXPERIMENT_NAME,
                "model_name": MODEL_KEY,
                "kind": "activity_liking_scores",
                "n": len(prompts),
                "n_repeats": n_repeats,
                "n_prompts": len(prompt_templates),
                "seed": seed,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "client_max_retries": CLIENT_MAX_RETRIES,
            }, rows)

    aligned = _align_rows_to_prompts(rows, prompts)
    out = pd.DataFrame(tasks)
    out["model_key"] = MODEL_KEY
    out["model"] = MAIN_MODEL["display_name"]
    out["reasoning"] = [row.get("reasoning", "") for row in aligned]
    out["answer"] = [row.get("answer", "") for row in aligned]
    out["error"] = [row.get("error") for row in aligned]
    out["score"] = out["answer"].apply(parse_score).astype("Int64")
    return out


def parse_score(raw: str | None) -> int | None:
    """Parse {"score": N}; accepts a bare integer fallback for robustness."""
    if not isinstance(raw, str) or not raw.strip():
        return None

    match = _JSON_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = {}
        score = obj.get("score")
        if isinstance(score, str):
            score = score.strip()
        try:
            value = int(score)
        except (TypeError, ValueError):
            value = None
        if value is not None and 0 <= value <= 100:
            return value

    stripped = raw.strip()
    bare = _BARE_SCORE_RE.fullmatch(stripped)
    if bare:
        value = int(bare.group(1))
        if 0 <= value <= 100:
            return value
    return None


def default_output_path() -> pathlib.Path:
    return result_path("activity_liking_scores", "csv")


def default_summary_output_path() -> pathlib.Path:
    return result_path("activity_liking_scores_summary", "csv")


def save_csv(df: pd.DataFrame, output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per activity with mean parsed score across repeats."""
    return (
        df.groupby(["activity_ix", "activity"], sort=False)
        .agg(
            n_outputs=("score", "size"),
            n_parsed=("score", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )


def report(df: pd.DataFrame) -> None:
    n = len(df)
    parsed = int(df["score"].notna().sum())
    failed = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"\nParsed scores: {parsed}/{n}")
    if failed:
        print(f"Skipped after timeout/error: {failed}/{n}")
    if parsed == 0:
        return
    overall = (
        df.dropna(subset=["score"])["score"]
        .agg(["count", "mean", "median", "min", "max"])
    )
    print("\nScore summary (all activities):")
    print(overall.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=CACHE_ONLY,
        help="Raise instead of sampling if the activity-score cache is missing.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=default_output_path(),
        help="CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=pathlib.Path,
        default=default_summary_output_path(),
        help="Per-activity summary CSV output path.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help=(
            "Number of independent samples per activity, spread (balanced) "
            "across the prompt variants."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PROMPT_SELECTION_SEED,
        help="Seed for the per-activity prompt-variant assignment.",
    )
    parser.add_argument(
        "--prompt",
        type=pathlib.Path,
        default=PROMPT_PATH,
        help="YAML prompt path containing score_prompts (or a single score_prompt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activities = load_activities()
    prompt_templates = load_score_prompts(args.prompt)
    print(
        f"Scoring {len(activities)} activities x {args.repeats} repeats "
        f"across {len(prompt_templates)} prompt variant(s) with {MODEL_KEY}..."
    )
    df = run_activity_scores(
        activities,
        prompt_templates=prompt_templates,
        n_repeats=args.repeats,
        seed=args.seed,
        cache_only=args.cache_only,
    )
    summary = summarize_scores(df)
    out = save_csv(df, args.output)
    summary_out = save_csv(summary, args.summary_output)
    report(df)
    print(f"\nSaved {len(df)} records -> {out}")
    print(f"Saved {len(summary)} summary records -> {summary_out}")


if __name__ == "__main__":
    main()
