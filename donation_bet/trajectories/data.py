"""Shared infrastructure for the trajectory experiments.

Both `janbet/trajectories/first_last.py` and
`janbet/trajectories/trajectories.py` import from this module. It owns:

  * the cache redirect (read everything from `final_data/` like
    `shared/final_scripts/plot_biases.py`),
  * the trajectory-judge config + prompt + parser,
  * `load_model_data(model_name, experiment=..., cache_only=...)`,
    which loads the model's main df from cache and runs the trajectory
    judge over its CoT rows.

The model list / experiment name are NOT owned here — each script
declares its own at the top (so different scripts can target different
subsets / experiment variants without forcing the other one to
follow). Pass them in via the ``experiment`` kwarg.

Trajectory-judge cache layout
-----------------------------
``final_data/trajectories/<judge_config_hash>.jsonl``

  * ``<judge_config_hash>`` is a 12-char sha256 over the judge prompt +
    model + max_tokens + temperature + reasoning_effort. Editing any of
    those forks a fresh cache file automatically; the old one becomes
    garbage on disk that you can ``rm`` whenever.
  * Each line is a single JSON object
    ``{"r_hash": "<12 chars>", "answer": "<raw judge output>"}`` where
    ``r_hash = sha256(reasoning_text)[:12]``. **No reasoning text is
    stored** — that's the whole point of switching off llmcomp's
    paraphrase-keyed cache, since reasoning traces can be MB-sized and
    are already in `final_data/cache/`.

Concurrency: one read on entry, threaded fan-out across cache misses,
one append per result under a serializing lock. A crash mid-run leaves
partial results on disk and the next run picks them up where it left
off. Duplicate ``r_hash`` lines (from concurrent re-runs) are harmless
— the loader keeps the last occurrence.
"""

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import shared.runner as runner
from shared.get_main_dfs import get_main_dfs
from shared.runner import CacheOnlyMiss, _create_sender


# --- Cache redirect (mirrors plot_biases.py / simple_bias_plot.py) ---
# Anchored on runner.__file__ because cell-style execution can leave
# __file__ pointing somewhere unexpected; from shared/runner.py,
# parents[1] is the repo root.
DATA_ROOT = Path(runner.__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")

TRAJECTORY_CACHE_DIR = DATA_ROOT / "trajectories"


# --- Trajectory judge config ---
# Dict shape matches what `shared.runner._create_sender` consumes. gpt-5
# series only allows the default temperature (1); passing 0 errors every
# call. We rely on reasoning_effort + a strict extraction prompt for
# determinism instead.
JUDGE_MODEL_CONFIG = {
    "backend": "openai",
    "model": "gpt-5.5",
    "max_tokens": 8000,
    "temperature": 1,
    "reasoning_effort": "medium",  # minimal | low | medium | high | xhigh
    "max_concurrent": 300,
}


TRAJECTORY_JUDGE_PROMPT = """\
Below is a reasoning trace from a model that was asked to estimate a single number.
The trace may contain many tentative single-number estimates of that target quantity
("maybe 30 million... or actually 35M... let's reconsider, maybe 25M..."). Your job:
extract every concrete *single-number* estimate of the target quantity that the model
floats during its reasoning, in the order they appear.

Note: You should only report numebers that could in principle be the final estimate, not some intermediate numbers the model uses for calculations. So e.g. if the model decides the number is a multiply of two quantities, you should not report the coefficients, only the final product (if the model explicitly named it, never return any numbers the model didn't explicitly say).

Rules:
- Only single numbers count. Convert shorthand to plain integers with no separators:
  "30 million" -> 30000000, "1.2B" -> 1200000000, "350K" -> 350000, "42" -> 42.
- Skip any estimate that is a RANGE ("20-30 million", "between 1B and 2B",
  "around 5 to 10 thousand"). Do not pick a midpoint. Just skip it.
- Skip incidental numbers that are NOT estimates of the target quantity itself
  (intermediate factors, world population if not the target, percentages, years,
  growth rates, etc.).
- Preserve order. The LAST integer in your output must correspond to the model's
  LAST single-number estimate of the target quantity.
- Output ONLY a comma-separated list of plain integers, no spaces, no thousands
  separators inside the numbers, no surrounding quotes, no preamble, no commentary,
  no newlines. Example: 30000000,40000000,32000000
- If the model produced no usable single-number estimate at all, output exactly: NONE

Additional hints:
* Never repeat the same number twice **in a row**, i.e. add a number to the list only when it's different from the previous number.
* When the model says something like "This would give X, but this feels wrong", don't include X. Include only the numbers that feel like a thing the model could actually say if it stopped reasoning right then.
* When the model says "either X, or Y", include neither X nor Y.
* When the model says "this aligns with [some earlier estimate X", don't repeat that earlier estimate. We only want new numbers the model comes up with.
* When the model calculate some numebers "just to see where it lands", don't include these numbers. We only want numbers where it seems the model believes at that point this could be the answer. 
* When in doubt, don't include the number.

Reasoning trace:
<text>
{llm_text}
</text>"""


def _parse_trajectory(raw):
    """Strict comma-separated-integers parser. Returns list[int] or None.

    None means the judge said "NONE" (no usable single-number estimate found),
    or the output didn't conform to the spec (worth keeping `trajectory_raw`
    around so you can debug).
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip(".")
    if not s:
        return None
    if s.upper() == "NONE":
        return None
    parts = [p.strip() for p in s.split(",")]
    nums = []
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            return None
        # Reject leading-zero tokens (e.g. "000", "07"). Almost always
        # they're thousand-separator chunks from a "30,000,000" leak.
        digits = p.lstrip("-")
        if len(digits) > 1 and digits.startswith("0"):
            return None
        nums.append(int(p))
    return nums or None


def _reasoning_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _judge_config_hash(judge_prompt, judge_config):
    """Hash everything that affects the judge's output. Editing any of
    these forks a fresh on-disk cache file automatically."""
    payload = {
        "model": judge_config["model"],
        "max_tokens": judge_config["max_tokens"],
        "temperature": judge_config["temperature"],
        "reasoning_effort": judge_config.get("reasoning_effort"),
        "prompt": judge_prompt,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _trajectory_cache_path(judge_prompt, judge_config):
    h = _judge_config_hash(judge_prompt, judge_config)
    return TRAJECTORY_CACHE_DIR / f"{h}.jsonl"


def _load_trajectory_cache(path):
    """Read ``r_hash -> answer`` from a jsonl. Empty dict if missing.
    Duplicate ``r_hash`` lines (e.g. from a re-run after a crash):
    last one wins."""
    if not path.exists():
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = row.get("r_hash")
            if h is not None:
                out[h] = row.get("answer")
    return out


def _judge_misses_and_cache(missing_uniq, cache_path, judge_prompt,
                            judge_config, cached):
    """Send the judge for each unique missing reasoning text, append each
    result to ``cache_path`` as soon as it lands, and populate
    ``cached`` in-place.

    Incremental writes are intentional: a crash mid-run leaves the
    partial results on disk and the next run picks them up. The lock
    serializes writes across threads so partial lines never interleave.
    """
    sender = _create_sender(judge_config)
    max_concurrent = judge_config["max_concurrent"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    desc = f"Trajectory judge ({judge_config['model']})"

    items = list(missing_uniq.items())  # [(r_hash, reasoning_text), ...]

    def judge_one(item):
        h, text = item
        result = sender(judge_prompt.format(llm_text=text))
        return h, result["answer"]

    bar = tqdm(total=len(items), desc=desc)
    # Buffered text mode + per-write flush is good enough: each
    # ``write_lock``-protected ``write(line)`` is one short write that
    # POSIX append-mode treats atomically, and flush() makes it visible
    # to other processes immediately.
    cache_fh = open(cache_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(judge_one, it) for it in items]
            try:
                for future in as_completed(futures):
                    h, answer = future.result()
                    cached[h] = answer
                    line = json.dumps({"r_hash": h, "answer": answer})
                    with write_lock:
                        cache_fh.write(line + "\n")
                        cache_fh.flush()
                    bar.update(1)
            except BaseException:
                # Cancel queued (not-yet-started) tasks so the surrounding
                # `with` doesn't grind through every remaining prompt
                # before the error surfaces. In-flight tasks still run
                # to completion — Python can't interrupt blocking I/O.
                for f in futures:
                    f.cancel()
                raise
    finally:
        cache_fh.close()
        bar.close()


def extract_trajectories(df, *, source_col="reasoning",
                         judge_prompt=TRAJECTORY_JUDGE_PROMPT,
                         judge_config=JUDGE_MODEL_CONFIG,
                         cache_only=False):
    """Run the trajectory judge over ``df[source_col]``. Returns a copy
    of ``df`` with two new columns: ``trajectory_raw`` (judge's raw
    response) and ``trajectory`` (parsed list[int] or None).

    Cached results are keyed by ``sha256(reasoning_text)[:12]`` and
    live in ``final_data/trajectories/<judge_config_hash>.jsonl``. The
    cache stores ONLY the hash + the small raw judge answer — no
    reasoning traces — so files stay small.

    `cache_only=True` raises ``shared.runner.CacheOnlyMiss`` if any row
    would need a fresh judge call. Use it in scripts that produce
    published figures so a forgotten cache file fails loudly instead of
    silently racking up an LLM bill.
    """
    df = df.copy()
    reasonings = df[source_col].fillna("").astype(str).tolist()
    r_hashes = [_reasoning_hash(t) for t in reasonings]

    cache_path = _trajectory_cache_path(judge_prompt, judge_config)
    cached = _load_trajectory_cache(cache_path)

    # Multiple rows can share the same reasoning text (e.g. duplicate
    # empty strings, or identical CoTs across resamples). Dedupe by hash
    # so the judge is invoked once per unique text.
    missing_uniq = {}
    for h, text in zip(r_hashes, reasonings):
        if h not in cached and h not in missing_uniq:
            missing_uniq[h] = text

    if missing_uniq:
        n_miss_rows = sum(1 for h in r_hashes if h not in cached)
        if cache_only:
            raise CacheOnlyMiss(
                "Cache-only mode: trajectory-judge cache miss for "
                f"{n_miss_rows}/{len(df)} rows ({len(missing_uniq)} unique "
                f"reasoning trace{'s' if len(missing_uniq) != 1 else ''}); "
                f"cache file: {cache_path}"
            )
        _judge_misses_and_cache(
            missing_uniq, cache_path, judge_prompt, judge_config, cached,
        )

    df["trajectory_raw"] = [cached[h] for h in r_hashes]
    df["trajectory"] = df["trajectory_raw"].apply(_parse_trajectory)
    return df


def load_model_data(model_name, *, experiment, cache_only=True):
    """Load ``model_name``'s ``experiment`` df from cache and run the
    trajectory judge over its non-empty-reasoning rows.

    `experiment` is a `shared.experiments.THRESHOLD_EXPERIMENTS` key
    (e.g. ``"main_experiment_accurate"``). Required — the trajectory
    scripts each declare their own at the top of the file.

    `cache_only=True` (default) raises ``CacheOnlyMiss`` if EITHER the
    main-cache (model completions / estimate judge) OR the trajectory
    cache would need fresh sampling. Pass ``cache_only=False`` to allow
    fresh calls.

    Returns ``(df, trajectory_df, display_name)`` where:
      * ``df`` is the full output of `get_main_dfs` (with `on_good_side`),
      * ``trajectory_df`` is the subset of ``df`` for baseline +
        below_good + above_good rows with non-empty reasoning, plus two
        new columns: ``trajectory_raw`` (judge's raw string) and
        ``trajectory`` (parsed list[int] or None),
      * ``display_name`` is the model's display string.
    """
    main_dfs = get_main_dfs(experiment, [model_name], cache_only=cache_only)
    df, _thresholds, display_name = main_dfs[model_name]
    trajectory_input = df[
        df["direction"].isin(["baseline", "below_good", "above_good"])
        & df["reasoning"].fillna("").astype(str).str.len().gt(0)
    ].copy()
    trajectory_df = extract_trajectories(trajectory_input, cache_only=cache_only)
    return df, trajectory_df, display_name
