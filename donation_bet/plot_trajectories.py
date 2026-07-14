# %%
"""Per-model trajectory plots: how a model's CoT-internal estimates
evolve over the reasoning trace, by direction.

Pick one model in `MODEL_NAME` below (its cached completions must exist).
Each row's CoT is run through the shared trajectory judge (see the
"Trajectory judge" section below), yielding an ordered list of
single-number estimates per trajectory. We then resample each trajectory
onto a fixed grid and plot per-direction central curves with optional
uncertainty bands.

The three "main" plots at the bottom are the ones that matter:
  * `plot_trajectory_diffs_overall`    -- `direction - baseline`
  * `plot_trajectory_offsets_overall`  -- `direction - threshold`
  * `plot_trajectory_offsets_per_variant` -- one subplot per prompt
  * `plot_trajectory_offsets_combined` -- all prompts overlaid, 3 subplots

The bottom of the file has a few secondary / experimental plots kept
around (commented out) -- uncomment a definition AND its call to use.

Figures are written (PDF only) into the giraffes section of the
gitignored Overleaf clone, under the ``trajectories`` subfolder.

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
    stored** -- that's the whole point of switching off llmcomp's
    paraphrase-keyed cache, since reasoning traces can be MB-sized and
    are already in `final_data/cache/`.
"""

import hashlib
import json
import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared import plot_style  # noqa: F401  applies shared figure sizing on import
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

import shared.runner as runner
from shared.get_main_dfs import get_main_dfs
from shared.runner import CacheOnlyMiss, _create_sender


# --- Cache redirect (mirrors plot_biases.py / bias_vs_cot_len.py) ---
# Anchored on runner.__file__ because cell-style execution can leave
# __file__ pointing somewhere unexpected; from shared/runner.py,
# parents[1] is the repo root.
DATA_ROOT = Path(runner.__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")

TRAJECTORY_CACHE_DIR = DATA_ROOT / "trajectories"
# Figure destination: every saved plot is written (PDF only) into the
# giraffes/trajectories section of the gitignored Overleaf clone.
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes" / "trajectories"


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
                # to completion -- Python can't interrupt blocking I/O.
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
    cache stores ONLY the hash + the small raw judge answer -- no
    reasoning traces -- so files stay small.

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
    (e.g. ``"main_experiment_accurate"``).

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


EXPERIMENT = "main_experiment_accurate"
CACHE_ONLY = True
MODEL_NAME = "qwen3.6-35"
# MODEL_NAME = "claude-opus-4.7-max"
# MODEL_NAME = "claude-opus-4.7-xhigh"
# MODEL_NAME = "gpt-5.5-high"
# MODEL_NAME = "kimi-k2.5"
# MODEL_NAME = "gemini-2.5-pro"

# %% --- Load completions for the chosen model and run the trajectory judge ---

df, trajectory_df, display_name = load_model_data(
    MODEL_NAME, experiment=EXPERIMENT, cache_only=CACHE_ONLY,
)
print(f"Loaded {len(trajectory_df)} trajectory rows "
      f"({display_name}, {EXPERIMENT}).")


# %% --- Quick sanity look ---

def trajectory_summary(trajectory_df, n=10):
    """Compact one-line view of the first n trajectories."""
    for _, row in trajectory_df.head(n).iterrows():
        traj = row["trajectory"]
        traj_str = "NONE" if traj is None else ",".join(str(t) for t in traj)
        print(f"prompt={row['prompt_key']:25s}  dir={row['direction']:11s}  "
              f"final_extract={row['estimate']!s:>14}  "
              f"traj={traj_str}")


trajectory_summary(trajectory_df)


# %% --- Debug: where do trajectories disappear between raw rows and `n=...`? ---

def debug_trajectory_counts(df, trajectory_df, directions=("baseline",
                                                           "below_good",
                                                           "above_good")):
    """Per-direction funnel from raw rows in ``df`` to the ``len>=2`` count
    that ends up in plot legends. Also breaks ``len>=2`` down by prompt_key
    so prompt-specific dropouts (e.g. judge mostly returning NONE for one
    prompt) jump out.
    """
    raw_dir = df[df["direction"].isin(directions)]
    has_reasoning_mask = (
        raw_dir["reasoning"].fillna("").astype(str).str.len().gt(0)
    )
    judge_input = raw_dir[has_reasoning_mask]

    def classify(t):
        if t is None:
            return "judge_none_or_unparseable"
        if not isinstance(t, list):
            return "weird_non_list"
        if len(t) == 0:
            return "empty_list"
        if len(t) == 1:
            return "len_1"
        return "len_ge_2"

    klass = trajectory_df["trajectory"].apply(classify)

    print("step1 = empty reasoning  |  step2 = judge NONE/unparseable  "
          "|  step3 = trajectory length 1")
    print(f"{'direction':12s}  {'raw':>5s}  {'step1':>6s}  "
          f"{'step2':>6s}  {'step3':>6s}  {'kept':>5s}")
    for d in directions:
        d_raw = raw_dir[raw_dir["direction"] == d]
        d_input = judge_input[judge_input["direction"] == d]
        d_klass = klass[trajectory_df["direction"] == d]
        n_raw = len(d_raw)
        step1 = len(d_raw) - len(d_input)
        c = d_klass.value_counts().to_dict()
        step2 = (c.get("judge_none_or_unparseable", 0)
                 + c.get("weird_non_list", 0)
                 + c.get("empty_list", 0))
        step3 = c.get("len_1", 0)
        n_kept = c.get("len_ge_2", 0)
        print(f"{d:12s}  {n_raw:5d}  {step1:6d}  "
              f"{step2:6d}  {step3:6d}  {n_kept:5d}")
        sanity = step1 + step2 + step3 + n_kept
        if sanity != n_raw:
            print(f"  !! mismatch: stages sum to {sanity}, raw={n_raw}")

    print("\nPer-prompt x direction kept (len>=2) trajectory counts:")
    by_pk = (
        trajectory_df.assign(_keep=trajectory_df["trajectory"].apply(
            lambda t: isinstance(t, list) and len(t) >= 2))
        .groupby(["prompt_key", "direction"])["_keep"].sum().unstack(fill_value=0)
    )
    print(by_pk)


debug_trajectory_counts(df, trajectory_df)


# %% --- Shared plotting helpers (resample, outlier filter, central, per-prompt curves) ---
# Resample: linearly interpolate each length-k trajectory onto a length-n_grid
# grid so all trajectories can be averaged pointwise. Length 1 trajectories
# are dropped (no shape to plot).
#
# Outlier filter: drop any trajectory containing a value outside
# [threshold/factor, threshold*factor] for that prompt. Symmetric on purpose:
# judge typos (an order-of-magnitude slip while extracting estimates) can
# land on either side of the truth, and both ruin a per-step mean. Pass
# `outlier_factor=None` to disable the value-range filter (length>=2 always
# enforced).

DIRECTIONS = ["baseline", "below_good", "above_good"]
DIR_COLORS = {
    "baseline":   "#7f7f7f",
    "below_good": "#1f77b4",
    "above_good": "#9467bd",
}
N_GRID = 1000  # dense grid; we draw smooth lines (no markers) on the plots


def _maybe_savefig(fig, filename):
    """Save ``fig`` to ``filename`` if given, creating parent dirs as needed."""
    if filename is None:
        return
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")


def _resample_trajectory(traj, n=N_GRID):
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        return None
    src = np.linspace(0, 1, len(arr))
    dst = np.linspace(0, 1, n)
    return np.interp(dst, src, arr)


def _make_outlier_filter(threshold, factor):
    """Predicate ok(traj) -> bool. Always rejects length<2; applies the
    symmetric [thr/factor, thr*factor] check when both inputs are given."""
    def _length_ok(traj):
        return isinstance(traj, list) and len(traj) >= 2

    if factor is None or threshold is None:
        return _length_ok

    upper = threshold * factor
    lower = threshold / factor

    def ok(traj):
        if not _length_ok(traj):
            return False
        for v in traj:
            if v > upper or v < lower:
                return False
        return True

    return ok


def _central_fn(name):
    """Map "mean" / "median" to the corresponding numpy reduction."""
    if name == "mean":
        return np.mean
    if name == "median":
        return np.median
    raise ValueError(f"central must be 'mean' or 'median', got {name!r}")


def _cap_ylim_to_centres(ax, centre_values):
    """Cap the auto-scaled y-axis to +-1.5x(extreme of any centre curve)
    so a wide error band at step ~0 doesn't dominate readability.

    Only contracts the y-axis -- if the natural limits are already
    tighter (e.g. ``band=None``), this is a no-op. Top cap kicks in only
    when at least one centre curve goes positive; bottom cap only when
    at least one goes negative. Centre lines themselves are always
    fully inside the resulting view; bands may get clipped, which is
    the whole point.

    Used on the band-shaded plots (``plot_trajectory_diffs_overall``,
    ``plot_trajectory_offsets_overall``, and per-subplot inside
    ``plot_trajectory_offsets_per_variant``). The per-variant version
    caps each subplot against its OWN centre curves -- subplots don't
    share a y-axis so a global cap would erase the natural scale
    differences across prompts.
    """
    if not centre_values:
        return
    pos_max = max(centre_values)
    neg_min = min(centre_values)
    natural_ymin, natural_ymax = ax.get_ylim()
    if pos_max > 0:
        ax.set_ylim(top=min(natural_ymax, 1.5 * pos_max))
    if neg_min < 0:
        ax.set_ylim(bottom=max(natural_ymin, 1.5 * neg_min))


def _compute_per_prompt_diff_curves(traj_df, n_grid, outlier_factor,
                                    normalize, central="mean",
                                    subtract="baseline"):
    """For each prompt_key, computes the per-step central trajectory
    (mean or median) for each direction, drops trajectories whose values
    fall outside [thr/outlier_factor, thr*outlier_factor], and returns::

        {prompt_key: {
            "threshold": float,
            "kept":       {dir: int},   # # trajectories surviving the filter
            "dropped":    {dir: int},   # # trajectories filtered out (len>=2 only)
            "length_sum": {dir: int},   # sum of len(traj) over kept trajectories
            "baseline":   ndarray | None,   # only when subtract="threshold"
            "below_good": ndarray | None,
            "above_good": ndarray | None,
        }}

    `subtract` controls what each direction's curve is computed against:
      - "baseline" (default): direction_centre - baseline_centre.
        ``baseline`` key is omitted (always 0).
      - "threshold": direction_centre - threshold (a constant scalar).
        ``baseline`` is included as a third curve.

    Diffs are divided by the threshold when ``normalize=True``.
    """
    if subtract not in ("baseline", "threshold"):
        raise ValueError(
            f"subtract must be 'baseline' or 'threshold', got {subtract!r}"
        )
    fn = _central_fn(central)
    out = {}
    for pk in sorted(traj_df["prompt_key"].dropna().unique()):
        sub = traj_df[traj_df["prompt_key"] == pk]
        thr_vals = sub["threshold"].dropna().unique()
        if len(thr_vals) == 0:
            continue
        threshold = float(thr_vals[0])
        ok = _make_outlier_filter(threshold, outlier_factor)

        centres = {}
        kept = {}
        dropped = {}
        length_sum = {}
        for direction in DIRECTIONS:
            d_sub = sub[sub["direction"] == direction]
            if len(d_sub) == 0:
                kept[direction] = 0
                dropped[direction] = 0
                length_sum[direction] = 0
                continue
            n_total = int(d_sub["trajectory"].apply(
                lambda t: isinstance(t, list) and len(t) >= 2,
            ).sum())
            d_kept = d_sub[d_sub["trajectory"].apply(ok)]
            kept[direction] = len(d_kept)
            dropped[direction] = n_total - len(d_kept)
            if len(d_kept) == 0:
                length_sum[direction] = 0
                continue
            length_sum[direction] = int(
                d_kept["trajectory"].apply(len).sum()
            )
            mat = np.vstack([
                _resample_trajectory(t, n_grid) for t in d_kept["trajectory"]
            ])
            centres[direction] = fn(mat, axis=0)

        if subtract == "baseline":
            if "baseline" not in centres:
                continue
            ref = centres["baseline"]
            keep_dirs = ("below_good", "above_good")
        else:  # subtract == "threshold"
            ref = threshold
            keep_dirs = ("baseline", "below_good", "above_good")

        entry = {"threshold": threshold,
                 "kept": kept, "dropped": dropped,
                 "length_sum": length_sum}
        for direction in keep_dirs:
            entry[direction] = None
        for direction in keep_dirs:
            if direction not in centres:
                continue
            diff = centres[direction] - ref
            if normalize:
                diff = diff / threshold
            entry[direction] = diff
        out[pk] = entry
    return out


# %% --- MAIN PLOTS (the three I care about) ---
# - plot_trajectory_diffs_overall:    direction - baseline   (2 lines, mean across prompts)
# - plot_trajectory_offsets_overall:  direction - threshold  (3 lines, mean across prompts)
# - plot_trajectory_offsets_combined: direction - threshold  (3 subplots, all prompts overlaid)


def plot_trajectory_diffs_overall(traj_df, n_grid=N_GRID,
                                  outlier_factor=None, band="iqr",
                                  central="mean", filename=None):
    """Two-line summary: across-prompt aggregate of the per-prompt diff
    curve, one line per direction. Each prompt is normalized by its
    threshold first so the cross-prompt aggregate is well-defined.

    `central` is "mean" or "median". It controls BOTH stages of
    aggregation: within each prompt x direction (across trajectories) and
    across prompts at each step.

    `band` controls the shaded uncertainty around each centre curve,
    computed across prompts at each step:
      - "iqr": 25%-75% percentile band (default)
      - "minmax": full envelope
      - "sem": +-1.96 * SEM (95% normal CI; mean only)
      - None: no band
    """
    fn = _central_fn(central)
    curves = _compute_per_prompt_diff_curves(
        traj_df, n_grid, outlier_factor, normalize=True, central=central,
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    grid_x = np.linspace(0, 1, n_grid)
    linestyles = {"below_good": "-", "above_good": "--"}
    colors = {"below_good": DIR_COLORS["below_good"],
              "above_good": DIR_COLORS["above_good"]}

    totals = {d: {"kept": 0, "length_sum": 0}
              for d in ("baseline", "below_good", "above_good")}
    for entry in curves.values():
        for d in totals:
            totals[d]["kept"] += entry["kept"][d]
            totals[d]["length_sum"] += entry["length_sum"][d]

    centre_values_all = []  # for y-axis capping; see _cap_ylim_to_centres
    for direction in ("below_good", "above_good"):
        stacked = np.vstack([
            entry[direction] for entry in curves.values()
            if entry[direction] is not None
        ]) if curves else np.empty((0, n_grid))
        if stacked.shape[0] == 0:
            continue
        centre_curve = fn(stacked, axis=0)
        centre_values_all.extend(centre_curve.tolist())
        color = colors[direction]
        if band is not None and stacked.shape[0] >= 2:
            if band == "iqr":
                lo, hi = np.percentile(stacked, [25, 75], axis=0)
            elif band == "minmax":
                lo, hi = stacked.min(axis=0), stacked.max(axis=0)
            elif band == "sem":
                if central != "mean":
                    raise ValueError(
                        f"band='sem' is only meaningful with central='mean', "
                        f"got central={central!r}"
                    )
                sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                lo = centre_curve - 1.96 * sem
                hi = centre_curve + 1.96 * sem
            else:
                raise ValueError(f"Unknown band={band!r}")
            ax.fill_between(grid_x, lo, hi, color=color, alpha=0.18)
        n_kept = totals[direction]["kept"]
        mean_len = (totals[direction]["length_sum"] / n_kept
                    if n_kept else float("nan"))
        ax.plot(grid_x, centre_curve, color=color,
                linestyle=linestyles[direction],
                label=(f"{direction} - baseline  "
                       f"(n={n_kept}, mean length={mean_len:.1f})"))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Normalized position in reasoning")
    ax.set_ylabel(f"(estimate - baseline) / threshold  ({central})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _cap_ylim_to_centres(ax, centre_values_all)

    fig.suptitle(display_name)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


def plot_trajectory_offsets_overall(traj_df, n_grid=N_GRID,
                                    outlier_factor=None, band="iqr",
                                    central="mean", filename=None):
    """Three-line summary: across-prompt aggregate of (direction - threshold)
    per direction. All prompts are normalized by their threshold first so
    the cross-prompt aggregate is well-defined.

    `central` controls both aggregation stages (across trajectories within
    prompt x direction, and across prompts at each step).

    `band`: "iqr" (default), "minmax", "sem" (mean only), or None.
    """
    fn = _central_fn(central)
    curves = _compute_per_prompt_diff_curves(
        traj_df, n_grid, outlier_factor, normalize=True,
        central=central, subtract="threshold",
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    grid_x = np.linspace(0, 1, n_grid)
    linestyles = {"baseline": "-", "below_good": "-", "above_good": "-"}

    totals = {d: {"kept": 0, "length_sum": 0} for d in DIRECTIONS}
    for entry in curves.values():
        for d in DIRECTIONS:
            totals[d]["kept"] += entry["kept"][d]
            totals[d]["length_sum"] += entry["length_sum"][d]

    centre_values_all = []  # for y-axis capping; see _cap_ylim_to_centres
    for direction in DIRECTIONS:
        stacked = np.vstack([
            entry[direction] for entry in curves.values()
            if entry[direction] is not None
        ]) if curves else np.empty((0, n_grid))
        if stacked.shape[0] == 0:
            continue
        centre_curve = fn(stacked, axis=0)
        centre_values_all.extend(centre_curve.tolist())
        color = DIR_COLORS[direction]
        if band is not None and stacked.shape[0] >= 2:
            if band == "iqr":
                lo, hi = np.percentile(stacked, [25, 75], axis=0)
            elif band == "minmax":
                lo, hi = stacked.min(axis=0), stacked.max(axis=0)
            elif band == "sem":
                if central != "mean":
                    raise ValueError(
                        f"band='sem' is only meaningful with central='mean', "
                        f"got central={central!r}"
                    )
                sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                lo = centre_curve - 1.96 * sem
                hi = centre_curve + 1.96 * sem
            else:
                raise ValueError(f"Unknown band={band!r}")
            ax.fill_between(grid_x, lo, hi, color=color, alpha=0.18)
        n_kept = totals[direction]["kept"]
        mean_len = (totals[direction]["length_sum"] / n_kept
                    if n_kept else float("nan"))
        ax.plot(grid_x, centre_curve, color=color,
                linestyle=linestyles[direction],
                label=(f"{direction}  "
                       f"(n={n_kept}, mean length={mean_len:.1f})"))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="both", which="major")
    ax.set_xlabel("Normalized position in reasoning")
    ax.set_ylabel(f"(estimate - threshold) / threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _cap_ylim_to_centres(ax, centre_values_all)

    # fig.suptitle(display_name, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


def plot_trajectory_offsets_per_variant(traj_df, n_grid=N_GRID,
                                        outlier_factor=None, band="iqr",
                                        central="mean", ncols=2,
                                        filename=None):
    """Per-variant version of `plot_trajectory_offsets_overall`. One
    subplot per `prompt_key` (default 2 columns, so 10 prompts -> 5 rows
    of 2). Each subplot does exactly what the overall plot does, but for
    a single prompt: it shows `(estimate - threshold) / threshold` for
    each direction, with the centre curve and `band` computed across that
    prompt's surviving trajectories (the across-prompt second stage is
    skipped -- there is only one prompt per subplot).

    `central` and `band` behave as in `plot_trajectory_offsets_overall`.
    """
    fn = _central_fn(central)
    prompts = sorted(traj_df["prompt_key"].dropna().unique())
    n = len(prompts)
    nrows = (n + ncols - 1) // ncols
    # sharey=False so each prompt's natural scale is preserved -- different
    # quantities (giraffe spot counts vs. total ages) sit on very different
    # ranges and a shared scale flattens most of them.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 3.5 * nrows),
                             squeeze=False, sharey=False)
    flat = axes.flatten()
    grid_x = np.linspace(0, 1, n_grid)
    linestyles = {"baseline": "-", "below_good": "-", "above_good": "-"}

    for ax, pk in zip(flat, prompts):
        sub = traj_df[traj_df["prompt_key"] == pk]
        thr_vals = sub["threshold"].dropna().unique()
        if len(thr_vals) == 0:
            ax.set_visible(False)
            continue
        threshold = float(thr_vals[0])
        ok = _make_outlier_filter(threshold, outlier_factor)

        centre_values_this_panel = []  # for _cap_ylim_to_centres
        for direction in DIRECTIONS:
            d_sub = sub[sub["direction"] == direction]
            d_kept = d_sub[d_sub["trajectory"].apply(ok)]
            if len(d_kept) == 0:
                continue
            stacked = np.vstack([
                _resample_trajectory(t, n_grid) for t in d_kept["trajectory"]
            ])
            stacked = (stacked - threshold) / threshold
            centre_curve = fn(stacked, axis=0)
            centre_values_this_panel.extend(centre_curve.tolist())
            color = DIR_COLORS[direction]
            if band is not None and stacked.shape[0] >= 2:
                if band == "iqr":
                    lo, hi = np.percentile(stacked, [25, 75], axis=0)
                elif band == "minmax":
                    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
                elif band == "sem":
                    if central != "mean":
                        raise ValueError(
                            f"band='sem' is only meaningful with central='mean', "
                            f"got central={central!r}"
                        )
                    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                    lo = centre_curve - 1.96 * sem
                    hi = centre_curve + 1.96 * sem
                else:
                    raise ValueError(f"Unknown band={band!r}")
                ax.fill_between(grid_x, lo, hi, color=color, alpha=0.18)
            n_kept = len(d_kept)
            mean_len = float(d_kept["trajectory"].apply(len).mean())
            ax.plot(grid_x, centre_curve, color=color,
                    linestyle=linestyles[direction],
                    label=(f"{direction}  "
                           f"(n={n_kept}, mean length={mean_len:.1f})"))

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.tick_params(axis="both", which="major")
        ax.set_xlabel("Normalized position in reasoning")
        ax.set_ylabel("(estimate - threshold) / threshold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(pk)
        _cap_ylim_to_centres(ax, centre_values_this_panel)

    for ax in flat[n:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


def plot_trajectory_offsets_combined(traj_df, n_grid=N_GRID,
                                     outlier_factor=None, normalize=True,
                                     central="mean", filename=None):
    """All prompts' offset curves overlaid, three subplots
    (baseline, below_good, above_good). Color = prompt (tab10). Each
    line's legend includes the direction's surviving trajectory count.
    """
    curves = _compute_per_prompt_diff_curves(
        traj_df, n_grid, outlier_factor, normalize,
        central=central, subtract="threshold",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=normalize)
    grid_x = np.linspace(0, 1, n_grid)
    cmap = plt.get_cmap("tab10")

    for ax, direction in zip(axes, DIRECTIONS):
        for pi, (pk, entry) in enumerate(curves.items()):
            curve = entry[direction]
            if curve is None:
                continue
            color = cmap(pi % 10)
            n = entry["kept"][direction]
            mean_len = (entry["length_sum"][direction] / n
                        if n else float("nan"))
            label = f"{pk}  (n={n}, mean length={mean_len:.1f})"
            ax.plot(grid_x, curve, color=color, linewidth=1.5,
                    alpha=0.9, label=label)

        ax.set_xlabel("Normalized position in reasoning")
        if normalize:
            ax.set_ylabel("(estimate - threshold) / threshold"
                          f"  ({central})")
        else:
            ax.axhline(0, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.set_ylabel(f"estimate {central} - threshold")
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{int(x):,}")
            )
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{direction} - threshold")
        ax.legend(loc="best")

    fig.suptitle(display_name)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


# %% --- (Interactive single-model preview removed) ---
# These on-screen previews plotted only the one MODEL_NAME selected above, which
# was misleading (a lone Qwen plot with no Claude counterpart) and saved nothing.
# The paper trajectory figures are produced by the generator loop below, which
# covers BOTH per-model figures (Claude Opus 4.7 max and Qwen3.6-35).


# %% --- Cross-model summary helpers (ported from janbet/trajectories/first_last.py) ---

def _on_good_side(direction, value, threshold):
    """Same convention as get_main_dfs._add_good_side."""
    if direction == "below_good":
        return value <= threshold
    if direction == "above_good":
        return value > threshold
    return False


def per_model_three_biases(df, traj_df):
    """``(first_bias, last_bias, final_bias)``, each = mean over prompts of
    ``2 * P(on_good_side) - 1``. ``final_bias`` uses the final answer (matches
    plot_biases.py); ``first_bias`` / ``last_bias`` use the first / last in-CoT
    estimate from ``traj_df`` (directional rows with a parseable trajectory)."""
    final_per_pk = []
    for pk in df["prompt_key"].dropna().unique():
        directional = df[(df["prompt_key"] == pk)
                         & df["direction"].isin(["below_good", "above_good"])]
        if len(directional) == 0:
            continue
        final_per_pk.append(2 * directional["on_good_side"].mean() - 1)
    final_bias = (float(pd.Series(final_per_pk).mean())
                  if final_per_pk else float("nan"))

    rows = []
    for _, r in traj_df.iterrows():
        traj = r["trajectory"]
        if not isinstance(traj, list) or len(traj) == 0:
            continue
        thr = r["threshold"]
        if pd.isna(thr) or r["direction"] not in ("below_good", "above_good"):
            continue
        rows.append({"prompt_key": r["prompt_key"], "direction": r["direction"],
                     "threshold": float(thr),
                     "first": float(traj[0]), "last": float(traj[-1])})
    fl = pd.DataFrame(rows)
    if fl.empty:
        return float("nan"), float("nan"), final_bias

    first_per_pk, last_per_pk = [], []
    for _pk, sub in fl.groupby("prompt_key"):
        first_good = sub.apply(
            lambda r: _on_good_side(r["direction"], r["first"], r["threshold"]),
            axis=1)
        last_good = sub.apply(
            lambda r: _on_good_side(r["direction"], r["last"], r["threshold"]),
            axis=1)
        first_per_pk.append(2 * first_good.mean() - 1)
        last_per_pk.append(2 * last_good.mean() - 1)
    first_bias = (float(pd.Series(first_per_pk).mean())
                  if first_per_pk else float("nan"))
    last_bias = (float(pd.Series(last_per_pk).mean())
                 if last_per_pk else float("nan"))
    return first_bias, last_bias, final_bias


def per_model_start_end_offset_gap(traj_df, central="median"):
    """``(gap_start, gap_end, n_prompts_with_both_directions)`` in threshold
    units. ``gap_*`` is ``above_good - below_good`` of the cross-prompt central
    ``(estimate - threshold) / threshold`` evaluated at the FIRST (u=0) and LAST
    (u=1) in-CoT estimate -- i.e. the endpoints of the orange/blue lines in
    ``plot_trajectory_offsets_overall``. Aggregates each direction across
    prompts BEFORE subtracting (median-of-differences != difference-of-medians)."""
    fn = np.median if central == "median" else np.mean
    per_pd = {}
    for (pk, direction), sub in traj_df.groupby(["prompt_key", "direction"]):
        if direction not in ("above_good", "below_good"):
            continue
        kept = sub[sub["trajectory"].apply(
            lambda t: isinstance(t, list) and len(t) >= 2)]
        if kept.empty:
            continue
        thr_vals = kept["threshold"].dropna().unique()
        if len(thr_vals) == 0 or float(thr_vals[0]) == 0:
            continue
        thr = float(thr_vals[0])
        firsts = kept["trajectory"].apply(lambda t: float(t[0])).to_numpy()
        lasts = kept["trajectory"].apply(lambda t: float(t[-1])).to_numpy()
        per_pd[(pk, direction)] = ((float(fn(firsts)) - thr) / thr,
                                   (float(fn(lasts)) - thr) / thr)

    def per_dir(direction, idx):
        return [v[idx] for (_, d), v in per_pd.items() if d == direction]

    above_first, below_first = per_dir("above_good", 0), per_dir("below_good", 0)
    above_last, below_last = per_dir("above_good", 1), per_dir("below_good", 1)
    if not (above_first and below_first):
        return float("nan"), float("nan"), 0
    gap_start = float(fn(above_first)) - float(fn(below_first))
    gap_end = float(fn(above_last)) - float(fn(below_last))
    pks_a = {pk for (pk, d) in per_pd if d == "above_good"}
    pks_b = {pk for (pk, d) in per_pd if d == "below_good"}
    return gap_start, gap_end, len(pks_a & pks_b)


def plot_first_last_final_bias(res_df, filename=None):
    """Three dots per model: bias of the FIRST in-CoT estimate, the LAST in-CoT
    estimate, and the FINAL answer (appendix ``first_last_final_bias.pdf``)."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(res_df))
    dx = 0.12
    ax.scatter(x - dx, res_df["bias_first"].to_numpy(), s=70, color="#1f77b4",
               zorder=3, label="first CoT estimate bias")
    ax.scatter(x, res_df["bias_last"].to_numpy(), s=70, color="#ff7f0e",
               zorder=3, label="last CoT estimate bias")
    ax.scatter(x + dx, res_df["bias_final"].to_numpy(), s=70, color="#2ca02c",
               zorder=3, label="final answer bias")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["display"].tolist(), rotation=30, ha="right")
    ax.tick_params(axis="y")
    ax.set_ylabel("bias")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _maybe_savefig(fig, filename)
    plt.show()


def plot_offset_gap_scatter(res_df, model_groups, filename=None):
    """2D scatter (appendix ``offset_gap_scatter.pdf``): x = above_good -
    below_good gap at the START of the CoT, y = same gap at the END, one
    labelled point per model, colored by family. Dashed y=x = no change during
    reasoning; points above grew the gap, points below shrank it."""
    group_of = {mk: gn for gn, ms in model_groups for mk in ms}
    groups_order = [gn for gn, _ in model_groups]
    cmap = plt.get_cmap("tab10")
    color_of = {gn: cmap(i % 10) for i, gn in enumerate(groups_order)}

    fig, ax = plt.subplots(figsize=(8, 8))
    xs = res_df["offset_gap_start"].to_numpy()
    ys = res_df["offset_gap_end"].to_numpy()
    for gn in groups_order:
        mask = np.array([group_of.get(mk) == gn for mk in res_df["model_key"]])
        if not mask.any():
            continue
        ax.scatter(xs[mask], ys[mask], s=80, color=color_of[gn],
                   edgecolor="white", linewidth=0.7, zorder=3, label=gn)
    for x, y, name in zip(xs, ys, res_df["display"]):
        if np.isnan(x) or np.isnan(y):
            continue
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=11, va="center", ha="left")

    finite = np.concatenate([xs[np.isfinite(xs)], ys[np.isfinite(ys)]])
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.05
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="gray", linewidth=0.8, linestyle="--",
            zorder=1, label="y = x (no change)")
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("gap at START of CoT  (above_good - below_good) / threshold")
    ax.set_ylabel("gap at END of CoT  (above_good - below_good) / threshold")
    ax.tick_params(axis="both")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_savefig(fig, filename)
    plt.show()


# %% --- Generate ALL paper trajectory figures (CACHE_ONLY-safe) ---
# Per-model figures (paper body + appendix) for the two models shown
# individually, plus the two cross-model summary figures. With CACHE_ONLY=True
# any model lacking a trajectory-judge cache is SKIPPED WITH A WARNING (never an
# API call), and the summaries are drawn from whatever IS cached -- migrate /
# judge the rest of the cache to fill them in. Saved names match the paper's
# figures/giraffes/trajectories/ references.
PER_MODEL_FIGURE_MODELS = ["claude-opus-4.7-max", "qwen3.6-35"]

# Cross-model summary model set (mirrors the canonical giraffes families).
SUMMARY_MODEL_GROUPS = [
    ("Claude", ["claude-opus-4.7-high", "claude-opus-4.7-xhigh",
                "claude-opus-4.7-max", "claude-opus-4.8-high",
                "claude-opus-4.8-max", "claude-fable-5-high"]),
    ("GPT", ["gpt-5.2-medium", "gpt-5.4-medium", "gpt-5.5-medium", "gpt-5.5-high"]),
    ("Gemini", ["gemini-2.5-pro", "gemini-3.1-pro-medium",
                "gemini-3.1-pro-high", "gemini-3.5-flash-high"]),
    ("Qwen", ["qwen3.5-35", "qwen3.6-35"]),
    ("Kimi", ["kimi-k2.5", "kimi-k2.6"]),
]
SUMMARY_MODELS = [mk for _, g in SUMMARY_MODEL_GROUPS for mk in g]

summary_rows = []
missing_models = []
for _model in SUMMARY_MODELS:
    try:
        _df, _traj_df, _disp = load_model_data(
            _model, experiment=EXPERIMENT, cache_only=CACHE_ONLY,
        )
    except CacheOnlyMiss:
        missing_models.append(_model)
        print(f"[skip] {_model}: no trajectory-judge cache (CACHE_ONLY).")
        continue

    _dir_df = _traj_df[_traj_df["direction"].isin(["below_good", "above_good"])]

    # Per-model figures (only the two models the paper shows individually).
    if _model in PER_MODEL_FIGURE_MODELS:
        plot_trajectory_offsets_overall(
            _traj_df, central="median",
            filename=str(FIG_DIR / f"trajectories_{_model}.pdf"),
        )
        plot_trajectory_offsets_per_variant(
            _traj_df, central="median",
            filename=str(FIG_DIR / f"trajectories_{_model}_per_variant.pdf"),
        )

    # Cross-model summary stats.
    bias_first, bias_last, bias_final = per_model_three_biases(_df, _dir_df)
    gap_start, gap_end, _n = per_model_start_end_offset_gap(_dir_df)
    summary_rows.append({
        "model_key": _model, "display": _disp,
        "bias_first": bias_first, "bias_last": bias_last, "bias_final": bias_final,
        "offset_gap_start": gap_start, "offset_gap_end": gap_end,
    })

missing_per_model = [m for m in PER_MODEL_FIGURE_MODELS if m in missing_models]
if missing_per_model:
    warnings.warn(
        "Per-model trajectory figures skipped (no trajectory-judge cache): "
        + ", ".join(missing_per_model), stacklevel=1,
    )
# The two cross-model summaries (first_last_final_bias, offset_gap_scatter) are
# only meaningful with the FULL model set, so they are drawn ONLY when every
# SUMMARY_MODELS entry is cached. If any are missing, skip BOTH entirely (rather
# than a misleading partial figure) and warn -- a colleague migrates the rest of
# the trajectory-judge cache, after which a re-run produces them.
summary_df = pd.DataFrame(summary_rows)
if missing_models:
    warnings.warn(
        f"Cross-model trajectory summaries (first_last_final_bias, "
        f"offset_gap_scatter) NOT drawn: they require all {len(SUMMARY_MODELS)} "
        f"models, but {len(missing_models)} lack a trajectory-judge cache "
        f"({', '.join(missing_models)}). Migrate / judge the rest, then re-run.",
        stacklevel=1,
    )
else:
    print(summary_df.to_string(index=False))
    plot_first_last_final_bias(
        summary_df, filename=str(FIG_DIR / "first_last_final_bias.pdf"),
    )
    plot_offset_gap_scatter(
        summary_df, SUMMARY_MODEL_GROUPS,
        filename=str(FIG_DIR / "offset_gap_scatter.pdf"),
    )

# %%
