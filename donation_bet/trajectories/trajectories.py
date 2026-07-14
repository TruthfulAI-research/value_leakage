# %%
"""Per-model trajectory plots: how a model's CoT-internal estimates
evolve over the reasoning trace, by direction.

Pick one model in `MODEL_NAME` below (must be in
`janbet.trajectories.data.MODELS` so the cached completions exist).
Each row's CoT is run through the shared trajectory judge (see
`data.py`), yielding an ordered list of single-number estimates per
trajectory. We then resample each trajectory onto a fixed grid and plot
per-direction central curves with optional uncertainty bands.

The three "main" plots at the bottom are the ones that matter:
  * `plot_trajectory_diffs_overall`    – `direction − baseline`
  * `plot_trajectory_offsets_overall`  – `direction − threshold`
  * `plot_trajectory_offsets_per_variant` – one subplot per prompt
  * `plot_trajectory_offsets_combined` – all prompts overlaid, 3 subplots

The bottom of the file has a few secondary / experimental plots kept
around (commented out) — uncomment a definition AND its call to use.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from donation_bet.trajectories.data import load_model_data

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


# %% --- Debug: where do trajectories disappear between raw rows and `n=…`? ---

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

    print("\nPer-prompt × direction kept (len>=2) trajectory counts:")
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
    "baseline":   "#90A4AE",
    "below_good": "#1f77b4",
    "above_good": "#ff7f0e",
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
    """Cap the auto-scaled y-axis to ±1.5×(extreme of any centre curve)
    so a wide error band at step ~0 doesn't dominate readability.

    Only contracts the y-axis — if the natural limits are already
    tighter (e.g. ``band=None``), this is a no-op. Top cap kicks in only
    when at least one centre curve goes positive; bottom cap only when
    at least one goes negative. Centre lines themselves are always
    fully inside the resulting view; bands may get clipped, which is
    the whole point.

    Used on the band-shaded plots (``plot_trajectory_diffs_overall``,
    ``plot_trajectory_offsets_overall``, and per-subplot inside
    ``plot_trajectory_offsets_per_variant``). The per-variant version
    caps each subplot against its OWN centre curves — subplots don't
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
    aggregation: within each prompt × direction (across trajectories) and
    across prompts at each step.

    `band` controls the shaded uncertainty around each centre curve,
    computed across prompts at each step:
      - "iqr": 25%-75% percentile band (default)
      - "minmax": full envelope
      - "sem": ±1.96 * SEM (95% normal CI; mean only)
      - None: no band
    """
    fn = _central_fn(central)
    curves = _compute_per_prompt_diff_curves(
        traj_df, n_grid, outlier_factor, normalize=True, central=central,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
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
                linestyle=linestyles[direction], linewidth=2.4,
                label=(f"{direction} − baseline  "
                       f"(n={n_kept}, mean length={mean_len:.1f})"))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Normalized position in reasoning")
    ax.set_ylabel(f"(estimate − baseline) / threshold  ({central})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _cap_ylim_to_centres(ax, centre_values_all)

    fig.suptitle(display_name, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


def plot_trajectory_offsets_overall(traj_df, n_grid=N_GRID,
                                    outlier_factor=None, band="iqr",
                                    central="mean", filename=None):
    """Three-line summary: across-prompt aggregate of (direction − threshold)
    per direction. All prompts are normalized by their threshold first so
    the cross-prompt aggregate is well-defined.

    `central` controls both aggregation stages (across trajectories within
    prompt × direction, and across prompts at each step).

    `band`: "iqr" (default), "minmax", "sem" (mean only), or None.
    """
    fn = _central_fn(central)
    curves = _compute_per_prompt_diff_curves(
        traj_df, n_grid, outlier_factor, normalize=True,
        central=central, subtract="threshold",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
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
                linestyle=linestyles[direction], linewidth=2.4,
                label=(f"{direction}  "
                       f"(n={n_kept}, mean length={mean_len:.1f})"))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("Normalized position in reasoning", fontsize=16)
    ax.set_ylabel(f"(estimate − threshold) / threshold", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc="best")
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
    a single prompt: it shows `(estimate − threshold) / threshold` for
    each direction, with the centre curve and `band` computed across that
    prompt's surviving trajectories (the across-prompt second stage is
    skipped — there is only one prompt per subplot).

    `central` and `band` behave as in `plot_trajectory_offsets_overall`.
    """
    fn = _central_fn(central)
    prompts = sorted(traj_df["prompt_key"].dropna().unique())
    n = len(prompts)
    nrows = (n + ncols - 1) // ncols
    # sharey=False so each prompt's natural scale is preserved — different
    # quantities (giraffe spot counts vs. total ages) sit on very different
    # ranges and a shared scale flattens most of them.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8.5 * ncols, 5 * nrows),
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
                    linestyle=linestyles[direction], linewidth=2.4,
                    label=(f"{direction}  "
                           f"(n={n_kept}, mean length={mean_len:.1f})"))

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xlabel("Normalized position in reasoning", fontsize=16)
        ax.set_ylabel("(estimate − threshold) / threshold", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, loc="best")
        ax.set_title(pk, fontsize=16)
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=normalize)
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
            ax.set_ylabel("(estimate − threshold) / threshold"
                          f"  ({central})")
        else:
            ax.axhline(0, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.set_ylabel(f"estimate {central} − threshold")
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{int(x):,}")
            )
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{direction} − threshold")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(display_name, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


# %% --- Run the three main plots ---

plot_trajectory_diffs_overall(trajectory_df, central="median")
plot_trajectory_offsets_overall(trajectory_df, central="median", filename=f"figures/trajectories/trajectories_{MODEL_NAME}.pdf")
plot_trajectory_offsets_per_variant(trajectory_df, central="median", filename=f"figures/trajectories/trajectories_{MODEL_NAME}_per_variant.pdf")
plot_trajectory_offsets_combined(trajectory_df, central="median")

# %%
