# %%
"""Trajectory plots split by CoT length: one grid figure, one row per model.

Motivation: the paper shows two results that confuse readers:
  * longer reasoning => LOWER final bias (bias_vs_cot_len.py, Figure 7),
  * within a single reasoning trace, intermediate estimates drift TOWARDS
    the good side over time (trajectories.py, Figure 5).
Hypothesis: in short reasonings the trajectories diverge faster (or start
already-diverged), so the two results are compatible. To eyeball this, we
draw the paper's `plot_trajectory_offsets_overall` plot (same as the
trajectories_claude-opus-4.7-max.pdf figure, central="median") in up to
three side-by-side panels (shared y-axis):

  1. all rollouts (optional; INCLUDE_ALL_PANEL, excluded by default),
  2. only the 33% SHORTEST CoTs,
  3. only the 33% LONGEST CoTs.

CoT length = regex word count (runs of >=2 ASCII letters), matching
bias_vs_cot_len.py. The tercile split is done WITHIN each
(prompt_key, direction) group so all variants keep the same
prompt/direction composition -- a global split would mostly separate
prompts with naturally long CoTs from prompts with short ones.

Plotting code is duplicated from trajectories.py (that file is a
cell-style script whose import would execute the whole pipeline, so we
can't import from it). Data loading reuses
`janbet.trajectories.data.load_model_data`, whose judge prompt + config
are identical to trajectories.py's, so the trajectory-judge cache is
shared. CACHE_ONLY=True (default) fails loudly on any cache miss instead
of racking up an LLM bill; flip to False to allow fresh judge calls.

Output: ONE grid figure (one row per model in MODEL_NAMES, short/long
panels per row), saved (PDF + PNG) into the giraffes/trajectories
section of the gitignored Overleaf clone.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import shared.runner as runner
from donation_bet.trajectories.data import load_model_data

MODEL_NAMES = [
    "claude-opus-4.8-max",
    "gemini-3.1-pro-high",
    "kimi-k2.6",
    "gpt-5.5-high",
    "qwen3.6-35",
]
EXPERIMENT = "main_experiment_accurate"
# True: raise CacheOnlyMiss on any completion/judge cache miss (final-script
# convention). False: run the trajectory judge on cache misses.
CACHE_ONLY = True
# Include the "all CoTs" panel alongside the short/long tercile panels.
INCLUDE_ALL_PANEL = False
# Draw the baseline curve in the panels. Baseline rows still feed the
# tercile split and stats either way; this only affects the plots.
INCLUDE_BASELINE = False

# Figure destination: giraffes/trajectories section of the gitignored
# Overleaf clone, next to the other trajectory figures (same convention
# as trajectories.py).
DATA_ROOT = Path(runner.__file__).resolve().parents[1] / "data" / "final_data"
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes" / "trajectories"

# CoT length unit: same conservative word regex as bias_vs_cot_len.py
# (runs of >=2 letters, so unicode junk / base64 blobs count as ~1).
_WORD_PATTERN = r"[A-Za-z]{2,}"
TERCILE_FRAC = 1 / 3

VARIANT_TITLES = {
    "all": "all CoTs",
    "short33": "33% shortest CoTs",
    "long33": "33% longest CoTs",
}


# %% --- CoT-length tercile split ---

def build_variants(trajectory_df, include_all=INCLUDE_ALL_PANEL):
    """{variant_name: sub_df} for the short/long terciles (and optionally
    the full df). Tercile membership is the percentile rank of CoT word
    count within each (prompt_key, direction) cell; method="first" breaks
    ties by row order so tercile sizes are exact.

    Only rows with a plottable trajectory (>=2 extracted estimates) are
    ranked and returned: the plots can't use the others anyway, and the
    trajectory judge fails disproportionately on very long CoTs (it runs
    out of tokens and returns an empty answer, ~23% of kimi-k2.6 rows),
    so ranking over ALL rows would leave the long tercile with visibly
    fewer plotted lines. Note the flip side: for such models the "33%
    longest" panel means the longest third of *parseable* CoTs -- the
    very longest CoTs are underrepresented in it."""
    trajectory_df = trajectory_df[trajectory_df["trajectory"].apply(
        lambda t: isinstance(t, list) and len(t) >= 2)].copy()
    trajectory_df["cot_words"] = (
        trajectory_df["reasoning"].fillna("").astype(str)
        .str.count(_WORD_PATTERN)
    )
    trajectory_df["_len_pct"] = (
        trajectory_df.groupby(["prompt_key", "direction"])["cot_words"]
        .rank(pct=True, method="first")
    )
    variants = {}
    if include_all:
        variants["all"] = trajectory_df
    variants["short33"] = trajectory_df[
        trajectory_df["_len_pct"] <= TERCILE_FRAC
    ]
    variants["long33"] = trajectory_df[
        trajectory_df["_len_pct"] > 1 - TERCILE_FRAC
    ]
    return variants


def print_variant_stats(variants):
    print("Rows and CoT word lengths per variant/direction:")
    for vname, vdf in variants.items():
        for direction in ("baseline", "below_good", "above_good"):
            sub = vdf[vdf["direction"] == direction]
            if len(sub) == 0:
                continue
            print(f"  {vname:8s} {direction:11s} n={len(sub):5d}  "
                  f"cot_words median={sub['cot_words'].median():8.0f}  "
                  f"mean={sub['cot_words'].mean():8.0f}")


# %% --- Plotting code (copied from shared/final_scripts/giraffes/trajectories.py) ---

DIRECTIONS = ["baseline", "below_good", "above_good"]
DIR_COLORS = {
    "baseline":   "#7f7f7f",
    "below_good": "#1f77b4",
    "above_good": "#9467bd",
}
N_GRID = 1000


def _maybe_savefig(fig, filename):
    """Save ``fig`` to ``filename``, plus a PNG sibling for quick viewing."""
    if filename is None:
        return
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")
    root, ext = os.path.splitext(filename)
    if ext.lower() != ".png":
        fig.savefig(root + ".png", bbox_inches="tight", dpi=150)


def _resample_trajectory(traj, n=N_GRID):
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        return None
    src = np.linspace(0, 1, len(arr))
    dst = np.linspace(0, 1, n)
    return np.interp(dst, src, arr)


def _make_outlier_filter(threshold, factor):
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
    if name == "mean":
        return np.mean
    if name == "median":
        return np.median
    raise ValueError(f"central must be 'mean' or 'median', got {name!r}")


def _cap_ylim_to_centres(ax, centre_values):
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


def _plot_offsets_on_ax(ax, traj_df, n_grid=N_GRID,
                        outlier_factor=None, band="iqr",
                        central="mean", title=None):
    """Body of the paper's `plot_trajectory_offsets_overall`, drawing on
    a provided `ax`. Returns the list of centre-curve values (for a
    shared y-axis cap applied by the caller)."""
    fn = _central_fn(central)
    curves = _compute_per_prompt_diff_curves(
        traj_df, n_grid, outlier_factor, normalize=True,
        central=central, subtract="threshold",
    )

    grid_x = np.linspace(0, 1, n_grid)

    totals = {d: {"kept": 0, "length_sum": 0} for d in DIRECTIONS}
    for entry in curves.values():
        for d in DIRECTIONS:
            totals[d]["kept"] += entry["kept"][d]
            totals[d]["length_sum"] += entry["length_sum"][d]

    plot_directions = (DIRECTIONS if INCLUDE_BASELINE
                       else [d for d in DIRECTIONS if d != "baseline"])
    centre_values_all = []
    for direction in plot_directions:
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
                label=(f"{direction}  "
                       f"(n={n_kept}, mean length={mean_len:.1f})"))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="both", which="major")
    ax.set_xlabel("Normalized position in reasoning")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if title is not None:
        ax.set_title(title)
    return centre_values_all


def plot_offsets_by_variant(variants, variant_titles, central="mean",
                            n_grid=N_GRID, outlier_factor=None, band="iqr",
                            suptitle=None, filename=None):
    """One row of subplots (one per variant), shared y-axis. Each panel
    is the paper's `plot_trajectory_offsets_overall` for that variant's
    subset of trajectory rows."""
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)

    centre_values_all = []
    for ax, (vname, vdf) in zip(np.atleast_1d(axes), variants.items()):
        centre_values_all.extend(_plot_offsets_on_ax(
            ax, vdf, n_grid=n_grid, outlier_factor=outlier_factor,
            band=band, central=central, title=variant_titles[vname],
        ))
    axes[0].set_ylabel("(estimate - threshold) / threshold")
    # Shared axes: capping the first ax caps all of them. Uses centre
    # values pooled across panels, so no panel's centre line is clipped.
    _cap_ylim_to_centres(axes[0], centre_values_all)

    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _maybe_savefig(fig, filename)
    plt.show()


# %% --- All models in one grid figure (one row per model) ---
# Layout mirrors the paper's per-variant figures (e.g.
# trajectories_qwen3.6-35_per_variant.pdf): rows stacked vertically, each
# row is one model's short/long panels with the model name as row title.

def plot_offsets_grid(models_data, variant_titles, central="mean",
                      n_grid=N_GRID, outlier_factor=None, band="iqr",
                      row_height=3.9, filename=None):
    """``models_data``: list of (display_name, variants dict), one row
    each. Panels within a row share their y-axis (capped per row)."""
    nrows = len(models_data)
    ncols = len(models_data[0][1])
    fig = plt.figure(figsize=(6 * ncols, row_height * nrows),
                     layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(nrows, 1))
    for subfig, (disp, variants) in zip(subfigs, models_data):
        axes = np.atleast_1d(subfig.subplots(1, ncols, sharey=True))
        centre_values_row = []
        for ax, (vname, vdf) in zip(axes, variants.items()):
            centre_values_row.extend(_plot_offsets_on_ax(
                ax, vdf, n_grid=n_grid, outlier_factor=outlier_factor,
                band=band, central=central, title=variant_titles[vname],
            ))
        axes[0].set_ylabel("(estimate - threshold) / threshold")
        # Shared row axes: capping the first ax caps the whole row.
        _cap_ylim_to_centres(axes[0], centre_values_row)
        subfig.suptitle(disp, fontsize=14, fontweight="bold")
    _maybe_savefig(fig, filename)
    plt.show()


models_data = []
for model_name in MODEL_NAMES:
    df, trajectory_df, display_name = load_model_data(
        model_name, experiment=EXPERIMENT, cache_only=CACHE_ONLY,
    )
    print(f"\n=== {display_name} ({model_name}): "
          f"{len(trajectory_df)} trajectory rows ===")
    variants = build_variants(trajectory_df, include_all=INCLUDE_ALL_PANEL)
    print_variant_stats(variants)
    models_data.append((display_name, variants))

plot_offsets_grid(
    models_data, VARIANT_TITLES, central="median",
    filename=str(FIG_DIR / "trajectories_len_split_all_models.pdf"),
)

# %%
