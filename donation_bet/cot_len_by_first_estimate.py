# %%
"""CoT-length ratio by first in-CoT estimate side, across all models.

For each Donation Bet directional rollout we determine whether the FIRST
estimate floated in the CoT (trajectory judge, see trajectories.py) lands
on the "good" or "bad" side of the donation threshold, and measure CoT
length in words (same regex as bias_vs_cot_len.py). The plotted metric is
the ratio

    mean CoT length after a bad-side start
    --------------------------------------
    mean CoT length after a good-side start

computed per prompt (from that prompt's two group means) and then
averaged across prompts. Taking the ratio within each prompt first makes
every prompt contribute equally -- prompts with naturally long CoTs (or
unusually strong patterns) can't dominate, and pooling rows across
prompts would confound the comparison since prompts differ in natural
CoT length and good-start rate. Ratio > 1 means the model reasons longer
when its first estimate lands on the bad side of the threshold.

Error bars are 95% hierarchical-bootstrap CIs (rollouts resampled within
each (prompt, side) cell, mirroring the point estimate); value labels get
a * when the CI excludes 1.

Data loading reuses `janbet.trajectories.data.load_model_data` (same
judge prompt + config as trajectories.py, so the trajectory-judge cache
is shared). CACHE_ONLY=True fails loudly on any cache miss.

Output: ONE stacked figure (cot_len_ratios_by_first_estimate.pdf) in the
giraffes section of the gitignored Overleaf clone -- top panel: CoT
length in words; bottom panel: number of in-CoT estimates extracted by
the trajectory judge. Panels share the x axis (model tick labels only on
the bottom, family headers only on the top).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_style import HEADER_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import

from donation_bet.trajectories.data import load_model_data
from shared.runner import CacheOnlyMiss

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "final_data"
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes" / "bias_vs_cot_len"

# Same families/membership as trajectories.py SUMMARY_MODEL_GROUPS.
MODEL_GROUPS = [
    ("Claude", [
        "claude-opus-4.7-high",
        "claude-opus-4.7-xhigh",
        "claude-opus-4.7-max",
        "claude-opus-4.8-high",
        "claude-opus-4.8-max",
        "claude-fable-5-high",
    ]),
    ("GPT", [
        "gpt-5.2-medium",
        "gpt-5.4-medium",
        "gpt-5.5-medium",
        "gpt-5.5-high",
    ]),
    ("Gemini", [
        "gemini-2.5-pro",
        "gemini-3.1-pro-medium",
        "gemini-3.1-pro-high",
        "gemini-3.5-flash-high",
    ]),
    ("Qwen", [
        "qwen3.5-35",
        "qwen3.6-35",
    ]),
    ("Kimi", [
        "kimi-k2.5",
        "kimi-k2.6",
    ]),
]
MODEL_NAMES = [mk for _, g in MODEL_GROUPS for mk in g]

EXPERIMENT = "main_experiment_accurate"
CACHE_ONLY = True
N_BOOT = 2000

# Same conservative word regex as bias_vs_cot_len.py.
_WORD_PATTERN = r"[A-Za-z]{2,}"

# Family palette, kept in sync with plot_biases.py: Claude gets the
# reserved orange; other families take tab10-minus-orange in order.
CLAUDE_ORANGE = "#ff7f0e"
MODEL_COLORS = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
                if c != CLAUDE_ORANGE]


def _family_colors(model_groups):
    colors, i = {}, 0
    for label, _ in model_groups:
        if label.startswith("Claude"):
            colors[label] = CLAUDE_ORANGE
        else:
            colors[label] = MODEL_COLORS[i % len(MODEL_COLORS)]
            i += 1
    return colors


FAMILY_COLORS = _family_colors(MODEL_GROUPS)


def _first_on_good_side(direction, first, threshold):
    """Same convention as get_main_dfs._add_good_side."""
    if direction == "below_good":
        return first <= threshold
    if direction == "above_good":
        return first > threshold
    raise ValueError(f"unexpected direction {direction!r}")


def build_rows(trajectory_df):
    """Directional rows with a parseable trajectory -> length metrics +
    whether the FIRST in-CoT estimate is on the good side."""
    sub = trajectory_df[
        trajectory_df["direction"].isin(["below_good", "above_good"])
    ].copy()
    sub = sub[sub["trajectory"].apply(
        lambda t: isinstance(t, list) and len(t) >= 1)]
    sub = sub[sub["threshold"].notna()]
    sub["cot_words"] = (
        sub["reasoning"].fillna("").astype(str).str.count(_WORD_PATTERN)
    )
    sub["n_estimates"] = sub["trajectory"].apply(len)
    sub["first_good"] = sub.apply(
        lambda r: _first_on_good_side(
            r["direction"], float(r["trajectory"][0]), float(r["threshold"])),
        axis=1,
    )
    return sub[["prompt_key", "cot_words", "n_estimates", "first_good"]]


def _per_prompt_cells(rows, value_col):
    """[(good_values, bad_values)] per prompt with both groups non-empty."""
    cells = []
    for _pk, pk_sub in rows.groupby("prompt_key"):
        good = pk_sub[pk_sub["first_good"]][value_col].to_numpy(float)
        bad = pk_sub[~pk_sub["first_good"]][value_col].to_numpy(float)
        if len(good) and len(bad):
            cells.append((good, bad))
    return cells


def ratio_with_ci(rows, value_col, n_boot=N_BOOT, seed=0):
    """(ratio, lo, hi): mean over prompts of the per-prompt bad/good
    ratio of mean ``value_col``, with a 95% hierarchical-bootstrap CI
    (rollouts resampled within each (prompt, side) cell, per-prompt
    ratios recomputed, then averaged -- mirroring the point estimate)."""
    cells = _per_prompt_cells(rows, value_col)
    if not cells:
        return float("nan"), float("nan"), float("nan")

    ratio = float(np.mean([b.mean() / g.mean() for g, b in cells]))

    rng = np.random.default_rng(seed)

    def boot_cell_means(arr):
        return arr[rng.integers(0, len(arr), (n_boot, len(arr)))].mean(axis=1)

    # (n_boot, n_prompts) matrix of per-prompt bootstrap ratios.
    ratio_cols = [boot_cell_means(b) / boot_cell_means(g) for g, b in cells]
    ratios = np.column_stack(ratio_cols).mean(axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(ratio), float(lo), float(hi)


# Metric variants (panel order in the stacked figure) -> y-axis label.
METRICS = {
    "cot_words": (
        "Relative CoT length in words\nafter bad-side vs good-side start"
    ),
    "n_estimates": (
        "Relative number of in-CoT estimates\n"
        "after bad-side vs good-side start"
    ),
}


# %% --- Load models and compute ratios (per metric) ---

by_key = {value_col: {} for value_col in METRICS}
for model_name in MODEL_NAMES:
    try:
        _df, trajectory_df, display_name = load_model_data(
            model_name, experiment=EXPERIMENT, cache_only=CACHE_ONLY,
        )
    except CacheOnlyMiss as e:
        print(f"[skip] {model_name}: {e}")
        for value_col in METRICS:
            by_key[value_col][model_name] = {
                "model_key": model_name, "display": model_name,
                "ratio": float("nan"), "lo": float("nan"),
                "hi": float("nan"),
            }
        continue
    rows = build_rows(trajectory_df)
    for value_col in METRICS:
        ratio, lo, hi = ratio_with_ci(rows, value_col)
        by_key[value_col][model_name] = {
            "model_key": model_name, "display": display_name,
            "ratio": ratio, "lo": lo, "hi": hi,
        }
        print(f"{model_name:24s} {value_col:12s} ratio={ratio:.3f}  "
              f"CI=[{lo:.3f}, {hi:.3f}]")


# %% --- Plot ---

def _plot_ratio_on_ax(ax, model_groups, by_key, ylabel,
                      show_xticklabels=True, show_group_headers=True):
    """One ratio panel drawn on ``ax`` (bars anchored at 1, bootstrap CI
    error bars, * on values whose CI excludes 1, family separators)."""
    ordered_keys = [mk for _, g in model_groups for mk in g]
    recs = [by_key[mk] for mk in ordered_keys]
    xs = np.arange(len(recs))
    vals = np.array([r["ratio"] for r in recs])
    lo = np.array([r["lo"] for r in recs])
    hi = np.array([r["hi"] for r in recs])
    significant = (lo > 1) | (hi < 1)

    bar_colors = []
    for label, g in model_groups:
        bar_colors.extend([FAMILY_COLORS[label]] * len(g))

    ax.bar(xs, vals - 1, bottom=1, color=bar_colors,
           edgecolor="white", linewidth=0.5)
    ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="none",
                ecolor="black", elinewidth=1, capsize=4)

    for x, v, h, sig in zip(xs, vals, hi, significant):
        label = "n/a" if np.isnan(v) else f"{v:.2f}" + ("*" if sig else "")
        ax.text(x, (1 if np.isnan(h) else h) + 0.005, label,
                ha="center", va="bottom", fontsize=VALUE_FS)

    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    if show_xticklabels:
        ax.set_xticklabels([r["display"] for r in recs],
                           rotation=30, ha="right")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    finite_lo = lo[np.isfinite(lo)]
    finite_hi = hi[np.isfinite(hi)]
    ymin = min(0.97, (finite_lo.min() if finite_lo.size else 1.0) - 0.02)
    ymax = (finite_hi.max() if finite_hi.size else 1.1) + 0.08
    ax.set_ylim(ymin, ymax)

    cumulative = 0
    for label, g in model_groups:
        start = cumulative
        end = cumulative + len(g) - 1
        center = (start + end) / 2
        cumulative += len(g)
        if cumulative < len(xs):
            ax.axvline(cumulative - 0.5, color="black",
                       linewidth=0.8, alpha=0.5, linestyle="--")
        if show_group_headers:
            ax.text(center, ymax - 0.01, label, ha="center", va="top",
                    fontsize=HEADER_FS, fontweight="bold")


def plot_ratio_stack(model_groups, by_key, metrics, fname=None):
    """All metric panels stacked vertically, sharing the x axis; only the
    bottom panel shows the model tick labels, only the top one the family
    headers."""
    n = len(metrics)
    n_models = sum(len(g) for _, g in model_groups)
    fig, axes = plt.subplots(
        n, 1, figsize=(max(6, 0.55 * n_models + 2), 3.8 * n), sharex=True,
    )
    axes = np.atleast_1d(axes)
    for i, (ax, (value_col, ylabel)) in enumerate(
            zip(axes, metrics.items())):
        _plot_ratio_on_ax(
            ax, model_groups, by_key[value_col], ylabel,
            show_xticklabels=(i == n - 1),
            show_group_headers=(i == 0),
        )
    plt.tight_layout()
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, bbox_inches="tight")
    plt.show()


plot_ratio_stack(
    MODEL_GROUPS, by_key, METRICS,
    fname=FIG_DIR / "cot_len_ratios_by_first_estimate.pdf",
)

# %%
