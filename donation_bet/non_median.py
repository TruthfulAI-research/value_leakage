# %%
"""Donation Bet with non-median thresholds: violin plot per scenario.

The main experiment always sets the threshold to the MEDIAN of the
model's baseline estimates. Here we additionally run the same prompts
with thresholds at the LO_PCT/HI_PCT percentiles (default 20/80) of the
baseline distribution, giving four new scenarios:

    below_good_20, below_good_80, above_good_20, above_good_80

plus the two old median scenarios (below_good_50, above_good_50), for
six violins total per model. Estimates are standardized per row as
``(estimate - threshold) / threshold`` -- the same normalization as the
trajectory plots (trajectories.py, normalize=True, subtract="threshold")
-- so all prompts can be pooled into one violin.

Mechanics: the runner's threshold spec ``{"pct_range": (lo, hi), "n": 2}``
already produces [p_lo, p_hi] via linspace, so we just clone
main_experiment_accurate with that spec and call run_thresholds_experiment
directly. We pass the ORIGINAL experiment_name for the estimate-judge
cache, so baseline judgings are reused and new direction judgings land in
the same shared cache. Baselines (and thus percentile computation) are
fully cached; only the 4 new direction scenarios need fresh sampling
(2 directions x 2 thresholds x n_per_threshold x 9 prompts rollouts per
model) the first time.

Figures go (PDF into the gitignored Overleaf clone under
giraffes/non_median, PNG sibling next to it).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared import plot_style  # noqa: F401  applies shared figure sizing on import
from donation_bet.bias_metrics import (
    balanced_bias_bootstrap_ci95,
)

import shared.runner as runner
from shared.experiments import THRESHOLD_EXPERIMENTS
from shared.get_main_dfs import get_main_dfs, _add_good_side
from shared.models import MODELS
from shared.runner import run_thresholds_experiment

# --- Cache redirect (mirrors plot_biases.py / trajectories.py) ---
DATA_ROOT = Path(runner.__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes" / "non_median"

MODEL_NAME = "claude-opus-4.8-high"
EXPERIMENT = "main_experiment_accurate"
LO_PCT = 20
HI_PCT = 80
# False: sample the missing percentile-threshold rollouts (the median data
# and all baselines are cached either way). True: fail loudly on any miss.
CACHE_ONLY = False
# Drop rows whose estimate is outside [threshold/f, threshold*f] before
# plotting -- same symmetric filter semantics as trajectories.py. A single
# order-of-magnitude judge slip otherwise stretches a violin into a line.
# Set to None to keep everything.
OUTLIER_FACTOR = 10


# %% --- Load the old (median) scenarios from cache ---

median_df, median_thresholds, display_name = get_main_dfs(
    EXPERIMENT, [MODEL_NAME], cache_only=True,
)[MODEL_NAME]


# %% --- Run/load the percentile-threshold scenarios ---
# Same prompts and sample sizes as EXPERIMENT, only the threshold spec
# differs. n=2 with pct_range=(LO, HI) makes linspace return exactly
# [p_LO, p_HI]. We reuse EXPERIMENT as experiment_name so the estimate
# judge cache is shared with the main experiment.

pct_experiment = {
    **THRESHOLD_EXPERIMENTS[EXPERIMENT],
    "thresholds": {"pct_range": (LO_PCT, HI_PCT), "n": 2},
}
pct_raw_df, pct_thresholds, _ = run_thresholds_experiment(
    MODELS[MODEL_NAME], pct_experiment, MODEL_NAME, EXPERIMENT,
    cache_only=CACHE_ONLY,
)
pct_df = _add_good_side(pct_raw_df)

print("Per-prompt thresholds (lo / median / hi):")
for pk in pct_thresholds:
    lo, hi = pct_thresholds[pk]
    med = median_thresholds[pk][0]
    print(f"  {pk:28s} {lo:>15,d}  {med:>15,d}  {hi:>15,d}")


# %% --- Build one long df: scenario label + standardized estimate ---

SCEN_MEDIAN = {"below_good": "below_median_good",
               "above_good": "above_median_good"}


def _directional(df):
    return df[df["direction"].isin(["below_good", "above_good"])].copy()


def build_scenarios(median_df, pct_df, pct_thresholds):
    """Rows: prompt_key, scenario, rel = (estimate - threshold)/threshold."""
    med = _directional(median_df)
    med["scenario"] = med["direction"].map(SCEN_MEDIAN)

    pct = _directional(pct_df)

    def pct_scenario(r):
        lo, hi = pct_thresholds[r["prompt_key"]]
        pct_label = LO_PCT if r["threshold"] == lo else HI_PCT
        side = r["direction"].removesuffix("_good")
        return f"{side}_{pct_label}_good"

    pct["scenario"] = pct.apply(pct_scenario, axis=1)

    both = pd.concat([med, pct], ignore_index=True)
    both["rel"] = (both["estimate"] - both["threshold"]) / both["threshold"]
    return both[["prompt_key", "direction", "scenario", "threshold",
                 "estimate", "rel", "on_good_side"]]


scen_df = build_scenarios(median_df, pct_df, pct_thresholds)

if OUTLIER_FACTOR is not None:
    ok = ((scen_df["estimate"] >= scen_df["threshold"] / OUTLIER_FACTOR)
          & (scen_df["estimate"] <= scen_df["threshold"] * OUTLIER_FACTOR))
    n_drop = int((~ok).sum())
    print(f"Outlier filter (factor {OUTLIER_FACTOR}): dropping "
          f"{n_drop}/{len(scen_df)} rows")
    scen_df = scen_df[ok]

SCENARIO_ORDER = [
    f"below_{LO_PCT}_good", f"above_{LO_PCT}_good",
    f"below_{HI_PCT}_good", f"above_{HI_PCT}_good",
    "below_median_good", "above_median_good",
]
# Vertical separator drawn before this scenario (percentiles | median).
SEPARATOR_BEFORE = "below_median_good"

print("\nPer-scenario summary:")
summary = (
    scen_df.groupby("scenario")
    .agg(n=("rel", "size"), rel_median=("rel", "median"),
         rel_mean=("rel", "mean"), p_good=("on_good_side", "mean"))
    .reindex(SCENARIO_ORDER)
)
summary["bias"] = 2 * summary["p_good"] - 1
print(summary.round(3).to_string())


# %% --- Violin plot ---

SCENARIO_COLORS = {
    f"below_{LO_PCT}_good": "#9ecae1",
    f"above_{LO_PCT}_good": "#bcbddc",
    f"below_{HI_PCT}_good": "#08519c",
    f"above_{HI_PCT}_good": "#54278f",
    "below_median_good":    "#3182bd",
    "above_median_good":    "#756bb1",
}

# Shared font/line settings for both violin plots.
YLABEL_FONTSIZE = 14
XTICK_FONTSIZE = 11
XTICK_ROTATION = 15
LEGEND_FONTSIZE = YLABEL_FONTSIZE
THRESHOLD_LINEWIDTH = 1.2


def plot_scenario_violins(scen_df, filename=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(SCENARIO_ORDER))
    data = [scen_df.loc[scen_df["scenario"] == s, "rel"].to_numpy()
            for s in SCENARIO_ORDER]

    parts = ax.violinplot(
        [d for d in data if len(d)],
        positions=[p for p, d in zip(positions, data) if len(d)],
        widths=0.8, showmedians=False, showextrema=False,
    )
    drawn = [s for s, d in zip(SCENARIO_ORDER, data) if len(d)]
    for body, s in zip(parts["bodies"], drawn):
        body.set_facecolor(SCENARIO_COLORS[s])
        body.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=THRESHOLD_LINEWIDTH,
               linestyle="--", label="threshold")
    sep_idx = SCENARIO_ORDER.index(SEPARATOR_BEFORE)
    ax.axvline(sep_idx - 0.5, color="gray", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{s}\n(n={len(d)})" for s, d in zip(SCENARIO_ORDER, data)],
        fontsize=XTICK_FONTSIZE, rotation=XTICK_ROTATION, ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("(estimate - threshold) / threshold",
                  fontsize=YLABEL_FONTSIZE)
    ax.set_ylim(top=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
        root, ext = os.path.splitext(filename)
        if ext.lower() != ".png":
            fig.savefig(root + ".png", bbox_inches="tight", dpi=150)
    plt.show()


plot_scenario_violins(
    scen_df,
    filename=str(FIG_DIR / f"non_median_violins_{MODEL_NAME}_"
                           f"p{LO_PCT}_p{HI_PCT}.pdf"),
)


# %% --- Single-prompt violin plot: absolute estimates + per-column thresholds ---

SELECTED_PROMPT = "v1_crochet_accurate"


def plot_scenario_violins_absolute(scen_df, prompt_key, filename=None):
    """Same layout as `plot_scenario_violins`, but for ONE prompt, with
    raw estimates on the y axis and each column's own threshold marked
    by a short horizontal line."""
    sub = scen_df[scen_df["prompt_key"] == prompt_key]
    lo_thr, hi_thr = pct_thresholds[prompt_key]
    med_thr = median_thresholds[prompt_key][0]
    scen_threshold = {
        f"below_{LO_PCT}_good": lo_thr, f"above_{LO_PCT}_good": lo_thr,
        f"below_{HI_PCT}_good": hi_thr, f"above_{HI_PCT}_good": hi_thr,
        "below_median_good": med_thr, "above_median_good": med_thr,
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(SCENARIO_ORDER))
    data = [sub.loc[sub["scenario"] == s, "estimate"].to_numpy(dtype=float)
            for s in SCENARIO_ORDER]

    parts = ax.violinplot(
        [d for d in data if len(d)],
        positions=[p for p, d in zip(positions, data) if len(d)],
        widths=0.8, showmedians=False, showextrema=False,
    )
    drawn = [s for s, d in zip(SCENARIO_ORDER, data) if len(d)]
    for body, s in zip(parts["bodies"], drawn):
        body.set_facecolor(SCENARIO_COLORS[s])
        body.set_alpha(0.7)

    for pos, s in zip(positions, SCENARIO_ORDER):
        ax.hlines(scen_threshold[s], pos - 0.45, pos + 0.45,
                  color="black", linewidth=THRESHOLD_LINEWIDTH,
                  linestyle="--", zorder=3)
    # One legend entry for the threshold marks.
    ax.plot([], [], color="black", linewidth=THRESHOLD_LINEWIDTH,
            linestyle="--", label="threshold")

    sep_idx = SCENARIO_ORDER.index(SEPARATOR_BEFORE)
    ax.axvline(sep_idx - 0.5, color="gray", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{s}\n(n={len(d)})" for s, d in zip(SCENARIO_ORDER, data)],
        fontsize=XTICK_FONTSIZE, rotation=XTICK_ROTATION, ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Number of half-stitches", fontsize=YLABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1e9:g}B")
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
        root, ext = os.path.splitext(filename)
        if ext.lower() != ".png":
            fig.savefig(root + ".png", bbox_inches="tight", dpi=150)
    plt.show()


plot_scenario_violins_absolute(
    scen_df, SELECTED_PROMPT,
    filename=str(FIG_DIR / f"non_median_absolute_{MODEL_NAME}_"
                           f"{SELECTED_PROMPT}_p{LO_PCT}_p{HI_PCT}.pdf"),
)


# %% --- Cross-model bias bars: p20 / median / p80 threshold scenarios ---
# Each bar pairs the below-good and above-good versions of one threshold
# scenario (e.g. below_20_good + above_20_good). Their good-side rates receive
# 50/50 weight, and questions receive equal weight, so unequal judge-parse
# survival cannot move the estimand away from its calibrated neutral point: an
# unbiased model lands on the good side about 20% of the time below p20 and 80%
# above p20.

BAR_MODELS = [
    "claude-opus-4.8-high",
    "gpt-5.5-medium",
    "gemini-3.1-pro-medium",
]

GROUP_OF_SCENARIO = {
    f"below_{LO_PCT}_good": str(LO_PCT), f"above_{LO_PCT}_good": str(LO_PCT),
    "below_median_good": "median", "above_median_good": "median",
    f"below_{HI_PCT}_good": str(HI_PCT), f"above_{HI_PCT}_good": str(HI_PCT),
}
GROUP_ORDER = [str(LO_PCT), "median", str(HI_PCT)]
GROUP_COLORS = {str(LO_PCT): "#a6bddb", "median": "#3690c0",
                str(HI_PCT): "#034e7b"}


def load_scen_df(model_key):
    """(scen_df, display_name) for one model: median + percentile scenarios,
    same pipeline as the single-model sections above."""
    m_df, m_thresholds, disp = get_main_dfs(
        EXPERIMENT, [model_key], cache_only=True,
    )[model_key]
    p_raw, p_thresholds, _ = run_thresholds_experiment(
        MODELS[model_key], pct_experiment, model_key, EXPERIMENT,
        cache_only=CACHE_ONLY,
    )
    sdf = build_scenarios(m_df, _add_good_side(p_raw), p_thresholds)
    return sdf, disp


bar_rows = []
for _model in BAR_MODELS:
    if _model == MODEL_NAME:
        _sdf, _disp = scen_df, display_name
    else:
        _sdf, _disp = load_scen_df(_model)
    rec = {"model_key": _model, "display": _disp}
    for g in GROUP_ORDER:
        scens = [s for s, gg in GROUP_OF_SCENARIO.items() if gg == g]
        rows = _sdf[_sdf["scenario"].isin(scens)]
        bias, err_low, err_high = balanced_bias_bootstrap_ci95(
            rows, prompt_keys=rows["prompt_key"].drop_duplicates(),
        )
        n = len(rows)
        rec[f"bias_{g}"] = bias
        rec[f"err_low_{g}"] = err_low
        rec[f"err_high_{g}"] = err_high
        rec[f"n_{g}"] = n
    bar_rows.append(rec)

bars_df = pd.DataFrame(bar_rows)
print(bars_df.round(3).to_string(index=False))


def plot_bias_bars(bars_df, filename=None):
    fig, ax = plt.subplots(figsize=(9, 2.25))
    x = np.arange(len(bars_df))
    w = 0.26
    for gi, g in enumerate(GROUP_ORDER):
        label = f"threshold at p{g}" if g != "median" else "threshold at median"
        ax.bar(x + (gi - 1) * w, bars_df[f"bias_{g}"], width=w,
               yerr=[bars_df[f"err_low_{g}"],
                     bars_df[f"err_high_{g}"]], capsize=3,
               color=GROUP_COLORS[g], label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bars_df["display"].tolist(),
                       fontsize=XTICK_FONTSIZE)
    ax.set_ylabel("Bias metric", fontsize=YLABEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
        root, ext = os.path.splitext(filename)
        if ext.lower() != ".png":
            fig.savefig(root + ".png", bbox_inches="tight", dpi=150)
    plt.show()


plot_bias_bars(
    bars_df,
    filename=str(FIG_DIR / f"non_median_bias_bars_p{LO_PCT}_p{HI_PCT}.pdf"),
)

# %%
