# %%
"""Framing-sweep bias figures (paper appendix "Sensitivity to different framings"):

    framing_bias_variants.pdf  -- base and the wording variants V1-V5
    framing_bias_charities.pdf -- charity pairs C1-C5 (zoomed y-axis)

Reads the cached rollouts of the ``main_experiment_small_f5_*`` experiments
(10 baseline + 10/direction/task, all 9 estimation questions) for four models
and plots the pooled value-leakage bias per framing with 95% bootstrap CIs
(2.5/97.5 percentiles over 5000 resamples of the pooled directional answers,
matching the paper's other bootstrap CIs).

The "base" bars are the good-cause/bad-cause results from the MAIN experiment
(fig:bias-per-model) and are kept as the constants below; the V/C bars are
computed from the framing-sweep caches.

Charity pairs are re-oriented to the models' average lean (mostly positive);
C5 splits. Run cache-only from the data submodule:

    python -m donation_bet.plot_framing_bias

Note: unlike the other donation_bet plots this intentionally does NOT import
shared.plot_style -- the paper figures were rendered with matplotlib defaults.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shared.runner as runner
from shared.get_main_dfs import get_main_dfs

# Use the release caches in the data submodule.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "framing"

COLORS = ["#5B8FF9", "#36A99B", "#9B6BD6", "#F2994A"]
NAMES = ["claude-opus-4.7-high", "claude-opus-4.8-high",
         "gemini-3.1-pro-high", "gpt-5.5-high"]

# (framing id, experiment name) -- see shared/prompts/thresholds.py
# _VARIANT_TEMPLATES for the exact prompt wordings.
FRAMING_EXPERIMENTS = {
    "V1": "main_experiment_small_f5_good",
    "V2": "main_experiment_small_f5_unicef",
    "V3": "main_experiment_small_f5_beer",
    "V4": "main_experiment_small_f5_naked",
    "V5": "main_experiment_small_f5_pick",
    "C1": "main_experiment_small_f5_dwb_greenpeace",
    "C2": "main_experiment_small_f5_redcross_unicef",
    "C3": "main_experiment_small_f5_mnm_amf",
    "C4": "main_experiment_small_f5_feeding_good360",
    "C5": "main_experiment_small_f5_wwf_tnc",
}

# Charity pairs re-oriented to the leaned-toward charity (positive bias).
FLIP = {"C2", "C3", "C5"}

# base taken from the MAIN experiment (fig:bias-per-model), read off that
# figure: value with 95% CI [lo, hi]. (V1-V5/C1-C5 come from the sweep caches.)
BASE_MAIN = {
    "claude-opus-4.7-high": (0.65, 0.55, 0.75),
    "claude-opus-4.8-high": (0.81, 0.72, 0.90),
    "gemini-3.1-pro-high":  (0.77, 0.67, 0.87),
    "gpt-5.5-high":         (0.18, 0.04, 0.32),
}


# %%
# Pooled directional outcomes per (framing, model): a 0/1 array over all
# directional answers, per task ordered ones-first (the bootstrap resamples
# with replacement, so only counts matter; the fixed order keeps the seeded
# draws identical to the original figure-generation run).
def load_arrays():
    arrays = {}
    for framing, experiment in FRAMING_EXPERIMENTS.items():
        main_dfs = get_main_dfs(experiment, NAMES, cache_only=True)
        for model in NAMES:
            df, _thresholds, _display = main_dfs[model]
            out = []
            for _prompt_key, sub in df.groupby("prompt_key"):
                sub_dir = sub[sub["direction"].isin(["below_good", "above_good"])]
                n = len(sub_dir)
                ng = int(sub_dir["on_good_side"].sum())
                if framing in FLIP:
                    ng = n - ng
                out += [1] * ng + [0] * (n - ng)
            arrays[framing, model] = np.array(out)
    return arrays


ARRAYS = load_arrays()


def boot(a, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    bs = 2 * rng.choice(a, size=(B, n), replace=True).mean(1) - 1
    return 2 * a.mean() - 1, *np.percentile(bs, [2.5, 97.5])


# %%
def panel(framings, labels, ylim, fname, zero=True):
    nb = len(NAMES)
    x = np.arange(len(framings))
    bw = 0.20
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    print(f"\n== {fname} ==")
    for i, m in enumerate(NAMES):
        vals = []
        lo = []
        hi = []
        for fr in framings:
            if fr == "base":
                b, l, h = BASE_MAIN[m]
            else:
                b, l, h = boot(ARRAYS[fr, m])
            vals.append(b)
            lo.append(b - l)
            hi.append(h - b)
            print(f"  {fr:<5}{m:<22} {b:+.3f}  CI[{l:+.3f},{h:+.3f}]")
        ax.bar(x + (i - (nb - 1) / 2) * bw, vals, bw, label=m, color=COLORS[i],
               edgecolor="black", linewidth=0.4, zorder=3, yerr=[lo, hi],
               capsize=2, error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="0.25"))
    if zero:
        ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, linespacing=1.25)
    ax.set_xlim(-0.5, len(framings) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(r"Value-leakage bias $p_{\mathrm{bias}}$", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.yaxis.grid(True, color="0.85", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              prop={"family": "monospace", "size": 9}, columnspacing=1.6,
              handletextpad=0.5)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


panel(["base", "V1", "V2", "V3", "V4", "V5"],
      ["base\ngood cause\nvs bad cause", "V1\ngood cause\nvs nothing",
       "V2\nUNICEF\nvs nothing", "V3\ngood cause\nvs beers",
       "V4\nnothing\nvs run naked", "V5\nuser picks\nvs friend picks"],
      (-0.2, 1.0), "framing_bias_variants")
panel(["C1", "C2", "C3", "C4", "C5"],
      ["C1\nDWB\nvs Greenpeace", "C2\nUNICEF\nvs Red Cross",
       "C3\nAMF\nvs Malaria No More", "C4\nFeeding America\nvs Good360",
       "C5\nNature Conservancy\nvs WWF"],
      (-0.3, 0.6), "framing_bias_charities")
print(f"\nwrote figures -> {FIG_DIR}")
