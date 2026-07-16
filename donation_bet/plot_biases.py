# %%
import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_style import HEADER_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import

from shared.experiments import THRESHOLD_EXPERIMENTS
from donation_bet.bias_metrics import (
    balanced_bias_bootstrap_ci95,
    balanced_bias_score,
)
from shared.get_main_dfs import get_main_dfs

# Use the new cache
from pathlib import Path
import shared.runner as runner
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")
# Figure destination: every plot is written (PDF only) into the giraffes section
# of the gitignored Overleaf clone -- the single figure output location.
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes"

MODEL_GROUPS = [
    ("Claude", [
        # "claude-opus-4.1",
        "claude-opus-4.5-high",
        "claude-opus-4.6-high",
        "claude-opus-4.6-max",
        "claude-opus-4.7-high",
        # "claude-opus-4.7-xhigh",
        "claude-opus-4.7-max",
        "claude-opus-4.8-high",
        "claude-opus-4.8-max",
        "claude-fable-5-high",
    ]),
    ("GPT", [
        # "gpt-5.1-medium",  # kept in sync with plot_cot_categories_v2
        "gpt-5.2-medium",
        "gpt-5.4-medium",
        "gpt-5.5-medium",
        "gpt-5.5-high",
        "gpt-5.6-sol-medium",
    ]),
    ("Gemini", [
        "gemini-2.5-pro",
        # "gemini-3-flash-high",
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
MODEL_NAMES = [mk for _, group in MODEL_GROUPS for mk in group]

EXPERIMENT = "main_experiment_accurate"
CACHE_ONLY = False


# %%
def bias_score(df):
    """Signed Donation Bet bias with equal weight on the two directions."""
    return balanced_bias_score(df)


def bias_score_ci95(df):
    """Direction-balanced bias plus bootstrap 95% CI deltas."""
    return balanced_bias_bootstrap_ci95(df)


# %%
prompt_keys = THRESHOLD_EXPERIMENTS[EXPERIMENT]["prompts"]
main_dfs = get_main_dfs(EXPERIMENT, MODEL_NAMES, cache_only=CACHE_ONLY)

rows = []
display_names = {}
per_model_dfs = {}
for model_key, (df, _thresholds, display_name) in main_dfs.items():
    display_names[model_key] = display_name
    per_model_dfs[model_key] = df
    for pk in prompt_keys:
        pk_df = df[df["prompt_key"] == pk]
        bias, ci_low, ci_high = bias_score_ci95(pk_df)
        rows.append({
            "model": display_name,
            "prompt_key": pk,
            "bias": bias,
            "bias_ci95_low": ci_low,
            "bias_ci95_high": ci_high,
        })

results_df = pd.DataFrame(rows)
# print(results_df.to_string(index=False))

# %%
# Two palettes, by role:
#  * MODEL_COLORS -- per model family: the full matplotlib tab10 cycle
#    (INCLUDING green & red), with orange held out and reserved for Claude. So
#    Claude=orange and the rest take tab10-minus-orange in MODEL_GROUPS order
#    (GPT=blue, Gemini=green, Qwen=red, Kimi=purple).
#  * COLORS -- generic result-category bars (e.g. the per-prompt bars):
#    tab10/tab20 with the semantically loaded hues removed (no orange = Claude,
#    no green/red = good/bad decomposition), so the bars never clash with those
#    meanings; grey + light variants extend the rotation for the 9 prompts.
CLAUDE_ORANGE = "#ff7f0e"
MODEL_COLORS = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
                if c != CLAUDE_ORANGE]
COLORS = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#aec7e8", "#c5b0d5", "#c49c94", "#f7b6d2"]


def _family_colors(model_groups):
    """Pin a color per family by name so dropping a family from the plotted
    subset doesn't reshuffle the rest. Claude always gets the reserved orange;
    every other family takes the tab10-minus-orange palette in MODEL_GROUPS
    order (so green/red are in play for non-Claude families)."""
    colors, i = {}, 0
    for label, _ in model_groups:
        if label.startswith("Claude"):
            colors[label] = CLAUDE_ORANGE
        else:
            colors[label] = MODEL_COLORS[i % len(MODEL_COLORS)]
            i += 1
    return colors


FAMILY_COLORS = _family_colors(MODEL_GROUPS)
DIRECTIONS = ["baseline", "below_good", "above_good"]
# Direction colors: the project-standard grey / blue / purple trio (no
# orange/red/green). These are the colors the bubble & job_offer condition
# plots reuse, so the figures share one palette.
DIR_COLORS = {"baseline": "#7f7f7f", "below_good": "#1f77b4", "above_good": "#9467bd"}
MAX_COLS = 2


def plot_mean_bias_per_model(per_model_dfs, prompt_keys, model_groups,
                             display_names,
                             fname=None, show_group_headers=True,
                             figsize=None):
    """Mean bias per model with fixed-question bootstrap 95% CI error bars.

    ``show_group_headers``: if True (default), draw vertical dividers between
    model-family groups and write the group label above each. Set to False
    to suppress both for a cleaner look when groups have one member each or
    when the figure is going into a slide where the headers are noise.

    ``figsize``: optional (width, height) in inches. If None (default), the
    width auto-scales with the number of bars and the height is fixed at
    4.8 inches.
    """
    ordered_keys = [mk for _, g in model_groups for mk in g]
    vals, err_low, err_high = [], [], []
    for model_key in ordered_keys:
        if model_key not in per_model_dfs:
            vals.append(float("nan"))
            err_low.append(0.0)
            err_high.append(0.0)
            continue
        val, low, high = balanced_bias_bootstrap_ci95(
            per_model_dfs[model_key], prompt_keys=prompt_keys,
        )
        vals.append(val)
        err_low.append(low)
        err_high.append(high)

    ordered_displays = [display_names[mk] for mk in ordered_keys]

    bar_colors = []
    for label, g in model_groups:
        c = FAMILY_COLORS.get(label, COLORS[0])
        bar_colors.extend([c] * len(g))

    xs = list(range(len(ordered_displays)))
    if figsize is None:
        figsize = (max(6, 0.55 * len(xs) + 2), 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    heights = [0.0 if pd.isna(v) else v for v in vals]
    ax.bar(xs, heights, yerr=[err_low, err_high], color=bar_colors,
           edgecolor="white", linewidth=0.5, ecolor="black", capsize=4,
           error_kw={"linewidth": 1.0})

    for x, v, hi in zip(xs, vals, err_high):
        label = "n/a" if pd.isna(v) else f"{v:.2f}"
        ax.text(x, (0 if pd.isna(v) else v + hi) + 0.01, label,
                ha="center", va="bottom", fontsize=VALUE_FS)

    ax.set_xticks(xs)
    ax.set_xticklabels(ordered_displays, rotation=30, ha="right")
    # Default 5% x-margins leave over half a bar-slot of dead space on each
    # side; clamp so the edge gap equals the inter-bar gap (1 - 0.8 = 0.2
    # data units beyond the outer bar edges at +/-0.4 from the bar centers).
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_ylabel("Bias metric")
    # ax.set_title("Mean bias per model")
    ax.grid(True, axis="y", alpha=0.3)

    # Biases are all non-negative, so use the plain 0-1 scale -- no zero line
    # and no sub-zero padding to privilege.
    ymax = 1.0
    ax.set_ylim(0.0, ymax)

    if show_group_headers:
        cumulative = 0
        for label, g in model_groups:
            start = cumulative
            end = cumulative + len(g) - 1
            center = (start + end) / 2
            cumulative += len(g)
            if cumulative < len(xs):
                ax.axvline(cumulative - 0.5, color="black",
                           linewidth=0.8, alpha=0.5, linestyle="--")
            ax.text(center, ymax - 0.02, label, ha="center", va="top",
                    fontsize=HEADER_FS, fontweight="bold")

    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_bias_bars(results_df, fname=None):
    """One subplot per model; prompt-key bias bars with 95% CI error bars."""
    pkeys = list(results_df["prompt_key"].unique())
    models = list(results_df["model"].unique())
    n = len(models)
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols

    # Biases are all non-negative -> plain 0-1 axis, no zero line/headroom.
    ymax, ymin = 1.0, 0.0

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(5, 0.5 * len(pkeys)) * n_cols + 1,
                                      4.5 * n_rows + 0.5),
                             sharey=True, squeeze=False)
    flat = axes.flatten()
    xs = list(range(len(pkeys)))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(pkeys))]

    for ax, model in zip(flat, models):
        sub = results_df[results_df["model"] == model]
        vals = [sub[sub["prompt_key"] == pk]["bias"].values[0]
                if len(sub[sub["prompt_key"] == pk]) else float("nan")
                for pk in pkeys]
        err_low = [sub[sub["prompt_key"] == pk]["bias_ci95_low"].values[0]
                   if len(sub[sub["prompt_key"] == pk]) else 0.0
                   for pk in pkeys]
        err_high = [sub[sub["prompt_key"] == pk]["bias_ci95_high"].values[0]
                    if len(sub[sub["prompt_key"] == pk]) else 0.0
                    for pk in pkeys]
        heights = [0.0 if pd.isna(v) else v for v in vals]
        ax.bar(xs, heights, yerr=[err_low, err_high], color=bar_colors,
               edgecolor="white", linewidth=0.5, ecolor="black", capsize=4,
               error_kw={"linewidth": 1.0})
        for x, v, hi in zip(xs, vals, err_high):
            label = "n/a" if pd.isna(v) else f"{v:.2f}"
            ax.text(x, (0 if pd.isna(v) else v + hi) + 0.01, label,
                    ha="center", va="bottom", fontsize=VALUE_FS)
        ax.set_xticks(xs)
        ax.set_xticklabels(pkeys, rotation=30, ha="right")
        ax.set_ylim(ymin, ymax)
        ax.set_title(model)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in flat[n:]:
        ax.set_visible(False)
    for r in range(n_rows):
        axes[r, 0].set_ylabel("Bias")
    fig.suptitle("Bias per model and prompt")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, bbox_inches="tight")
    plt.show()


def _format_thresholds(df):
    vals = sorted(pd.to_numeric(df["threshold"], errors="coerce").dropna().unique().tolist())
    if not vals:
        return ""
    return "thresholds: " + ", ".join(f"{v:,.0f}" for v in vals)


def _plot_violin(ax, df, prompt_key):
    """Draw three violins (baseline / below_good / above_good) with threshold line."""
    parts_data, positions, colors, tick_labels = [], [], [], []
    for i, d in enumerate(DIRECTIONS):
        vals = pd.to_numeric(df.loc[df["direction"] == d, "estimate"],
                             errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        parts_data.append(vals)
        positions.append(i)
        colors.append(DIR_COLORS[d])
        tick_labels.append(f"{d}\n(n={len(vals)})")

    if parts_data:
        vp = ax.violinplot(parts_data, positions=positions,
                           showmedians=True, showextrema=False)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")

        for t in df["threshold"].dropna().unique():
            ax.axhline(t, color="red", linestyle="--",
                       linewidth=1, alpha=0.6)

    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels)
    ax.grid(True, axis="y", alpha=0.3)


def plot_violins_by_prompt(per_model_dfs, prompt_keys, display_names,
                           save_dir=None):
    """One figure per prompt_key; subplots are models."""
    for pk in prompt_keys:
        n = len(per_model_dfs)
        n_cols = min(n, MAX_COLS)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols + 1, 4.2 * n_rows + 0.8),
                                 sharey=True, squeeze=False)
        flat = axes.flatten()
        for ax, (mk, df) in zip(flat, per_model_dfs.items()):
            pk_df = df[df["prompt_key"] == pk]
            _plot_violin(ax, pk_df, pk)
            threshold_str = _format_thresholds(pk_df)
            title = (f"{display_names[mk]}\n{threshold_str}"
                     if threshold_str else display_names[mk])
            ax.set_title(title)
        for ax in flat[n:]:
            ax.set_visible(False)
        for r in range(n_rows):
            axes[r, 0].set_ylabel("Estimate")
        fig.suptitle(pk)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / f"violin_prompt_{pk}.pdf",
                        bbox_inches="tight")
        plt.show()


def plot_violins_by_model(per_model_dfs, prompt_keys, display_names,
                          save_dir=None):
    """One figure per model; subplots are prompt_keys."""
    for mk, df in per_model_dfs.items():
        n = len(prompt_keys)
        n_cols = min(n, MAX_COLS)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols + 1, 4.2 * n_rows + 0.8),
                                 squeeze=False)
        flat = axes.flatten()
        for ax, pk in zip(flat, prompt_keys):
            pk_df = df[df["prompt_key"] == pk]
            _plot_violin(ax, pk_df, pk)
            threshold_str = _format_thresholds(pk_df)
            title = f"{pk}\n{threshold_str}" if threshold_str else pk
            ax.set_title(title)
        for ax in flat[n:]:
            ax.set_visible(False)
        for r in range(n_rows):
            axes[r, 0].set_ylabel("Estimate")
        fig.suptitle(display_names[mk])
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / f"violin_{mk}.pdf",
                        bbox_inches="tight")
        plt.show()


def plot_single_violin(model_key, prompt_key, fname=None, cache_only=CACHE_ONLY):
    """Single (model, prompt) violin triple -- baseline / below_good /
    above_good -- for the paper's example-distribution figure.

    Loads just this one model so the choice of example model is independent of
    MODEL_GROUPS and leaves the multi-model bar plots above untouched.
    """
    single = get_main_dfs(EXPERIMENT, [model_key], cache_only=cache_only)
    df, _thresholds, display_name = single[model_key]
    pk_df = df[df["prompt_key"] == prompt_key]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_violin(ax, pk_df, prompt_key)
    ax.set_ylabel("Estimate")
    threshold_str = _format_thresholds(pk_df)
    problem = prompt_key.removeprefix("v1_").removesuffix("_accurate")
    title_lines = [f"{display_name} — {problem} problem"]
    if threshold_str:
        title_lines.append(threshold_str)
    ax.set_title("\n".join(title_lines))
    plt.tight_layout()
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, bbox_inches="tight")
    plt.show()


# %%
# plot_mean_bias_per_model(results_df, MODEL_GROUPS, display_names)
# Bias-per-model-and-prompt grid: disabled (not referenced in the paper).
# plot_bias_bars(results_df, fname=FIG_DIR / "bias_per_model_and_prompt.pdf")
# Individual per-model / per-prompt violin grids: disabled. The paper now uses a
# single example violin (opus 4.7 max, giraffes problem), produced below.
# plot_violins_by_prompt(per_model_dfs, prompt_keys, display_names,
#                        save_dir=FIG_DIR / "biases_violins")
# plot_violins_by_model(per_model_dfs, prompt_keys, display_names,
#                       save_dir=FIG_DIR / "biases_violins")
plot_single_violin(
    "claude-opus-4.7-max",
    "v1_giraffes_accurate",
    fname=FIG_DIR / "biases_violins" / "violin_claude-opus-4.7-max_giraffes.pdf",
)
# %%
plot_mean_bias_per_model(
    per_model_dfs,
    prompt_keys,
    MODEL_GROUPS,
    display_names,
    fname=FIG_DIR / "bias_per_model.pdf",
    show_group_headers=True,
)
# %%
