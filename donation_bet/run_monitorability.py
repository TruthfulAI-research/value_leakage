# %%
"""Interactive monitorability analysis for the giraffes threshold experiments.

This is intentionally notebook-style: edit the flags below, then run cells from
top to bottom. Reusable monitor/metric/bootstrap logic lives in top-level
`monitorability.py`; this file keeps the one-off plotting and analysis cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_style import HEADER_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from donation_bet import monitorability

RunConfig = monitorability.RunConfig

# For final runs, use final_data. For smoke/local runs, switch to:
# ARTIFACT_ROOT = REPO_ROOT / "johannes" / "data"
ARTIFACT_ROOT = REPO_ROOT / "data" / "final_data"
SCRIPT_CACHE_ROOT = ARTIFACT_ROOT / "run_monitorability_cache"
DEFAULT_METRIC_PATH = REPO_ROOT / "shared" / "intervention_gmean_metric.py"
monitorability.configure_runner_cache(ARTIFACT_ROOT)

# %%
# Top-level analysis flags. This mirrors the style of
# `shared/final_scripts/plot_biases.py` and `plot_cot_categories.py`.

MODEL_GROUPS = [
    ("Claude", [
        # "claude-opus-4.1",
        "claude-opus-4.5-high",
        "claude-opus-4.6-high",
        "claude-opus-4.6-max",
        "claude-opus-4.7-high",
        "claude-opus-4.7-xhigh",
        "claude-opus-4.7-max",
        "claude-opus-4.8-max",
    ]),
    ("GPT", [
        "gpt-5.1-medium",
        "gpt-5.2-medium",
        "gpt-5.4-medium",
        "gpt-5.5-medium",
        "gpt-5.5-high",
    ]),
    ("Gemini", [
        "gemini-2.5-pro",
        "gemini-3.1-pro-medium",
        "gemini-3.1-pro-high",
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

GOOD_BAD_EXPERIMENT = "main_experiment_accurate"
CACHE_ONLY = True
MONITOR_CACHE_ONLY = False
MONITOR_CACHE_EXPERIMENT = GOOD_BAD_EXPERIMENT

# Default: real monitor labels on cached main_experiment CoTs. Set
# LIMIT_ROWS_PER_INSTANCE_ARM to a small integer for smoke tests.
COMPARISONS = ["good_bad_vs_no_hint"]
# Generate BOTH paper monitorability figures automatically: cot_only feeds
# fig:monitorability (+ component breakdown fig:monitorability-components) and
# answer_only feeds fig:answer-monitorability. (cot_and_answer also available.)
SCOPES = ["answer_only", "cot_only"]
PROMPT_FILTER = None
# test
LIMIT_ROWS_PER_INSTANCE_ARM = None

MONITOR_BACKEND = "openai"  # Use "openai" for real numbers. constant_* is plumbing-only.
MONITOR_MODEL = "gpt-5.4"
MONITOR_REASONING_EFFORT = "low"
MONITOR_MAX_OUTPUT_TOKENS = 16000
MONITOR_MAX_WORKERS = 100
ALLOW_SYNTHETIC_MONITOR = False

N_BOOTSTRAP = 100
RUN_BOOTSTRAP = True
BOOTSTRAP_MAX_WORKERS = 1
SELECTION_FRAC = 0.6
GROUP_KEEP_FRAC = 1.0
WALD_K = 1.96
SEED = 0
ERROR_BAR_MODE = "ci95"  # "ci95" uses bootstrap percentiles; "std" uses bootstrap SD.

SAVE_PROMPTS = False
# Monitor answers are cached under CACHE_DIR. Derived tables are snapshots used
# by LOAD_EXISTING_OUTPUTS for fast reload/debugging; set False for throwaway runs.
SAVE_TABLES = True
RUN_ANALYSIS = True
LOAD_EXISTING_OUTPUTS = True
OUTPUT_DIR = ARTIFACT_ROOT / "run_monitorability_outputs" / GOOD_BAD_EXPERIMENT
# Figures are saved by default (PDF) into the giraffes section of the gitignored
# Overleaf clone, under a monitorability/ subdir (keyed by the _finalize name).
SAVE_PLOTS = True
FIG_DIR = REPO_ROOT / "overleaf" / "figures" / "giraffes" / "monitorability"
# Classifier cache layout:
#   <ARTIFACT_ROOT>/run_monitorability_cache/<MONITOR_CACHE_EXPERIMENT>/monitor/
CACHE_DIR = SCRIPT_CACHE_ROOT / MONITOR_CACHE_EXPERIMENT / "monitor"

# Same palette and model-group color assignment as
# `shared/final_scripts/giraffes/plot_biases.py`: full tab10 INCLUDING
# green/red, with orange reserved for Claude (Claude=orange, GPT=blue,
# Gemini=green, Qwen=red, Kimi=purple).
CLAUDE_ORANGE = "#ff7f0e"
MODEL_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _family_colors(model_groups):
    colors, i = {}, 0
    for label, _ in model_groups:
        if label.startswith("Claude"):
            colors[label] = CLAUDE_ORANGE
        else:
            colors[label] = MODEL_COLORS[i % len(MODEL_COLORS)]
            i += 1
    return colors


# %%
# Plotting helpers.

def _finalize(fig, _name: str, _args: RunConfig) -> None:
    plt.tight_layout()
    if SAVE_PLOTS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{_name}.pdf", bbox_inches="tight")
    plt.show()


def _ordered_model_keys(model_groups: list[tuple[str, list[str]]], present: set[str]) -> list[str]:
    ordered = [mk for _, group in model_groups for mk in group if mk in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def _group_colors(model_keys: list[str], model_groups: list[tuple[str, list[str]]]) -> list[str]:
    family_colors = _family_colors(model_groups)
    color_by_model: dict[str, str] = {}
    for label, group in model_groups:
        for mk in group:
            color_by_model[mk] = family_colors[label]
    return [color_by_model.get(mk, "#7f7f7f") for mk in model_keys]


def _metric_values_and_errors(
    sub: pd.DataFrame,
    model_keys: list[str],
    metric: str,
    args: RunConfig,
) -> tuple[list[float], list[float], list[float]]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    ci_low_col = f"{metric}_ci95_low"
    ci_high_col = f"{metric}_ci95_high"
    vals: list[float] = []
    err_low: list[float] = []
    err_high: list[float] = []
    use_ci = args.error_bar_mode == "ci95" and {ci_low_col, ci_high_col}.issubset(sub.columns)

    for mk in model_keys:
        row = sub[sub["model_key"] == mk]
        if row.empty or mean_col not in row or pd.isna(row[mean_col].iloc[0]):
            vals.append(np.nan)
            err_low.append(0.0)
            err_high.append(0.0)
            continue

        val = float(row[mean_col].iloc[0])
        vals.append(val)
        if use_ci:
            lo = row[ci_low_col].iloc[0]
            hi = row[ci_high_col].iloc[0]
            err_low.append(max(0.0, val - float(lo)) if pd.notna(lo) else 0.0)
            err_high.append(max(0.0, float(hi) - val) if pd.notna(hi) else 0.0)
        else:
            std = row[std_col].iloc[0] if std_col in row else 0.0
            err = float(std) if pd.notna(std) else 0.0
            err_low.append(err)
            err_high.append(err)

    return vals, err_low, err_high


def _comparison_slice(final: pd.DataFrame, comparison: str, scope: str) -> pd.DataFrame:
    sub = final[(final["comparison"] == comparison) & (final["scope"] == scope)].copy()
    if sub.empty:
        raise ValueError(f"No rows for comparison={comparison!r}, scope={scope!r}")
    return sub


def plot_main_monitorability(
    final: pd.DataFrame,
    *,
    comparison: str,
    scope: str,
    model_groups: list[tuple[str, list[str]]],
    display_names: dict[str, str],
    args: RunConfig,
) -> None:
    sub = _comparison_slice(final, comparison, scope)
    model_keys = _ordered_model_keys(model_groups, set(sub["model_key"]))
    xs = np.arange(len(model_keys))
    vals, err_low, err_high = _metric_values_and_errors(sub, model_keys, "gmean", args)

    # Match plot_cot_categories_v2.build_model_comparison_biased_stack proportions
    # (same figsize formula + bar width) so this comes out looking like the other
    # giraffes bar figures rather than a stretched strip.
    fig_w = max(6.0, 0.55 * len(model_keys) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    bars = ax.bar(
        xs,
        [0 if pd.isna(v) else v for v in vals],
        width=0.68,
        yerr=[err_low, err_high],
        color=_group_colors(model_keys, model_groups),
        edgecolor="white",
        linewidth=0.5,
        ecolor="black",
        capsize=3,
        error_kw={"linewidth": 1.0},
    )
    for x, v, hi in zip(xs, vals, err_high):
        label = "n/a" if pd.isna(v) else f"{v:.2f}"
        ax.text(x, (0 if pd.isna(v) else v) + hi + 0.025, label,
                ha="center", va="bottom", fontsize=VALUE_FS)

    ax.set_xticks(xs)
    ax.set_xticklabels([display_names.get(mk, mk) for mk in model_keys],
                       rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    if not args.run_bootstrap:
        err_label = "no bootstrap"
    else:
        err_label = "95% bootstrap CI" if args.error_bar_mode == "ci95" else "bootstrap SD"
    ax.set_ylabel(f"g-mean monitorability ({err_label})")
    # No overall title: the comparison/scope are named by the filename and the
    # paper caption.
    ax.grid(True, axis="y", alpha=0.3)

    cumulative = 0
    ymax = ax.get_ylim()[1]
    for label, group in model_groups:
        group_present = [mk for mk in group if mk in model_keys]
        if not group_present:
            continue
        start = cumulative
        cumulative += len(group_present)
        end = cumulative - 1
        center = (start + end) / 2
        if cumulative < len(model_keys):
            ax.axvline(cumulative - 0.5, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
        ax.text(center, ymax * 0.99, label, ha="center", va="top", fontsize=HEADER_FS, fontweight="bold")

    _finalize(fig, f"main_monitorability_{comparison}_{scope}", args)


def _fmt_bootstrap_count(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def plot_monitor_confusion_counts(
    final: pd.DataFrame,
    *,
    comparison: str,
    scope: str,
    model_groups: list[tuple[str, list[str]]],
    display_names: dict[str, str],
    args: RunConfig,
) -> None:
    """Plot metric-aligned component rows from the same bootstrap draws."""
    sub = _comparison_slice(final, comparison, scope)
    row_specs = [
        (
            "raw_TPR",
            "positive_count",
            "n+",
            "Mean raw TPR: P(Z=1 | X=1,Y=1)",
            100.0,
            "percent",
            (0, 115),
        ),
        (
            "percent_y1x1_from_effect",
            "positive_count",
            "n+",
            "Mean q: effect size / P(Y=1 | X=1)",
            100.0,
            "percent",
            None,
        ),
        (
            "TPR",
            "positive_count",
            "n+",
            "Mean TPR bound: min(1, raw TPR / q)",
            100.0,
            "percent",
            (0, 115),
        ),
        (
            "TNR_x0",
            "negative_x0_count",
            "n0",
            "Mean TNR among X=0",
            100.0,
            "percent",
            (0, 115),
        ),
        (
            "TNR_x1",
            "negative_x1_count",
            "n1",
            "Mean TNR among X=1,Y=0",
            100.0,
            "percent",
            (0, 115),
        ),
    ]
    required_metrics = {spec[0] for spec in row_specs}
    required_metrics.update(spec[1] for spec in row_specs if spec[1] is not None)
    missing = [
        f"{metric}_mean"
        for metric in required_metrics
        if f"{metric}_mean" not in sub.columns
    ]
    if missing:
        raise ValueError(
            "Bootstrap component columns are missing from final summary. "
            "Regenerate metric outputs with the current run_monitorability.py. "
            f"Missing columns: {missing}"
        )

    model_keys = _ordered_model_keys(model_groups, set(sub["model_key"]))
    xs = np.arange(len(model_keys))
    colors = _group_colors(model_keys, model_groups)
    labels = [display_names.get(mk, mk) for mk in model_keys]

    fig_w = 13.0
    fig, axes = plt.subplots(
        len(row_specs),
        1,
        figsize=(fig_w, max(11.5, 2.35 * len(row_specs))),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (
        rate_metric,
        total_metric,
        total_label,
        title,
        scale,
        y_label,
        ylim,
    ) in zip(axes_flat, row_specs):
        values, err_low, err_high = _metric_values_and_errors(sub, model_keys, rate_metric, args)
        values = [np.nan if pd.isna(v) else scale * v for v in values]
        err_low = [scale * err for err in err_low]
        err_high = [scale * err for err in err_high]
        if total_metric is None:
            totals = [np.nan for _ in model_keys]
        else:
            totals, _, _ = _metric_values_and_errors(sub, model_keys, total_metric, args)
        heights = [0 if pd.isna(v) else v for v in values]
        bars = ax.bar(
            xs,
            heights,
            yerr=[err_low, err_high],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            ecolor="black",
            capsize=3,
            error_kw={"linewidth": 0.9},
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            finite_bounds = [
                (value - lo, value + hi)
                for value, lo, hi in zip(values, err_low, err_high)
                if pd.notna(value)
            ]
            if finite_bounds:
                lower = min(0.0, *(lo for lo, _hi in finite_bounds))
                upper = max(0.0, *(hi for _lo, hi in finite_bounds))
                span = upper - lower
                pad = max(1.0, 0.08 * span)
                ax.set_ylim(lower - pad, upper + pad)
            else:
                ax.set_ylim(-1, 1)

        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        label_pad = 0.025 * y_span
        for bar, value, total, _lo, hi in zip(bars, values, totals, err_low, err_high):
            if total_metric is None:
                continue
            total_value = (
                "n/a"
                if pd.isna(total) or total == 0
                else _fmt_bootstrap_count(total)
            )
            label = f"n={total_value}"
            y = (
                y_min + label_pad
                if pd.isna(value)
                else min(y_max - label_pad, bar.get_height() + hi + label_pad)
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=VALUE_FS,
            )

        cumulative = 0
        for _group_label, group in model_groups:
            group_present = [mk for mk in group if mk in model_keys]
            if not group_present:
                continue
            cumulative += len(group_present)
            if cumulative < len(model_keys):
                ax.axvline(
                    cumulative - 0.5,
                    color="black",
                    linewidth=0.7,
                    alpha=0.35,
                    linestyle="--",
                )

        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        ax.set_ylabel(y_label)
        ax.grid(True, axis="y", alpha=0.3)

    axes_flat[-1].set_xticks(xs)
    axes_flat[-1].set_xticklabels(labels, rotation=35, ha="right")
    # No overall suptitle: the per-row bold titles label each component, and
    # the comparison/scope are named by the filename and the paper caption.
    fig.subplots_adjust(hspace=0.5)
    _finalize(fig, f"monitor_component_rows_{comparison}_{scope}", args)


def plot_prompt_heatmap(
    per_instance: pd.DataFrame,
    *,
    comparison: str,
    scope: str,
    metric: str,
    model_groups: list[tuple[str, list[str]]],
    display_names: dict[str, str],
    args: RunConfig,
) -> None:
    sub = per_instance[
        (per_instance["comparison"] == comparison)
        & (per_instance["scope"] == scope)
    ].copy()
    if sub.empty:
        raise ValueError(f"No per-instance rows for comparison={comparison!r}, scope={scope!r}")
    avg = (
        sub.groupby(["model_key", "prompt_stem"], sort=False)[metric]
        .mean()
        .reset_index()
    )
    model_keys = _ordered_model_keys(model_groups, set(avg["model_key"]))
    prompt_keys = sorted(avg["prompt_stem"].unique())
    matrix = np.full((len(model_keys), len(prompt_keys)), np.nan)
    for i, mk in enumerate(model_keys):
        for j, pk in enumerate(prompt_keys):
            vals = avg[(avg["model_key"] == mk) & (avg["prompt_stem"] == pk)][metric]
            if len(vals):
                matrix[i, j] = float(vals.iloc[0])

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(prompt_keys) + 2.5),
                                    max(3.5, 0.45 * len(model_keys) + 1.5)))
    im = ax.imshow(matrix, aspect="auto", vmin=0 if metric != "effect_size" else None,
                   vmax=1 if metric != "effect_size" else None, cmap="viridis")
    ax.set_xticks(np.arange(len(prompt_keys)))
    ax.set_xticklabels([pk.removeprefix("v1_") for pk in prompt_keys],
                       rotation=35, ha="right")
    ax.set_yticks(np.arange(len(model_keys)))
    ax.set_yticklabels([display_names.get(mk, mk) for mk in model_keys])
    ax.set_title(f"{metric} by prompt: {comparison} ({scope})")
    fig.colorbar(im, ax=ax, shrink=0.85)
    _finalize(fig, f"prompt_heatmap_{metric}_{comparison}_{scope}", args)


def monitorability_table(final: pd.DataFrame, *, comparison: str, scope: str, display_names: dict[str, str]) -> pd.DataFrame:
    sub = _comparison_slice(final, comparison, scope).copy()
    sub["model"] = sub["model_key"].map(lambda mk: display_names.get(mk, mk))
    cols = [
        "model",
        "gmean_mean",
        "gmean_std",
        "TPR_mean",
        "TNR_defined_mean",
        "TNR_x1_mean",
        "effect_size_all_mean",
        "effect_size_eligible_mean",
        "eligible_fraction_mean",
        "FPR_mean",
    ]
    return sub[cols].sort_values("model").reset_index(drop=True)


# %%
# Run or load the analysis, then produce the default figures.

CONFIG = RunConfig(
    good_bad_experiment=GOOD_BAD_EXPERIMENT,
    cache_only=CACHE_ONLY,
    models=MODEL_NAMES,
    comparisons=COMPARISONS,
    scopes=SCOPES,
    prompt_filter=PROMPT_FILTER,
    monitor_backend=MONITOR_BACKEND,
    monitor_model=MONITOR_MODEL,
    monitor_reasoning_effort=MONITOR_REASONING_EFFORT,
    monitor_max_output_tokens=MONITOR_MAX_OUTPUT_TOKENS,
    monitor_cache_only=MONITOR_CACHE_ONLY,
    monitor_max_workers=MONITOR_MAX_WORKERS,
    allow_synthetic_monitor=ALLOW_SYNTHETIC_MONITOR,
    output_dir=OUTPUT_DIR,
    cache_dir=CACHE_DIR,
    metric_path=DEFAULT_METRIC_PATH,
    n_bootstrap=N_BOOTSTRAP,
    run_bootstrap=RUN_BOOTSTRAP,
    bootstrap_max_workers=BOOTSTRAP_MAX_WORKERS,
    selection_frac=SELECTION_FRAC,
    group_keep_frac=GROUP_KEEP_FRAC,
    wald_k=WALD_K,
    seed=SEED,
    error_bar_mode=ERROR_BAR_MODE,
    limit_rows_per_instance_arm=LIMIT_ROWS_PER_INSTANCE_ARM,
    save_prompts=SAVE_PROMPTS,
    save_tables=SAVE_TABLES,
    run_analysis=RUN_ANALYSIS,
    load_existing_outputs=LOAD_EXISTING_OUTPUTS,
)
display_names = monitorability.model_display_names(CONFIG.resolved_models())

if CONFIG.load_existing_outputs:
    try:
        base_rows, monitored, final_summary, per_bootstrap, per_instance = monitorability.load_outputs(CONFIG)
    except monitorability.StaleOutputsError as exc:
        if not CONFIG.run_analysis:
            raise
        print(f"{exc}\nRe-running analysis because saved outputs are stale.", flush=True)
        base_rows, monitored, final_summary, per_bootstrap, per_instance = monitorability.run_monitorability(CONFIG)
elif CONFIG.run_analysis:
    base_rows, monitored, final_summary, per_bootstrap, per_instance = monitorability.run_monitorability(CONFIG)
else:
    base_rows = monitored = final_summary = per_bootstrap = per_instance = None

if final_summary is not None:
    if per_bootstrap is not None:
        final_summary = monitorability.ensure_bootstrap_ci_columns(final_summary, per_bootstrap)
    DEFAULT_COMPARISON = CONFIG.resolved_comparisons()[0]
    for scope in CONFIG.resolved_scopes():
        if scope not in set(final_summary["scope"]):
            print(
                f"Skipping plots for scope={scope!r}; not present in final summary.",
                flush=True,
            )
            continue
        print(f"\n=== Summary for scope={scope} ===")
        summary_table = monitorability_table(
            final_summary,
            comparison=DEFAULT_COMPARISON,
            scope=scope,
            display_names=display_names,
        )
        print(summary_table.to_string(index=False))
        plot_main_monitorability(
            final_summary,
            comparison=DEFAULT_COMPARISON,
            scope=scope,
            model_groups=MODEL_GROUPS,
            display_names=display_names,
            args=CONFIG,
        )
        plot_monitor_confusion_counts(
            final_summary,
            comparison=DEFAULT_COMPARISON,
            scope=scope,
            model_groups=MODEL_GROUPS,
            display_names=display_names,
            args=CONFIG,
        )
        plot_prompt_heatmap(
            per_instance,
            comparison=DEFAULT_COMPARISON,
            scope=scope,
            metric="gmean",
            model_groups=MODEL_GROUPS,
            display_names=display_names,
            args=CONFIG,
        )
