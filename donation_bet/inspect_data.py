# %%
"""Data-quality inspection for threshold experiments.

Reads the same cache tree as plot_biases.py (final_data/) and produces
diagnostic plots focused on judge-side failures, i.e. rows where the
judge returned <final_estimate>UNKNOWN</final_estimate> or anything else
that didn't parse into a number (see shared.runner._parse_tagged_estimate).
These rows have NaN ``estimate`` in the raw df returned by
``get_main_dfs(..., raw=True)``.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_style import HEADER_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import

import shared.runner as runner
from shared.get_main_dfs import get_main_dfs
from shared.view_results import view_results

# Same cache redirect as plot_biases.py.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")

# These diagnostic plots are saved by default (PDF) into the giraffes section of
# the gitignored Overleaf clone, under an inspect_data/ subdir. Each plot
# function takes fname=_fig("<name>") at its call site below.
SAVE_PLOTS = globals().get("SAVE_PLOTS", True)
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes" / "inspect_data"


def _fig(name):
    """overleaf .../giraffes/inspect_data/<name>.pdf, or None when SAVE_PLOTS is off."""
    if not SAVE_PLOTS:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / f"{name}.pdf"

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
    ("Gemini", [
        "gemini-2.5-pro",
        "gemini-3.1-pro-medium",
        # "gemini-3.1-pro-high",
    ]),
    ("Qwen", [
        "qwen3.5-35",
        "qwen3.6-35",
    ]),
    ("Kimi", [
        "kimi-k2.5",
        # "kimi-k2.6",
    ]),
]
MODEL_NAMES = [mk for _, group in MODEL_GROUPS for mk in group]

EXPERIMENT = "main_experiment_accurate"
CACHE_ONLY = True


# %%
raw_dfs = get_main_dfs(EXPERIMENT, MODEL_NAMES, cache_only=CACHE_ONLY, raw=True)


# %% ============================================================
# UNKNOWN ESTIMATE ANALYSIS
# ============================================================
# Two summary tables. UNKNOWN = judge couldn't parse a number (estimate
# is NaN in the raw df). Ratios are over the rows actually present in
# the cache, which under cache_only=True is the full experiment spec.
overall_rows = []
per_pk_rows = []
display_names = {}
prompt_keys = None
for model_key, (df, _thresholds, display_name) in raw_dfs.items():
    display_names[model_key] = display_name
    if prompt_keys is None:
        prompt_keys = list(df["prompt_key"].drop_duplicates())
    n_total = len(df)
    n_unknown = int(df["estimate"].isna().sum())
    overall_rows.append({
        "model_key": model_key,
        "model": display_name,
        "n_unknown": n_unknown,
        "n_total": n_total,
        "ratio_unknown": (n_unknown / n_total) if n_total else float("nan"),
    })
    for pk in prompt_keys:
        sub = df[df["prompt_key"] == pk]
        n_total_pk = len(sub)
        n_unknown_pk = int(sub["estimate"].isna().sum())
        per_pk_rows.append({
            "model_key": model_key,
            "model": display_name,
            "prompt_key": pk,
            "n_unknown": n_unknown_pk,
            "n_total": n_total_pk,
            "ratio_unknown": (n_unknown_pk / n_total_pk) if n_total_pk else float("nan"),
        })

overall_df = pd.DataFrame(overall_rows)
per_pk_df = pd.DataFrame(per_pk_rows)
# print(overall_df.to_string(index=False))


# %%
def view_unknown(model_key=None, prompt_key=None):
    """Open Question.view on rows whose judge estimate is NaN (UNKNOWN).

    Both filters are optional:
    - no args: every UNKNOWN row across all loaded models.
    - model_key only: UNKNOWN rows for that one model.
    - prompt_key only: UNKNOWN rows for that one prompt across all models.
    - both: intersection.

    Adds a leading ``model`` column so multi-model views stay legible in
    the viewer.
    """
    chunks = []
    for mk, (df, _t, dn) in raw_dfs.items():
        if model_key is not None and mk != model_key:
            continue
        sub = df[df["estimate"].isna()]
        if prompt_key is not None:
            sub = sub[sub["prompt_key"] == prompt_key]
        if len(sub):
            sub = sub.copy()
            sub.insert(0, "model", dn)
            chunks.append(sub)
    if not chunks:
        print("No UNKNOWN rows match the given filters.")
        return
    combined = pd.concat(chunks, ignore_index=True)
    print(f"Opening viewer on {len(combined)} UNKNOWN rows.")
    view_results(combined)


# %%
# Per-family model palette (matches plot_biases.py): full tab10 INCLUDING
# green/red, with orange reserved for Claude.
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


# Restricted palette for generic result-category bars (the per-prompt bars):
# no orange (Claude) / green / red (good-bad), so they never clash with those
# reserved meanings.
COLORS = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#aec7e8", "#c5b0d5", "#c49c94", "#f7b6d2"]
MAX_COLS = 2


def _annotate_bar(ax, x, value, n_unknown, n_total, ymax, fontsize=VALUE_FS):
    """Print '<pct>% (k/N)' above each bar, falling back to 'n/a' for empty groups."""
    if pd.isna(value):
        label = "n/a"
        y = 0.0
    else:
        label = f"{value * 100:.1f}%\n({n_unknown}/{n_total})"
        y = value
    ax.text(x, y + ymax * 0.01, label, ha="center", va="bottom", fontsize=fontsize)


def plot_unknown_per_model(overall_df, model_groups, fname=None):
    """Bar chart: fraction of UNKNOWN-estimate rows per model.

    Bars are coloured by family group (same colour-per-group scheme as
    plot_biases.plot_mean_bias_per_model).
    """
    ordered_keys = [mk for _, g in model_groups for mk in g]
    ordered = overall_df.set_index("model_key").reindex(ordered_keys)
    vals = ordered["ratio_unknown"].tolist()
    n_unk = ordered["n_unknown"].tolist()
    n_tot = ordered["n_total"].tolist()
    labels = ordered["model"].tolist()

    family_colors = _family_colors(model_groups)
    bar_colors = []
    for label, g in model_groups:
        c = family_colors[label]
        bar_colors.extend([c] * len(g))

    xs = list(range(len(labels)))
    fig_w = max(6, 0.55 * len(xs) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    heights = [0.0 if pd.isna(v) else v for v in vals]
    ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

    finite = [v for v in vals if pd.notna(v)]
    ymax = max(0.05, (max(finite) if finite else 0.0) * 1.25)
    ax.set_ylim(0, ymax)

    for x, v, k, n in zip(xs, vals, n_unk, n_tot):
        _annotate_bar(ax, x, v, k, n, ymax, fontsize=VALUE_FS)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="y")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _p: f"{y * 100:.1f}%")
    )
    ax.set_ylabel("UNKNOWN-estimate ratio")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Judge UNKNOWN / unparseable ratio per model")

    # Family dividers (matches plot_biases convention).
    cumulative = 0
    for label, g in model_groups:
        start = cumulative
        end = cumulative + len(g) - 1
        center = (start + end) / 2
        cumulative += len(g)
        if cumulative < len(xs):
            ax.axvline(cumulative - 0.5, color="black",
                       linewidth=0.8, alpha=0.5, linestyle="--")
        ax.text(center, ymax * 0.97, label, ha="center", va="top",
                fontsize=HEADER_FS, fontweight="bold")

    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_unknown_per_model_by_prompt(per_pk_df, prompt_keys, display_names,
                                     model_groups, fname=None):
    """One subplot per model, MAX_COLS columns; bars are UNKNOWN ratio per prompt_key.

    Subplots share y-axis so models are directly comparable.
    """
    ordered_keys = [mk for _, g in model_groups for mk in g
                    if mk in display_names]
    n = len(ordered_keys)
    if n == 0:
        return
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols

    pkeys = list(prompt_keys)
    xs = list(range(len(pkeys)))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(pkeys))]

    finite_vals = per_pk_df["ratio_unknown"].dropna().tolist()
    ymax = max(0.05, (max(finite_vals) if finite_vals else 0.0) * 1.25)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 0.55 * len(pkeys)) * n_cols + 1, 4.2 * n_rows + 0.8),
        sharey=True, squeeze=False,
    )
    flat = axes.flatten()

    for ax, mk in zip(flat, ordered_keys):
        sub = per_pk_df[per_pk_df["model_key"] == mk].set_index("prompt_key")
        sub = sub.reindex(pkeys)
        vals = sub["ratio_unknown"].tolist()
        n_unk = sub["n_unknown"].tolist()
        n_tot = sub["n_total"].tolist()
        heights = [0.0 if pd.isna(v) else v for v in vals]
        ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

        for x, v, k, ntot in zip(xs, vals, n_unk, n_tot):
            _annotate_bar(ax, x, v, int(k) if pd.notna(k) else 0,
                          int(ntot) if pd.notna(ntot) else 0, ymax, fontsize=VALUE_FS)

        ax.set_xticks(xs)
        ax.set_xticklabels(pkeys, rotation=30, ha="right")
        ax.set_ylim(0, ymax)
        ax.set_title(display_names[mk])
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _p: f"{y * 100:.1f}%")
        )
        # ylabel + tick labels on every subplot so individual screenshots
        # remain self-explanatory (sharey=True hides ticks by default).
        ax.set_ylabel("UNKNOWN ratio")
        ax.tick_params(labelleft=True)

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Judge UNKNOWN / unparseable ratio per model and prompt")
    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
plot_unknown_per_model(overall_df, MODEL_GROUPS, fname=_fig("unknown_per_model"))
plot_unknown_per_model_by_prompt(per_pk_df, prompt_keys, display_names,
                                 MODEL_GROUPS, fname=_fig("unknown_per_model_by_prompt"))
# %%
view_unknown("claude-opus-4.7-max")


# %% ============================================================
# COT LENGTH ANALYSIS
# ============================================================
# Restrict to rows where the judge produced a valid estimate (UNKNOWNs
# are handled in the section above), then count rows whose reasoning is
# shorter than SHORT_COT_CHARS. Empty / missing reasoning counts as
# length 0 → "short". Bump SHORT_COT_CHARS and re-run to vary the
# threshold.
SHORT_COT_CHARS = 200

short_overall_rows = []
short_per_pk_rows = []
for model_key, (df, _t, dn) in raw_dfs.items():
    df_ok = df[df["estimate"].notna()]
    cot_len = df_ok["reasoning"].fillna("").str.len()
    is_short = cot_len < SHORT_COT_CHARS
    n_total = len(df_ok)
    n_short = int(is_short.sum())
    short_overall_rows.append({
        "model_key": model_key,
        "model": dn,
        "n_short": n_short,
        "n_total": n_total,
        "ratio_short": (n_short / n_total) if n_total else float("nan"),
    })
    for pk in prompt_keys:
        pk_mask = df_ok["prompt_key"] == pk
        n_total_pk = int(pk_mask.sum())
        n_short_pk = int((is_short & pk_mask).sum())
        short_per_pk_rows.append({
            "model_key": model_key,
            "model": dn,
            "prompt_key": pk,
            "n_short": n_short_pk,
            "n_total": n_total_pk,
            "ratio_short": (n_short_pk / n_total_pk) if n_total_pk else float("nan"),
        })

short_overall_df = pd.DataFrame(short_overall_rows)
short_per_pk_df = pd.DataFrame(short_per_pk_rows)


# %%
def view_short_cot(model_key=None, prompt_key=None, short_chars=None):
    """Open Question.view on estimate-valid rows whose CoT is shorter than
    ``short_chars`` (defaults to SHORT_COT_CHARS).

    Both filters are optional:
    - no args: every short-CoT row across all loaded models.
    - model_key only: short-CoT rows for that one model.
    - prompt_key only: short-CoT rows for that one prompt across all models.
    - both: intersection.

    Adds a leading ``model`` column so multi-model views stay legible.
    """
    threshold = SHORT_COT_CHARS if short_chars is None else short_chars
    chunks = []
    for mk, (df, _t, dn) in raw_dfs.items():
        if model_key is not None and mk != model_key:
            continue
        sub = df[df["estimate"].notna()]
        sub = sub[sub["reasoning"].fillna("").str.len() < threshold]
        if prompt_key is not None:
            sub = sub[sub["prompt_key"] == prompt_key]
        if len(sub):
            sub = sub.copy()
            sub.insert(0, "model", dn)
            chunks.append(sub)
    if not chunks:
        print("No short-CoT rows match the given filters.")
        return
    combined = pd.concat(chunks, ignore_index=True)
    print(f"Opening viewer on {len(combined)} rows with CoT < {threshold} chars.")
    view_results(combined)


# %%
def _annotate_short_bar(ax, x, value, n_short, n_total, ymax, fontsize=VALUE_FS):
    """Print '<pct>% (k/N)' above each bar, falling back to 'n/a' for empty groups."""
    if pd.isna(value):
        label = "n/a"
        y = 0.0
    else:
        label = f"{value * 100:.1f}%\n({n_short}/{n_total})"
        y = value
    ax.text(x, y + ymax * 0.01, label, ha="center", va="bottom", fontsize=fontsize)


def plot_short_cot_per_model(short_overall_df, model_groups, short_chars,
                             fname=None):
    """Bar chart: fraction of rows whose reasoning is shorter than ``short_chars``.

    Mirrors plot_unknown_per_model but on the estimate-valid subset.
    """
    ordered_keys = [mk for _, g in model_groups for mk in g]
    ordered = short_overall_df.set_index("model_key").reindex(ordered_keys)
    vals = ordered["ratio_short"].tolist()
    n_short = ordered["n_short"].tolist()
    n_tot = ordered["n_total"].tolist()
    labels = ordered["model"].tolist()

    family_colors = _family_colors(model_groups)
    bar_colors = []
    for label, g in model_groups:
        c = family_colors[label]
        bar_colors.extend([c] * len(g))

    xs = list(range(len(labels)))
    fig_w = max(6, 0.55 * len(xs) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    heights = [0.0 if pd.isna(v) else v for v in vals]
    ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

    finite = [v for v in vals if pd.notna(v)]
    ymax = max(0.05, (max(finite) if finite else 0.0) * 1.25)
    ax.set_ylim(0, ymax)

    for x, v, k, n in zip(xs, vals, n_short, n_tot):
        _annotate_short_bar(ax, x, v, k, n, ymax, fontsize=VALUE_FS)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="y")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _p: f"{y * 100:.1f}%")
    )
    ax.set_ylabel(f"Ratio with CoT < {short_chars} chars")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Short-CoT ratio (< {short_chars} chars) per model")

    cumulative = 0
    for label, g in model_groups:
        start = cumulative
        end = cumulative + len(g) - 1
        center = (start + end) / 2
        cumulative += len(g)
        if cumulative < len(xs):
            ax.axvline(cumulative - 0.5, color="black",
                       linewidth=0.8, alpha=0.5, linestyle="--")
        ax.text(center, ymax * 0.97, label, ha="center", va="top",
                fontsize=HEADER_FS, fontweight="bold")

    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_short_cot_per_model_by_prompt(short_per_pk_df, prompt_keys,
                                       display_names, model_groups,
                                       short_chars, fname=None):
    """One subplot per model, MAX_COLS columns; bars are short-CoT ratio per prompt_key."""
    ordered_keys = [mk for _, g in model_groups for mk in g
                    if mk in display_names]
    n = len(ordered_keys)
    if n == 0:
        return
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols

    pkeys = list(prompt_keys)
    xs = list(range(len(pkeys)))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(pkeys))]

    finite_vals = short_per_pk_df["ratio_short"].dropna().tolist()
    ymax = max(0.05, (max(finite_vals) if finite_vals else 0.0) * 1.25)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 0.55 * len(pkeys)) * n_cols + 1, 4.2 * n_rows + 0.8),
        sharey=True, squeeze=False,
    )
    flat = axes.flatten()

    for ax, mk in zip(flat, ordered_keys):
        sub = short_per_pk_df[short_per_pk_df["model_key"] == mk].set_index("prompt_key")
        sub = sub.reindex(pkeys)
        vals = sub["ratio_short"].tolist()
        n_short = sub["n_short"].tolist()
        n_tot = sub["n_total"].tolist()
        heights = [0.0 if pd.isna(v) else v for v in vals]
        ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

        for x, v, k, ntot in zip(xs, vals, n_short, n_tot):
            _annotate_short_bar(ax, x, v, int(k) if pd.notna(k) else 0,
                                int(ntot) if pd.notna(ntot) else 0,
                                ymax, fontsize=VALUE_FS)

        ax.set_xticks(xs)
        ax.set_xticklabels(pkeys, rotation=30, ha="right")
        ax.set_ylim(0, ymax)
        ax.set_title(display_names[mk])
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _p: f"{y * 100:.1f}%")
        )
        # ylabel + tick labels on every subplot so individual screenshots
        # remain self-explanatory (sharey=True hides ticks by default).
        ax.set_ylabel(f"CoT < {short_chars} chars")
        ax.tick_params(labelleft=True)

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Short-CoT ratio (< {short_chars} chars) per model and prompt",    )
    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
plot_short_cot_per_model(short_overall_df, MODEL_GROUPS, SHORT_COT_CHARS,
                         fname=_fig("short_cot_per_model"),)
plot_short_cot_per_model_by_prompt(short_per_pk_df, prompt_keys, display_names,
                                   MODEL_GROUPS, SHORT_COT_CHARS,
                                   fname=_fig("short_cot_per_model_by_prompt"))
# %%
view_short_cot("qwen3.5-35")


# %% ============================================================
# OUTLIER ANALYSIS
# ============================================================
# Frequency of estimates that exceed OUTLIER_MULTIPLIER * threshold.
# Restricts to estimate-valid rows with a non-null threshold (i.e. drops
# baseline rows, which have no threshold to compare against). Bump
# OUTLIER_MULTIPLIER and re-run the section to tighten/loosen the bound.
OUTLIER_MULTIPLIER = 3

outlier_overall_rows = []
outlier_per_pk_rows = []
for model_key, (df, _t, dn) in raw_dfs.items():
    df_ok = df[df["estimate"].notna() & df["threshold"].notna()]
    is_out = df_ok["estimate"] > OUTLIER_MULTIPLIER * df_ok["threshold"]
    n_total = len(df_ok)
    n_out = int(is_out.sum())
    outlier_overall_rows.append({
        "model_key": model_key,
        "model": dn,
        "n_outlier": n_out,
        "n_total": n_total,
        "ratio_outlier": (n_out / n_total) if n_total else float("nan"),
    })
    for pk in prompt_keys:
        pk_mask = df_ok["prompt_key"] == pk
        n_total_pk = int(pk_mask.sum())
        n_out_pk = int((is_out & pk_mask).sum())
        outlier_per_pk_rows.append({
            "model_key": model_key,
            "model": dn,
            "prompt_key": pk,
            "n_outlier": n_out_pk,
            "n_total": n_total_pk,
            "ratio_outlier": (n_out_pk / n_total_pk) if n_total_pk else float("nan"),
        })

outlier_overall_df = pd.DataFrame(outlier_overall_rows)
outlier_per_pk_df = pd.DataFrame(outlier_per_pk_rows)


# %%
def view_outliers(model_key=None, prompt_key=None, multiplier=None):
    """Open Question.view on rows whose estimate exceeds ``multiplier`` *
    that row's threshold (defaults to OUTLIER_MULTIPLIER). Useful for
    spot-checking suspected unit / scale confusion in the judge or model.
    """
    m = OUTLIER_MULTIPLIER if multiplier is None else multiplier
    chunks = []
    for mk, (df, _t, dn) in raw_dfs.items():
        if model_key is not None and mk != model_key:
            continue
        sub = df[df["estimate"].notna() & df["threshold"].notna()]
        sub = sub[sub["estimate"] > m * sub["threshold"]]
        if prompt_key is not None:
            sub = sub[sub["prompt_key"] == prompt_key]
        if len(sub):
            sub = sub.copy()
            sub.insert(0, "model", dn)
            chunks.append(sub)
    if not chunks:
        print("No outlier rows match the given filters.")
        return
    combined = pd.concat(chunks, ignore_index=True)
    print(f"Opening viewer on {len(combined)} rows with estimate > {m}x threshold.")
    view_results(combined)


# %%
def _annotate_outlier_bar(ax, x, value, n_outlier, n_total, ymax, fontsize=VALUE_FS):
    """Print '<pct>% (k/N)' above each bar; 'n/a' for empty groups."""
    if pd.isna(value):
        label = "n/a"
        y = 0.0
    else:
        label = f"{value * 100:.1f}%\n({n_outlier}/{n_total})"
        y = value
    ax.text(x, y + ymax * 0.01, label, ha="center", va="bottom", fontsize=fontsize)


def plot_outlier_per_model(outlier_overall_df, model_groups, multiplier,
                           fname=None):
    """Bar chart: fraction of estimates greater than ``multiplier`` * threshold."""
    ordered_keys = [mk for _, g in model_groups for mk in g]
    ordered = outlier_overall_df.set_index("model_key").reindex(ordered_keys)
    vals = ordered["ratio_outlier"].tolist()
    n_out = ordered["n_outlier"].tolist()
    n_tot = ordered["n_total"].tolist()
    labels = ordered["model"].tolist()

    family_colors = _family_colors(model_groups)
    bar_colors = []
    for label, g in model_groups:
        c = family_colors[label]
        bar_colors.extend([c] * len(g))

    xs = list(range(len(labels)))
    fig_w = max(6, 0.55 * len(xs) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    heights = [0.0 if pd.isna(v) else v for v in vals]
    ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

    finite = [v for v in vals if pd.notna(v)]
    ymax = max(0.05, (max(finite) if finite else 0.0) * 1.25)
    ax.set_ylim(0, ymax)

    for x, v, k, n in zip(xs, vals, n_out, n_tot):
        _annotate_outlier_bar(ax, x, v, k, n, ymax, fontsize=VALUE_FS)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="y")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _p: f"{y * 100:.1f}%")
    )
    ax.set_ylabel(f"Ratio with estimate > {multiplier}x threshold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Outlier ratio (estimate > {multiplier}x threshold) per model")

    cumulative = 0
    for label, g in model_groups:
        start = cumulative
        end = cumulative + len(g) - 1
        center = (start + end) / 2
        cumulative += len(g)
        if cumulative < len(xs):
            ax.axvline(cumulative - 0.5, color="black",
                       linewidth=0.8, alpha=0.5, linestyle="--")
        ax.text(center, ymax * 0.97, label, ha="center", va="top",
                fontsize=HEADER_FS, fontweight="bold")

    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_outlier_per_model_by_prompt(outlier_per_pk_df, prompt_keys,
                                     display_names, model_groups, multiplier,
                                     fname=None):
    """One subplot per model; bars = outlier ratio per prompt_key."""
    ordered_keys = [mk for _, g in model_groups for mk in g
                    if mk in display_names]
    n = len(ordered_keys)
    if n == 0:
        return
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols

    pkeys = list(prompt_keys)
    xs = list(range(len(pkeys)))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(pkeys))]

    finite_vals = outlier_per_pk_df["ratio_outlier"].dropna().tolist()
    ymax = max(0.05, (max(finite_vals) if finite_vals else 0.0) * 1.25)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 0.55 * len(pkeys)) * n_cols + 1, 4.2 * n_rows + 0.8),
        sharey=True, squeeze=False,
    )
    flat = axes.flatten()

    for ax, mk in zip(flat, ordered_keys):
        sub = outlier_per_pk_df[outlier_per_pk_df["model_key"] == mk].set_index("prompt_key")
        sub = sub.reindex(pkeys)
        vals = sub["ratio_outlier"].tolist()
        n_out = sub["n_outlier"].tolist()
        n_tot = sub["n_total"].tolist()
        heights = [0.0 if pd.isna(v) else v for v in vals]
        ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

        for x, v, k, ntot in zip(xs, vals, n_out, n_tot):
            _annotate_outlier_bar(ax, x, v, int(k) if pd.notna(k) else 0,
                                  int(ntot) if pd.notna(ntot) else 0,
                                  ymax, fontsize=VALUE_FS)

        ax.set_xticks(xs)
        ax.set_xticklabels(pkeys, rotation=30, ha="right")
        ax.set_ylim(0, ymax)
        ax.set_title(display_names[mk])
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _p: f"{y * 100:.1f}%")
        )
        # ylabel + tick labels on every subplot so individual screenshots
        # remain self-explanatory (sharey=True hides ticks by default).
        ax.set_ylabel(f"Estimate > {multiplier}x threshold")
        ax.tick_params(labelleft=True)

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Outlier ratio (estimate > {multiplier}x threshold) per model and prompt",    )
    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
plot_outlier_per_model(outlier_overall_df, MODEL_GROUPS, OUTLIER_MULTIPLIER,
                       fname=_fig("outlier_per_model"))
plot_outlier_per_model_by_prompt(outlier_per_pk_df, prompt_keys, display_names,
                                 MODEL_GROUPS, OUTLIER_MULTIPLIER,
                                 fname=_fig("outlier_per_model_by_prompt"))
# %%
view_outliers("qwen3.5-35")


# %% ============================================================
# VARIANCE ANALYSIS
# ============================================================
# Within-cell spread of estimates, normalized by the cell's threshold.
# A "cell" is (prompt_key, direction, threshold); with n_per_threshold
# samples per cell, we compute std(estimate, ddof=1) / threshold and
# average those ratios across cells. Cells with only one sample (std
# undefined) are dropped. Baseline rows have no threshold and are
# excluded. Higher numbers = noisier model on that prompt.
var_overall_rows = []
var_per_pk_rows = []
for model_key, (df, _t, dn) in raw_dfs.items():
    df_ok = df[df["estimate"].notna() & df["threshold"].notna()]
    if not len(df_ok):
        var_overall_rows.append({
            "model_key": model_key, "model": dn,
            "mean_norm_std": float("nan"), "n_cells": 0,
        })
        for pk in prompt_keys:
            var_per_pk_rows.append({
                "model_key": model_key, "model": dn, "prompt_key": pk,
                "mean_norm_std": float("nan"), "n_cells": 0,
            })
        continue
    agg = (
        df_ok.groupby(["prompt_key", "direction", "threshold"])["estimate"]
        .agg(["std", "count"]).reset_index()
    )
    agg["norm_std"] = agg["std"] / agg["threshold"]
    valid = agg.dropna(subset=["norm_std"])
    var_overall_rows.append({
        "model_key": model_key,
        "model": dn,
        "mean_norm_std": valid["norm_std"].mean() if len(valid) else float("nan"),
        "n_cells": len(valid),
    })
    for pk in prompt_keys:
        pk_valid = valid[valid["prompt_key"] == pk]
        var_per_pk_rows.append({
            "model_key": model_key,
            "model": dn,
            "prompt_key": pk,
            "mean_norm_std": pk_valid["norm_std"].mean() if len(pk_valid) else float("nan"),
            "n_cells": len(pk_valid),
        })

var_overall_df = pd.DataFrame(var_overall_rows)
var_per_pk_df = pd.DataFrame(var_per_pk_rows)


# %%
def _annotate_variance_bar(ax, x, value, n_cells, ymax, fontsize=VALUE_FS):
    """Print '<value> (N cells)' above each bar; 'n/a' for empty groups."""
    if pd.isna(value):
        label = "n/a"
        y = 0.0
    else:
        label = f"{value:.3f}\n({n_cells} cells)"
        y = value
    ax.text(x, y + ymax * 0.01, label, ha="center", va="bottom", fontsize=fontsize)


def plot_variance_per_model(var_overall_df, model_groups, fname=None):
    """Bar chart: mean within-cell std / threshold, per model."""
    ordered_keys = [mk for _, g in model_groups for mk in g]
    ordered = var_overall_df.set_index("model_key").reindex(ordered_keys)
    vals = ordered["mean_norm_std"].tolist()
    n_cells = ordered["n_cells"].tolist()
    labels = ordered["model"].tolist()

    family_colors = _family_colors(model_groups)
    bar_colors = []
    for label, g in model_groups:
        c = family_colors[label]
        bar_colors.extend([c] * len(g))

    xs = list(range(len(labels)))
    fig_w = max(6, 0.55 * len(xs) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    heights = [0.0 if pd.isna(v) else v for v in vals]
    ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

    finite = [v for v in vals if pd.notna(v)]
    ymax = max(0.05, (max(finite) if finite else 0.0) * 1.25)
    ax.set_ylim(0, ymax)

    for x, v, c in zip(xs, vals, n_cells):
        _annotate_variance_bar(ax, x, v, int(c) if pd.notna(c) else 0,
                               ymax, fontsize=VALUE_FS)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="y")
    ax.set_ylabel("Mean(std(estimate) / threshold)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Within-cell normalized std per model (lower = more consistent)")

    cumulative = 0
    for label, g in model_groups:
        start = cumulative
        end = cumulative + len(g) - 1
        center = (start + end) / 2
        cumulative += len(g)
        if cumulative < len(xs):
            ax.axvline(cumulative - 0.5, color="black",
                       linewidth=0.8, alpha=0.5, linestyle="--")
        ax.text(center, ymax * 0.97, label, ha="center", va="top",
                fontsize=HEADER_FS, fontweight="bold")

    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_variance_per_model_by_prompt(var_per_pk_df, prompt_keys,
                                      display_names, model_groups, fname=None):
    """One subplot per model; bars = mean normalized std per prompt_key."""
    ordered_keys = [mk for _, g in model_groups for mk in g
                    if mk in display_names]
    n = len(ordered_keys)
    if n == 0:
        return
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols

    pkeys = list(prompt_keys)
    xs = list(range(len(pkeys)))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(pkeys))]

    finite_vals = var_per_pk_df["mean_norm_std"].dropna().tolist()
    ymax = max(0.05, (max(finite_vals) if finite_vals else 0.0) * 1.25)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 0.55 * len(pkeys)) * n_cols + 1, 4.2 * n_rows + 0.8),
        sharey=True, squeeze=False,
    )
    flat = axes.flatten()

    for ax, mk in zip(flat, ordered_keys):
        sub = var_per_pk_df[var_per_pk_df["model_key"] == mk].set_index("prompt_key")
        sub = sub.reindex(pkeys)
        vals = sub["mean_norm_std"].tolist()
        cells = sub["n_cells"].tolist()
        heights = [0.0 if pd.isna(v) else v for v in vals]
        ax.bar(xs, heights, color=bar_colors, edgecolor="white", linewidth=0.5)

        for x, v, c in zip(xs, vals, cells):
            _annotate_variance_bar(ax, x, v, int(c) if pd.notna(c) else 0,
                                   ymax, fontsize=VALUE_FS)

        ax.set_xticks(xs)
        ax.set_xticklabels(pkeys, rotation=30, ha="right")
        ax.set_ylim(0, ymax)
        ax.set_title(display_names[mk])
        ax.grid(True, axis="y", alpha=0.3)
        # ylabel + tick labels on every subplot so individual screenshots
        # remain self-explanatory (sharey=True hides ticks by default).
        ax.set_ylabel("std(estimate) / threshold")
        ax.tick_params(labelleft=True)

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Within-cell normalized std per model and prompt")
    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
plot_variance_per_model(var_overall_df, MODEL_GROUPS, fname=_fig("variance_per_model"))
plot_variance_per_model_by_prompt(var_per_pk_df, prompt_keys, display_names,
                                  MODEL_GROUPS, fname=_fig("variance_per_model_by_prompt"))
# %%
