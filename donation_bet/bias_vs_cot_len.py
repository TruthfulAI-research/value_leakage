# %%
"""Bias vs. CoT length: does a model's good-side bias change with how much
it reasoned? Buckets directional rows by per-model (or per-model-per-prompt)
CoT-length quantiles and plots bias against the bucket's median length.

Figures are written (PDF only) into the giraffes section of the gitignored
Overleaf clone, under the ``giraffes/bias_vs_cot_len`` subfolder. Both figures
measure CoT length in regex-counted words, so the filenames use the ``words``
prefix:
  * ``bias_vs_cot_words_absolute.pdf``           -- across-models overview
  * ``bias_vs_cot_words_per_prompt_<model>.pdf`` -- one per model in
    ``PER_PROMPT_MODELS`` (paper appendix; one line per prompt variant)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared import plot_style  # noqa: F401  applies shared figure sizing on import

from shared.get_main_dfs import get_main_dfs

# Use the new cache. Anchored on runner.__file__ (which lives at
# shared/runner.py, so parents[1] = repo root) so this script can move
# around and still find the cache, and so cell-style execution -- where
# __file__ can be undefined / wrong -- doesn't silently mis-resolve the
# path.
from pathlib import Path
import shared.runner as runner
DATA_ROOT = Path(runner.__file__).resolve().parents[1] / "data" / "final_data"
runner.CACHE_DIR = str(DATA_ROOT / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")
# Figure destination: every plot is written (PDF only) into the
# giraffes/bias_vs_cot_len section of the gitignored Overleaf clone.
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "giraffes" / "bias_vs_cot_len"


def _maybe_savefig(fig, filename):
    """Save ``fig`` to ``filename`` if given, creating parent dirs as needed."""
    if filename is None:
        return
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")


# Kept in sync with `plot_biases.py` MODEL_GROUPS (same family labels and
# membership) so the two figures share one model list and one per-family
# palette. Edit both together.
MODEL_GROUPS = [
    ("Claude", [
        # "claude-opus-4.1",
        # "claude-opus-4.5-high",
        # "claude-opus-4.6-high",
        # "claude-opus-4.6-max",
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

# Single-model per-prompt (appendix) figures: each model here gets one
# `bias_vs_cot_words_per_prompt_<model>.pdf` (one line per estimate category).
# They are loaded into `main_dfs` and plotted on their own, but kept OUT of
# MODEL_GROUPS so they do NOT appear in the across-models overview. Referenced
# by the paper appendix (sec:reasoning-length-vs-bias).
PER_PROMPT_MODELS = ["claude-opus-4.7-xhigh"]

EXPERIMENT = "main_experiment_accurate"
N_BUCKETS = 5
CACHE_ONLY = True


# Length-extraction strategy used by `per_model_bucket_df` and friends.
# "chars" is the simple character count. "words" counts runs of >=2
# ASCII letters via regex (vectorized in pandas). The regex form is
# deliberately conservative: CoTs sometimes contain long unbroken
# garbage runs (unicode noise, repeated chars without spaces, base64-
# like blobs) that would otherwise count as a single huge "word" under
# `split()`; under the regex such junk contributes at most 1 to the
# count. The `{2,}` filters out single-letter algebra variables in
# math / code so e.g. ``x = 1 + y`` doesn't inflate the count.
_WORD_PATTERN = r"[A-Za-z]{2,}"
_LENGTH_KIND_LABEL = {"chars": "characters", "words": "words"}


def _cot_length_series(reasoning_series, length_kind):
    s = reasoning_series.fillna("").astype(str)
    if length_kind == "chars":
        return s.str.len()
    if length_kind == "words":
        return s.str.count(_WORD_PATTERN)
    raise ValueError(f"unknown length_kind={length_kind!r}; "
                     f"expected one of {sorted(_LENGTH_KIND_LABEL)}")


# %%
def bias_score(df):
    """Pooled good-side bias in [-1, 1]; same definition as plot_biases.py."""
    directional = df[df["direction"].isin(["below_good", "above_good"])]
    if len(directional) == 0:
        return float("nan")
    return 2 * directional["on_good_side"].mean() - 1


# %%
main_dfs = get_main_dfs(
    EXPERIMENT,
    list(dict.fromkeys(MODEL_NAMES + PER_PROMPT_MODELS)),
    cache_only=CACHE_ONLY,
)


# %%
def _balanced_bias_and_se(sub):
    """Direction-balanced bucket bias: P(good | below_good) +
    P(good | above_good) - 1, i.e. both threshold directions weighted
    equally regardless of the bucket's composition. CoT length correlates
    with direction (e.g. above_good CoTs tend to be longer for some
    models), so a pooled estimate would conflate the length effect with
    direction composition drifting across buckets. Equal weighting keeps
    the estimand identical to the by-design ~50/50 headline bias.

    Returns (bias, se) or (nan, nan) if either direction is missing.
    ``se`` propagates the two binomial standard errors.
    """
    below = sub.loc[sub["direction"] == "below_good", "on_good_side"]
    above = sub.loc[sub["direction"] == "above_good", "on_good_side"]
    if len(below) == 0 or len(above) == 0:
        return float("nan"), float("nan")
    pb, pa = below.mean(), above.mean()
    se = float(np.sqrt(pb * (1 - pb) / len(below)
                       + pa * (1 - pa) / len(above)))
    return float(pb + pa - 1), se


def per_model_bucket_df(main_dfs, n_buckets=N_BUCKETS, length_kind="chars"):
    """Long-form: one row per (model, bucket). Buckets are per-model quantiles
    of CoT length, computed over directional rows pooled across prompts.
    `length_kind` selects the unit ("chars" = `str.len()`, "words" = runs of
    >=2 letters; see `_cot_length_series`). Bias is the direction-balanced
    estimator (see `_balanced_bias_and_se`); buckets missing a direction
    entirely are dropped.

    Deliberately NOT equal-weighted per prompt (unlike the headline bias
    numbers): a length bucket's prompt mix shifts with length by
    construction -- long CoTs come disproportionately from particular
    prompts, and that composition is part of what the length axis means.
    For the within-prompt view use `per_model_prompt_bucket_df` (the
    per-prompt appendix figures).
    """
    rows = []
    for model_key, (df, _thresholds, display_name) in main_dfs.items():
        directional = df[df["direction"].isin(["below_good", "above_good"])].copy()
        if len(directional) == 0:
            continue
        directional["cot_len"] = _cot_length_series(
            directional["reasoning"], length_kind,
        )

        try:
            directional["bucket"] = pd.qcut(
                directional["cot_len"], q=n_buckets,
                labels=False, duplicates="drop",
            )
        except ValueError:
            # All zeros / not enough distinct values: skip this model.
            continue

        for b, sub in directional.groupby("bucket"):
            bias, se = _balanced_bias_and_se(sub)
            if np.isnan(bias):
                continue
            rows.append({
                "model_key": model_key,
                "model": display_name,
                "bucket": int(b),
                "n": len(sub),
                "median_len": float(sub["cot_len"].median()),
                "min_len": int(sub["cot_len"].min()),
                "max_len": int(sub["cot_len"].max()),
                "bias": bias,
                "bias_se": se,
            })
    return pd.DataFrame(rows)


# %%
buckets_df = per_model_bucket_df(main_dfs, length_kind="words")
print(buckets_df.to_string(index=False))


# %%
# Distinguishable colors keyed by MODEL_GROUPS family. Family base colors
# are derived exactly as in `shared/final_scripts/giraffes/plot_biases.py`
# (Claude -> reserved orange; every other family takes the orange-free
# categorical palette in MODEL_GROUPS order) so the per-line palette here
# uses the same per-family hues as the per-model bars there. Each family is
# rendered as a 3-stop "light tint -> base color -> dark shade" gradient;
# members are sampled symmetrically around t=0.5, so a single-member family
# lands on the exact plot_biases family color and multi-member families
# straddle it. Markers handle within-family disambiguation when sibling hues
# are close (e.g. opus-4.7-high vs opus-4.7-max).
from matplotlib.colors import to_rgb, LinearSegmentedColormap

# Per-family palette (kept in sync with plot_biases.py): the full tab10 cycle
# INCLUDING green/red, with orange held out and reserved for Claude. With
# MODEL_GROUPS in canonical order this gives Claude=orange, GPT=blue,
# Gemini=green, Qwen=red, Kimi=purple.
CLAUDE_ORANGE = "#ff7f0e"
MODEL_COLORS = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
                if c != CLAUDE_ORANGE]


def _family_colors(model_groups):
    """Pin a base color per family by name (same rule as plot_biases.py):
    Claude families get the reserved orange; every other family takes the
    tab10-minus-orange palette in MODEL_GROUPS order."""
    colors, i = {}, 0
    for label, _ in model_groups:
        if label.startswith("Claude"):
            colors[label] = CLAUDE_ORANGE
        else:
            colors[label] = MODEL_COLORS[i % len(MODEL_COLORS)]
            i += 1
    return colors


# Positional list (one base color per group, in MODEL_GROUPS order) so the
# index-based `_model_color(group_idx, ...)` below stays unchanged.
FAMILY_BASE_COLORS = [
    _family_colors(MODEL_GROUPS)[label] for label, _ in MODEL_GROUPS
]
MEMBER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _build_family_cmap(base_hex, light_blend=0.7, dark_blend=0.55):
    """Sequential colormap passing through `base_hex` at t=0.5.

    `light_blend`: how far to blend toward white at t=0 (higher = lighter).
    `dark_blend`:  how far to blend toward black at t=1 (higher = darker).
    """
    base = to_rgb(base_hex)
    light = tuple(c + (1 - c) * light_blend for c in base)
    dark = tuple(c * (1 - dark_blend) for c in base)
    return LinearSegmentedColormap.from_list(
        f"fam_{base_hex}", [light, base, dark], N=256,
    )


_FAMILY_CMAPS = [_build_family_cmap(c) for c in FAMILY_BASE_COLORS]


def _model_color(group_idx, member_idx, group_size):
    cmap = _FAMILY_CMAPS[group_idx % len(_FAMILY_CMAPS)]
    # Symmetric around t=0.5 so single-member families return the exact
    # plot_biases family color, and multi-member families pass through
    # it on average. Spread widens with group size, capped so the
    # extremes don't bleach (t<0.05) or blacken (t>0.95).
    if group_size == 1:
        t = 0.5
    else:
        center = 0.5
        half = min(0.40, 0.08 * group_size)
        t = center - half + 2 * half * (member_idx / (group_size - 1))
    return cmap(t)


def _model_marker(member_idx):
    return MEMBER_MARKERS[member_idx % len(MEMBER_MARKERS)]


def plot_bias_vs_cot_len_absolute(buckets_df, model_groups, log_x=True,
                                  show_errorbars=True, filename=None,
                                  length_kind="chars"):
    """Same buckets, but x is the bucket's median CoT length in the chosen
    unit ("chars" or "words"; only affects the axis label -- must match the
    `length_kind` used to build ``buckets_df``). Log x by default because
    length distributions span orders of magnitude across families.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for gi, (group_label, members) in enumerate(model_groups):
        for mi, mk in enumerate(members):
            sub = buckets_df[buckets_df["model_key"] == mk].sort_values("bucket")
            if sub.empty:
                continue
            display = sub["model"].iloc[0]
            color = _model_color(gi, mi, len(members))
            marker = _model_marker(mi)
            xs = sub["median_len"].to_numpy()
            ys = sub["bias"].to_numpy()
            if log_x:
                # Log x can't show 0; nudge zero-length buckets to 1 so they
                # don't disappear (they're rare but happen for non-thinking
                # models).
                xs = np.where(xs <= 0, 1.0, xs)
            if show_errorbars:
                ax.errorbar(xs, ys, yerr=sub["bias_se"].to_numpy(),
                            marker=marker,
                            color=color, label=display,
                            capsize=2, alpha=0.9)
            else:
                ax.plot(xs, ys, marker=marker, color=color,
                        label=display)

    ax.axhline(0, color="black", linewidth=0.6)
    if log_x:
        ax.set_xscale("log")
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 5), numticks=200)
        )
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(FuncFormatter(
            lambda x, _: f"{int(x):,}" if x >= 1 else ""
        ))
        ax.tick_params(axis="x", which="minor", labelsize=8, labelrotation=30)
        ax.tick_params(axis="x", which="major", labelrotation=30)
    ax.tick_params(axis="y", which="major")
    unit = _LENGTH_KIND_LABEL.get(length_kind, length_kind)
    ax.set_xlabel(f"CoT length ({unit}; bucket median)")
    ax.set_ylabel("Bias metric")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left",
              bbox_to_anchor=(1.02, 0.5), ncol=1, framealpha=0.9)
    plt.tight_layout()
    _maybe_savefig(fig, filename)
    plt.show()


# %%
plot_bias_vs_cot_len_absolute(
    buckets_df, MODEL_GROUPS, show_errorbars=False, length_kind="words",
    filename=str(FIG_DIR / "bias_vs_cot_words_absolute.pdf"),
)


# %%
def print_bucket_lengths(buckets_df, model_groups):
    """Quick summary: per-model median CoT length per bucket."""
    ordered = [mk for _, g in model_groups for mk in g]
    pivot = buckets_df.pivot(index="model_key", columns="bucket",
                             values="median_len").reindex(ordered)
    pivot.columns = [f"q{int(c) + 1}_med_len" for c in pivot.columns]
    print(pivot.to_string())


print_bucket_lengths(buckets_df, MODEL_GROUPS)


# %%
def per_model_prompt_bucket_df(main_dfs, n_buckets=N_BUCKETS,
                               length_kind="chars"):
    """Long-form: one row per (model, prompt_key, bucket). Buckets are
    per-(model, prompt_key) quantiles of CoT length, so each subplot's
    quintiles describe that prompt's own distribution rather than a
    pooled-across-prompts one (which would slice into very sparse cells).
    `length_kind` matches `per_model_bucket_df`.
    """
    rows = []
    for model_key, (df, _thresholds, display_name) in main_dfs.items():
        directional = df[df["direction"].isin(["below_good", "above_good"])].copy()
        if len(directional) == 0:
            continue
        directional["cot_len"] = _cot_length_series(
            directional["reasoning"], length_kind,
        )
        for pk, pk_sub in directional.groupby("prompt_key"):
            pk_sub = pk_sub.copy()
            try:
                pk_sub["bucket"] = pd.qcut(
                    pk_sub["cot_len"], q=n_buckets,
                    labels=False, duplicates="drop",
                )
            except ValueError:
                continue
            for b, sub in pk_sub.groupby("bucket"):
                bias, se = _balanced_bias_and_se(sub)
                if np.isnan(bias):
                    continue
                rows.append({
                    "model_key": model_key,
                    "model": display_name,
                    "prompt_key": pk,
                    "bucket": int(b),
                    "n": len(sub),
                    "median_len": float(sub["cot_len"].median()),
                    "min_len": int(sub["cot_len"].min()),
                    "max_len": int(sub["cot_len"].max()),
                    "bias": bias,
                    "bias_se": se,
                })
    return pd.DataFrame(rows)


def plot_bias_vs_cot_len_absolute_for_model(prompt_buckets_df, model_key,
                                            model_display, log_x=True,
                                            show_errorbars=True,
                                            filename=None,
                                            length_kind="chars"):
    """Single-model version of `plot_bias_vs_cot_len_absolute`. One line
    per `prompt_key` (estimate category), all on the same axes. Buckets
    come from `per_model_prompt_bucket_df` (per-(model, prompt_key)
    quantiles), so each line spans that prompt's own CoT length range.
    `length_kind` matches its meaning in `plot_bias_vs_cot_len_absolute`.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.get_cmap("tab10")

    sub_model = prompt_buckets_df[prompt_buckets_df["model_key"] == model_key]
    prompts = sorted(sub_model["prompt_key"].dropna().unique())

    for pi, pk in enumerate(prompts):
        sub = sub_model[sub_model["prompt_key"] == pk].sort_values("bucket")
        if sub.empty:
            continue
        color = cmap(pi % 10)
        xs = sub["median_len"].to_numpy()
        ys = sub["bias"].to_numpy()
        if log_x:
            xs = np.where(xs <= 0, 1.0, xs)
        label = pk[len("v1_"):] if pk.startswith("v1_") else pk
        if show_errorbars:
            ax.errorbar(xs, ys, yerr=sub["bias_se"].to_numpy(),
                        marker="o", color=color, label=label,
                        capsize=2, alpha=0.9)
        else:
            ax.plot(xs, ys, marker="o", color=color,
                    label=label)

    ax.axhline(0, color="black", linewidth=0.6)
    if log_x:
        ax.set_xscale("log")
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 5), numticks=200)
        )
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:g}"
        ))
        ax.xaxis.set_minor_formatter(FuncFormatter(
            lambda x, _: f"{int(x):,}" if x >= 1 else ""
        ))
        ax.tick_params(axis="x", which="minor", labelsize=8, labelrotation=30)
        ax.tick_params(axis="x", which="major", labelrotation=30)
    ax.tick_params(axis="y", which="major")
    unit = _LENGTH_KIND_LABEL.get(length_kind, length_kind)
    ax.set_xlabel(f"CoT length ({unit}; bucket median)")
    ax.set_ylabel("Bias metric")
    ax.set_title(model_display)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    plt.tight_layout()
    _maybe_savefig(fig, filename)
    plt.show()


# %% --- Per-model (per-prompt) appendix figures ---
# One figure per model in PER_PROMPT_MODELS: one line per prompt (estimate
# category), CoT length in words (matching the across-models overview above).
# These are the appendix `bias_vs_cot_words_per_prompt_<model>.pdf` figures
# referenced by the paper; the models here are kept out of MODEL_GROUPS, so
# they appear only in their own per-prompt figure and not the overview.
prompt_buckets_df_words = per_model_prompt_bucket_df(main_dfs, length_kind="words")
for model_key in PER_PROMPT_MODELS:
    _df, _thresholds, display_name = main_dfs[model_key]
    plot_bias_vs_cot_len_absolute_for_model(
        prompt_buckets_df_words, model_key, display_name,
        show_errorbars=False, length_kind="words",
        filename=str(FIG_DIR / f"bias_vs_cot_words_per_prompt_{model_key}.pdf"),
    )

# %%
