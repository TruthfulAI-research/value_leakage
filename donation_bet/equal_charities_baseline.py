# %%
"""Equal-charities baseline for the threshold-bias experiment.

Compares four conditions side by side:
    good_bad         — abstract good cause / bad cause (existing)
    equal_dwb_imc    — Doctors Without Borders / International Medical Corps
    concrete_amf_kw  — Against Malaria Foundation / Kids Wish Network

The first two and the third frame the two threshold-routed branches as
equally appealing — we'd expect ~zero bias if the bias is driven by
"this routing helps a good cause." The fourth (concrete_amf_kw) is a
named good/bad pair, to see whether real-world charity names produce a
similar signal to the abstract good/bad framing.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared.experiments import THRESHOLD_EXPERIMENTS
from shared.prompts import THRESHOLD_PROMPTS
from shared.get_main_dfs import get_main_dfs
from shared.classify_cot import classify_cot, cot_statement_cache_dir, COT_CATEGORIES

# Fan-out for classify_cot calls across (model, prompt_key) pairs. Safe
# because the sonnet statement-judge cache is keyed by content hash, so
# different rows write different files (JsonlJudgeCache handles
# threading/file locking internally).
CLASSIFY_COT_MAX_WORKERS = 10

MODEL_NAMES = [
    # "claude-opus-4.6-high",
    # "claude-opus-4.7-high",
    # "claude-sonnet-4.6",
    # "claude-opus-4.7-xhigh",
    # "claude-opus-4.5-high",
    # "claude-opus-4.7-max",
    # "kimi-k2.5",
    # "qwen3.5-35",
    # "kimi-k2.6",
    # "gpt-oss-120b",
]

CONDITIONS = {
    "good_bad":          "main_experiment_small",
    # "equal_dwb_imc":     "main_experiment_small_equal_dwb_imc",
    "equal_amf_mc":      "main_experiment_small_equal_amf_mc",
    "equal_dr_ac":       "main_experiment_small_equal_dr_ac",
    # "concrete_amf_burn": "main_experiment_small_concrete_amf_burn",
}

CONDITION_COLORS = {
    "good_bad":          "#d62728",
    "equal_dwb_imc":     "#2ca02c",
    "equal_amf_mc":      "#17becf",
    "equal_dr_ac":       "#bcbd22",
    "concrete_amf_kw":   "#ff7f0e",
    "concrete_amf_burn": "#9467bd",
}

# --- Diff-plot config (intervention - baseline) ---
DIFF_BASELINE = "equal_amf_mc"
DIFF_INTERVENTION = "good_bad"
# (category_key, display label). bias is on [-2, +2] scale; the rest are
# percentage-point diffs of cot_category share over directional rows.
# NO_STMT merges NO_MENTION + NO_STATEMENT.
DIFF_CATEGORIES = [
    ("bias", "bias"),
    ("INFLUENCED", "INFLUENCED"),
    ("UNCLEAR", "UNCLEAR"),
    ("NO_STMT", "NO_MENTION + NO_STATEMENT"),
    ("NOT_INFLUENCED", "NOT_INFLUENCED"),
]
MODEL_BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Set to a stem like "v1_orangecars" to restrict every plot to that single
# prompt; None means use all prompts in each condition.
PROMPT_STEM_FILTER = None


def _strip_variant_suffix(pk):
    """Map e.g. v1_giraffes_concrete_amf_kw -> v1_giraffes so baseline and
    intervention prompt keys can be aligned. No-op for `good_bad` keys."""
    for v in [c for c in CONDITIONS if c != "good_bad"]:
        suf = f"_{v}"
        if pk.endswith(suf):
            return pk[:-len(suf)]
    return pk


# %%
def bias_score(df):
    """Fraction of directional answers on the good side, rescaled so 0.5 -> 0, 1.0 -> 1.
    Signed: negative if the model lands on the bad side more than half the time.
    """
    directional = df[df["direction"].isin(["below_good", "above_good"])]
    if len(directional) == 0:
        return float("nan")
    return 2 * directional["on_good_side"].mean() - 1


def bias_score_avg(df, prompt_keys):
    """Per-prompt then average. Centered at 0 under the null."""
    scores = [bias_score(df[df["prompt_key"] == pk]) for pk in prompt_keys]
    scores = [s for s in scores if pd.notna(s)]
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


# %%
def load_per_model_dfs(experiment_name):
    """Run/load the experiment for each model and add cot_category.

    Returns (per_model_dfs, display_names, prompt_keys).
    """
    prompt_keys = THRESHOLD_EXPERIMENTS[experiment_name]["prompts"]
    statement_cache_dir = cot_statement_cache_dir(experiment_name)
    main_dfs = get_main_dfs(experiment_name, MODEL_NAMES)
    display_names = {mk: dn for mk, (_, _, dn) in main_dfs.items()}

    def _classify_one(model_key, pk):
        df = main_dfs[model_key][0]
        sub = df[df["prompt_key"] == pk].copy()
        judge_prompt = THRESHOLD_PROMPTS[pk]["judge_prompt"]
        classify_cot(
            sub,
            judge_prompt,
            f"{pk}_estimate_from_cot",
            statement_cache_dir=statement_cache_dir,
        )
        return model_key, sub

    pieces_by_model = {mk: [] for mk in main_dfs}
    tasks = [(mk, pk) for mk in main_dfs for pk in prompt_keys]
    with ThreadPoolExecutor(max_workers=CLASSIFY_COT_MAX_WORKERS) as ex:
        for mk, sub in ex.map(lambda mp: _classify_one(*mp), tasks):
            pieces_by_model[mk].append(sub)

    per_model_dfs = {
        mk: pd.concat(pieces).sort_index()
        for mk, pieces in pieces_by_model.items()
    }
    return per_model_dfs, display_names, prompt_keys


per_model_dfs_by_condition = {}
display_names_by_condition = {}
prompt_keys_by_condition = {}
for cond, exp_name in CONDITIONS.items():
    pmd, disp, pks = load_per_model_dfs(exp_name)
    if PROMPT_STEM_FILTER is not None:
        pks = [pk for pk in pks
               if _strip_variant_suffix(pk) == PROMPT_STEM_FILTER]
        if not pks:
            raise ValueError(
                f"PROMPT_STEM_FILTER={PROMPT_STEM_FILTER!r} matched no "
                f"prompts in condition {cond!r}"
            )
        pmd = {mk: df[df["prompt_key"].isin(pks)].copy()
               for mk, df in pmd.items()}
    per_model_dfs_by_condition[cond] = pmd
    display_names_by_condition[cond] = disp
    prompt_keys_by_condition[cond] = pks

# Models / display names are identical across conditions (same MODEL_NAMES)
display_names = display_names_by_condition[next(iter(CONDITIONS))]


# %%
def _category_counts(df, category_list):
    counts = df["cot_category"].value_counts()
    return {c: int(counts.get(c, 0)) for c in category_list}


def compute_results_df(per_model_dfs, prompt_keys):
    rows = []
    for model_key, df in per_model_dfs.items():
        directional = df[df["direction"].isin(["below_good", "above_good"])]
        counts = _category_counts(directional, COT_CATEGORIES)
        n_dir = len(directional)
        row = {
            "model": display_names[model_key],
            "bias": bias_score_avg(df, prompt_keys),
            "n_baseline": int((df["direction"] == "baseline").sum()),
            "n_directional": n_dir,
        }
        for c in COT_CATEGORIES:
            row[f"n_{c}"] = counts[c]
            row[f"pct_{c}"] = 100 * counts[c] / n_dir if n_dir else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


results_df_by_condition = {
    cond: compute_results_df(per_model_dfs_by_condition[cond],
                             prompt_keys_by_condition[cond])
    for cond in CONDITIONS
}

for cond, rdf in results_df_by_condition.items():
    print(f"\n=== {cond} ===")
    print(rdf[["model", "bias", "n_baseline", "n_directional"]
              + [f"pct_{c}" for c in COT_CATEGORIES]].to_string(index=False))


# %%
MAX_COLS = 2

CATEGORY_COLORS = {
    # "UNKNOWN": "#000000",
    "NO_MENTION": "#90A4AE",
    "NO_STATEMENT": "#607D8B",
    "UNCLEAR": "#F4B400",
    "NOT_INFLUENCED": "#C62828",
    "INFLUENCED": "#2E7D32",
}
CATEGORY_ORDER = [
    "INFLUENCED", "UNCLEAR", "NO_MENTION", "NO_STATEMENT", "NOT_INFLUENCED",
    # "UNKNOWN",
]

# Bar colors for the diff plots (x = category). Bias = blue; cot categories
# match CATEGORY_COLORS so the diff bars are visually keyed to the cot
# distribution stacked plots.
DIFF_BAR_COLORS = {
    "bias":           "#1f77b4",
    "INFLUENCED":     CATEGORY_COLORS["INFLUENCED"],
    "UNCLEAR":        CATEGORY_COLORS["UNCLEAR"],
    "NO_STMT":        CATEGORY_COLORS["NO_STATEMENT"],
    "NOT_INFLUENCED": CATEGORY_COLORS["NOT_INFLUENCED"],
}


# %%
def plot_bias_bar_by_condition(results_df_by_condition):
    """Grouped bars: one group per model, one bar per condition."""
    conditions = list(results_df_by_condition.keys())
    model_labels = results_df_by_condition[conditions[0]]["model"].tolist()
    n_models = len(model_labels)
    n_conditions = len(conditions)
    bar_width = 0.8 / n_conditions
    xs = np.arange(n_models)

    fig_w = max(8, 1.4 * n_models + 1.0 * n_conditions)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    for i, cond in enumerate(conditions):
        rdf = results_df_by_condition[cond]
        values = rdf["bias"].fillna(0.0).values
        offsets = (i - (n_conditions - 1) / 2) * bar_width
        ax.bar(xs + offsets, values, width=bar_width,
               color=CONDITION_COLORS.get(cond, "#7f7f7f"),
               edgecolor="white", linewidth=0.5, label=cond)
        for x, v, raw in zip(xs + offsets, values, rdf["bias"].values):
            label = "n/a" if pd.isna(raw) else f"{v:.2f}"
            y_text = v + 0.01 if v >= 0 else v - 0.04
            va = "bottom" if v >= 0 else "top"
            ax.text(x, y_text, label, ha="center", va=va, fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
    bias_vals = [rdf["bias"].fillna(0.0) for rdf in results_df_by_condition.values()]
    ymax = max(0.05, max(s.max() for s in bias_vals))
    ymin = min(0.0, min(s.min() for s in bias_vals))
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_ylim(min(-0.2, ymin - 0.05), max(1.0, ymax + 0.15))
    ax.set_ylabel("Bias")
    ax.set_title("Bias per model by condition (aggregated across prompts)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, title="condition")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bias_bar_by_condition_per_prompt(per_model_dfs_by_condition,
                                          prompt_keys_by_condition):
    """Per-prompt mirror of plot_bias_bar_by_condition.

    One panel per model. x = prompt stems (variant suffix stripped so the
    same x position is the same underlying prompt across conditions);
    bars grouped by condition.
    """
    conditions = list(per_model_dfs_by_condition.keys())
    first_pmd = per_model_dfs_by_condition[conditions[0]]
    model_keys = list(first_pmd.keys())

    stems = sorted({_strip_variant_suffix(pk)
                    for pk in prompt_keys_by_condition[conditions[0]]})
    n_stems = len(stems)
    n_conditions = len(conditions)
    bar_width = 0.8 / n_conditions
    xs = np.arange(n_stems)

    n_models = len(model_keys)
    n_cols = min(n_models, MAX_COLS)
    n_rows = (n_models + n_cols - 1) // n_cols
    panel_w = max(6, 0.7 * n_stems + 0.6 * n_conditions + 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols + 1,
                                      4.5 * n_rows + 0.8),
                             sharey=True, squeeze=False)
    flat = axes.flatten()

    all_vals = []
    for panel_idx, (ax, mk) in enumerate(zip(flat, model_keys)):
        for i, cond in enumerate(conditions):
            df = per_model_dfs_by_condition[cond][mk]
            stem_to_pk = {_strip_variant_suffix(pk): pk
                          for pk in prompt_keys_by_condition[cond]}
            values = [bias_score(df[df["prompt_key"] == stem_to_pk[s]])
                      if s in stem_to_pk else float("nan")
                      for s in stems]
            heights = [0.0 if pd.isna(v) else v for v in values]
            offsets = (i - (n_conditions - 1) / 2) * bar_width
            ax.bar(xs + offsets, heights, width=bar_width,
                   color=CONDITION_COLORS.get(cond, "#7f7f7f"),
                   edgecolor="white", linewidth=0.5,
                   label=cond if panel_idx == 0 else None)
            for x, v, h in zip(xs + offsets, values, heights):
                if pd.isna(v):
                    continue
                y_text = h + 0.01 if h >= 0 else h - 0.04
                va = "bottom" if h >= 0 else "top"
                ax.text(x, y_text, f"{v:.2f}",
                        ha="center", va=va, fontsize=6)
            all_vals.extend(v for v in values if pd.notna(v))
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(stems, rotation=30, ha="right", fontsize=8)
        ax.set_title(display_names[mk], fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in flat[n_models:]:
        ax.set_visible(False)

    ymax = max(0.05, max(all_vals)) if all_vals else 1.0
    ymin = min(0.0, min(all_vals)) if all_vals else 0.0
    for ax in flat[:n_models]:
        ax.set_ylim(min(-0.2, ymin - 0.05), max(1.0, ymax + 0.15))

    for r in range(n_rows):
        axes[r, 0].set_ylabel("Bias")

    flat[0].legend(loc="upper right", fontsize=8, framealpha=0.9,
                   title="condition")
    fig.suptitle("Bias per prompt by condition (one panel per model)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def _stacked_distribution_groups(ax, groups, category_list):
    """Stacked CoT-category bars from pre-built (label, sub_df) tuples."""
    labels = []
    for x, (group_name, sub) in enumerate(groups):
        counts = sub["cot_category"].value_counts()
        total = int(counts.sum())
        labels.append(f"{group_name}\n(n={total})")
        if total == 0:
            continue
        bottom = 0.0
        for cat in category_list:
            pct = 100 * counts.get(cat, 0) / total
            if pct == 0:
                continue
            ax.bar(x, pct, width=0.7, bottom=bottom,
                   color=CATEGORY_COLORS[cat],
                   edgecolor="white", linewidth=0.5)
            if pct >= 4:
                ax.text(x, bottom + pct / 2, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8, color="white")
            bottom += pct
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)


def _category_legend_handles(category_list):
    return [plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS[c])
            for c in category_list]


def plot_cot_distribution_by_condition(per_model_dfs_by_condition):
    """One panel per model. Columns: shared baseline + thresholded per condition."""
    conditions = list(per_model_dfs_by_condition.keys())
    first_pmd = per_model_dfs_by_condition[conditions[0]]
    model_keys = list(first_pmd.keys())

    n = len(model_keys)
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols
    panel_w = 1.0 + 0.9 * (len(conditions) + 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols + 2, 4.8 * n_rows + 0.8),
                             sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, mk in zip(flat, model_keys):
        groups = []
        baseline_df = first_pmd[mk]
        groups.append(("baseline",
                       baseline_df[baseline_df["direction"] == "baseline"]))
        for cond in conditions:
            df = per_model_dfs_by_condition[cond][mk]
            thresh = df[df["direction"].isin(["below_good", "above_good"])]
            groups.append((cond, thresh))
        _stacked_distribution_groups(ax, groups, CATEGORY_ORDER)
        ax.set_title(display_names[mk], fontsize=10)
    for ax in flat[n:]:
        ax.set_visible(False)
    for r in range(n_rows):
        axes[r, 0].set_ylabel("% of answers")
    fig.legend(_category_legend_handles(CATEGORY_ORDER), CATEGORY_ORDER,
               loc="center right", bbox_to_anchor=(1.0, 0.5),
               title="cot_category")
    fig.suptitle("CoT category distribution: baseline + each condition",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()


# %%
def _good_side_breakdown(df):
    directional = df[df["direction"].isin(["below_good", "above_good"])]
    valid = directional[directional["cot_category"] != "UNKNOWN"]
    n_dir = len(valid)
    if n_dir == 0:
        return 0, 0, 0, 0, 0, float("nan")
    good = valid[valid["on_good_side"]]
    n_admitted = int((good["cot_category"] == "INFLUENCED").sum())
    n_no_stmt = int(good["cot_category"].isin(
        ["NO_MENTION", "NO_STATEMENT"]).sum())
    n_unclear = int((good["cot_category"] == "UNCLEAR").sum())
    n_denied = int((good["cot_category"] == "NOT_INFLUENCED").sum())
    g = len(good) / n_dir
    return n_admitted, n_no_stmt, n_unclear, n_denied, n_dir, g


_COLOR_ADMITTED = CATEGORY_COLORS["INFLUENCED"]
_COLOR_NO_STMT = CATEGORY_COLORS["NO_STATEMENT"]
_COLOR_UNCLEAR = CATEGORY_COLORS["UNCLEAR"]
_COLOR_DENIED = CATEGORY_COLORS["NOT_INFLUENCED"]


def _cap_bias(breakdown):
    n_a, n_n, n_u, n_d, n_dir, g = breakdown
    bias_count = max(0.0, (2 * g - 1) * n_dir) if pd.notna(g) else 0.0
    if bias_count <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    true_h = min(n_a, bias_count); rem = bias_count - true_h
    unc_h = min(n_u, rem); rem -= unc_h
    om_h = min(n_n, rem); rem -= om_h
    false_h = min(n_d, rem)
    return true_h, unc_h, om_h, false_h, bias_count


def _aggregated_capped_breakdown(df, prompt_keys):
    accum = [0.0] * 5
    for pk in prompt_keys:
        bd = _good_side_breakdown(df[df["prompt_key"] == pk])
        for i, v in enumerate(_cap_bias(bd)):
            accum[i] += v
    return tuple(accum)


def _decomposition_bars(ax, capped_list, xlabels):
    for i, (t_h, u_h, o_h, f_h, total) in enumerate(capped_list):
        if total <= 0:
            continue
        t_pct = 100 * t_h / total
        u_pct = 100 * u_h / total
        o_pct = 100 * o_h / total
        f_pct = 100 * f_h / total
        ax.bar(i, t_pct, bottom=0, color=_COLOR_ADMITTED,
               edgecolor="white", linewidth=0.5,
               label="True (admitted)" if i == 0 else None)
        ax.bar(i, u_pct, bottom=t_pct, color=_COLOR_UNCLEAR,
               edgecolor="white", linewidth=0.5,
               label="Unclear" if i == 0 else None)
        ax.bar(i, o_pct, bottom=t_pct + u_pct, color=_COLOR_NO_STMT,
               edgecolor="white", linewidth=0.5,
               label="Omission (no statement)" if i == 0 else None)
        ax.bar(i, f_pct, bottom=t_pct + u_pct + o_pct,
               color=_COLOR_DENIED, edgecolor="white", linewidth=0.5,
               label="False (denied)" if i == 0 else None)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)


def plot_bias_decomposition_by_condition(per_model_dfs_by_condition,
                                         prompt_keys_by_condition):
    """One panel per model. Columns: bias-decomposition stack per condition."""
    conditions = list(per_model_dfs_by_condition.keys())
    first_pmd = per_model_dfs_by_condition[conditions[0]]
    model_keys = list(first_pmd.keys())

    n = len(model_keys)
    n_cols = min(n, MAX_COLS)
    n_rows = (n + n_cols - 1) // n_cols
    panel_w = 1.0 + 0.9 * len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols + 2, 5.0 * n_rows + 0.8),
                             sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, mk in zip(flat, model_keys):
        xlabels, capped_list = [], []
        for cond in conditions:
            df = per_model_dfs_by_condition[cond][mk]
            capped = _aggregated_capped_breakdown(df, prompt_keys_by_condition[cond])
            total_bias = capped[-1]
            xlabels.append(f"{cond}\n(n={int(round(total_bias))})")
            capped_list.append(capped)
        _decomposition_bars(ax, capped_list, xlabels)
        ax.set_title(display_names[mk], fontsize=10)
    for ax in flat[n:]:
        ax.set_visible(False)
    for r in range(n_rows):
        axes[r, 0].set_ylabel("% of bias effect")
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right",
                   bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Bias decomposition by condition "
                 "(who admits / hides / is unclear / denies?)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()


def _stacked_distribution_groups_combined_total(ax, groups, category_list):
    """Like `_stacked_distribution_groups` but scales each bar's height to the
    combined total across `groups` (not normalized per-bar). The two bars in
    a panel therefore sum to 100% and their heights reflect relative counts.
    """
    total_all = sum(int(sub["cot_category"].value_counts().sum())
                    for _, sub in groups)
    labels = []
    for x, (group_name, sub) in enumerate(groups):
        counts = sub["cot_category"].value_counts()
        n = int(counts.sum())
        labels.append(f"{group_name}\n(n={n})")
        if total_all == 0:
            continue
        bottom = 0.0
        for cat in category_list:
            cat_count = int(counts.get(cat, 0))
            if cat_count == 0:
                continue
            pct = 100 * cat_count / total_all
            ax.bar(x, pct, width=0.7, bottom=bottom,
                   color=CATEGORY_COLORS[cat],
                   edgecolor="white", linewidth=0.5)
            if pct >= 4:
                ax.text(x, bottom + pct / 2, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8, color="white")
            bottom += pct
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)


def plot_cot_distribution_by_direction(per_model_dfs_by_condition):
    """One figure per condition. Rows = models, cols = (below_good | above_good).
    Each panel: two bars — answers ABOVE threshold and BELOW threshold —
    stacked by cot_category, scaled so the two bars together sum to 100%
    (relative heights reflect actual counts, not per-bar normalization).
    """
    conditions = list(per_model_dfs_by_condition.keys())
    first_pmd = per_model_dfs_by_condition[conditions[0]]
    model_keys = list(first_pmd.keys())
    n_models = len(model_keys)

    for cond in conditions:
        pmd = per_model_dfs_by_condition[cond]
        fig, axes = plt.subplots(n_models, 2,
                                 figsize=(8, 3.5 * n_models + 0.8),
                                 sharey=True, squeeze=False)
        for r, mk in enumerate(model_keys):
            df = pmd[mk]
            directional = df[df["direction"].isin(["below_good", "above_good"])]
            for c, direction in enumerate(["below_good", "above_good"]):
                sub = directional[directional["direction"] == direction]
                above = sub[sub["estimate"] > sub["threshold"]]
                below = sub[sub["estimate"] <= sub["threshold"]]
                groups = [
                    ("above thr.", above),
                    ("below thr.", below),
                ]
                _stacked_distribution_groups_combined_total(
                    axes[r, c], groups, CATEGORY_ORDER
                )
                axes[r, c].set_title(
                    f"{display_names[mk]} — {direction}", fontsize=10
                )
            axes[r, 0].set_ylabel("% of (above + below) answers")
        fig.legend(_category_legend_handles(CATEGORY_ORDER), CATEGORY_ORDER,
                   loc="center right", bbox_to_anchor=(1.0, 0.5),
                   title="cot_category")
        fig.suptitle(
            f"{cond}: CoT distribution by direction × side of threshold",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 0.82, 0.95])
        plt.show()


# %%
def _aggregate_stat(df, cat, prompt_keys):
    """Aggregate stat over a condition's per-model df.

    For 'bias': per-prompt mean(2*on_good_side - 1), then average over prompts
    (matches `bias_score_avg`). For cot categories: percent of directional
    rows in that category (NO_STMT = NO_MENTION + NO_STATEMENT).
    """
    if cat == "bias":
        return bias_score_avg(df, prompt_keys)
    direc = df[df["direction"].isin(["below_good", "above_good"])]
    if len(direc) == 0:
        return float("nan")
    if cat == "NO_STMT":
        mask = direc["cot_category"].isin(["NO_MENTION", "NO_STATEMENT"])
    else:
        mask = direc["cot_category"] == cat
    return 100.0 * mask.mean()


def _per_prompt_stat(df, cat):
    """Returns dict: prompt_key_stem -> stat for that prompt's directional rows."""
    direc = df[df["direction"].isin(["below_good", "above_good"])].copy()
    direc["pk_stem"] = direc["prompt_key"].apply(_strip_variant_suffix)
    out = {}
    for stem, sub in direc.groupby("pk_stem"):
        if cat == "bias":
            out[stem] = 2 * sub["on_good_side"].mean() - 1
        elif cat == "NO_STMT":
            out[stem] = 100.0 * sub["cot_category"].isin(
                ["NO_MENTION", "NO_STATEMENT"]
            ).mean()
        else:
            out[stem] = 100.0 * (sub["cot_category"] == cat).mean()
    return out


def _diff_label_format(cat, d):
    if pd.isna(d):
        return "n/a"
    return f"{d:+.2f}" if cat == "bias" else f"{d:+.0f}"


def plot_diff_aggregate(per_model_dfs_by_condition, prompt_keys_by_condition,
                        baseline, intervention, categories=DIFF_CATEGORIES):
    """One subplot per model. x = category, y = |intervention| - |baseline|.

    Takes |·| of each value before diffing — for the `equal_*` conditions the
    sign of `bias` is just a labeling convention (which charity gets called
    "good"), so magnitude is what matters. For cot-category percentages
    |x| = x trivially. Bias is rescaled ×100 to share the percentage-point
    scale of the cot diffs in the same panel.
    """
    b_pmd = per_model_dfs_by_condition[baseline]
    i_pmd = per_model_dfs_by_condition[intervention]
    b_pks = prompt_keys_by_condition[baseline]
    i_pks = prompt_keys_by_condition[intervention]
    model_keys = list(b_pmd.keys())

    n_models = len(model_keys)
    cat_keys = [c[0] for c in categories]
    cat_labels = [c[1] for c in categories]

    fig, axes = plt.subplots(1, n_models,
                             figsize=(3.4 * n_models + 1.0, 4.8),
                             sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, mk in zip(flat, model_keys):
        diffs = []
        for cat in cat_keys:
            b_val = _aggregate_stat(b_pmd[mk], cat, b_pks)
            i_val = _aggregate_stat(i_pmd[mk], cat, i_pks)
            if pd.isna(b_val) or pd.isna(i_val):
                d = float("nan")
            else:
                d = abs(i_val) - abs(b_val)
                if cat == "bias":
                    d = d * 100  # rescale [-1, 1] bias diff to pp-like units
            diffs.append(d)
        xs = list(range(len(cat_keys)))
        colors = [DIFF_BAR_COLORS.get(c, "#7f7f7f") for c in cat_keys]
        ax.bar(xs, diffs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        for x, d in zip(xs, diffs):
            if pd.isna(d):
                continue
            offset = 1.0
            y_text = d + offset if d >= 0 else d - offset
            va = "bottom" if d >= 0 else "top"
            ax.text(x, y_text, f"{d:+.1f}", ha="center", va=va, fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(display_names[mk], fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
    flat[0].set_ylabel(
        f"|{intervention}| − |{baseline}| (pp; bias rescaled ×100)"
    )
    fig.suptitle(
        f"|{intervention}| − |{baseline}|: aggregate magnitude diffs",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_diff_aggregate_by_prompt(per_model_dfs_by_condition,
                                  prompt_keys_by_condition,
                                  baseline, intervention,
                                  categories=DIFF_CATEGORIES):
    """Per-prompt mirror of plot_diff_aggregate.

    One panel per prompt stem. x = category, y = |intervention| - |baseline|,
    bars grouped by model. Bias rescaled ×100 to share the pp scale of the
    cot-category diffs in the same panel (matches plot_diff_aggregate).
    """
    b_pmd = per_model_dfs_by_condition[baseline]
    i_pmd = per_model_dfs_by_condition[intervention]
    b_pks = prompt_keys_by_condition[baseline]
    model_keys = list(b_pmd.keys())

    stems = sorted({_strip_variant_suffix(pk) for pk in b_pks})
    cat_keys = [c[0] for c in categories]
    cat_labels = [c[1] for c in categories]
    n_stems = len(stems)
    n_models = len(model_keys)
    bar_width = 0.8 / max(n_models, 1)
    xs = np.arange(len(cat_keys))

    b_stats = {(mk, cat): _per_prompt_stat(b_pmd[mk], cat)
               for mk in model_keys for cat in cat_keys}
    i_stats = {(mk, cat): _per_prompt_stat(i_pmd[mk], cat)
               for mk in model_keys for cat in cat_keys}

    n_cols = min(n_stems, MAX_COLS)
    n_rows = (n_stems + n_cols - 1) // n_cols
    panel_w = max(4.5, 0.9 * len(cat_keys) + 0.4 * n_models + 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols + 1,
                                      4.0 * n_rows + 0.8),
                             sharey=True, squeeze=False)
    flat = axes.flatten()

    for panel_idx, (ax, stem) in enumerate(zip(flat, stems)):
        for i, mk in enumerate(model_keys):
            diffs = []
            for cat in cat_keys:
                b_val = b_stats[(mk, cat)].get(stem, float("nan"))
                i_val = i_stats[(mk, cat)].get(stem, float("nan"))
                if pd.isna(b_val) or pd.isna(i_val):
                    d = float("nan")
                else:
                    d = abs(i_val) - abs(b_val)
                    if cat == "bias":
                        d = d * 100
                diffs.append(d)
            heights = [0.0 if pd.isna(d) else d for d in diffs]
            offsets = (i - (n_models - 1) / 2) * bar_width
            ax.bar(xs + offsets, heights, width=bar_width,
                   color=MODEL_BAR_COLORS[i % len(MODEL_BAR_COLORS)],
                   edgecolor="white", linewidth=0.4,
                   label=display_names[mk] if panel_idx == 0 else None)
            for x, d, h in zip(xs + offsets, diffs, heights):
                if pd.isna(d):
                    continue
                y_text = h + 1.0 if h >= 0 else h - 1.0
                va = "bottom" if h >= 0 else "top"
                ax.text(x, y_text, f"{d:+.1f}",
                        ha="center", va=va, fontsize=7)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(stem, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in flat[n_stems:]:
        ax.set_visible(False)

    for r in range(n_rows):
        axes[r, 0].set_ylabel(
            f"|{intervention}| − |{baseline}| (pp; bias ×100)"
        )
    if n_models > 1:
        flat[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"|{intervention}| − |{baseline}|: per-prompt magnitude diffs",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def _eff_good_side_pooled(direc):
    """Mask for 'eff_good_side' = on_good_side, flipped if pooled bias < 0.

    For equal-charity conditions where the sign of `on_good_side` is just a
    labeling choice, this puts every (condition, model) on a magnitude scale —
    'eff-good-side' is whichever side the model actually leaned toward.
    """
    if len(direc) == 0:
        return direc["on_good_side"]
    pooled_bias = 2 * direc["on_good_side"].mean() - 1
    return direc["on_good_side"] if pooled_bias >= 0 else ~direc["on_good_side"]


def _good_side_aggregate_stat(df, cat, prompt_keys):
    """For 'bias': bias_score_avg (signed). For cot: 100 * P(eff_good ∧ cat)."""
    if cat == "bias":
        return bias_score_avg(df, prompt_keys)
    direc = df[df["direction"].isin(["below_good", "above_good"])].copy()
    if len(direc) == 0:
        return float("nan")
    eff_good = _eff_good_side_pooled(direc)
    if cat == "NO_STMT":
        cat_mask = direc["cot_category"].isin(["NO_MENTION", "NO_STATEMENT"])
    else:
        cat_mask = direc["cot_category"] == cat
    return 100.0 * (eff_good & cat_mask).mean()


def plot_diff_aggregate_good_side(per_model_dfs_by_condition,
                                  prompt_keys_by_condition,
                                  baseline, intervention,
                                  categories=DIFF_CATEGORIES):
    """One subplot per model. x = category.

    Cat bars: Δ % of directional rows that are (eff-good-side ∧ in cat).
        eff-good-side flips `on_good_side` if the pooled signed bias is
        negative for that (condition, model) — magnitude-only, matching the
        |·| convention for `equal_*` conditions.

    All bars are scaled ×2 (so cot bars are 2·Δpp, bias bar is Δ|bias|×100).
    The relative heights are unchanged from the unscaled version; only the
    axis units shift.

    Reading: positive bias bar = the intervention shifted answers more
    toward the model's preferred side; positive cat bar = the intervention
    increased the share of (preferred-side AND cat-x) rows.
    """
    b_pmd = per_model_dfs_by_condition[baseline]
    i_pmd = per_model_dfs_by_condition[intervention]
    b_pks = prompt_keys_by_condition[baseline]
    i_pks = prompt_keys_by_condition[intervention]
    model_keys = list(b_pmd.keys())

    n_models = len(model_keys)
    cat_keys = [c[0] for c in categories]
    cat_labels = [c[1] for c in categories]

    fig, axes = plt.subplots(1, n_models,
                             figsize=(3.4 * n_models + 1.0, 4.8),
                             sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, mk in zip(flat, model_keys):
        diffs = []
        for cat in cat_keys:
            b_val = _good_side_aggregate_stat(b_pmd[mk], cat, b_pks)
            i_val = _good_side_aggregate_stat(i_pmd[mk], cat, i_pks)
            if pd.isna(b_val) or pd.isna(i_val):
                d = float("nan")
            elif cat == "bias":
                d = (abs(i_val) - abs(b_val)) * 100
            else:
                d = (i_val - b_val) * 2
            diffs.append(d)
        xs = list(range(len(cat_keys)))
        colors = [DIFF_BAR_COLORS.get(c, "#7f7f7f") for c in cat_keys]
        ax.bar(xs, diffs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        for x, d in zip(xs, diffs):
            if pd.isna(d):
                continue
            offset = 1.0
            y_text = d + offset if d >= 0 else d - offset
            va = "bottom" if d >= 0 else "top"
            ax.text(x, y_text, f"{d:+.1f}", ha="center", va=va, fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(display_names[mk], fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
    flat[0].set_ylabel(
        "2·Δ pp of total rows in (eff-good ∧ cat); bias bar = Δ|bias|×100"
    )
    fig.suptitle(
        f"|{intervention}| − |{baseline}|: good-side-conditional composition",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def _per_prompt_good_side_stat(df, cat):
    """Per-prompt mirror of `_good_side_aggregate_stat`.

    Returns dict: prompt_key_stem -> stat. eff-good is computed *per prompt*
    (flips on_good_side if that prompt's pooled bias is negative). For 'bias'
    returns signed bias on [-1, 1]; for cot returns 100 * P(eff_good ∧ cat).
    """
    direc = df[df["direction"].isin(["below_good", "above_good"])].copy()
    direc["pk_stem"] = direc["prompt_key"].apply(_strip_variant_suffix)
    out = {}
    for stem, sub in direc.groupby("pk_stem"):
        if cat == "bias":
            out[stem] = 2 * sub["on_good_side"].mean() - 1
            continue
        eff_good = _eff_good_side_pooled(sub)
        if cat == "NO_STMT":
            cat_mask = sub["cot_category"].isin(["NO_MENTION", "NO_STATEMENT"])
        else:
            cat_mask = sub["cot_category"] == cat
        out[stem] = 100.0 * (eff_good & cat_mask).mean()
    return out


def plot_diff_by_prompt(per_model_dfs_by_condition, prompt_keys_by_condition,
                        baseline, intervention, categories=DIFF_CATEGORIES):
    """Per-prompt mirror of plot_diff_aggregate_good_side.

    Grid: rows = prompts, cols = models. One panel per (prompt, model).
    x = category. Cat bars: Δ % of directional rows (eff-good-side ∧ in cat),
    eff-good determined per (prompt, model). All bars scaled ×2 (cot bars are
    2·Δpp, bias bar is Δ|bias|×100) to match the aggregate plot.
    """
    b_pmd = per_model_dfs_by_condition[baseline]
    i_pmd = per_model_dfs_by_condition[intervention]
    b_pks = prompt_keys_by_condition[baseline]
    model_keys = list(b_pmd.keys())

    stems = sorted({_strip_variant_suffix(pk) for pk in b_pks})
    cat_keys = [c[0] for c in categories]
    cat_labels = [c[1] for c in categories]
    n_stems = len(stems)
    n_models = len(model_keys)
    xs = np.arange(len(cat_keys))

    b_stats = {(mk, cat): _per_prompt_good_side_stat(b_pmd[mk], cat)
               for mk in model_keys for cat in cat_keys}
    i_stats = {(mk, cat): _per_prompt_good_side_stat(i_pmd[mk], cat)
               for mk in model_keys for cat in cat_keys}

    panel_w = max(3.5, 0.7 * len(cat_keys) + 1.0)
    fig, axes = plt.subplots(n_stems, n_models,
                             figsize=(panel_w * n_models + 1.0,
                                      3.0 * n_stems + 1.2),
                             sharey=True, squeeze=False)

    for r, stem in enumerate(stems):
        for c, mk in enumerate(model_keys):
            ax = axes[r, c]
            diffs = []
            for cat in cat_keys:
                b_val = b_stats[(mk, cat)].get(stem, float("nan"))
                i_val = i_stats[(mk, cat)].get(stem, float("nan"))
                if pd.isna(b_val) or pd.isna(i_val):
                    d = float("nan")
                elif cat == "bias":
                    d = (abs(i_val) - abs(b_val)) * 100
                else:
                    d = (i_val - b_val) * 2
                diffs.append(d)
            heights = [0.0 if pd.isna(d) else d for d in diffs]
            colors = [DIFF_BAR_COLORS.get(c, "#7f7f7f") for c in cat_keys]
            ax.bar(xs, heights, color=colors,
                   edgecolor="white", linewidth=0.4)
            for x, d, h in zip(xs, diffs, heights):
                if pd.isna(d):
                    continue
                y_text = h + 1.0 if h >= 0 else h - 1.0
                va = "bottom" if h >= 0 else "top"
                ax.text(x, y_text, f"{d:+.1f}",
                        ha="center", va=va, fontsize=7)
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(display_names[mk], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{stem}\n(2·Δpp; bias ×100)", fontsize=9)

    fig.suptitle(
        f"|{intervention}| − |{baseline}|: per-prompt good-side composition",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# %%
plot_bias_bar_by_condition(results_df_by_condition)
plot_bias_bar_by_condition_per_prompt(per_model_dfs_by_condition,
                                      prompt_keys_by_condition)
plot_cot_distribution_by_condition(per_model_dfs_by_condition)
plot_bias_decomposition_by_condition(per_model_dfs_by_condition,
                                     prompt_keys_by_condition)
plot_cot_distribution_by_direction(per_model_dfs_by_condition)
plot_diff_aggregate(per_model_dfs_by_condition, prompt_keys_by_condition,
                    DIFF_BASELINE, DIFF_INTERVENTION)
plot_diff_aggregate_by_prompt(per_model_dfs_by_condition, prompt_keys_by_condition,
                              DIFF_BASELINE, DIFF_INTERVENTION)
plot_diff_aggregate_good_side(per_model_dfs_by_condition, prompt_keys_by_condition,
                              DIFF_BASELINE, DIFF_INTERVENTION)
plot_diff_by_prompt(per_model_dfs_by_condition, prompt_keys_by_condition,
                    DIFF_BASELINE, DIFF_INTERVENTION)
# %%
