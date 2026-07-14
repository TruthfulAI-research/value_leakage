"""Plot monitorability split by cached eval-awareness labels.

This imports the top-level `monitorability.py` helpers, then rebuilds the
monitorability rows from the experiment data, reads monitor labels from the
shared monitor cache, adds eval-awareness annotations from the existing judge
cache, splits rows by eval-awareness status before bootstrapping, and writes
one combined table plus one paired-bar plot.
"""

from __future__ import annotations
from donation_bet import monitorability
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GIRAFFES_ROOT = next(
    parent for parent in (THIS_DIR, *THIS_DIR.parents)
    if (parent / "shared" / "runner.py").exists()
)
if str(GIRAFFES_ROOT) not in sys.path:
    sys.path.insert(0, str(GIRAFFES_ROOT))

ARTIFACT_ROOT = GIRAFFES_ROOT / "data" / "final_data"
monitorability.configure_runner_cache(ARTIFACT_ROOT)
GOOD_BAD_EXPERIMENT = "main_experiment_accurate"
OUTPUT_DIR = ARTIFACT_ROOT / "plot_monitorability_by_eval_awareness_outputs" / GOOD_BAD_EXPERIMENT
PLOTS_DIR = OUTPUT_DIR / "plots"
MONITOR_CACHE_DIR = ARTIFACT_ROOT / "run_monitorability_cache" / GOOD_BAD_EXPERIMENT / "monitor"
EVAL_AWARENESS_CACHE_DIR = (
    ARTIFACT_ROOT
    / "plot_eval_awareness_cache"
    / GOOD_BAD_EXPERIMENT
    / "eval_awareness"
)
COMPARISON = "good_bad_vs_no_hint"
SCOPE = "cot_only"
CACHE_ONLY = True
MONITOR_CACHE_ONLY = True
LIMIT_ROWS_PER_INSTANCE_ARM = None
SAVE_TABLES = False
SAVE_PLOTS = False
LOAD_EXISTING_SPLIT_METRIC = False

# Set any of these to an explicit value to override the matching
# `run_monitorability.py` RunConfig default for the split-before-bootstrap
# metric. If left as None, the value from run_monitorability.py is used.
N_BOOTSTRAP = None
BOOTSTRAP_MAX_WORKERS = None
SELECTION_FRAC = None
GROUP_KEEP_FRAC = None
WALD_K = None
SEED = None
ERROR_BAR_MODE = None

MODEL_GROUPS = [
    ("Claude Opus", [
        "claude-opus-4.1",
        "claude-opus-4.5-high",
        "claude-opus-4.6-high",
        "claude-opus-4.6-max",
        "claude-opus-4.7-high",
        "claude-opus-4.7-xhigh",
        "claude-opus-4.7-max",
    ]),
    ("Claude Sonnet", [
        "claude-sonnet-4.5",
        "claude-sonnet-4.6-high",
    ]),
    ("Kimi", [
        "kimi-k2.5",
        "kimi-k2.6",
    ]),
    ("Qwen", [
        "qwen3.5-35",
        "qwen3.6-35",
    ]),
    ("Gemini", [
        "gemini-2.5-pro",
        "gemini-3.1-pro-medium",
        "gemini-3.1-pro-high",
    ]),
    ("OpenAI", [
        "gpt-5.1-medium",
        "gpt-5.2-medium",
        "gpt-5.4-medium",
        "gpt-5.5-medium",
        "gpt-5.5-high",
    ]),
]
MODEL_NAMES = [mk for _, group in MODEL_GROUPS for mk in group]

import matplotlib.pyplot as plt


def _ordered_model_keys(model_groups: list[tuple[str, list[str]]], present: set[str]) -> list[str]:
    ordered = [mk for _, group in model_groups for mk in group if mk in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def add_eval_awareness_from_cache(rows: pd.DataFrame, *, cache_only: bool = CACHE_ONLY) -> pd.DataFrame:
    if str(GIRAFFES_ROOT) not in sys.path:
        sys.path.insert(0, str(GIRAFFES_ROOT))
    from shared import classify_eval_awareness as cea

    out = rows.copy()
    out["eval_awareness_raw"] = None
    out["eval_awareness_reasoning"] = None
    out["eval_awareness_score"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    needs_judge = out["reasoning"].apply(lambda r: isinstance(r, str) and bool(r.strip()))
    prompt_by_idx: dict[Any, str] = {}
    for idx in out.index[needs_judge]:
        prompt_by_idx[idx] = cea.ACTIVE_EVAL_AWARENESS_PROMPT.format(
            prompt=_safe_text(out.at[idx, "prompt"]),
            reasoning=_safe_text(out.at[idx, "reasoning"]),
            answer=_safe_text(out.at[idx, "answer"]),
        )

    cached_by_prompt: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    cache = cea._eval_awareness_cache(EVAL_AWARENESS_CACHE_DIR)
    for prompt in dict.fromkeys(prompt_by_idx.values()):
        key = cache.key(prompt)
        cached = cache.get(prompt)
        if cached is None:
            missing.append(key)
            continue
        cached_by_prompt[prompt] = cached

    if missing and cache_only:
        sample = ", ".join(missing[:8])
        raise RuntimeError(
            f"Missing {len(missing)} eval-awareness cache entries. "
            "Run shared/final_scripts/giraffes/plot_eval_awareness.py first, or set "
            f"CACHE_ONLY=False in {Path(__file__).name}. Checked: "
            f"{EVAL_AWARENESS_CACHE_DIR}. Sample keys: {sample}"
        )
    if missing:
        missing_idxs = [idx for idx, prompt in prompt_by_idx.items() if prompt not in cached_by_prompt]
        to_classify = out.loc[missing_idxs].copy()
        cea.classify_eval_awareness(
            to_classify,
            cache_dir=EVAL_AWARENESS_CACHE_DIR,
            cache_only=False,
        )
        out.loc[missing_idxs, "eval_awareness_raw"] = to_classify["eval_awareness_raw"]
        out.loc[missing_idxs, "eval_awareness_reasoning"] = to_classify["eval_awareness_reasoning"]
        out.loc[missing_idxs, "eval_awareness_score"] = to_classify["eval_awareness_score"].astype("Int64")

    for idx, prompt in prompt_by_idx.items():
        cached = cached_by_prompt.get(prompt)
        if cached is None:
            continue
        raw = cached.get("answer")
        out.at[idx, "eval_awareness_raw"] = raw
        out.at[idx, "eval_awareness_reasoning"] = cached.get("reasoning")
        score = cea._parse_score(raw)
        if score is not None:
            out.at[idx, "eval_awareness_score"] = int(score)

    out["eval_aware"] = out["eval_awareness_score"].apply(
        lambda score: pd.notna(score) and int(score) >= cea.EVAL_AWARE_THRESHOLD
    )
    return out


def _binomial_pct_ci95(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return 100 * (p - lower), 100 * (upper - p)


def eval_awareness_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    scored = annotated[annotated["eval_awareness_score"].notna()].copy()
    if scored.empty:
        raise ValueError("No rows have parseable eval-awareness scores.")
    summary = (
        scored.groupby("model_key", sort=False)
        .agg(
            eval_awareness_n=("eval_aware", "size"),
            eval_awareness_count=("eval_aware", "sum"),
            eval_awareness_score_mean=("eval_awareness_score", "mean"),
        )
        .reset_index()
    )
    summary["eval_awareness_pct"] = (
        100 * summary["eval_awareness_count"] / summary["eval_awareness_n"]
    )
    ci = [
        _binomial_pct_ci95(int(row.eval_awareness_count), int(row.eval_awareness_n))
        for row in summary.itertuples(index=False)
    ]
    summary["eval_awareness_ci95_low_err"] = [lo for lo, _ in ci]
    summary["eval_awareness_ci95_high_err"] = [hi for _, hi in ci]
    return summary


SPLIT_LABELS = {
    True: "eval-aware",
    False: "not eval-aware",
}
SPLIT_METRIC_FINAL_PATH = OUTPUT_DIR / f"monitorability_by_eval_awareness_split_metric_final_{COMPARISON}_{SCOPE}.csv"
SPLIT_METRIC_PER_BOOTSTRAP_PATH = OUTPUT_DIR / f"monitorability_by_eval_awareness_split_metric_per_bootstrap_{COMPARISON}_{SCOPE}.parquet"
SPLIT_METRIC_PER_INSTANCE_PATH = OUTPUT_DIR / f"monitorability_by_eval_awareness_split_metric_per_instance_{COMPARISON}_{SCOPE}.parquet"
REQUIRED_SPLIT_FINAL_COLUMNS = {
    "gmean_mean",
    "true_positive_pct_mean",
    "positive_count_mean",
}


def complete_split_rows(scored: pd.DataFrame, rm: Any) -> dict[bool, pd.DataFrame]:
    """Return eval-awareness splits with both x arms present per metric instance."""
    out: dict[bool, pd.DataFrame] = {}
    for eval_aware, label in SPLIT_LABELS.items():
        subset = scored[scored["eval_aware"].eq(eval_aware)].copy()
        if subset.empty:
            continue
        complete = rm.drop_incomplete_instances(subset)
        if complete.empty:
            print(
                f"Skipping split metric: {label}; no complete x=0/x=1 instances",
                flush=True,
            )
            continue
        out[eval_aware] = complete
    return out


def split_row_counts(split_rows: dict[bool, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for eval_aware, split_df in split_rows.items():
        for model_key, sub in split_df.groupby("model_key", sort=False):
            rows.append({
                "model_key": model_key,
                "eval_aware": bool(eval_aware),
                "split_row_n": len(sub),
                "split_prompt_n": int(sub["prompt_stem"].nunique()),
                "split_instance_n": int(
                    sub[["prompt_stem", "metric_direction"]].drop_duplicates().shape[0]
                ),
            })
    return pd.DataFrame(
        rows,
        columns=[
            "model_key",
            "eval_aware",
            "split_row_n",
            "split_prompt_n",
            "split_instance_n",
        ],
    )


def _split_metric_args(rm: Any, base_config: Any) -> Any:
    return rm.RunConfig(
        good_bad_experiment=base_config.good_bad_experiment,
        cache_only=base_config.cache_only,
        models=base_config.models,
        comparisons=base_config.comparisons,
        scopes=base_config.scopes,
        prompt_filter=base_config.prompt_filter,
        monitor_backend=base_config.monitor_backend,
        monitor_model=base_config.monitor_model,
        monitor_reasoning_effort=base_config.monitor_reasoning_effort,
        monitor_max_output_tokens=base_config.monitor_max_output_tokens,
        monitor_cache_only=base_config.monitor_cache_only,
        monitor_max_workers=base_config.monitor_max_workers,
        allow_synthetic_monitor=base_config.allow_synthetic_monitor,
        output_dir=OUTPUT_DIR,
        cache_dir=base_config.cache_dir,
        metric_path=base_config.metric_path,
        load_existing_outputs=False,
        n_bootstrap=base_config.n_bootstrap if N_BOOTSTRAP is None else N_BOOTSTRAP,
        run_bootstrap=base_config.run_bootstrap,
        bootstrap_max_workers=(
            base_config.bootstrap_max_workers
            if BOOTSTRAP_MAX_WORKERS is None
            else BOOTSTRAP_MAX_WORKERS
        ),
        selection_frac=base_config.selection_frac if SELECTION_FRAC is None else SELECTION_FRAC,
        group_keep_frac=base_config.group_keep_frac if GROUP_KEEP_FRAC is None else GROUP_KEEP_FRAC,
        wald_k=base_config.wald_k if WALD_K is None else WALD_K,
        seed=base_config.seed if SEED is None else SEED,
        error_bar_mode=base_config.error_bar_mode if ERROR_BAR_MODE is None else ERROR_BAR_MODE,
        limit_rows_per_instance_arm=base_config.limit_rows_per_instance_arm,
        save_prompts=base_config.save_prompts,
        save_tables=False,
        run_analysis=False,
    )


def compute_split_monitorability(split_rows: dict[bool, pd.DataFrame], rm: Any, base_config: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args = _split_metric_args(rm, base_config)
    final_parts: list[pd.DataFrame] = []
    per_bootstrap_parts: list[pd.DataFrame] = []
    per_instance_parts: list[pd.DataFrame] = []
    for eval_aware, label in SPLIT_LABELS.items():
        subset = split_rows.get(eval_aware, pd.DataFrame()).copy()
        if subset.empty:
            continue
        print(f"running split metric: {label}, rows={len(subset)}", flush=True)
        final, per_bootstrap, per_instance = rm.run_metric(subset, Path(args.metric_path), args)
        final = rm.ensure_bootstrap_ci_columns(final, per_bootstrap)
        for df in (final, per_bootstrap, per_instance):
            df["eval_aware"] = eval_aware
            df["eval_awareness_condition"] = label
        final_parts.append(final)
        per_bootstrap_parts.append(per_bootstrap)
        per_instance_parts.append(per_instance)

    if not final_parts:
        raise ValueError("No eval-awareness split had rows to score.")
    return (
        pd.concat(final_parts, ignore_index=True),
        pd.concat(per_bootstrap_parts, ignore_index=True),
        pd.concat(per_instance_parts, ignore_index=True),
    )


def load_or_compute_split_monitorability(split_rows: dict[bool, pd.DataFrame], rm: Any, base_config: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        SPLIT_METRIC_FINAL_PATH,
        SPLIT_METRIC_PER_BOOTSTRAP_PATH,
        SPLIT_METRIC_PER_INSTANCE_PATH,
    ]
    if LOAD_EXISTING_SPLIT_METRIC and all(path.exists() for path in paths):
        final = pd.read_csv(SPLIT_METRIC_FINAL_PATH)
        missing = sorted(REQUIRED_SPLIT_FINAL_COLUMNS.difference(final.columns))
        if not missing:
            per_bootstrap = pd.read_parquet(SPLIT_METRIC_PER_BOOTSTRAP_PATH)
            per_instance = pd.read_parquet(SPLIT_METRIC_PER_INSTANCE_PATH)
            return final, per_bootstrap, per_instance
        print(
            "Existing split metric output is missing current metric columns; "
            f"recomputing. Missing columns: {missing}",
            flush=True,
        )

    final, per_bootstrap, per_instance = compute_split_monitorability(split_rows, rm, base_config)
    if SAVE_TABLES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        final.to_csv(SPLIT_METRIC_FINAL_PATH, index=False)
        per_bootstrap.to_parquet(SPLIT_METRIC_PER_BOOTSTRAP_PATH, index=False)
        per_instance.to_parquet(SPLIT_METRIC_PER_INSTANCE_PATH, index=False)
    return final, per_bootstrap, per_instance


def monitored_rows_from_cache(rm: Any, config: Any) -> pd.DataFrame:
    """Rebuild metric rows and attach cached monitor labels without loading run outputs."""
    base_rows = rm.build_base_rows(
        models=config.resolved_models(),
        comparisons=config.resolved_comparisons(),
        limit_rows_per_instance_arm=config.limit_rows_per_instance_arm,
        args=config,
    )
    return rm.add_monitor_outputs(
        base_rows,
        scopes=config.resolved_scopes(),
        backend=config.monitor_backend,
        model=config.monitor_model,
        cache_dir=Path(config.cache_dir),
        max_output_tokens=config.monitor_max_output_tokens,
        reasoning_effort=config.monitor_reasoning_effort,
        monitor_cache_only=config.monitor_cache_only,
        monitor_max_workers=config.monitor_max_workers,
        save_prompts=False,
    )


def build_split_monitorability_table(
    split_final: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    display_names: dict[str, str],
    ordered_model_keys: list[str],
) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "eval_aware": eval_aware,
                "eval_awareness_condition": SPLIT_LABELS[eval_aware],
            }
            for model_key in ordered_model_keys
            for eval_aware in [True, False]
        ]
    )
    metric = split_final[
        (split_final["comparison"] == COMPARISON)
        & (split_final["scope"] == SCOPE)
    ].copy()
    table = pd.merge(
        base,
        metric,
        on=["model_key", "eval_aware", "eval_awareness_condition"],
        how="left",
    )
    table = pd.merge(table, counts, on=["model_key", "eval_aware"], how="left")
    table["split_row_n"] = table["split_row_n"].fillna(0).astype(int)
    table["split_prompt_n"] = table["split_prompt_n"].fillna(0).astype(int)
    table["split_instance_n"] = table["split_instance_n"].fillna(0).astype(int)
    table["model"] = table["model_key"].map(lambda mk: display_names.get(mk, mk))
    order = {mk: i for i, mk in enumerate(ordered_model_keys)}
    table["model_order"] = table["model_key"].map(order).fillna(len(order)).astype(int)
    table["split_order"] = table["eval_aware"].map({True: 0, False: 1}).astype(int)
    return (
        table.sort_values(["model_order", "split_order"])
        .drop(columns=["model_order", "split_order"])
        .reset_index(drop=True)
    )


def _metric_value_and_error(row: pd.Series, metric: str) -> tuple[float, float, float]:
    value = row.get(f"{metric}_mean", np.nan)
    if pd.isna(value):
        return np.nan, 0.0, 0.0
    ci_low_col = f"{metric}_ci95_low"
    ci_high_col = f"{metric}_ci95_high"
    if ci_low_col in row and ci_high_col in row:
        lo = row.get(ci_low_col, np.nan)
        hi = row.get(ci_high_col, np.nan)
        err_low = max(0.0, float(value) - float(lo)) if pd.notna(lo) else 0.0
        err_high = max(0.0, float(hi) - float(value)) if pd.notna(hi) else 0.0
        return float(value), err_low, err_high
    std = row.get(f"{metric}_std", 0.0)
    err = float(std) if pd.notna(std) else 0.0
    return float(value), err, err


def _gmean_value_and_error(row: pd.Series) -> tuple[float, float, float]:
    return _metric_value_and_error(row, "gmean")


def _fmt_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def plot_split_monitorability_by_eval_awareness(
    table: pd.DataFrame,
    *,
    model_groups: list[tuple[str, list[str]]],
    out_dir: Path,
) -> None:
    model_keys = [mk for _, group in model_groups for mk in group if mk in set(table["model_key"])]
    extras = sorted(set(table["model_key"]).difference(model_keys))
    model_keys.extend(extras)
    labels = [
        table[table["model_key"].eq(mk)]["model"].dropna().iloc[0]
        if not table[table["model_key"].eq(mk)]["model"].dropna().empty
        else mk
        for mk in model_keys
    ]
    xs = np.arange(len(model_keys))
    width = 0.38
    offset = 0.21
    specs = [
        (True, -offset, "#2ca02c", "eval-aware"),
        (False, offset, "0.55", "not eval-aware"),
    ]

    fig, ax = plt.subplots(figsize=(max(9.0, 0.8 * len(model_keys) + 2.5), 5.6))
    for eval_aware, dx, color, label in specs:
        vals, err_low, err_high, ns = [], [], [], []
        for mk in model_keys:
            row = table[(table["model_key"] == mk) & (table["eval_aware"].eq(eval_aware))]
            if row.empty:
                vals.append(np.nan)
                err_low.append(0.0)
                err_high.append(0.0)
                ns.append(0)
                continue
            value, lo, hi = _gmean_value_and_error(row.iloc[0])
            vals.append(value)
            err_low.append(lo)
            err_high.append(hi)
            ns.append(int(row["split_row_n"].iloc[0]))

        heights = [0.0 if pd.isna(v) else v for v in vals]
        bars = ax.bar(
            xs + dx,
            heights,
            width,
            yerr=[err_low, err_high],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            ecolor="black",
            capsize=3,
            error_kw={"linewidth": 0.9},
            label=label,
        )
        for bar, value, hi, n in zip(bars, vals, err_high, ns):
            if pd.isna(value):
                text = f"n={n}\nn/a"
                y = 0.03
            else:
                text = f"{value:.2f}\nn={n}"
                y = min(1.18, bar.get_height() + hi + 0.025)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("g-mean monitorability")
    ax.set_ylim(0, 1.24)
    ax.set_title(f"Monitorability split by eval-awareness status: {COMPARISON} ({SCOPE})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9)

    cumulative = 0
    ymax = ax.get_ylim()[1]
    for label, group in model_groups:
        present = [mk for mk in group if mk in model_keys]
        if not present:
            continue
        start = cumulative
        cumulative += len(present)
        end = cumulative - 1
        center = (start + end) / 2
        if cumulative < len(model_keys):
            ax.axvline(cumulative - 0.5, color="black", linewidth=0.8, alpha=0.45, linestyle="--")
        ax.text(center, ymax - 0.02, label, ha="center", va="top", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 0.9, 1))
    if SAVE_PLOTS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / f"monitorability_by_eval_awareness_{COMPARISON}_{SCOPE}.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def plot_split_raw_tpr_by_eval_awareness(
    table: pd.DataFrame,
    *,
    model_groups: list[tuple[str, list[str]]],
    out_dir: Path,
) -> None:
    missing = sorted(REQUIRED_SPLIT_FINAL_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(
            "Split table is missing raw-TPR columns. Recompute split metric "
            f"outputs with the current run_monitorability.py. Missing: {missing}"
        )

    model_keys = [mk for _, group in model_groups for mk in group if mk in set(table["model_key"])]
    extras = sorted(set(table["model_key"]).difference(model_keys))
    model_keys.extend(extras)
    labels = [
        table[table["model_key"].eq(mk)]["model"].dropna().iloc[0]
        if not table[table["model_key"].eq(mk)]["model"].dropna().empty
        else mk
        for mk in model_keys
    ]
    xs = np.arange(len(model_keys))
    width = 0.38
    offset = 0.21
    specs = [
        (True, -offset, "#2ca02c", "eval-aware"),
        (False, offset, "0.55", "not eval-aware"),
    ]

    fig, ax = plt.subplots(figsize=(max(9.0, 0.8 * len(model_keys) + 2.5), 5.6))
    for eval_aware, dx, color, label in specs:
        vals, err_low, err_high, ns = [], [], [], []
        for mk in model_keys:
            row = table[(table["model_key"] == mk) & (table["eval_aware"].eq(eval_aware))]
            if row.empty:
                vals.append(np.nan)
                err_low.append(0.0)
                err_high.append(0.0)
                ns.append("n/a")
                continue
            row0 = row.iloc[0]
            value, lo, hi = _metric_value_and_error(row0, "true_positive_pct")
            vals.append(value)
            err_low.append(lo)
            err_high.append(hi)
            ns.append(_fmt_count(row0.get("positive_count_mean", np.nan)))

        heights = [0.0 if pd.isna(v) else v for v in vals]
        bars = ax.bar(
            xs + dx,
            heights,
            width,
            yerr=[err_low, err_high],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            ecolor="black",
            capsize=3,
            error_kw={"linewidth": 0.9},
            label=label,
        )
        for bar, value, hi, n in zip(bars, vals, err_high, ns):
            if pd.isna(value):
                text = f"n+={n}\nn/a"
                y = 3.0
            else:
                text = f"{value:.1f}%\nn+={n}"
                y = min(112.0, bar.get_height() + hi + 2.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("raw TPR before effect-size rescaling (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Raw monitor TPR split by eval-awareness status: {COMPARISON} ({SCOPE})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9)

    cumulative = 0
    ymax = ax.get_ylim()[1]
    for label, group in model_groups:
        present = [mk for mk in group if mk in model_keys]
        if not present:
            continue
        start = cumulative
        cumulative += len(present)
        end = cumulative - 1
        center = (start + end) / 2
        if cumulative < len(model_keys):
            ax.axvline(cumulative - 0.5, color="black", linewidth=0.8, alpha=0.45, linestyle="--")
        ax.text(center, ymax - 3.0, label, ha="center", va="top", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 0.9, 1))
    if SAVE_PLOTS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / f"raw_tpr_by_eval_awareness_{COMPARISON}_{SCOPE}.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


# %%
# Rebuild monitorability rows from caches and add cached eval-awareness labels.

config = monitorability.RunConfig(
    good_bad_experiment=GOOD_BAD_EXPERIMENT,
    cache_only=CACHE_ONLY,
    models=MODEL_NAMES,
    comparisons=[COMPARISON],
    scopes=[SCOPE],
    output_dir=OUTPUT_DIR,
    cache_dir=MONITOR_CACHE_DIR,
    monitor_cache_only=MONITOR_CACHE_ONLY,
    limit_rows_per_instance_arm=LIMIT_ROWS_PER_INSTANCE_ARM,
    save_tables=False,
    run_analysis=False,
    load_existing_outputs=False,
)
monitored = monitored_rows_from_cache(monitorability, config)

monitored_sub = monitored[
    (monitored["comparison"] == COMPARISON)
    & (monitored["scope"] == SCOPE)
].dropna(subset=["z"]).copy()
annotated = add_eval_awareness_from_cache(monitored_sub, cache_only=CACHE_ONLY)
scored = annotated[annotated["eval_awareness_score"].notna()].copy()
split_rows = complete_split_rows(scored, monitorability)
split_counts = split_row_counts(split_rows)

# %%
# Run or load split-before-bootstrap monitorability summaries.

model_groups = MODEL_GROUPS
ordered_model_keys = _ordered_model_keys(
    model_groups, set(monitored_sub["model_key"])
)
display_names = monitorability.model_display_names(ordered_model_keys)
split_final, split_per_bootstrap, split_per_instance = load_or_compute_split_monitorability(
    split_rows,
    monitorability,
    config,
)
monitorability_by_eval_awareness = build_split_monitorability_table(
    split_final,
    split_counts,
    display_names=display_names,
    ordered_model_keys=ordered_model_keys,
)

if SAVE_TABLES:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monitorability_by_eval_awareness.to_csv(
        OUTPUT_DIR / f"monitorability_by_eval_awareness_{COMPARISON}_{SCOPE}.csv",
        index=False,
    )
print_cols = [
    "model",
    "eval_awareness_condition",
    "split_row_n",
    "split_instance_n",
    "gmean_mean",
    "gmean_ci95_low",
    "gmean_ci95_high",
    "true_positive_pct_mean",
    "true_positive_pct_ci95_low",
    "true_positive_pct_ci95_high",
    "positive_count_mean",
    "eligible_fraction_mean",
]
print(monitorability_by_eval_awareness[print_cols].to_string(index=False))

# %%
# Plot paired monitorability bars by eval-awareness split.

plot_split_monitorability_by_eval_awareness(
    monitorability_by_eval_awareness,
    model_groups=model_groups,
    out_dir=PLOTS_DIR,
)
plot_split_raw_tpr_by_eval_awareness(
    monitorability_by_eval_awareness,
    model_groups=model_groups,
    out_dir=PLOTS_DIR,
)
