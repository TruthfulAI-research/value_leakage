"""Paper-styled three-panel score-vs-selection scatter, one figure per model.

For every model under RESULTS_DIR (with a stated-liking score summary) and the three tool settings
(no tools, unix time, coin flip), each panel scatters every activity's mean stated
liking score (x) against its selection rate (y) in the supposedly-random pick, with
a least-squares fit line and the Pearson r + 95% CI printed inside. A strong
positive r means the "random" pick is really driven by the model's preferences.

This reproduces harry's `score_vs_selection_all_tool_settings_nonrefusal_paper.pdf`
(pipeline.write_paper_score_selection_figure on the `_nonrefusal` cut): every
pairwise decision is kept EXCEPT randomness-judge refusals, so overt bias
(random_in_reasoning == false) is included; selection_rate still drops choice-judge
refusals from the denominator.

Self-contained (like plot_biases.py / plot_model_comparison.py): reads harry's
cached results directly and re-implements the selection-rate / correlation helpers.
Writes one PDF per model, straight into the paper's Overleaf clone, named by model:
  overleaf/figures/preferences/<model_slug>.pdf   (e.g. gpt_55_xhigh.pdf)

Usage (from the repo root):
  uv run python choosing_activities/plot_score_vs_selection.py
"""
import math
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import plot_style  # noqa: F401  applies shared figure sizing on import
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
HARRY_ROOT = REPO_ROOT / "choosing_activities"
RESULTS_DIR = REPO_ROOT / "data" / "choosing_activities" / "results"

# One scatter PDF per model, written straight into the gitignored Overleaf clone
# as overleaf/figures/preferences/<model_slug>.pdf (e.g. gpt_55_xhigh.pdf). Every
# model under RESULTS_DIR that has a stated-liking score summary is plotted.
OUT_DIR = REPO_ROOT / "overleaf" / "figures" / "preferences"

# Tool settings, left -> right. (results-dir key, panel title)
SETTINGS = [
    ("no_tools", "No tools"),
    ("unix_time", "Unix time tool"),
    ("coin_flip", "Coin flip tool"),
]

# Randomness verdicts kept for the "nonrefusal" cut: everything the randomness
# judge did NOT call a refusal (so overt bias, `false`, is kept).
NON_REFUSAL_LABELS = ("true", "hedged", "false")

POINT_COLOR, FIT_COLOR = "#2a6f97", "#b0b0b0"


# --- selection-rate / correlation helpers (self-contained, mirroring
# harry/activity_preferences/pipeline.py and shared/.../plot_biases.py) ---------
def _randomness_label(value) -> str:
    """Normalise harry's random_in_reasoning verdict to a lowercase string.

    pd.read_json turns the JSON booleans into Python bools, so handle those too.
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        n = value.strip().lower()
        if n in {"true", "false", "refusal", "hedged"}:
            return n
        if n == "unclear":  # legacy label -> hedged
            return "hedged"
    return "hedged"


def _cut_rows(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Rows whose randomness verdict is in `labels` (None = keep all)."""
    if labels is None:
        return df
    mask = df["random_in_reasoning"].apply(lambda v: _randomness_label(v) in labels)
    return df[mask]


def _summarize_selection_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per activity: selection_rate = picks / decisive appearances.

    Refusal pairs are dropped from the denominator (decisive = judgment in 1/2),
    matching harry's pipeline.summarize_selection_rates.
    """
    appearances = pd.concat(
        [
            df[["activity_1", "judgment"]].rename(columns={"activity_1": "activity"}),
            df[["activity_2", "judgment"]].rename(columns={"activity_2": "activity"}),
        ],
        ignore_index=True,
    )
    appearances["decisive"] = appearances["judgment"].isin([1, 2])
    appearance_summary = (
        appearances.groupby("activity", sort=False)
        .agg(
            n_appearances=("activity", "size"),
            n_decisive_appearances=("decisive", "sum"),
        )
        .reset_index()
    )
    pick_summary = (
        df[df["judgment"].isin([1, 2])]
        .groupby("picked_name", sort=False)
        .size()
        .rename("n_picked")
        .reset_index()
        .rename(columns={"picked_name": "activity"})
    )
    summary = appearance_summary.merge(pick_summary, on="activity", how="left")
    summary["n_picked"] = summary["n_picked"].fillna(0).astype(int)
    summary["selection_rate"] = (
        summary["n_picked"] / summary["n_decisive_appearances"].replace({0: pd.NA})
    )
    return summary


def _merge_scores(selection_summary: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    merged = selection_summary.merge(
        scores[["activity", "mean_score"]], on="activity", how="left"
    )
    return merged.dropna(subset=["mean_score", "selection_rate"])


def _pearson_ci(r: float, n: int, confidence: float = 0.95):
    """Fisher z-transform CI for a Pearson r over `n` paired activities."""
    if n <= 3 or pd.isna(r):
        return float("nan"), float("nan")
    zcrit = NormalDist().inv_cdf(0.5 + confidence / 2)
    r_clamped = max(min(float(r), 0.999999999), -0.999999999)
    z = math.atanh(r_clamped)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - zcrit * se), math.tanh(z + zcrit * se)


def _load_setting_df(model: str, setting_key: str):
    """Pipeline rows for one model/setting, or None if that file is absent."""
    jsonl = RESULTS_DIR / model / "pipeline" / setting_key / "pipeline.jsonl"
    if not jsonl.exists():
        return None
    return pd.read_json(jsonl, lines=True)


def _load_scores(model: str):
    """Stated-liking score summary for one model, or None if absent."""
    score_path = (
        RESULTS_DIR / model / "activity_preferences"
        / "activity_liking_scores_summary.csv"
    )
    if not score_path.exists():
        return None
    return pd.read_csv(score_path)


def _slug(model: str) -> str:
    """Overleaf filename stem: gpt-5.5-xhigh -> gpt_55_xhigh."""
    return model.replace("-", "_").replace(".", "")


def _discover_models() -> list:
    """Every results-dir model that has a stated-liking score summary, sorted."""
    return sorted(
        p.name for p in RESULTS_DIR.iterdir()
        if p.is_dir()
        and (p / "activity_preferences" / "activity_liking_scores_summary.csv").exists()
    )


def _style_axis(ax) -> None:
    """Boxed white panel with #333 spines, matching the paper figure."""
    ax.grid(False)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", length=5, width=1.0,
                   colors="#333333")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#333333")
        sp.set_linewidth(0.9)


def _plot_model(model: str, scores: pd.DataFrame) -> None:
    n_panels = len(SETTINGS)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.7 * n_panels, 4.5), sharex=True, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    print(f"\n{model}:")
    for ax, (skey, slabel) in zip(axes, SETTINGS):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Activity preference score", labelpad=6)
        _style_axis(ax)
        ax.set_box_aspect(1)
        ax.set_title(slabel, fontweight="semibold", pad=8)

        df = _load_setting_df(model, skey)
        plot_df = (
            _merge_scores(
                _summarize_selection_rates(_cut_rows(df, NON_REFUSAL_LABELS)), scores)
            if df is not None else None
        )
        if plot_df is None or plot_df.empty:
            ax.text(0.5, 0.5, "no scored selections", transform=ax.transAxes,
                    ha="center", va="center", color="#999999",
                    fontsize=plot_style.ANNOT_FS)
            continue
        xs = plot_df["mean_score"].to_numpy(dtype=float)
        ys = (plot_df["selection_rate"] * 100).to_numpy(dtype=float)
        ax.scatter(xs, ys, s=26, alpha=0.6, color=POINT_COLOR,
                   edgecolor="white", linewidth=0.4)
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.array([xs.min(), xs.max()])
            ax.plot(xline, slope * xline + intercept,
                    color=FIT_COLOR, linewidth=1.6, zorder=3)

        n = len(plot_df)
        r = float(plot_df["mean_score"].corr(plot_df["selection_rate"])) if n >= 2 else float("nan")
        lo, hi = _pearson_ci(r, n)
        r_text = "n/a" if pd.isna(r) else f"{r:.2f}"
        ci_text = "" if pd.isna(lo) else f" [{lo:.2f}, {hi:.2f}]"
        ax.text(0.05, 0.95, f"$r$ = {r_text}{ci_text}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=plot_style.ANNOT_FS)
        print(f"  {slabel:16} r = {r_text}{ci_text}  (n_activities={n})")

    axes[0].set_ylabel("Selection rate (%)", labelpad=6)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.92, wspace=0.14)

    out = OUT_DIR / f"{_slug(model)}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    models = _discover_models()
    if not models:
        sys.exit(f"no models with score summaries under {RESULTS_DIR}")
    print(f"plotting {len(models)} model(s): {', '.join(models)}")
    for model in models:
        scores = _load_scores(model)
        if scores is None:
            print(f"skip {model}: missing stated-liking score summary")
            continue
        _plot_model(model, scores)


if __name__ == "__main__":
    main()
