"""Preference-leak bias vs reasoning effort / CoT length: GPT-5.5 & Claude Opus 4.8.

Equivalent to harry's `correlation_vs_effort_opus_3panel.pdf`, but comparing
GPT-5.5 against Claude Opus 4.8. One line per model, a panel per tool setting
(no-tools / unix-time / coin-flip), plotting Bias = r(stated mean_score,
selection_rate) with a Fisher-z 95% CI on each point.

The same code produces two sister figures, differing only in the x-axis:
  1. correlation_vs_effort_gpt55_opus48_3panel.pdf  -- x = reasoning-effort ladder
  2. correlation_vs_tokens_gpt55_opus48_3panel.pdf  -- x = mean CoT length (words)
     (mean word count of the `reasoning` trace over the same non-refusal rows)

Cut: the "non-refusal" cut in EVERY panel -- every pairwise decision is kept
EXCEPT randomness-judge refusals (so overt bias, random_in_reasoning == false, is
included). selection_rate still drops choice-judge refusals from the denominator.

r is over the ~100 activity means, so n is ~100 regardless of how many variations a
run used. Missing (effort, model) points are skipped -- GPT-5.5 has no `max` and
Opus 4.8 has no `instant`, so each line spans its own effort range.

Self-contained (like plot_biases.py / plot_model_comparison.py): reads harry's
cached results directly and re-implements the selection-rate / correlation helpers.
Writes PDF only, straight into the preferences section of the Overleaf clone.

Usage (from the repo root):
  uv run python choosing_activities/plot_correlation_vs_effort.py
"""
import math
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import plot_style  # noqa: F401  applies shared figure sizing on import
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "choosing_activities" / "results"
# Write straight into the preferences section of the Overleaf clone. Two figures:
# bias vs reasoning-effort ladder, and bias vs mean reasoning length (in words).
PREF_DIR = REPO_ROOT / "overleaf" / "figures" / "preferences"
OUT_EFFORT = PREF_DIR / "correlation_vs_effort_gpt55_opus48_3panel.pdf"
OUT_TOKENS = PREF_DIR / "correlation_vs_tokens_gpt55_opus48_3panel.pdf"

# Reasoning-effort ladder (union of both models), low -> high, equally spaced.
EFFORTS = ["instant", "low", "medium", "high", "xhigh", "max"]

# One line per model: (legend label, key template, colour, marker). Colours are
# the family palette from shared/final_scripts/giraffes/plot_biases.py (tab10
# pinned by family): Claude = blue, GPT = brown.
FAMILIES = [
    ("Claude Opus 4.8", "claude-opus-4.8-{}", "#1f77b4", "o"),
    ("GPT-5.5",         "gpt-5.5-{}",         "#8c564b", "s"),
]

# (setting key, panel title). Same non-refusal cut in every panel.
SETTINGS = [
    ("no_tools",  "No-tools"),
    ("unix_time", "Unix-time tool"),
    ("coin_flip", "Coin-flip tool"),
]

# Randomness verdicts kept: everything the randomness judge did NOT call a refusal.
NON_REFUSAL_LABELS = ("true", "hedged", "false")


# --- selection-rate / correlation helpers (self-contained, mirroring
# harry/activity_preferences/pipeline.py and shared/.../plot_biases.py) ---------
def _randomness_label(value) -> str:
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
    if labels is None:
        return df
    mask = df["random_in_reasoning"].apply(lambda v: _randomness_label(v) in labels)
    return df[mask]


def _summarize_selection_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per activity: selection_rate = picks / decisive appearances."""
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


def _pearson_ci(r: float, n: int, confidence: float = 0.95):
    if n <= 3 or pd.isna(r):
        return float("nan"), float("nan")
    zcrit = NormalDist().inv_cdf(0.5 + confidence / 2)
    r_clamped = max(min(float(r), 0.999999999), -0.999999999)
    z = math.atanh(r_clamped)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - zcrit * se), math.tanh(z + zcrit * se)


def _word_count(text) -> int:
    """Word count of a reasoning trace (None / NaN -> 0)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    return len(str(text).split())


def stats(model_key: str, setting: str):
    """Bias r + Fisher-z CI + mean CoT length (words) over the non-refusal rows.

    Returns (r, lo, hi, mean_words) or None if files/rows are missing.
    """
    pipe = RESULTS_DIR / model_key / "pipeline" / setting / "pipeline.jsonl"
    score_path = (RESULTS_DIR / model_key / "activity_preferences"
                  / "activity_liking_scores_summary.csv")
    if not pipe.exists() or not score_path.exists():
        return None
    df = pd.read_json(pipe, lines=True)
    if df.empty:
        return None
    df = _cut_rows(df, NON_REFUSAL_LABELS)
    if df.empty:
        return None
    score = pd.read_csv(score_path)
    merged = _summarize_selection_rates(df).merge(
        score[["activity", "mean_score"]], on="activity", how="left"
    ).dropna(subset=["mean_score", "selection_rate"])
    n = len(merged)
    if n < 2:
        return None
    r = float(merged["mean_score"].corr(merged["selection_rate"]))
    if pd.isna(r):
        return None
    lo, hi = _pearson_ci(r, n)
    # Mean CoT length per response. Empty reasoning traces are counted as 0 words
    # (deliberate: this is reasoning emitted per response, so a no-CoT run like
    # gpt-5.5-instant correctly sits at 0), not averaged over non-empty rows only.
    mean_words = float(df["reasoning"].map(_word_count).mean())
    return r, lo, hi, mean_words


def draw(ax, setting: str, *, x_mode: str) -> None:
    """Draw one panel. x_mode = "effort" (categorical ladder) or "tokens" (mean
    CoT length in words)."""
    ax.axhline(0, color="#9a9a9a", ls=":", lw=1.1, zorder=1)  # "no correlation"
    for label, key_tmpl, colour, marker in FAMILIES:
        pts = []  # (x, r, lo, hi), in effort order
        for idx, effort in enumerate(EFFORTS):
            res = stats(key_tmpl.format(effort), setting)
            if res is None:
                continue
            r, lo, hi, words = res
            pts.append((idx if x_mode == "effort" else words, r, lo, hi))
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])  # left -> right along the x-axis
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo_err = [p[1] - p[2] for p in pts]
        hi_err = [p[3] - p[1] for p in pts]
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err], fmt="none", ecolor=colour,
                    elinewidth=1.1, capsize=3.5, capthick=0.9, alpha=0.55, zorder=2)
        ax.plot(xs, ys, marker=marker, ms=7.5, lw=2.2, color=colour, label=label,
                markeredgecolor="white", markeredgewidth=0.9, zorder=3)
    if x_mode == "effort":
        ax.set_xticks(range(len(EFFORTS)))
        ax.set_xticklabels(EFFORTS)
        ax.set_xlim(-0.35, len(EFFORTS) - 0.65)
    else:
        # Fixed, shared x-axis across all three panels (0 always shown).
        ax.set_xlim(0, 250)
        ax.set_xticks(range(0, 251, 50))
    ax.grid(False)
    ax.set_facecolor("white")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", length=5, width=1.0,
                   colors="#333333")


def make_figure(x_mode: str, xlabel: str, out: Path) -> None:
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(16, 5.2), sharey=True)
    for ax, (setting, label) in zip(axes, SETTINGS):
        draw(ax, setting, x_mode=x_mode)
        ax.set_title(label, fontweight="semibold", pad=8)
        ax.set_xlabel(xlabel, labelpad=5)
    axes[0].set_ylabel("Bias", labelpad=6)

    # Legend inside the upper-right of the coin-flip (rightmost) panel.
    axes[-1].legend(loc="upper right", frameon=True, framealpha=0.9,
                    title="Model", title_fontsize=11)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    make_figure("effort", "Reasoning effort", OUT_EFFORT)
    make_figure("tokens", "CoT length (words)", OUT_TOKENS)


if __name__ == "__main__":
    main()
