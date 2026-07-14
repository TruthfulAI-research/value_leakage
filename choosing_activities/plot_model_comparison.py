"""Grouped bar chart comparing score/selection correlation across models & tool settings.

For the models below (grouped by family) and three tool settings (no tools, unix
time, coin flip), plot the Pearson correlation between each activity's mean stated
liking score and its selection rate. A strong positive r means the model steers
"random" picks toward activities it prefers.

We use the "overall excluding refusals" cut: every pairwise decision is kept
EXCEPT randomness-judge refusals, so overt bias (random_in_reasoning == false) is
included. This is more inclusive than the paper's main figure (which keeps only
true+hedged). Within each cut, selection_rate still drops choice-judge refusals
from the denominator (handled by summarize_selection_rates).

Self-contained (like plot_biases.py): reads harry's cached results directly and
re-implements the selection-rate / correlation helpers; the only shared/ dependency
is `shared.models` for display-name labels. Writes PDF only (figures-PDF-only
convention) straight into the paper's Overleaf clone so the figure used in the
paper stays in sync:
  overleaf/figures/preferences/model_comparison_correlation.pdf

Usage (from the repo root):
  uv run python choosing_activities/plot_model_comparison.py
"""
import math
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
from shared.plot_style import HEADER_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import
import pandas as pd

from shared.models import MODELS as MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
HARRY_ROOT = REPO_ROOT / "choosing_activities"
RESULTS_DIR = REPO_ROOT / "data" / "choosing_activities" / "results"
# Write straight into the gitignored Overleaf clone at the repo root so the paper
# figure is regenerated in place.
OUT = REPO_ROOT / "overleaf" / "figures" / "preferences" / "model_comparison_correlation.pdf"

# Models, left -> right on the x-axis, grouped by family. Each entry is
# (model_key, family): the family name is drawn along the top and adjacent
# families are separated by a dashed vertical line. The per-bar x tick is the
# canonical full display name from the shared registry. Keep families contiguous
# so the headers/separators line up.
MODELS = [
    ("claude-opus-4.7-xhigh", "Claude"),
    ("claude-opus-4.7-max",   "Claude"),
    ("claude-opus-4.8-xhigh", "Claude"),
    ("claude-opus-4.8-max",   "Claude"),
    ("gpt-5.5-high",          "GPT"),
    ("gpt-5.5-xhigh",         "GPT"),
    ("gemini-3.1-pro-high",   "Gemini"),
]


def _display(model_key: str) -> str:
    """Axis label for a model: the shared.models display_name (same labels as the
    other final-scripts figures), falling back to the key."""
    return MODEL_REGISTRY.get(model_key, {}).get("display_name", model_key)


DISPLAY_NAMES = [_display(k) for k, _ in MODELS]

# Tool settings, in legend / within-group order. (key, legend label, colour).
# Colours aligned with ai_bubble/plot_probabilities.py's project-standard trio
# (grey / blue / purple from plot_biases.DIR_COLORS): blue for no-tools, purple
# for unix-time, grey for coin-flip. No orange (reserved for Claude) / green / red.
SETTINGS = [
    ("no_tools", "No tools", "#1f77b4"),
    ("unix_time", "Unix time tool", "#9467bd"),
    ("coin_flip", "Coin flip tool", "#7f7f7f"),
]

# Randomness verdicts kept for the "overall excluding refusals" cut: everything
# the randomness judge did NOT call a refusal (so overt bias, `false`, is kept).
NON_REFUSAL_LABELS = ("true", "hedged", "false")


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


def _pearson_ci(r: float, n: int, confidence: float = 0.95):
    """Fisher z-transform CI for a Pearson r over `n` paired activities."""
    if n <= 3 or pd.isna(r):
        return float("nan"), float("nan")
    zcrit = NormalDist().inv_cdf(0.5 + confidence / 2)
    r_clamped = max(min(float(r), 0.999999999), -0.999999999)
    z = math.atanh(r_clamped)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - zcrit * se), math.tanh(z + zcrit * se)


def correlation(model_key: str, setting_key: str) -> tuple[float, float, float]:
    """Bias (Pearson r of mean stated score vs selection rate) + 95% CI.

    Cut: every pairwise decision EXCEPT randomness-judge refusals (overt bias,
    i.e. random_in_reasoning == false, is kept). The CI is the Fisher-z interval
    over activities — the codebase's standard.
    """
    jsonl = RESULTS_DIR / model_key / "pipeline" / setting_key / "pipeline.jsonl"
    if not jsonl.exists():
        sys.exit(f"missing pipeline jsonl: {jsonl}")
    df = pd.read_json(jsonl, lines=True)

    score_path = (
        RESULTS_DIR / model_key / "activity_preferences"
        / "activity_liking_scores_summary.csv"
    )
    if not score_path.exists():
        sys.exit(f"missing stated-liking score summary: {score_path}")
    score = pd.read_csv(score_path)

    sel = _summarize_selection_rates(_cut_rows(df, NON_REFUSAL_LABELS))
    merged = sel.merge(
        score[["activity", "mean_score"]], on="activity", how="left"
    ).dropna(subset=["mean_score", "selection_rate"])
    n = len(merged)
    r = float(merged["mean_score"].corr(merged["selection_rate"])) if n >= 2 else float("nan")
    lo, hi = _pearson_ci(r, n)
    return r, lo, hi


def main() -> None:
    # stats[i][j] = (r, lo, hi) for model i, setting j
    stats = [[correlation(mk, sk) for sk, _, _ in SETTINGS] for mk, _ in MODELS]
    for name, row in zip(DISPLAY_NAMES, stats):
        print(f"{name:22}  " + "  ".join(
            f"{sl}={r:+.2f}[{lo:+.2f},{hi:+.2f}]"
            for (_, sl, _), (r, lo, hi) in zip(SETTINGS, row)))

    n_models = len(MODELS)
    n_settings = len(SETTINGS)
    # Bars within a model touch (no intra-group gap); group_w < 1 leaves
    # whitespace between models. Model centres are 1.0 apart, so the gap between
    # adjacent groups is (1 - group_w); we reuse that same gap as the left/right
    # margin to the y-axis below (xlim).
    group_w = 0.78
    gap = 0.0
    bar_w = (group_w - gap * (n_settings - 1)) / n_settings
    x = list(range(n_models))

    fig, ax = plt.subplots(figsize=(1.9 * n_models, 4.5))

    # y grid lines only, behind the bars (kept faint so the bars dominate).
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.7)
    ax.xaxis.grid(False)

    for j, (skey, slabel, colour) in enumerate(SETTINGS):
        offsets = [xi - group_w / 2 + (bar_w + gap) * j + bar_w / 2 for xi in x]
        heights = [stats[i][j][0] for i in range(n_models)]
        yerr = [
            [stats[i][j][0] - stats[i][j][1] for i in range(n_models)],  # lower
            [stats[i][j][2] - stats[i][j][0] for i in range(n_models)],  # upper
        ]
        ax.bar(
            offsets, heights, bar_w, color=colour, label=slabel,
            edgecolor="none", zorder=3,
            yerr=yerr, capsize=3,
            error_kw=dict(ecolor="#000000", elinewidth=1.0, capthick=1.0, zorder=4),
        )
        # Value labels above the upper error-bar cap; for bars that sit below
        # zero we clamp the label to just above the axis so it clears the bar
        # (and never renders a "-0.00").
        for i, xpos in enumerate(offsets):
            r, _, hi = stats[i][j]
            txt = f"{r:.2f}".replace("-0.00", "0.00")
            ax.text(xpos, max(hi, 0.0) + 0.035, txt, ha="center", va="bottom",
                    fontsize=VALUE_FS, color="#000000", zorder=5)

    ax.axhline(0, color="#000000", linewidth=0.9, zorder=4)

    # Family headers along the top + dashed separators between adjacent families.
    families = [fam for _, fam in MODELS]
    spans = []  # (family, first_idx, last_idx) for each contiguous run
    start = 0
    for i in range(1, n_models + 1):
        if i == n_models or families[i] != families[start]:
            spans.append((families[start], start, i - 1))
            start = i
    # Headers sit INSIDE the plot box near the top.
    for fam, i0, i1 in spans:
        ax.text((i0 + i1) / 2, 0.95, fam, ha="center", va="top",
                fontsize=HEADER_FS, fontweight="bold", color="#000000", zorder=6)
    for a, b in zip(spans, spans[1:]):
        ax.axvline((a[2] + b[1]) / 2, color="#888888", linestyle=(0, (4, 4)),
                   linewidth=1.0, zorder=2)

    # Left/right margin to the y-axis = the inter-model gap (1 - group_w), so the
    # whitespace before the first group and after the last matches the gaps
    # between groups.
    ax.set_xlim(group_w / 2 - 1, n_models - group_w / 2)

    # Horizontal x labels, full names, wrapped onto two lines. Split at the hyphen
    # nearest the middle (minimising the longer line, e.g. "claude-opus\n4-7-xhigh"
    # rather than "claude-opus-4-7\nxhigh") so the widest label stays narrow enough
    # that adjacent labels don't collide at the larger font size below.
    def _wrap(name: str) -> str:
        hyphens = [i for i, c in enumerate(name) if c == "-"]
        if not hyphens:
            return name
        i = min(hyphens, key=lambda k: max(k, len(name) - k - 1))
        return name[:i] + "\n" + name[i + 1:]

    # Font size for the y-axis title, y tick labels and x tick labels. Chosen so
    # that, after LaTeX scales this figure to \linewidth, these appear the SAME
    # apparent size as the FS=12 text in the covertness figure
    # (covertness_pooled_reasoning.pdf). Both are included at \linewidth, so
    # apparent_pt = pt * linewidth / cropped_pdf_width; matching the covertness
    # FS=12 needs 12 * (W_barchart / W_covertness) = 12 * 950.1/755.6 ~= 15.1.
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(n) for n in DISPLAY_NAMES],
                       rotation=0, ha="center")
    ax.set_ylabel("Bias metric", labelpad=2)
    ax.set_ylim(-0.5, 1.0)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.tick_params(axis="both", which="major", length=5, width=1.0,
                   colors="#000000")
    # Box the plot: all four spines on, drawn above the bars.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.9)
        spine.set_zorder(10)

    # Legend to the right of the plot, vertically centred (single column). The
    # family headers stay inside the axes near the top.
    ax.legend(frameon=True, loc="center left",
              bbox_to_anchor=(1.02, 0.5), ncol=1,
              handlelength=1.3, handletextpad=0.6).set_zorder(11)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"\nwrote {OUT}")

#%%
if __name__ == "__main__":
    main()
