"""Per-outcome distribution box plots (appendix companions to ranking.png).

The ranking bar plots show only means ± SEM; these box plots show the full
per-sample distribution so it's visible whether a mean is moved by outliers or
reflects consistent behavior. One box per outcome, sorted by mean (same order
logic as the ranking plots), colored by payout category.

Layout mirrors plot_rankings.py:

    results/<slug>/agentic/data.csv   --> results/<slug>/agentic/box.png    (n_send_attempts)
    results/<slug>/liking/data.csv    --> results/<slug>/liking/box.png     (rating)
    results/<slug>/number/data.csv    --> results/<slug>/number/box.png     (metric, task=="number")

ELO has no per-sample scalar (it aggregates pairwise matches), so it has no
box plot. Example:

    python scripts/plot_boxplots.py --model openai/gpt-5.5
    python scripts/plot_boxplots.py --model openrouter/google/gemini-3.1-pro-preview \
        --agentic results/gemini-3.1-pro/agentic/data.csv --out-root results/gemini-3.1-pro
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_correlation import CATEGORY_COLORS  # type: ignore

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "final_data" / "agentic_effort"
_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _slug(model: str) -> str:
    return model.split("/")[-1]


def plot_box(df: pd.DataFrame, value_col: str, label: str, model: str, out_path: Path) -> None:
    """df needs columns: payout, payout_category, <value_col> (one row per sample)."""
    df = df.dropna(subset=[value_col])
    if df.empty:
        print(f"no data for {out_path}")
        return

    order = (
        df.groupby("payout")[value_col].mean().sort_values(ascending=True).index.tolist()
    )
    groups = [df.loc[df["payout"] == p, value_col].to_numpy() for p in order]
    categories = [df.loc[df["payout"] == p, "payout_category"].iloc[0] for p in order]
    ns = [len(g) for g in groups]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.36 * len(order) + 1.5)))
    boxes = ax.boxplot(
        groups, vert=False, patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
        medianprops={"color": "black"},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for patch, cat in zip(boxes["boxes"], categories):
        patch.set_facecolor(CATEGORY_COLORS.get(cat, "#333"))
        patch.set_alpha(0.85)

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{p}  (n={n})" for p, n in zip(order, ns)], fontsize=8)
    ax.set_xlabel(label)
    ax.set_title(f"{model} | {label} | per-outcome distribution")
    ax.grid(True, axis="x", alpha=0.3)

    present = list(dict.fromkeys(categories))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS.get(c, "#333")) for c in present]
    ax.legend(handles, present, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(order)} boxes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--agentic", default=None, help="Override path to agentic CSV.")
    parser.add_argument("--liking", default=None)
    parser.add_argument("--number", default=None)
    parser.add_argument("--out-root", default=None,
                        help="Output root (default: results/<slug(model)>/).")
    args = parser.parse_args()

    root = _DATA_ROOT / _slug(args.model)          # raw data (submodule)
    out_root = Path(args.out_root) if args.out_root else _RESULTS_ROOT / _slug(args.model)  # figures: local

    agentic_csv = Path(args.agentic or root / "agentic" / "data.csv")
    if agentic_csv.exists():
        df = pd.read_csv(agentic_csv)
        # No model filter: results/<slug>/agentic/data.csv is single-model, and
        # its `model` column holds the routing id (not the display slug).
        # Persistence distribution among rollouts that engaged (>= 1 send);
        # 0-send rollouts are refusals, analysed separately.
        df = df[df["n_send_attempts"] >= 1]
        df = df.rename(columns={"outcome": "payout"})
        plot_box(df, "n_send_attempts", "Agentic persistence (send attempts on failing send_btc, engaged rollouts)",
                 args.model, out_root / "agentic" / "box.png")

    liking_csv = Path(args.liking or root / "liking" / "data.csv")
    if liking_csv.exists():
        df = pd.read_csv(liking_csv)
        df = df[df["model"] == args.model] if "model" in df.columns else df
        df = df.rename(columns={"outcome": "payout"})
        plot_box(df, "rating", "Stated liking (0-100 rating)",
                 args.model, out_root / "liking" / "box.png")

    number_csv = Path(args.number or root / "number" / "data.csv")
    if number_csv.exists():
        df = pd.read_csv(number_csv)
        if "model" in df.columns:
            df = df[df["model"] == args.model]
        if "task" in df.columns:
            df = df[df["task"] == "number"]
        plot_box(df, "metric", "Competition effort (digits in response)",
                 args.model, out_root / "number" / "box.png")


if __name__ == "__main__":
    main()
