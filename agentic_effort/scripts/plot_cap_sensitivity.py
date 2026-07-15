"""Cap-sensitivity check: per-outcome mean send attempts as a function of the
turn cap, computed offline from the uncapped transcripts. Shows the outcome
ordering is stable in the cap, so the paper's 300-turn cap is not doing any
work beyond bounding the tail. Only gemini-3.1-pro is affected (other models
never exceed 50 turns)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_correlation import CATEGORY_COLORS  # type: ignore


def sends_within(transcript_json: str, cap: int) -> int:
    t = json.loads(transcript_json)
    return sum(1 for turn in t[:cap] for c in turn.get("tools", []) if c.get("name") == "send_btc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/gemini-3.1-pro/agentic/data_uncapped.csv")
    parser.add_argument("--out", default="results/gemini-3.1-pro/agentic/cap_sensitivity.png")
    parser.add_argument("--caps", nargs="+", type=int, default=[50, 100, 150, 200, 250, 300])
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    rows = []
    for _, r in df.iterrows():
        # Persistence is among engaged rollouts; drop refusals (never send at all).
        if sends_within(r["transcript_json"], 10**9) == 0:
            continue
        for cap in args.caps:
            rows.append({"outcome": r["outcome"], "payout_category": r["payout_category"],
                         "cap": cap, "sends": sends_within(r["transcript_json"], cap)})
    long = pd.DataFrame(rows)
    means = long.groupby(["outcome", "payout_category", "cap"])["sends"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    for (outcome, cat), g in means.groupby(["outcome", "payout_category"]):
        g = g.sort_values("cap")
        ax.plot(g["cap"], g["sends"], marker="o", markersize=3,
                color=CATEGORY_COLORS.get(cat, "#333"), alpha=0.85, label=outcome)
        ax.annotate(outcome, (g["cap"].iloc[-1], g["sends"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=6, va="center")
    ax.set_xlabel("turn cap")
    ax.set_ylabel("mean send attempts within cap")
    ax.set_title("gemini-3.1-pro | outcome ordering is stable in the turn cap")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
