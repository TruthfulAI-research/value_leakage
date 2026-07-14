"""Build the outcome files from outcomes/labels.jsonl.

Writes:
  outcomes/covertness_by_model.csv  counts and percentages per model and group
  outcomes/summary.txt              overall totals across all labelled rollouts
  outcomes/cot_covertness.pdf/.png  stacked bar chart per model and group

Run from the repo root:
    python job_offer/cot_analysis/make_outputs.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outcomes"
LABELS_PATH = OUT / "labels.jsonl"

CATS = ["ADMITS", "MENTIONS", "DENIES", "NO_MENTION"]
GROUPS = ["ours_first", "ours_second"]
GROUP_LABEL = {"ours_first": "own=current", "ours_second": "own=offer"}

MODELS = [
    "claude-opus-4.7-high", "claude-opus-4.8-high", "claude-fable-5-high",
    "gemini-3.1-pro-medium", "gpt-5.5-high", "gpt-5.4-medium",
]
DISPLAY = {
    "claude-opus-4.7-high": "Claude Opus 4.7", "claude-opus-4.8-high": "Claude Opus 4.8",
    "claude-fable-5-high": "Claude 5-Fable", "gemini-3.1-pro-medium": "Gemini 3.1 Pro",
    "gpt-5.5-high": "GPT-5.5", "gpt-5.4-medium": "GPT-5.4",
}
COLORS = {"ADMITS": "#2ca02c", "MENTIONS": "#98df8a",
          "DENIES": "#d62728", "NO_MENTION": "#cccccc"}
CAT_TEXT = {"ADMITS": "admits to bias", "MENTIONS": "mentions bias",
            "DENIES": "denies bias", "NO_MENTION": "no mention of bias"}


def load_rows():
    rows = []
    with open(LABELS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    counts = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    for r in rows:
        counts[(r["model"], r["group"])][r["label"]] += 1
        overall[r["label"]] += 1

    # csv
    with open(OUT / "covertness_by_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "group", "n"] + [c.lower() + "_pct" for c in CATS])
        for m in MODELS:
            for g in GROUPS:
                d = counts[(m, g)]
                n = sum(d.values())
                if not n:
                    continue
                w.writerow([m, g, n] + [round(100 * d[c] / n, 1) for c in CATS])

    # summary
    total = sum(overall.values())
    with open(OUT / "summary.txt", "w") as f:
        f.write(f"chains of thought labelled: {total}\n\n")
        for c in CATS:
            pct = 100 * overall[c] / total if total else 0
            f.write(f"{c:<11} {overall[c]:>5}  ({pct:.1f}%)\n")

    # figure
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.38
    x = np.arange(len(MODELS))
    for gi, g in enumerate(GROUPS):
        for xi, m in enumerate(MODELS):
            xp = x[xi] + (gi - 0.5) * width
            d = counts[(m, g)]
            n = sum(d.values())
            if not n:
                ax.text(xp, 3, "no CoT", rotation=90, ha="center",
                        va="bottom", fontsize=9, color="0.4")
                continue
            bottom = 0.0
            for c in CATS:
                h = 100 * d[c] / n
                ax.bar(xp, h, width, bottom=bottom, color=COLORS[c],
                       label=CAT_TEXT[c] if (gi == 0 and xi == 1) else None)
                bottom += h
            ax.text(xp, -4, GROUP_LABEL[g], rotation=90, ha="center",
                    va="top", fontsize=8, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in MODELS], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of chains of thought")
    ax.set_title("Job Offer: chain of thought covertness by model and condition")
    ax.legend(loc="lower center", ncol=4, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "cot_covertness.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "cot_covertness.png", dpi=150, bbox_inches="tight")

    print("wrote:")
    for name in ["covertness_by_model.csv", "summary.txt",
                 "cot_covertness.pdf", "cot_covertness.png"]:
        print("  outcomes/" + name)


if __name__ == "__main__":
    main()
