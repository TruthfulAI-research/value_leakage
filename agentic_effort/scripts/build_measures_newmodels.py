"""Build results/<slug>/correlations/measures.csv for the newly-added models.

These models only have ELO (from run_elo_newmodels.py) and agentic data, so the
full 5-measure correlation pipeline in plot_correlation.py isn't applicable.
We assemble a measures.csv with the columns the results-explorer widget reads
(payout, payout_category, <measure>_mean, <measure>_sem), populating elo_* and
agentic_* where data exists.

The widget maps bare-charity agentic rows onto amount-keyed rows itself, so we
just emit whatever rows each measure produces and outer-merge them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_correlation import measure_elo, measure_agentic, measure_liking  # type: ignore

SLUGS = ["claude-opus-4-8", "gpt-5.5", "gemini-3.1-pro"]


def build_one(slug: str) -> None:
    root = Path("results") / slug
    measures = []

    liking_csv = root / "liking" / "data.csv"
    if liking_csv.exists():
        # Written by run_liking_newmodels.py with model column = display slug.
        measures.append(measure_liking(str(liking_csv), slug))

    elo_csv = root / "elo" / "elo.csv"
    if elo_csv.exists():
        # elo.csv is tagged with the display slug in the `model` column.
        measures.append(measure_elo(str(elo_csv), slug))

    agentic_csv = root / "agentic" / "data.csv"
    if agentic_csv.exists():
        # The agentic `model` column may hold a routing id, not the slug, so
        # pass the value actually present to avoid an empty filter.
        df = pd.read_csv(agentic_csv)
        model_val = df["model"].iloc[0] if "model" in df.columns and len(df) else slug
        measures.append(measure_agentic(str(agentic_csv), model_val))

    if not measures:
        print(f"[{slug}] no measures found, skipping")
        return

    merged = None
    for m in measures:
        d = m.df.rename(columns={"mean": f"{m.name}_mean", "sem": f"{m.name}_sem"})
        merged = d if merged is None else merged.merge(
            d, on=["payout", "payout_category"], how="outer"
        )

    out_dir = root / "correlations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "measures.csv"
    merged.to_csv(out_path, index=False)
    print(f"[{slug}] wrote {out_path}  ({len(merged)} rows, cols={list(merged.columns)})")


def main() -> None:
    for slug in SLUGS:
        build_one(slug)


if __name__ == "__main__":
    main()
