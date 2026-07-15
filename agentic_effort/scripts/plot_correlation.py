"""Scatter plots across four per-outcome preference-elicitation measures for a given model.

  digits  — mean digits in the `number` task response  (long_outputs)
  fermi   — P(above | above-variant) − P(above | below-variant)  (fermi)
  liking  — mean stated 0–100 rating of the outcome  (stated_liking)
  elo     — iterative ELO rating from pairwise match-ups  (elo_ranking)

Generates every pairwise scatter (C(4,2) = 6 plots) plus a combined grid, all
with both-axis error bars.
"""
from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORY_COLORS = {
    "choose":                "#2ca02c",
    "none":                  "#7f7f7f",
    "ea_global_health":      "#1f77b4",
    "ea_animal":             "#17becf",
    "ea_meta":               "#aec7e8",
    "ea_ai_safety":          "#9467bd",
    "neutral_mainstream":    "#bcbd22",
    "political_left":        "#6baed6",
    "political_right":       "#fdae6b",
    "harmful_animals":       "#d62728",
    "harmful_health":        "#e377c2",
    "harmful_authoritarian": "#e6550d",
    "harmful_extremist":     "#8c564b",
}


@dataclass
class Measure:
    name: str
    label: str
    df: pd.DataFrame  # columns: payout, payout_category, mean, sem


def measure_digits(long_csv: str, model: str, task: str = "number") -> Measure:
    df = pd.read_csv(long_csv)
    d = df[(df["model"] == model) & (df["task"] == task)]
    agg = (
        d.groupby(["payout", "payout_category"])["metric"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / agg["count"].pow(0.5)
    agg = agg[["payout", "payout_category", "mean", "sem"]]
    return Measure("digits", f"Mean digits in response  (long_outputs: {task})", agg)


def measure_fermi(fermi_csv: str, model: str) -> Measure:
    df = pd.read_csv(fermi_csv)
    f = df[(df["model"] == model) & (df["direction"].isin(["above", "below"]))]
    per = (
        f.dropna(subset=["estimate"])
        .groupby(["payout", "payout_category", "direction"])
        .agg(above_rate=("above_threshold", "mean"), n=("above_threshold", "size"))
        .reset_index()
    )
    wide = per.pivot_table(
        index=["payout", "payout_category"],
        columns="direction",
        values=["above_rate", "n"],
    ).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in wide.columns.to_flat_index()]
    wide["mean"] = wide["above_rate_above"] - wide["above_rate_below"]
    var_a = wide["above_rate_above"] * (1 - wide["above_rate_above"]) / wide["n_above"]
    var_b = wide["above_rate_below"] * (1 - wide["above_rate_below"]) / wide["n_below"]
    wide["sem"] = (var_a + var_b).pow(0.5)
    return Measure(
        "fermi",
        "Fermi preference  (P(above | above-variant) − P(above | below-variant))",
        wide[["payout", "payout_category", "mean", "sem"]],
    )


def measure_liking(liking_csv: str, model: str) -> Measure:
    df = pd.read_csv(liking_csv)
    d = df[df["model"] == model].dropna(subset=["rating"])
    agg = (
        d.groupby(["outcome", "payout_category"])["rating"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"outcome": "payout"})
    )
    agg["sem"] = agg["std"] / agg["count"].pow(0.5)
    return Measure("liking", "Stated liking  (mean 0–100 rating)",
                   agg[["payout", "payout_category", "mean", "sem"]])


def measure_elo(elo_csv: str, model: str | None = None) -> Measure:
    df = pd.read_csv(elo_csv)
    if model and "model" in df.columns:
        df = df[df["model"] == model]
    agg = df.rename(columns={"outcome": "payout", "elo": "mean", "elo_sem": "sem"})
    return Measure("elo", "ELO rating  (pairwise preferences)",
                   agg[["payout", "payout_category", "mean", "sem"]])


def measure_agentic(agentic_csv: str, model: str, column: str = "n_send_attempts") -> Measure:
    """Agentic-effort (persistence) measure: mean n_send_attempts among rollouts
    that engaged, i.e. sent at least once. Rollouts with 0 send attempts are
    *refusals* (the model declined to send at all) and are analysed separately
    (see scripts/refusals_table.py); mixing them in would conflate "won't do it"
    with "how hard it tries once engaged". Outcomes with no senders (e.g. a
    100%%-refused recipient) produce no row."""
    df = pd.read_csv(agentic_csv)
    if "model" in df.columns:
        df = df[df["model"] == model]
    df = df[df[column] >= 1]
    agg = (
        df.groupby(["outcome", "payout_category"])[column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"outcome": "payout"})
    )
    agg["sem"] = agg["std"] / agg["count"].pow(0.5)
    return Measure("agentic",
                   f"Agentic effort  (mean {column}, retries on failing send_btc)",
                   agg[["payout", "payout_category", "mean", "sem"]])


_AMOUNT_SUFFIX = re.compile(r"_\d+$")


def _has_amount_suffix(df: pd.DataFrame) -> bool:
    return df["payout"].astype(str).str.contains(_AMOUNT_SUFFIX).any()


def _aggregate_by_charity(df: pd.DataFrame) -> pd.DataFrame:
    """Strip `_\\d+$` from payout names and aggregate mean/sem to charity level."""
    out = df.copy()
    out["payout"] = out["payout"].astype(str).str.replace(_AMOUNT_SUFFIX, "", regex=True)
    def combine_sem(s):
        s = s.fillna(0).to_numpy(dtype=float)
        if len(s) == 0:
            return np.nan
        return float(np.sqrt(np.sum(s ** 2)) / len(s))
    agg = (
        out.groupby(["payout", "payout_category"])
        .agg(mean=("mean", "mean"), sem=("sem", combine_sem))
        .reset_index()
    )
    return agg


def _align_granularity(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """If one df is charity-level and the other has amount suffixes, aggregate the
    amount-level one to charity level so they can be merged."""
    a_amt, b_amt = _has_amount_suffix(a), _has_amount_suffix(b)
    if a_amt and not b_amt:
        return _aggregate_by_charity(a), b
    if b_amt and not a_amt:
        return a, _aggregate_by_charity(b)
    return a, b


def _pearson_spearman(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    ok = x.notna() & y.notna()
    return x[ok].corr(y[ok]), x[ok].corr(y[ok], method="spearman")


def scatter(ax, mx: Measure, my: Measure, title_prefix: str = "") -> tuple[float, float, int]:
    a_df, b_df = _align_granularity(mx.df, my.df)
    a = a_df.rename(columns={"mean": "x_mean", "sem": "x_sem"})
    b = b_df.rename(columns={"mean": "y_mean", "sem": "y_sem"})
    merged = a.merge(b, on=["payout", "payout_category"], how="inner")

    for _, row in merged.iterrows():
        color = CATEGORY_COLORS.get(row["payout_category"], "#333")
        ax.errorbar(
            row["x_mean"], row["y_mean"],
            xerr=row["x_sem"], yerr=row["y_sem"],
            fmt="o", markersize=6, color=color,
            ecolor=color, elinewidth=0.8, capsize=2, alpha=0.9,
            markeredgecolor="black", markeredgewidth=0.4,
        )
        ax.annotate(
            row["payout"], (row["x_mean"], row["y_mean"]),
            fontsize=6, alpha=0.7,
            xytext=(4, 3), textcoords="offset points",
        )

    r_p, r_s = _pearson_spearman(merged["x_mean"], merged["y_mean"])
    ax.set_xlabel(mx.label, fontsize=9)
    ax.set_ylabel(my.label, fontsize=9)
    ax.set_title(f"{title_prefix}Pearson r = {r_p:.2f}, Spearman ρ = {r_s:.2f}, n = {len(merged)}",
                 fontsize=10)
    ax.grid(True, alpha=0.25)
    return r_p, r_s, len(merged)


def save_pair(mx: Measure, my: Measure, model: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    r_p, r_s, n = scatter(ax, mx, my, title_prefix=f"{model}   |   ")
    merged = mx.df.merge(my.df, on="payout_category", how="inner")
    cats_used = sorted(set(merged["payout_category"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS.get(c, "#333")) for c in cats_used]
    ax.legend(handles, cats_used, fontsize=7, loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}  (Pearson={r_p:.3f}, Spearman={r_s:.3f}, n={n})")


def save_grid(measures: list[Measure], model: str, out_path: Path, n_cols: int = 3) -> None:
    pairs = list(itertools.combinations(measures, 2))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6.5))
    axes = axes.flatten()
    for ax, (mx, my) in zip(axes, pairs):
        scatter(ax, mx, my, title_prefix=f"{model}   |   ")
    for ax in axes[len(pairs):]:
        ax.axis("off")
    cats_used = sorted(set().union(*[set(m.df["payout_category"]) for m in measures]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS.get(c, "#333")) for c in cats_used]
    fig.legend(handles, cats_used, ncol=min(len(cats_used), 7),
               loc="lower center", fontsize=9)
    fig.subplots_adjust(bottom=0.08)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def _slug(model: str) -> str:
    return model.split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--long",    default=None)
    parser.add_argument("--fermi",   default=None)
    parser.add_argument("--liking",  default=None)
    parser.add_argument("--elo",     default=None,
                        help="Path to the elo.csv (not matches.csv) produced by run_elo_ranking.")
    parser.add_argument("--agentic", default=None)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to results/<slug(model)>/correlations/.")
    args = parser.parse_args()

    slug = _slug(args.model)
    root = Path("results") / slug
    defaults = {
        "long":    root / "number" / "data.csv",
        "fermi":   root / "fermi" / "data.csv",
        "liking":  root / "liking" / "data.csv",
        "elo":     root / "elo" / "elo.csv",
        "agentic": root / "agentic" / "data.csv",
    }

    out_dir = Path(args.out_dir) if args.out_dir else root / "correlations"
    out_dir.mkdir(parents=True, exist_ok=True)

    measures: list[Measure] = []
    for name, ctor in [("long", measure_digits), ("fermi", measure_fermi),
                        ("liking", measure_liking), ("elo", measure_elo),
                        ("agentic", measure_agentic)]:
        path = Path(getattr(args, name) or defaults[name])
        if not path.exists():
            continue
        if name == "long":
            measures.append(measure_digits(str(path), args.model))
        else:
            measures.append(ctor(str(path), args.model))

    if len(measures) < 2:
        print("need at least 2 measures; got", len(measures))
        return

    for mx, my in itertools.combinations(measures, 2):
        save_pair(mx, my, args.model, out_dir / f"{mx.name}_vs_{my.name}.png")
    save_grid(measures, args.model, out_dir / "grid.png")

    merged = None
    for m in measures:
        df = m.df.rename(columns={"mean": f"{m.name}_mean", "sem": f"{m.name}_sem"})
        merged = df if merged is None else merged.merge(df, on=["payout", "payout_category"], how="outer")
    if merged is not None:
        csv_path = out_dir / "measures.csv"
        merged.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
