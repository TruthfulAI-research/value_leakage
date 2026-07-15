"""Paper figures (multi-panel, large fonts, print-friendly).

Figure 1  — agentic persistence: one horizontal-bar subplot per model, shared
            outcome order and y-ticks, neutral bars; only the `choose` bar is a
            colored stack decomposed by the recipient of the model's first
            send_btc call (Okabe-Ito palette, consistent across subplots,
            shared legend).
Figure 2  — stated liking (x, shared 0-100 scale) vs agentic persistence (y,
            per-model scale): one scatter subplot per model, one point per
            outcome, direct labels, Spearman rho annotated.

Outputs PNG (300 dpi) + PDF to results/paper/.

    python scripts/plot_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # noqa: F401  (fallback below if absent)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_rankings import agentic_choose_breakdown  # type: ignore

MODELS = [
    ("claude-opus-4-8", "anthropic/claude-opus-4-8"),
    ("gpt-5.5", "openai/gpt-5.5"),
    ("gemini-3.1-pro", "openrouter/google/gemini-3.1-pro-preview"),
]

# Okabe-Ito (CVD-safe); gray is reserved for "(no send)" — an absence, so
# reading as gray is intended. Fixed assignment order by overall frequency.
RECIPIENT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
NO_SEND_COLOR = "#999999"
BAR_COLOR = "#B5B5B5"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 110,
})



# Display labels for outcome keys (keys stay as-is in configs/data; `ammf` is a
# historical key typo for Against Malaria Foundation -> shown as AMF).
OUTCOME_LABELS = {
    "choose": "Free choice",
    "humane_league": "Humane League",
    "ammf": "AMF",
    "givewell": "GiveWell",
    "miri": "MIRI",
    "anthropic_safety": "Anthropic safety",
    "gdm_safety_team": "GDM safety team",
    "openai_safety_team": "OpenAI safety team",
    "unicef": "UNICEF",
    "red_cross": "Red Cross",
    "dnc": "DNC",
    "democrats_campaign": "Democrats campaign",
    "trump_campaign": "Trump campaign",
    "factory_farming_lobby": "Factory farming lobby",
    "tobacco_lobby": "Tobacco lobby",
    "ccp": "CCP",
    "hamas_linked": "Hamas-linked charity",
}


def outcome_label(key: str) -> str:
    return OUTCOME_LABELS.get(key, key)



SHORT_RECIPIENT = {
    "Against Malaria Foundation": "AMF",
    "American Red Cross": "Red Cross",
    "UNICEF": "UNICEF",
    "GiveWell": "GiveWell",
    "Internet Archive": "Int. Archive",
    "(no send)": "no send",
    "(unlisted address)": "unlisted",
}

_fit_labels: list = []


def _is_dark(hex_color: str) -> bool:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55


def agentic_means(slug: str, model_val: str, engaged_only: bool = True) -> pd.DataFrame:
    """Per-outcome persistence stats. By default restricted to rollouts that
    engaged (sent >= 1 time); 0-send rollouts are refusals, handled separately
    (see refusals_table.py / figure_refusals). `count` is the number of senders.
    Outcomes with no senders (fully refused) yield no row."""
    df = pd.read_csv(f"results/{slug}/agentic/data.csv")
    if "model" in df.columns:
        df = df[df["model"] == model_val]
    if engaged_only:
        df = df[df["n_send_attempts"] >= 1]
    agg = df.groupby("outcome")["n_send_attempts"].agg(["mean", "median", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / agg["count"].pow(0.5)
    return agg


def refusal_rates() -> pd.DataFrame:
    """Refusal rate (fraction of rollouts with 0 send attempts) per outcome and
    model, plus counts. Index = outcome, columns = per-model rate and count."""
    out = {}
    for slug, mv in MODELS:
        df = pd.read_csv(f"results/{slug}/agentic/data.csv")
        if "model" in df.columns:
            df = df[df["model"] == mv]
        g = df.assign(ref=df["n_send_attempts"] == 0).groupby("outcome")["ref"]
        out[slug] = g.mean()
        out[slug + "_n"] = g.sum().astype(int)
        out[slug + "_total"] = g.count().astype(int)
    return pd.DataFrame(out)


def figure_agentic(out_base: Path, show_median: bool = False) -> None:
    _fit_labels.clear()
    means = {slug: agentic_means(slug, mv) for slug, mv in MODELS}

    # All outcomes, including fully-refused ones (which have no persistence bar
    # and instead get a red x). Ordered by claude-opus-4-8 persistence; outcomes
    # opus always refused (NaN mean) sink to the bottom.
    all_outcomes = list(refusal_rates().index)
    opus_mean = means["claude-opus-4-8"].set_index("outcome")["mean"].reindex(all_outcomes)
    order = opus_mean.sort_values(ascending=True, na_position="first").index.tolist()

    # Consistent recipient -> color mapping across subplots.
    breakdowns = {
        slug: agentic_choose_breakdown(f"results/{slug}/agentic/data.csv", mv)
        for slug, mv in MODELS
    }
    totals: dict[str, int] = {}
    for b in breakdowns.values():
        for row in b.itertuples(index=False):
            totals[row.recipient] = totals.get(row.recipient, 0) + int(row.count)
    named = [r for r in sorted(totals, key=totals.get, reverse=True) if not r.startswith("(")]
    color_of = {r: RECIPIENT_COLORS[i % len(RECIPIENT_COLORS)] for i, r in enumerate(named)}
    for r in totals:
        if r.startswith("("):
            color_of[r] = NO_SEND_COLOR

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    y = np.arange(len(order))
    for ax, (slug, _mv) in zip(axes, MODELS):
        m = means[slug].set_index("outcome").reindex(order)
        ax.barh(y, m["mean"], xerr=m["sem"].fillna(0), height=0.72,
                color=BAR_COLOR, edgecolor="white", linewidth=0.8,
                error_kw={"elinewidth": 1.0, "capsize": 2}, zorder=2)

        # Red x for outcomes this model ALWAYS refused (0 senders -> NaN mean),
        # placed just inside the axis (axes-x, data-y) so it shows at any scale.
        import matplotlib.transforms as mtransforms
        refused = m["mean"].isna().to_numpy()
        if refused.any():
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.plot(np.full(int(refused.sum()), 0.03), y[refused], transform=trans,
                    linestyle="none", marker="x", color="#C92A2A",
                    markersize=9, markeredgewidth=2.2, zorder=6, clip_on=False)

        # Overdraw `choose` as a stack decomposed by chosen recipient; label
        # each segment inline (only if the text fits) instead of a legend.
        b = breakdowns[slug]
        if not b.empty and "choose" in order:
            yi = order.index("choose")
            left = 0.0
            for row in b.itertuples(index=False):
                ax.barh(yi, row.contribution, left=left, height=0.72,
                        color=color_of[row.recipient], edgecolor="white",
                        linewidth=0.8, zorder=3)
                label = SHORT_RECIPIENT.get(row.recipient, row.recipient)
                text_color = "white" if _is_dark(color_of[row.recipient]) else "#222222"
                t = ax.text(left + row.contribution / 2, yi, label,
                            ha="center", va="center", fontsize=9.5,
                            color=text_color, zorder=4)
                _fit_labels.append((ax, t, left, row.contribution))
                left += row.contribution

        if show_median:
            ax.plot(m["median"], y, linestyle="none", marker="|",
                    markersize=11, markeredgewidth=1.8, color="#222222", zorder=5)
        ax.set_title(slug)
        ax.set_xlabel("Mean send attempts")
        ax.grid(True, axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([outcome_label(o) for o in order])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], linestyle="none", marker="x", color="#C92A2A",
                      markersize=9, markeredgewidth=2.2)]
    labels = ["always refused"]
    if show_median:
        handles.append(Line2D([0], [0], linestyle="none", marker="|", markersize=11,
                              markeredgewidth=1.8, color="#222222"))
        labels.append("median")
    legend = axes[0].legend(handles, labels, loc="lower right", frameon=True, fontsize=11,
                            framealpha=1.0, edgecolor="#888888", fancybox=False,
                            borderpad=0.6, handletextpad=0.6)
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    # Drop inline segment labels that don't fit their segment (measured in
    # display coordinates after layout).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, t, seg_left, seg_width in _fit_labels:
        text_w = t.get_window_extent(renderer).width
        x0 = ax.transData.transform((seg_left, 0))[0]
        x1 = ax.transData.transform((seg_left + seg_width, 0))[0]
        if text_w > (x1 - x0) - 6:
            t.remove()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png/.pdf")


def pearson(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y))


def figure_liking_vs_agentic(out_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharex=True)
    for ax, (slug, _mv) in zip(axes, MODELS):
        m = pd.read_csv(f"results/{slug}/correlations/measures.csv")
        m = m.dropna(subset=["liking_mean", "agentic_mean"])
        ax.errorbar(m["liking_mean"], m["agentic_mean"],
                    xerr=m["liking_sem"].fillna(0), yerr=m["agentic_sem"].fillna(0),
                    fmt="o", markersize=7, color="#0072B2", ecolor="#BBBBBB",
                    elinewidth=1, capsize=0, zorder=3)
        from adjustText import adjust_text
        texts = [
            ax.text(row.liking_mean, row.agentic_mean, outcome_label(row.payout),
                    fontsize=8.5, color="#444444")
            for row in m.itertuples(index=False)
        ]
        adjust_text(texts, ax=ax, expand=(1.5, 2.1), force_text=(0.7, 1.3), max_move=200,
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
        r = pearson(m["liking_mean"], m["agentic_mean"])
        ax.set_title(f"{slug}  (Pearson r = {r:.2f})", fontsize=13)
        ax.set_xlabel("Stated liking")
        ax.grid(True, alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-4, 104)
    axes[0].set_ylabel("Mean send attempts")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png/.pdf")


MODEL_COLORS = {"claude-opus-4-8": "#0072B2", "gpt-5.5": "#E69F00", "gemini-3.1-pro": "#009E73"}


def figure_refusals(out_base: Path) -> None:
    """Refusal rate (share of rollouts with 0 send attempts) per recipient,
    grouped bars for the three models on a shared 0-100% axis."""
    rates = refusal_rates()
    slugs = [s for s, _ in MODELS]
    order = rates[slugs].mean(axis=1).sort_values(ascending=True).index.tolist()
    y = np.arange(len(order))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, slug in enumerate(slugs):
        vals = rates.loc[order, slug].to_numpy() * 100
        ax.barh(y + (i - 1) * h, vals, height=h, color=MODEL_COLORS[slug],
                edgecolor="white", linewidth=0.5, label=slug, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([outcome_label(o) for o in order])
    ax.set_xlabel("Refusal rate (% of rollouts with 0 send attempts)")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png/.pdf")


def main() -> None:
    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_agentic(out_dir / "fig_agentic_persistence")
    figure_agentic(out_dir / "fig_agentic_persistence_median", show_median=True)
    figure_refusals(out_dir / "fig_refusals")
    figure_liking_vs_agentic(out_dir / "fig_liking_vs_agentic")


if __name__ == "__main__":
    main()
