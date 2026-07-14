# %%
"""Mean-probability plots for the motivated-reasoning experiments.

Loads (or samples) the bubble_v1 and marcus_v1 rollouts, extracts
probabilities with the Sonnet judge, and plots per-model bar groups
(baseline / own company / other companies). Experiment definitions, caches,
and pipeline live in `motivated_reasoning.py` (same directory).

With CACHE_ONLY=False, any model missing rollouts (e.g. claude-opus-4.8-max
before its first run) is sampled live: 2 experiments x 3 paraphrases x 7
conditions x 1000 rollouts = 42,000 requests per model.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_style import ANNOT_FS, HEADER_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from shared.models import MODELS
from ai_company_questions.motivated_reasoning import (
    COMPANIES,
    CONDITION_GROUPS,
    EXPERIMENTS,
    MODEL_GROUPS,
    MODEL_KEYS,
    ORIGIN_COMPANY,
    add_condition_group,
    compute_bias_metrics,
    get_experiment_df,
    pair_gap_table,
    pair_significance,
    summarize_experiment,
)
from shared.cluster_stats import equal_weight_summary

# Sampling enabled: running this notebook fills in any model with missing
# rollouts — currently claude-opus-4.8-max, whose first run is 2 experiments
# x 3 paraphrases x 7 conditions x 1000 rollouts = 42,000 max-effort
# requests plus ~42k Sonnet judge calls. Everything already migrated is a
# cache hit. Set to True for a pure cache read (no API calls).
CACHE_ONLY = True
# Drop models whose rollout cache is incomplete instead of raising, so the
# plot works before claude-opus-4.8-max has been sampled. Only effective
# together with CACHE_ONLY; with CACHE_ONLY=False misses are sampled live.
SKIP_MISSING_MODELS = True

# Figure destination: every plot is written (PDF only) into the ai_bubble section
# of the gitignored Overleaf clone -- the single figure output location.
FIG_DIR = Path(__file__).resolve().parents[1] / "overleaf" / "figures" / "ai_bubble"

GROUP_LABELS = {"baseline": "baseline", "origin": "Own company",
                "other": "Other companies"}
# Project-standard palette (matches giraffes/plot_biases.py DIR_COLORS): the
# highlighted own-company bar is blue, the neutral baseline/other bars are grey.
# No orange/red/green (those carry Claude / good-bad meaning elsewhere).
GROUP_COLORS = {"baseline": "#7f7f7f", "origin": "#1f77b4",
                "other": "#7f7f7f"}
TITLES = {
    "bubble_v1": "AI Bubble Mean Probability by Company",
    "marcus_v1": "AGI Tweet Mean Probability by Company",
}
# Short titles for the side-by-side figure.
SHORT_TITLES = {"bubble_v1": "AI Bubble", "marcus_v1": "AGI Tweet"}
# Key for the significance stars drawn over the origin-vs-other pair. Pure
# star definitions -- the test itself (equal-weighted mean of the
# per-paraphrase paired gaps, rollout-level SEs, paraphrases fixed) is
# described in the paper captions, not on the figure. In-axes legends can
# pass a shorter ``sig_caption`` listing only the levels that occur.
SIG_CAPTION = "* p<.05   ** p<.01   *** p<.001"
# Which y-direction is "favorable to the company" per experiment: a bursting
# bubble is bad news (lower is favorable), achieving AGI is good news (higher
# is favorable). Drives the preference arrow on the combined figure only;
# experiments absent from this map get no arrow.
FAVORABLE_UP = {"bubble_v1": False, "marcus_v1": True}

# Absolute distance (inches) of the violin "more favorable" arrow and y-label
# from the axes, held constant across figure widths so the gap to the ticks
# stays tight on both the ~7in main figure and the ~13in appendix figures.
# (The bar overview figure keeps its own axes-fraction defaults.)
ARROW_OFFSET_IN = 0.38
LABEL_OFFSET_IN = 0.55


def plot_mean_probability_per_model(summary_df, experiment_name, model_groups,
                                    fname=None, show_group_headers=True,
                                    figsize=None, include_baseline=True,
                                    title=None, ax=None, show_legend=True,
                                    ymax=None, sig=None):
    """Grouped bars: one group per model; bars = baseline / origin / other.

    Layout follows ``giraffes/plot_biases.py::plot_mean_bias_per_model``:
    models are arranged in model-family groups with dashed dividers and a
    bold family label above each group (suppress with
    ``show_group_headers=False``). Unlike plot_biases, color encodes the
    condition group (baseline / origin / other), not the family.

    ``figsize``: optional (width, height) in inches. If None (default), the
    width auto-scales with the number of models.

    ``ax``: draw onto an existing axis (for multi-panel figures). When given,
    the function does not create/save/show a figure and leaves the legend to
    the caller unless ``show_legend`` is True.

    ``sig``: model -> (p, stars) from ``pair_significance`` (which needs the
    per-rollout frame, so it cannot be recomputed from ``summary_df`` here).
    None -> no significance brackets.
    """
    exp = EXPERIMENTS[experiment_name]
    groups = (CONDITION_GROUPS if include_baseline
              else [g for g in CONDITION_GROUPS if g != "baseline"])
    plotted = set(summary_df["model"])
    nonempty = [(label, [mk for mk in group if mk in plotted])
                for label, group in model_groups]
    nonempty = [(label, group) for label, group in nonempty if group]
    present = [mk for _, group in nonempty for mk in group]

    n_groups = len(groups)
    # When the (grey) baseline bar is shown, recolour "other companies" with the
    # project's third standard hue (purple, the grey/blue/purple trio from
    # plot_biases.DIR_COLORS) so it doesn't collide with the grey baseline.
    # Without baseline, "other" stays grey so the highlighted own-company bar
    # (blue) reads against a neutral comparison.
    group_colors = dict(GROUP_COLORS)
    if include_baseline:
        group_colors["other"] = "#9467bd"
    # Bars fill ~80% of each model's slot, leaving a small inter-model gap
    # regardless of how many condition bars there are (2 vs 3).
    width = 0.8 / n_groups
    owns_fig = ax is None
    if owns_fig:
        if figsize is None:
            figsize = (max(4, 1.0 * len(present) + 1.2), 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    bar_tops = []
    for j, group in enumerate(groups):
        xs, vals, errs = [], [], []
        for i, mk in enumerate(present):
            sub = summary_df[(summary_df["model"] == mk)
                             & (summary_df["condition_group"] == group)]
            xs.append(i + (j - (n_groups - 1) / 2) * width)
            vals.append(sub["mean"].iloc[0] if len(sub) else float("nan"))
            errs.append(sub["ci95"].iloc[0] if len(sub) else 0.0)
        heights = [0.0 if pd.isna(v) else v for v in vals]
        ax.bar(xs, heights, width=width, yerr=errs,
               color=group_colors[group], label=GROUP_LABELS[group],
               edgecolor="white", linewidth=0.5, ecolor="black", capsize=3,
               error_kw={"linewidth": 1.0})
        for x, v, e in zip(xs, vals, errs):
            if pd.notna(v):
                ax.text(x, v + e + 0.005, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=VALUE_FS)
                bar_tops.append(v + e)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(
        [MODELS[mk].get("display_name", mk) for mk in present],
        rotation=20, ha="right",
    )
    ax.set_ylabel(exp["ylabel"])
    if title is not None:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if "origin" in groups and "other" in groups:
            handles = handles + [Line2D([], [], linestyle="none", marker="none")]
            labels = labels + [SIG_CAPTION]
        ax.legend(handles, labels, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0))

    if ymax is None:
        peak = max(bar_tops) if bar_tops else 1.0
        # Extra headroom (vs the pre-stars 0.15/0.08) leaves room for the
        # significance brackets that sit above the tallest pair.
        ymax = peak + (0.17 if show_group_headers else 0.12)
    ax.set_ylim(0, ymax)

    # Significance stars over the own-company-vs-other-companies pair (the
    # fixed-cells test behind pair_significance; key = SIG_CAPTION). The
    # bracket spans only those two bars — that difference is the
    # own-company-bias effect this plot is about.
    if sig is not None and "origin" in groups and "other" in groups:
        j_o, j_t = groups.index("origin"), groups.index("other")
        for i, mk in enumerate(present):
            stars = sig.get(mk, (float("nan"), ""))[1]
            if not stars:
                continue
            o = summary_df[(summary_df["model"] == mk)
                           & (summary_df["condition_group"] == "origin")]
            t = summary_df[(summary_df["model"] == mk)
                           & (summary_df["condition_group"] == "other")]
            x_o = i + (j_o - (n_groups - 1) / 2) * width
            x_t = i + (j_t - (n_groups - 1) / 2) * width
            top = max(o["mean"].iloc[0] + o["ci95"].iloc[0],
                      t["mean"].iloc[0] + t["ci95"].iloc[0])
            y = top + 0.045
            tick = 0.012
            ax.plot([x_o, x_o, x_t, x_t], [y - tick, y, y, y - tick],
                    color="black", linewidth=1.0)
            ax.text((x_o + x_t) / 2, y, stars, ha="center", va="bottom",
                    fontsize=ANNOT_FS)

    if show_group_headers:
        cumulative = 0
        for label, group in nonempty:
            start = cumulative
            end = cumulative + len(group) - 1
            center = (start + end) / 2
            cumulative += len(group)
            if cumulative < len(present):
                ax.axvline(cumulative - 0.5, color="black",
                           linewidth=0.8, alpha=0.5, linestyle="--")
            ax.text(center, ymax - 0.01, label, ha="center", va="top",
                    fontsize=HEADER_FS, fontweight="bold")

    if owns_fig:
        if fname is not None:
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.tight_layout()
        plt.show()


def plot_violin_probability_per_model(df, experiment_name, model_groups,
                                      fname=None, show_group_headers=True,
                                      figsize=None, title=None, ax=None,
                                      show_legend=True, include_baseline=False,
                                      show_pref_arrow=True, legend_outside=False,
                                      ytop=1.12, ybottom=0.0,
                                      headers_above_top=False,
                                      means_below_families=(),
                                      sig_caption=SIG_CAPTION):
    """Violins of the per-rollout probability spread: one slot per model, with
    the mean marked.

    The distribution counterpart to ``plot_mean_probability_per_model``: instead
    of a single mean bar per condition it draws the full kernel-density violin of
    the judged probabilities. By default two violins (own company vs other
    companies, baseline omitted); with ``include_baseline=True`` three
    (baseline / own / other). ``df`` is the per-rollout frame from
    ``get_experiment_df`` (needs columns model, paraphrase, condition, p). The
    layout (family groups, dashed dividers, bold headers, significance stars,
    "more favorable" preference arrow, palette) matches the bar plot so the two
    are drop-in comparable. The mean marked on each violin (line + printed
    number) is the equal-weighted mean over the fixed paraphrase x condition
    cells -- the same aggregation as the bar plots (``summarize_experiment``) --
    so the two figure families print identical numbers; the violin body itself
    remains the pooled kept-after-judge distribution.
    """
    exp = EXPERIMENTS[experiment_name]
    # Two violins (own vs other) by default; three (baseline / own / other)
    # with include_baseline, matching plot_mean_probability_per_model.
    groups = (CONDITION_GROUPS if include_baseline
              else [g for g in CONDITION_GROUPS if g != "baseline"])
    # Recolour "other" to the project's third hue (purple) when the grey
    # baseline violin is shown, so the two neutral conditions don't collide.
    group_colors = dict(GROUP_COLORS)
    if include_baseline:
        group_colors["other"] = "#9467bd"
    df = add_condition_group(df.copy()).dropna(subset=["p"])
    plotted = set(df["model"])
    nonempty = [(label, [mk for mk in group if mk in plotted])
                for label, group in model_groups]
    nonempty = [(label, group) for label, group in nonempty if group]
    present = [mk for _, group in nonempty for mk in group]

    n_groups = len(groups)
    width = 0.8 / n_groups
    owns_fig = ax is None
    if owns_fig:
        if figsize is None:
            figsize = (max(4, 1.0 * len(present) + 1.2), 4.5)
        fig, ax = plt.subplots(figsize=figsize)

    # One violin per (model, group); colour = condition group (same palette as
    # the bar plot), mean marked with a short black line. `series` keeps each
    # array around for the significance brackets below. The marked mean is the
    # equal-weighted mean over the paraphrase x condition cells (identical to
    # the bar plots), not the pooled mean of the violin's rollouts, so unequal
    # judge-parse survival across cells does not reweight it.
    eq_mean = {
        (mk, group): equal_weight_summary(
            df[(df["model"] == mk) & (df["condition_group"] == group)],
            "p", ["paraphrase", "condition"])["mean"]
        for group in groups for mk in present
    }
    series = {}
    for j, group in enumerate(groups):
        positions, datasets, cell_models = [], [], []
        for i, mk in enumerate(present):
            vals = df[(df["model"] == mk)
                      & (df["condition_group"] == group)]["p"].to_numpy()
            series[(mk, group)] = vals
            if len(vals):
                positions.append(i + (j - (n_groups - 1) / 2) * width)
                datasets.append(vals)
                cell_models.append(mk)
        if not datasets:
            continue
        parts = ax.violinplot(datasets, positions=positions,
                              widths=width * 0.9, showmeans=False,
                              showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(group_colors[group])
            body.set_edgecolor("white")
            body.set_alpha(0.85)
        # Draw the mean lines a bit wider than the violin body so the mean
        # reads clearly (same stretched geometry the ``showmeans`` lines had:
        # centre +- 0.65 x the violin width).
        half = width * 0.9 * 0.65
        for pos, mk in zip(positions, cell_models):
            m = eq_mean[(mk, group)]
            ax.plot([pos - half, pos + half], [m, m],
                    color="black", linewidth=2)

    # Print the numeric mean in black just above each violin's top (below the
    # significance bracket, whose height leaves room for it), so the violins
    # carry the same at-a-glance numbers as the bar plots without the text
    # sitting on the violin bodies. Families listed in
    # ``means_below_families`` get the number below the violin's bottom
    # instead -- for groups whose violin tops leave no clean space above.
    family_of = {mk: label for label, group in nonempty for mk in group}
    for j, group in enumerate(groups):
        for i, mk in enumerate(present):
            vals = series.get((mk, group))
            if vals is None or not len(vals):
                continue
            pos = i + (j - (n_groups - 1) / 2) * width
            m = eq_mean[(mk, group)]
            if family_of.get(mk) in means_below_families:
                ax.text(pos, float(vals.min()) - 0.003, f"{m:.2f}",
                        ha="center", va="top", fontsize=VALUE_FS,
                        color="black")
            else:
                ax.text(pos, float(vals.max()) + 0.003, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=VALUE_FS,
                        color="black")

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(
        [MODELS[mk].get("display_name", mk) for mk in present],
        rotation=20, ha="right",
    )
    # The violins show the full per-rollout distribution, not a mean, so drop the
    # "Mean " prefix the shared ylabel carries for the bar plots.
    ax.set_ylabel(exp["ylabel"].removeprefix("Mean "))
    if title is not None:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    # Probabilities live in [0, 1]; the KDE can spill slightly past 1, so leave
    # a little headroom for it plus the family headers / significance brackets.
    # ``ytop`` controls that headroom: the default 1.12 keeps the appendix
    # figures unchanged; the main-text violin passes a tighter value so the
    # probability axis tops out close to 1 with little empty margin.
    # ``ybottom`` crops empty space below the violins (the main-text bubble_v1
    # violins all sit above ~0.3 with only hair-thin outlier tails lower); it
    # must stay 0 wherever violin bodies reach low values (e.g. the AGI-tweet
    # claude-opus-4.6-max violins center around 0.16).
    ax.set_ylim(ybottom, ytop)

    # "more favorable" preference arrow on the left, pointing the
    # company-favorable way (down for the bubble, up for AGI), matching the
    # overview bar figure. Experiments absent from FAVORABLE_UP get no arrow.
    if show_pref_arrow:
        up_better = FAVORABLE_UP.get(experiment_name)
        if up_better is not None:
            # Place the arrow/label a constant absolute distance left of the
            # ticks regardless of figure width: a fixed axes-fraction offset
            # leaves a tiny gap on a narrow figure but a huge one on the wide
            # (~13in) appendix figures. Convert a target inch offset to the axes
            # fraction via the (pre-layout) axes width.
            fig_w = ax.figure.get_size_inches()[0]
            ax_w_in = max(1e-6, ax.get_position().width * fig_w)
            _add_preference_arrow(
                ax, up_better,
                x_arrow=-ARROW_OFFSET_IN / ax_w_in,
                x_label=-LABEL_OFFSET_IN / ax_w_in,
            )

    if show_legend:
        handles = [Patch(facecolor=group_colors[g], edgecolor="white")
                   for g in groups]
        labels = [GROUP_LABELS[g] for g in groups]
        if "origin" in groups and "other" in groups:
            handles.append(Line2D([], [], linestyle="none", marker="none"))
            # ``sig_caption`` lets tight in-axes legends use a shorter key
            # (the full test description lives in the paper caption); the
            # outside legends keep the self-contained SIG_CAPTION default.
            labels.append(sig_caption)
        if legend_outside:
            # One horizontal row centred above the axes, so it never overlaps
            # the violins (the wide appendix figures). The owns_fig
            # finalisation reserves a top margin for it inside the figure so
            # bbox_inches="tight" keeps everything within figsize. With
            # headers_above_top the family labels occupy the strip right above
            # the spine, so the legend moves up to clear them.
            ax.legend(handles, labels, loc="lower center",
                      bbox_to_anchor=(0.5, 1.10 if headers_above_top else 1.01),
                      ncol=len(labels))
        else:
            # Inside the axes, centred roughly under the GPT group. When the
            # bold family headers sit inside the axes the legend drops a
            # little to clear them; with headers_above_top it can hug the top
            # spine. (Used only by the main-text bubble_v1 violin; the
            # appendix figures use legend_outside. The x-anchor is tuned for
            # the bubble_v1 layout, and that call passes a short
            # ``sig_caption`` so the box clears the Gemini mean labels.)
            ax.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.66, 0.9975 if headers_above_top
                                      else 0.92))

    # Significance stars over the own-vs-other pair (same fixed-cells test
    # as the bar plot); the bracket sits above the taller violin of the
    # pair, high enough that its down-ticks clear the mean numbers on the
    # violin tops.
    if "origin" in groups and "other" in groups:
        j_o, j_t = groups.index("origin"), groups.index("other")
        sig = pair_significance(df)
        for i, mk in enumerate(present):
            stars = sig.get(mk, (float("nan"), ""))[1]
            o, t = series.get((mk, "origin")), series.get((mk, "other"))
            if not stars or o is None or t is None or not len(o) or not len(t):
                continue
            x_o = i + (j_o - (n_groups - 1) / 2) * width
            x_t = i + (j_t - (n_groups - 1) / 2) * width
            y = min(max(float(o.max()), float(t.max())) + 0.068, ytop - 0.08)
            tick = 0.012
            ax.plot([x_o, x_o, x_t, x_t], [y - tick, y, y, y - tick],
                    color="black", linewidth=1.0)
            ax.text((x_o + x_t) / 2, y, stars, ha="center", va="bottom",
                    fontsize=ANNOT_FS)

    if show_group_headers:
        cumulative = 0
        for label, group in nonempty:
            start = cumulative
            end = cumulative + len(group) - 1
            center = (start + end) / 2
            cumulative += len(group)
            if cumulative < len(present):
                ax.axvline(cumulative - 0.5, color="black",
                           linewidth=0.8, alpha=0.5, linestyle="--")
            if headers_above_top:
                # Sit the labels just above the top spine (outside the axes) so
                # the data axis can top out exactly at 1.0 with no inside margin.
                # bbox_inches="tight" keeps them in the saved figure.
                ax.text(center, ytop + 0.01, label, ha="center", va="bottom",
                        fontsize=HEADER_FS, fontweight="bold", clip_on=False)
            else:
                ax.text(center, ytop - 0.01, label, ha="center", va="top",
                        fontsize=HEADER_FS, fontweight="bold")

    if owns_fig:
        if legend_outside:
            # Lay arrow/y-label (left) and the horizontal legend row (top)
            # WITHIN the figure bounds, so bbox_inches="tight" never extends the
            # saved width past figsize -- it only trims, keeping the total
            # <= ~13in. (Don't use tight_layout here: it ignores the
            # out-of-axes legend/arrow.)
            fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.86)
            if fname is not None:
                Path(fname).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.show()
        else:
            if fname is not None:
                Path(fname).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.tight_layout()
            plt.show()


def _add_preference_arrow(ax, up_better, *, x_arrow=-0.085, x_label=-0.12):
    """One real arrow + a 'more favorable' label between the ticks and y-label.

    The arrow points the company-favorable way (up for AGI when
    ``up_better=True``, down for the bubble when False); the label is centered
    on the axis, both in the default text colour. The arrow and label sit just
    left of the ticks, with the y-label pulled in just outboard of them.

    ``x_arrow``/``x_label`` are the arrow and y-label x-positions in axes
    fraction. The defaults suit the ~6-7in bar/violin panels; callers on much
    wider figures pass values closer to 0 so the gap to the ticks doesn't blow
    up (axes-fraction offsets scale with the axes width).
    """
    x = x_arrow
    # Stack label + arrow on the axis: label on the far side from favorable,
    # arrow pointing the favorable way. The arrow tail sits well clear of the
    # rotated label text (which is fairly tall) so the two don't touch.
    if up_better:                       # AGI: arrow up (top), label below
        text_y, y_tail, y_head = 0.40, 0.58, 0.74
    else:                               # bubble: label on top, arrow down
        text_y, y_tail, y_head = 0.60, 0.42, 0.26
    ax.text(x, text_y, "more favorable", transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=plt.rcParams["ytick.labelsize"])
    ax.annotate("", xy=(x, y_head), xytext=(x, y_tail),
                xycoords="axes fraction", annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.yaxis.set_label_coords(x_label, 0.5)


def plot_experiments_side_by_side(summaries, model_groups, *,
                                  include_baseline=False, fname=None,
                                  sigs=None):
    """One figure, the experiments as side-by-side panels (short titles).

    ``summaries`` maps experiment_name -> summary_df (from
    ``summarize_experiment``); ``sigs`` maps experiment_name -> the
    ``pair_significance`` dict for its significance brackets (None -> no
    brackets). A single shared legend is drawn for the whole figure.
    ``include_baseline=False`` (default) shows origin vs other only.
    """
    experiments = [e for e in EXPERIMENTS if e in summaries]
    n = len(experiments)
    # Per-panel width; kept in sync with covertness.plot_bias_decomposition
    # so the two side-by-side figures come out the same width.
    per_w = max(4.0, 0.7 * len(MODEL_KEYS) + 1.6)
    fig, axes = plt.subplots(
        1, n, figsize=(per_w * n, 4.5), squeeze=False,
    )
    # Shared y-scale: tallest (bar + CI) across both panels, plus room for
    # the bold family headers.
    groups = (CONDITION_GROUPS if include_baseline
              else [g for g in CONDITION_GROUPS if g != "baseline"])
    peak = 0.0
    for s in summaries.values():
        tops = (s[s["condition_group"].isin(groups)]["mean"]
                + s[s["condition_group"].isin(groups)]["ci95"]).dropna()
        if len(tops):
            peak = max(peak, float(tops.max()))
    shared_ymax = peak + 0.17
    for ax, experiment_name in zip(axes[0], experiments):
        plot_mean_probability_per_model(
            summaries[experiment_name], experiment_name, model_groups,
            include_baseline=include_baseline,
            title=SHORT_TITLES.get(experiment_name, experiment_name),
            ax=ax, show_legend=False, ymax=shared_ymax,
            sig=None if sigs is None else sigs.get(experiment_name),
        )
        up_better = FAVORABLE_UP.get(experiment_name)
        if up_better is not None:
            _add_preference_arrow(ax, up_better)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Ride the star key in the same centered row, after the colour entries
    # (invisible handle, so it reads as plain text).
    if "origin" in groups and "other" in groups:
        handles = handles + [Line2D([], [], linestyle="none", marker="none")]
        labels = labels + [SIG_CAPTION]
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=len(labels))
    plt.tight_layout()
    # Reserve room left of each panel (figure margin + inter-panel gap) for the
    # preference arrow/label and the y-label; overrides tight_layout, which
    # doesn't account for the out-of-axes text. wspace is the floor that still
    # fits the right panel's y-label/arrow without hitting the left panel.
    fig.subplots_adjust(left=0.10, wspace=0.22)
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_company_bars(df, experiment_name, model_key, *, fname=None,
                          ax=None, title=None, show_legend=True):
    """Per-model, all-company bars: baseline + one bar per company.

    The per-model breakdown behind the cross-model
    ``plot_mean_probability_per_model`` overview: for a single model it shows
    the mean judged probability for the baseline and each of the six company
    conditions, equal-weighting the three paraphrases (so judge-parse
    survival doesn't reweight them); the 95% CI combines the within-paraphrase
    rollout SEs with the paraphrases held fixed
    (``cluster_stats.equal_weight_summary``). The own company is coloured blue
    and tagged ``(own)``, the baseline grey, and the five other companies
    purple, matching the grey/blue/purple palette of the overview bars.
    ``df`` is the per-rollout frame from ``get_experiment_df`` (needs columns
    model, paraphrase, condition, p).
    """
    exp = EXPERIMENTS[experiment_name]
    origin = ORIGIN_COMPANY.get(model_key)
    order = ["baseline"] + COMPANIES
    other_color = "#9467bd"
    sub = df[df["model"] == model_key].dropna(subset=["p"])

    xs, heights, errs, colors, labels = [], [], [], [], []
    for i, cond in enumerate(order):
        s = equal_weight_summary(sub[sub["condition"] == cond], "p",
                                 "paraphrase")
        mean, ci = s["mean"], (s["ci95"] if s["n"] else 0.0)
        xs.append(i)
        heights.append(0.0 if pd.isna(mean) else mean)
        errs.append(ci)
        if cond == "baseline":
            colors.append(GROUP_COLORS["baseline"])
            labels.append("baseline")
        elif cond == origin:
            colors.append(GROUP_COLORS["origin"])
            labels.append(f"{cond}\n(own)")
        else:
            colors.append(other_color)
            labels.append(cond)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(order) + 1.5), 3.5))

    ax.bar(xs, heights, width=0.7, yerr=errs, color=colors,
           edgecolor="white", linewidth=0.5, ecolor="black", capsize=3,
           error_kw={"linewidth": 1.0})
    bar_tops = []
    for x, h, e in zip(xs, heights, errs):
        if h or e:
            ax.text(x, h + e + 0.005, f"{h:.2f}", ha="center", va="bottom",
                    fontsize=VALUE_FS)
            bar_tops.append(h + e)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(exp["ylabel"])
    ax.set_title(title if title is not None
                 else MODELS[model_key].get("display_name", model_key))
    ax.grid(True, axis="y", alpha=0.3)
    peak = max(bar_tops) if bar_tops else 1.0
    ax.set_ylim(0, peak + 0.08)

    if show_legend:
        handles = [Patch(facecolor=GROUP_COLORS["baseline"], edgecolor="white"),
                   Patch(facecolor=GROUP_COLORS["origin"], edgecolor="white"),
                   Patch(facecolor=other_color, edgecolor="white")]
        ax.legend(handles, ["baseline", "Own company", "Other companies"],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))

    if owns_fig:
        if fname is not None:
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.tight_layout()
        plt.show()


# %%
experiment_dfs = {}
for experiment_name in EXPERIMENTS:
    experiment_dfs[experiment_name] = get_experiment_df(
        experiment_name, MODEL_KEYS,
        cache_only=CACHE_ONLY, skip_missing=SKIP_MISSING_MODELS,
    )

# %%
summaries = {}
sigs = {}
for experiment_name, df in experiment_dfs.items():
    summary = summarize_experiment(df)
    summaries[experiment_name] = summary
    sigs[experiment_name] = pair_significance(df)
    print(f"\n=== {experiment_name} ===")
    print(summary.to_string(index=False))
    # The numbers behind the significance stars: per-paraphrase paired
    # origin-vs-other gaps (with rollout-level SEs) and their equal-weighted
    # combination (fixed-cells z test).
    print(f"\n=== {experiment_name} (paraphrase-paired gaps) ===")
    print(pair_gap_table(df).to_string(index=False))
    # Per-experiment cross-model mean bars are not used in the paper (the
    # side-by-side overview below is the only mean-bar figure we keep). Kept
    # here, commented out, for ad-hoc use.
    # plot_mean_probability_per_model(summary, experiment_name, MODEL_GROUPS,
    #                                 title=None, sig=sigs[experiment_name],
    #                                 fname=FIG_DIR / f"mean_probability_{experiment_name}.pdf")
    # plot_mean_probability_per_model(summary, experiment_name, MODEL_GROUPS,
    #                                 title=None, include_baseline=False,
    #                                 sig=sigs[experiment_name],
    #                                 fname=FIG_DIR / f"mean_probability_{experiment_name}_no_baseline.pdf")

# %%
# PAPER FIGURE (appendix, fig:rlb-headline-bars): both experiments side by side,
# own company vs other companies (short titles, preference arrow).
plot_experiments_side_by_side(summaries, MODEL_GROUPS, sigs=sigs,
                              fname=FIG_DIR / "mean_probability_side_by_side.pdf")

# %%
# PAPER FIGURE (main text, fig:bubble-v1-violin): AI Bubble drawn as violins
# (the full per-rollout probability spread behind each mean), own company vs
# other companies (baseline omitted).
plot_violin_probability_per_model(
    experiment_dfs["bubble_v1"], "bubble_v1", MODEL_GROUPS,
    include_baseline=False,
    # Short key: the in-axes legend would cover the Gemini mean labels with
    # the full three-tier SIG_CAPTION; list only the levels that occur in
    # this panel (recheck when the data changes). The LaTeX caption spells
    # out the test.
    sig_caption="*** p<0.001",
    ytop=0.9,  # crop above the violins too (tallest body tops out at 0.75)
    ybottom=0.1,  # crop the empty space below the violins (bodies sit > 0.3)
    # With the 0.1-0.9 crop the Gemini family header no longer fits inside
    # the axes (its group's significance stars reach the top), so the family
    # labels float just above the top spine. That also lets the in-axes
    # legend hug the top spine, high enough to clear the GPT mean numbers.
    headers_above_top=True,
    fname=FIG_DIR / "violin_probability_bubble_v1.pdf",
)

# %%
# PAPER FIGURES (appendix, fig:rlb-cross-violin / fig:gm-cross-violin): the same
# violins but with the baseline condition shown too (baseline / own / other),
# one figure per experiment.
for experiment_name in EXPERIMENTS:
    plot_violin_probability_per_model(
        experiment_dfs[experiment_name], experiment_name, MODEL_GROUPS,
        include_baseline=True,
        # Slight headroom above 1: the AGI-tweet gemini violins reach 0.85, and
        # with the mean numbers on the violin tops the significance stars would
        # otherwise run into the bold family headers.
        ytop=1.06,
        figsize=(13, 4.5),  # wider than the auto width (~7.2 for 6 models)
        legend_outside=True,
        fname=FIG_DIR / f"violin_probability_{experiment_name}_with_baseline.pdf",
    )

# %%
# PAPER FIGURES (appendix, fig:rlb-permodel / fig:gm-permodel): per-model,
# all-company bar plots (baseline + six companies, own highlighted) — one PDF
# per model per experiment.
for experiment_name, exp_df in experiment_dfs.items():
    present = set(exp_df["model"])
    for model_key in MODEL_KEYS:
        if model_key not in present:
            continue
        plot_per_company_bars(
            exp_df, experiment_name, model_key,
            show_legend=False,  # legend is explained in the figure caption
            fname=FIG_DIR
            / f"per_company_bars_{experiment_name}_{model_key}.pdf",
        )

# %%
# The AGI Tweet own-vs-other violin (no baseline) is not used in the paper; the
# appendix uses the with-baseline version generated above. Kept commented out.
# plot_violin_probability_per_model(
#     experiment_dfs["marcus_v1"], "marcus_v1", MODEL_GROUPS,
#     include_baseline=False,
#     fname=FIG_DIR / "violin_probability_marcus_v1.pdf",
# )

# %%
# Per-paraphrase breakdown (printed only; the plots equal-weight paraphrases).
for experiment_name, df in experiment_dfs.items():
    by_para = summarize_experiment(df, by_paraphrase=True)
    pivot = by_para.pivot_table(
        index=["model", "paraphrase"], columns="condition_group",
        values="mean",
    )[CONDITION_GROUPS]
    print(f"\n=== {experiment_name} (per paraphrase) ===")
    print(pivot.to_string())

# %%
# Median-threshold bias metric (origin vs pooled other labs).
for experiment_name, df in experiment_dfs.items():
    bias_df = compute_bias_metrics(df, experiment_name)
    print(f"\n=== {experiment_name} (bias metric) ===")
    print(bias_df.to_string(index=False))

# %%
# Per-paraphrase bias metrics: the pooled threshold mixes paraphrase
# distributions with unequal judge-parse survival, and the per-paraphrase
# numbers can flip sign relative to the pooled headline (e.g. opus-4.6-max
# bubble_v1: -0.27 / +0.56 / +0.55 vs pooled +0.20).
for experiment_name, df in experiment_dfs.items():
    bias_df = compute_bias_metrics(df, experiment_name, by_paraphrase=True)
    print(f"\n=== {experiment_name} (bias metric per paraphrase) ===")
    print(bias_df.to_string(index=False))

# %%
