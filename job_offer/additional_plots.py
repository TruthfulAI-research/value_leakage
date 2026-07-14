# %%
"""Additional Job Offer plots (not in the paper body).

Currently: the DISTRIBUTION of per-answer scores for a single model in
the three main scenario groups (own company = current job / own company
= job offer / own company not involved), as one violin per scenario.
The per-answer score is the mean 0-100 stay<->leave judge score over the
papers extracted from that answer -- exactly the rollout-level quantity
the paper's bar plots average (see eval.py).

Implementation note: this script IMPORTS eval.py, which executes its
whole (fully cached) pipeline and re-saves the paper figures as a side
effect -- cheap (~seconds) and byte-identical, but be aware. It reuses
eval's `_final_rollout` dataframe (one row per valid answer with its
mean score and scenario group) and the shared group colors/labels.

Figure: overleaf/figures/job_offer/answer_score_violins_<model>.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

import job_offer.eval as ev

MODEL_KEY = "claude-fable-5-high"


# %% --- Violin plot of per-answer scores by scenario group ---

def plot_answer_score_violins(model_key, filename=None):
    sub = ev._final_rollout[ev._final_rollout["model"] == model_key]
    data = [sub.loc[sub["group"] == g, "score"].to_numpy()
            for g in ev.GROUP_ORDER]

    fig, ax = plt.subplots(figsize=(7, 3))
    positions = np.arange(len(ev.GROUP_ORDER))
    parts = ax.violinplot(
        [d for d in data if len(d)],
        positions=[p for p, d in zip(positions, data) if len(d)],
        widths=0.8, showmedians=False, showextrema=False,
    )
    drawn = [g for g, d in zip(ev.GROUP_ORDER, data) if len(d)]
    for body, g in zip(parts["bodies"], drawn):
        body.set_facecolor(ev.FINAL_GROUP_COLORS[g])
        body.set_alpha(0.7)

    # Mean of each distribution as a short horizontal black line.
    for pos, d in zip(positions, data):
        if len(d):
            ax.hlines(d.mean(), pos - 0.25, pos + 0.25,
                      color="black", linewidth=1.5, zorder=3)
    ax.plot([], [], color="black", linewidth=1.5, label="mean")

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [ev.FINAL_GROUP_LABELS[g].replace(" = ", " =\n")
         .replace(" not ", "\nnot ") + f"\n(n={len(d)})"
         for g, d in zip(ev.GROUP_ORDER, data)],
        fontsize=9,
    )
    ev._style_final_yaxis(ax)
    ax.set_ylim(10, 60)
    ax.set_yticks(range(10, 61, 10))
    ax.legend(loc="lower right")
    # ax.set_title(model_key)
    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
    plt.show()


plot_answer_score_violins(
    MODEL_KEY,
    filename=str(ev._figures_dir / f"answer_score_violins_{MODEL_KEY}.pdf"),
)

# %%
