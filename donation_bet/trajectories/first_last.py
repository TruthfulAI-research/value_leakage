# %%
"""One-off experiment: per model, gap between above_good and below_good
in the FIRST trajectory estimate vs the LAST trajectory estimate.

Pulls model completions + estimate-judge results from the isolated
`final_data/` cache (same as `shared/final_scripts/plot_biases.py`),
runs the shared trajectory judge from `janbet.trajectories.data` over
each model's CoT rows, and produces three bar plots per (model):

  * (first vs last) `above_good − below_good` median, in threshold units,
  * (first vs last vs final-answer) bias in the `plot_biases.py` sense
    (`2·P(on_good_side) − 1`, mean over prompts),
  * the same three quantities expressed as fractions
    (`(bias + 1) / 2 = P(on_good_side)`).

Each estimate is divided by its prompt's threshold so prompts share a
scale. Per (prompt, direction) cell we take the MEDIAN first / last over
that cell's trajectories; per prompt we take `above_good − below_good`;
finally we take the MEDIAN across prompts -> one "first" number and one
"last" number per model.

Quick-and-dirty by design.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from donation_bet.trajectories.data import load_model_data


# Mirrors shared/final_scripts/plot_biases.py. Edit freely — this script
# owns its own model list / experiment / cache mode (the sibling
# `trajectories.py` keeps its own copy so they can drift).
MODEL_GROUPS = [
    ("Claude Opus", [
        "claude-opus-4.1",
        "claude-opus-4.5-high",
        "claude-opus-4.6-high",
        "claude-opus-4.6-max",
        "claude-opus-4.7-high",
        "claude-opus-4.7-xhigh",
        "claude-opus-4.7-max",
    ]),
    ("Claude Sonnet", [
        "claude-sonnet-4.5",
        "claude-sonnet-4.6-high",
    ]),
    ("Kimi", [
        "kimi-k2.5",
        "kimi-k2.6",
    ]),
    ("Qwen", [
        "qwen3.5-35",
        "qwen3.6-35",
    ]),
    ("Gemini", [
        "gemini-2.5-pro",
        "gemini-3.1-pro-medium",
        # "gemini-3.1-pro-high",
    ]),
    ("OpenAI", [
        "gpt-5.1-medium",
        "gpt-5.2-medium",
        "gpt-5.4-medium",
        "gpt-5.5-medium",
        "gpt-5.5-high",
    ]),
]
MODELS = [mk for _, group in MODEL_GROUPS for mk in group]

EXPERIMENT = "main_experiment_accurate"
CACHE_ONLY = True


# --- Bias (plot_biases.py style) on first / last / final answer ---
def _on_good_side(direction, value, threshold):
    """Same convention as get_main_dfs._add_good_side."""
    if direction == "below_good":
        return value <= threshold
    if direction == "above_good":
        return value > threshold
    return False


def per_model_three_biases(df, traj_df):
    """Returns (first_bias, last_bias, final_bias). All three use the
    plot_biases.py method: bias_score = 2 * P(on_good_side) - 1 per
    prompt_key (over directional rows), then mean across prompts.

    `final_bias` matches plot_mean_bias_per_model exactly (uses full df).
    `first_bias` / `last_bias` apply the same formula but with the first
    / last extracted trajectory estimate in place of `estimate`. They are
    necessarily over a subset of rows (only rows with a parseable
    trajectory of length >= 1), so the row sets differ slightly from
    `final_bias`'s — that's the inevitable cost of needing a CoT to
    extract a "first" / "last" from.
    """
    # --- final answer bias (plot_biases.py exact replica) ---
    final_per_pk = []
    for pk in df["prompt_key"].dropna().unique():
        pk_df = df[df["prompt_key"] == pk]
        directional = pk_df[pk_df["direction"].isin(["below_good",
                                                     "above_good"])]
        if len(directional) == 0:
            continue
        final_per_pk.append(2 * directional["on_good_side"].mean() - 1)
    final_bias = (float(pd.Series(final_per_pk).mean())
                  if final_per_pk else float("nan"))

    # --- first / last from trajectory rows ---
    rows = []
    for _, r in traj_df.iterrows():
        traj = r["trajectory"]
        if not isinstance(traj, list) or len(traj) == 0:
            continue
        thr = r["threshold"]
        if pd.isna(thr):
            continue
        if r["direction"] not in ("below_good", "above_good"):
            continue
        rows.append({
            "prompt_key": r["prompt_key"],
            "direction": r["direction"],
            "threshold": float(thr),
            "first": float(traj[0]),
            "last": float(traj[-1]),
        })
    fl = pd.DataFrame(rows)
    if fl.empty:
        return float("nan"), float("nan"), final_bias

    first_per_pk = []
    last_per_pk = []
    for _pk, sub in fl.groupby("prompt_key"):
        first_good = sub.apply(
            lambda r: _on_good_side(r["direction"], r["first"], r["threshold"]),
            axis=1,
        )
        last_good = sub.apply(
            lambda r: _on_good_side(r["direction"], r["last"], r["threshold"]),
            axis=1,
        )
        first_per_pk.append(2 * first_good.mean() - 1)
        last_per_pk.append(2 * last_good.mean() - 1)
    first_bias = (float(pd.Series(first_per_pk).mean())
                  if first_per_pk else float("nan"))
    last_bias = (float(pd.Series(last_per_pk).mean())
                 if last_per_pk else float("nan"))
    return first_bias, last_bias, final_bias


# --- Per-model first/last (above - below) gap ---
def first_last_gap(traj_df):
    """Returns (first_diff, last_diff, n_prompts_with_both_directions).

    Each estimate is divided by its prompt's threshold; for each prompt
    we take the MEDIAN across trajectories per direction; for each prompt
    we take above_good - below_good; finally we take the MEDIAN over
    prompts. All in threshold units.
    """
    rows = []
    for _, r in traj_df.iterrows():
        traj = r["trajectory"]
        if not isinstance(traj, list) or len(traj) == 0:
            continue
        thr = r["threshold"]
        if pd.isna(thr) or float(thr) == 0:
            continue
        rows.append({
            "prompt_key": r["prompt_key"],
            "direction": r["direction"],
            "threshold": float(thr),
            "first": float(traj[0]),
            "last": float(traj[-1]),
        })
    fl = pd.DataFrame(rows)
    if fl.empty:
        return float("nan"), float("nan"), 0

    fl["first_norm"] = fl["first"] / fl["threshold"]
    fl["last_norm"] = fl["last"] / fl["threshold"]

    per_pd = (
        fl.groupby(["prompt_key", "direction"])
          .agg(first_n=("first_norm", "median"),
               last_n=("last_norm", "median"))
          .reset_index()
    )

    diffs = []
    for pk, sub in per_pd.groupby("prompt_key"):
        a = sub[sub["direction"] == "above_good"]
        b = sub[sub["direction"] == "below_good"]
        if a.empty or b.empty:
            continue
        diffs.append({
            "prompt_key": pk,
            "first": float(a["first_n"].iloc[0]) - float(b["first_n"].iloc[0]),
            "last": float(a["last_n"].iloc[0]) - float(b["last_n"].iloc[0]),
        })
    if not diffs:
        return float("nan"), float("nan"), 0
    d = pd.DataFrame(diffs)
    return float(d["first"].median()), float(d["last"].median()), len(d)


# --- Endpoints of plot_trajectory_offsets_overall (above − below at u=0/u=1) ---
def per_model_start_end_offset_gap(traj_df, central="median"):
    """Replicates plot_trajectory_offsets_overall from trajectories.py at
    u=0 and u=1, then takes (above_good − below_good) at each endpoint.

    Returns ``(gap_start, gap_end, n_prompts_with_both_directions)`` in
    threshold units. Matches the trajectories plot exactly (so requires
    len>=2 per trajectory and aggregates each direction across prompts
    BEFORE subtracting, since median-of-differences ≠ difference-of-
    medians in general).
    """
    fn = np.median if central == "median" else np.mean

    # Per (prompt, direction): (first_norm, last_norm), both
    # = (central_over_trajectories(endpoint) − threshold) / threshold.
    per_pd = {}
    grouped = traj_df.groupby(["prompt_key", "direction"])
    for (pk, direction), sub in grouped:
        if direction not in ("above_good", "below_good"):
            continue
        kept = sub[sub["trajectory"].apply(
            lambda t: isinstance(t, list) and len(t) >= 2,
        )]
        if kept.empty:
            continue
        thr_vals = kept["threshold"].dropna().unique()
        if len(thr_vals) == 0 or float(thr_vals[0]) == 0:
            continue
        thr = float(thr_vals[0])
        firsts = kept["trajectory"].apply(lambda t: float(t[0])).to_numpy()
        lasts = kept["trajectory"].apply(lambda t: float(t[-1])).to_numpy()
        per_pd[(pk, direction)] = (
            (float(fn(firsts)) - thr) / thr,
            (float(fn(lasts)) - thr) / thr,
        )

    def per_dir(direction, idx):
        return [v[idx] for (_, d), v in per_pd.items() if d == direction]

    above_first = per_dir("above_good", 0)
    below_first = per_dir("below_good", 0)
    above_last = per_dir("above_good", 1)
    below_last = per_dir("below_good", 1)

    if not (above_first and below_first):
        return float("nan"), float("nan"), 0

    gap_start = float(fn(above_first)) - float(fn(below_first))
    gap_end = float(fn(above_last)) - float(fn(below_last))
    pks_above = {pk for (pk, d) in per_pd if d == "above_good"}
    pks_below = {pk for (pk, d) in per_pd if d == "below_good"}
    return gap_start, gap_end, len(pks_above & pks_below)


# %% --- Run ---
results = []
for model in MODELS:
    print(f"\n=== {model} ===")
    df, traj_df, display_name = load_model_data(
        model, experiment=EXPERIMENT, cache_only=CACHE_ONLY,
    )
    print(f"  rows for judge (incl. baseline, for cache parity): {len(traj_df)}")
    traj_df = traj_df[traj_df["direction"].isin(["below_good", "above_good"])]
    first_diff, last_diff, n_prompts = first_last_gap(traj_df)
    print(f"  first_diff = {first_diff:.4f}, last_diff = {last_diff:.4f} "
          f"(across {n_prompts} prompts with both directions)")
    bias_first, bias_last, bias_final = per_model_three_biases(df, traj_df)
    print(f"  bias_first = {bias_first:.4f}, bias_last = {bias_last:.4f}, "
          f"bias_final = {bias_final:.4f}")
    gap_start, gap_end, n_gap = per_model_start_end_offset_gap(traj_df)
    print(f"  offset_gap: start = {gap_start:.4f}, end = {gap_end:.4f} "
          f"(across {n_gap} prompts with both directions)")
    results.append({
        "model_key": model,
        "display": display_name,
        "first": first_diff,
        "last": last_diff,
        "n_prompts": n_prompts,
        "bias_first": bias_first,
        "bias_last": bias_last,
        "bias_final": bias_final,
        "offset_gap_start": gap_start,
        "offset_gap_end": gap_end,
    })


# %%
res_df = pd.DataFrame(results)
print()
print(res_df.to_string(index=False))


# %% --- Plot the three good-side fractions (= (bias + 1) / 2) ---
def plot_three_fracs(res_df):
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(res_df))
    width = 0.27
    frac_first = ((res_df["bias_first"] + 1) / 2).to_numpy()
    frac_last = ((res_df["bias_last"] + 1) / 2).to_numpy()
    frac_final = ((res_df["bias_final"] + 1) / 2).to_numpy()
    ax.bar(x - width, frac_first, width,
           label="First trajectory estimate", color="#1f77b4")
    ax.bar(x, frac_last, width,
           label="Last trajectory estimate", color="#ff7f0e")
    ax.bar(x + width, frac_final, width,
           label="Final answer (plot_biases.py)", color="#2ca02c")
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["display"].tolist(),
                       rotation=30, ha="right")
    ax.set_ylabel("Fraction on good side  (mean over prompts of P(on_good_side))")
    ax.set_title("Fraction on good side per model: "
                 "first vs last trajectory estimate vs final answer")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_three_fracs(res_df)

# # %%

# # %% --- Plot ---
# def plot_first_last_diffs(res_df):
#     fig, ax = plt.subplots(figsize=(11, 6))
#     x = np.arange(len(res_df))
#     width = 0.4
#     ax.bar(x - width / 2, res_df["first"].to_numpy(), width,
#            label="First estimate (above − below)", color="#1f77b4")
#     ax.bar(x + width / 2, res_df["last"].to_numpy(), width,
#            label="Last estimate (above − below)", color="#ff7f0e")
#     ax.axhline(0, color="black", linewidth=0.6)
#     ax.set_xticks(x)
#     ax.set_xticklabels(res_df["display"].tolist(),
#                        rotation=30, ha="right")
#     ax.set_ylabel("(above_good − below_good) median, in threshold units")
#     ax.set_title("Trajectory first vs last estimate: above_good − below_good "
#                  "(median, in threshold units)")
#     ax.legend()
#     ax.grid(True, axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# plot_first_last_diffs(res_df)


# # %% --- Plot the three biases (first / last / final-answer) ---
def plot_three_biases(res_df):
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(res_df))
    width = 0.27
    ax.bar(x - width, res_df["bias_first"].to_numpy(), width,
           label="First trajectory estimate", color="#1f77b4")
    ax.bar(x, res_df["bias_last"].to_numpy(), width,
           label="Last trajectory estimate", color="#ff7f0e")
    ax.bar(x + width, res_df["bias_final"].to_numpy(), width,
           label="Final answer (plot_biases.py)", color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["display"].tolist(),
                       rotation=30, ha="right")
    ax.set_ylabel("Bias  (mean over prompts of 2·P(on_good_side) − 1)")
    ax.set_title("Bias per model: first vs last trajectory estimate "
                 "vs final answer")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_three_biases(res_df)


# %% --- Headline new comparison: first / last-CoT / final bias ---
def plot_first_vs_final_bias(res_df, filename=None):
    """Three dots per model: bias from the FIRST in-CoT estimate, the
    LAST in-CoT estimate, and the FINAL answer."""
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(res_df))
    bias_first = res_df["bias_first"].to_numpy()
    bias_last = res_df["bias_last"].to_numpy()
    bias_final = res_df["bias_final"].to_numpy()

    # Small horizontal offset (~dot diameter) so the three dots for each
    # model sit side-by-side instead of overlapping.
    dx = 0.12

    ax.scatter(x - dx, bias_first, s=70, color="#1f77b4", zorder=3,
               label="first CoT estimate bias")
    ax.scatter(x, bias_last, s=70, color="#ff7f0e", zorder=3,
               label="last CoT estimate bias")
    ax.scatter(x + dx, bias_final, s=70, color="#2ca02c", zorder=3,
               label="final answer bias")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["display"].tolist(),
                       rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("bias", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if filename:
        import os
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
   
    plt.show()


plot_first_vs_final_bias(res_df, filename="figures/trajectories/first_last_final_bias.pdf")


# %% --- Endpoints of the trajectory offset plot: gap at start vs end ---
def plot_offset_gap_start_end(res_df, filename=None):
    """Two dots per model: above_good − below_good evaluated at the
    start (u=0) and end (u=1) of the CoT, in threshold units. This is
    the gap between the orange and blue lines in
    ``plot_trajectory_offsets_overall`` (trajectories.py) at those
    endpoints.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(res_df))
    gap_start = res_df["offset_gap_start"].to_numpy()
    gap_end = res_df["offset_gap_end"].to_numpy()

    dx = 0.10  # ~dot diameter; only two dots per model so a bit tighter.
    ax.scatter(x - dx, gap_start, s=70, color="#1f77b4", zorder=3,
               label="gap at start of CoT")
    ax.scatter(x + dx, gap_end, s=70, color="#ff7f0e", zorder=3,
               label="gap at end of CoT")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["display"].tolist(),
                       rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("(above_good − below_good) / threshold", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if filename:
        import os
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


plot_offset_gap_start_end(
    res_df,
    filename="figures/trajectories/offset_gap_start_end.pdf",
)


# %% --- Same two numbers, but as a 2D scatter ---
def plot_offset_gap_scatter(res_df, filename=None):
    """Same two numbers as plot_offset_gap_start_end, but as a 2D
    scatter: x = gap at start of CoT, y = gap at end of CoT, one
    labelled point per model. Colored by model group. The dashed
    diagonal is y=x (no change during reasoning); points ABOVE it had
    their gap grow inside the CoT, points BELOW had it shrink.
    """
    group_of = {mk: gn for gn, ms in MODEL_GROUPS for mk in ms}
    groups_order = [gn for gn, _ in MODEL_GROUPS]
    cmap = plt.get_cmap("tab10")
    color_of = {gn: cmap(i % 10) for i, gn in enumerate(groups_order)}

    fig, ax = plt.subplots(figsize=(10, 9))
    xs = res_df["offset_gap_start"].to_numpy()
    ys = res_df["offset_gap_end"].to_numpy()

    # Plot one scatter per group so the legend picks up group colors.
    for gn in groups_order:
        mask = np.array([group_of.get(mk) == gn for mk in res_df["model_key"]])
        if not mask.any():
            continue
        ax.scatter(xs[mask], ys[mask], s=80, color=color_of[gn],
                   edgecolor="white", linewidth=0.7, zorder=3, label=gn)

    # Per-point labels, offset slightly so they don't sit on the dot.
    for x, y, name in zip(xs, ys, res_df["display"]):
        if np.isnan(x) or np.isnan(y):
            continue
        ax.annotate(name, (x, y),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=11, va="center", ha="left")

    # Square the axes around the data so y=x reads as a true 45° line.
    finite = np.concatenate([xs[np.isfinite(xs)], ys[np.isfinite(ys)]])
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.05
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="gray", linewidth=0.8, linestyle="--",
            zorder=1, label="y = x (no change)")
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.axvline(0, color="black", linewidth=0.6, zorder=1)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("gap at START of CoT  "
                  "(above_good − below_good) / threshold", fontsize=15)
    ax.set_ylabel("gap at END of CoT  "
                  "(above_good − below_good) / threshold", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        import os
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


plot_offset_gap_scatter(
    res_df,
    filename="figures/trajectories/offset_gap_scatter.pdf",
)


# %%
