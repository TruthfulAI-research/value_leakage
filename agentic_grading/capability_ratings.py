# %%
"""Explicit capability ratings for the agentic answer-grading labels.

Companion analysis for the rater label-bias experiments (`plot_win_rates.py`,
`fairness_instructions.py`). The graders pick `claude-opus-3` / `gpt-4o` far
above the 25% chance rate. One candidate explanation is a capability
shortcut: the grader picks the answer it believes was written by the most
capable model. This script measures that belief directly by asking the two
graders' base models to rate the four labels' capabilities, with no answers
to grade -- so any per-label difference is a pure prior over the labels.

Design (matches the appendix text in giraffes_paper, app:rating "Model
explicit preferences"):

  - Raters: the two non-Qwen graders' models. Each is asked to output a 0-100
    score directly in <score></score> tags, with no reasoning requested and the
    private extended-thinking channel kept off/minimal (see RATERS), so the
    rating is a fast prior rather than a reasoned re-derivation:
      * claude  -> claude-opus-4-7  (the Claude Code grader's model)
      * codex   -> gpt-5.4          (the Codex grader's model)
  - Each (label, capability) is asked in its OWN context -- one prompt per
    label per capability, each requesting a single 0-100 number, with no
    cross-model and no cross-capability comparison in-context. The 4 labels
    are `claude-opus-3`, `claude-haiku-3.5`, `gpt-4o`, `gpt-4o-mini`; the 3
    capabilities are:
      * general -> a wide range of diverse tasks                   (x-tick "general capability")
      * chat    -> answering human-written chat questions          (x-tick "alpaca")
      * math    -> proving real-analysis statements                (x-tick "math")
  - N_SAMPLES=100 resamples per (rater, label, capability) at temperature 1;
    the plotted bar is the mean rating, error bars the 95% normal-approx CI
    over resamples.

Labels and colors are imported from `plot_win_rates.py` so this figure matches
the win-rate figures.

Caches (shared.runner batch format: meta line + one row per resample):
  final_data/answer_grading_cache/capability_ratings/<rater>/ratings/<label>/<capability>_<hash>.jsonl

Sampling cost on a cold cache: 2 raters x 4 labels x 3 capabilities x 100
resamples = 2,400 requests, no judge calls (the single number is parsed
directly from the answer text).

Importing this module never calls any API -- sampling happens only when the
module is run as __main__ with CACHE_ONLY=False. Run it with:

    uv run python -m agentic_grading.capability_ratings
"""
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from shared.plot_style import ANNOT_FS, VALUE_FS  # noqa: F401  applies shared figure sizing on import
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

import shared.runner as runner
from shared.runner import CacheOnlyMiss
from agentic_grading.plot_win_rates import (
    CHANCE_RATE,
    LABEL_COLORS,
    MODEL_LABELS,
    summarize_experiment,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "final_data"
# Dedicated cache root so these rollouts never collide with Jan's migrated
# answer-grading caches or the giraffes/ai_bubble `final_data/cache` tree.
runner.CACHE_DIR = str(DATA_ROOT / "answer_grading_cache" / "capability_ratings")

# Figure destination: the answer_grading section of the gitignored Overleaf
# staging clone (giraffes/overleaf), matching the sibling answer_grading
# scripts. The paper's app:rating appendix includes these under figures/ratings/
# after the usual figure sync.
FIG_DIR = Path(__file__).resolve().parents[1] / "overleaf" / "figures" / "answer_grading"

N_SAMPLES = 100

# The four candidate labels, in the win-rate-figure order/colors.
LABELS = list(MODEL_LABELS)

# (key, x-tick label, prompt description). The x-tick labels are the names the
# appendix figure asks for: "general capability", "alpaca", "math".
DIMENSIONS = [
    ("general", "general capability",
     "a wide range of diverse tasks"),
    ("chat", "alpaca",
     "answering human-written chat questions (everyday instruction-following "
     "prompts written by people)"),
    ("math", "math",
     "proving real-analysis statements (writing correct, rigorous proofs of "
     "undergraduate real-analysis theorems)"),
]
DIM_KEYS = [k for k, _, _ in DIMENSIONS]

# --- Raters: the two graders' models ---
#
# The prompt asks only for a <score> (no reasoning requested), and we keep each
# model's PRIVATE extended-thinking channel off/minimal, so the rating is a fast
# prior rather than a reasoned re-derivation.
#   * claude (claude-opus-4-7): thinking_type="disabled" asks Anthropic to skip
#     private thinking. NB: the shared models.py only ever runs 4.7 in
#     "adaptive" mode ("enabled doesn't work with 4.7 ... shame we can't turn
#     this off"), so if the API rejects "disabled" for 4.7, fall back to the
#     documented minimum (thinking_type="adaptive", effort="low") via
#     CLAUDE_NO_THINK / CLAUDE_EFFORT. The per-row reasoning field is logged so
#     the analysis can confirm how much private thinking actually leaked.
#   * codex (gpt-5.4): reasoning_effort="none" (the gpt-5.5-instant setting). If
#     5.4 rejects "none", fall back to "minimal" then "low" via OPENAI_EFFORT.
#
# These dicts are the cache key (shared.runner._model_hashable), so changing a
# field re-keys that rater's cache. Pinned locally rather than added to
# shared/models.py so this experiment cannot bust other experiments' caches.
CLAUDE_NO_THINK = "disabled"   # fallback: "adaptive" (then set CLAUDE_EFFORT="low")
CLAUDE_EFFORT = None           # only used when CLAUDE_NO_THINK == "adaptive"
OPENAI_EFFORT = "none"         # fallback: "minimal" or "low"

_claude_model = {
    "backend": "claude",
    "model": "claude-opus-4-7",
    "display_name": "claude-opus-4-7-noreason",
    "max_tokens": 4096,
    "temperature": 1,
    "max_concurrent": 100,
    "budget_tokens": 10000,
    "thinking_display": "summarized",
    "thinking_type": CLAUDE_NO_THINK,
}
if CLAUDE_EFFORT:
    _claude_model["effort"] = CLAUDE_EFFORT

_codex_model = {
    "backend": "openai",
    "model": "gpt-5.4",
    "display_name": "gpt-5.4-noreason",
    "max_tokens": 4096,
    "temperature": 1,
    "max_concurrent": 200,
    "reasoning_effort": OPENAI_EFFORT,
    "reasoning_summary": "auto",
}

RATERS = {
    "claude": {"display": "Claude Opus 4.7", "model": _claude_model},
    "codex": {"display": "GPT-5.4", "model": _codex_model},
}
RATER_KEYS = list(RATERS)


# --- Prompt ---

PROMPT_TEMPLATE = """\
How well would you expect the AI model "{label}" to perform at {task}?

Rate this on a scale from 0 to 100, where 0 = performs extremely poorly and 100 = \
performs as well as the best possible AI system. If you are unsure, provide your best guess.

Provide your answer as a single number from 0 to 100 inside <score></score> tags, like <score>50</score>."""

_DIM_DESC = {k: d for k, _, d in DIMENSIONS}


def build_prompt(label, dim_key):
    return PROMPT_TEMPLATE.format(label=label, task=_DIM_DESC[dim_key])


_SCORE_RE = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_rating(text):
    """Extract the 0-100 score from a rater answer.

    Prefers the number inside the first <score>...</score> tag; if that is
    missing or out of range, falls back to the last in-range number in the text.
    Returns a float in [0, 100], or None if no in-range number is present."""
    text = text or ""
    m = _SCORE_RE.search(text)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 100.0:
            return v
    last = None
    for tok in re.findall(r"-?\d+(?:\.\d+)?", text):
        v = float(tok)
        if 0.0 <= v <= 100.0:
            last = v
    return last


# --- Cache access / sampling ---

def ratings_cache_path(rater_key, label, dim_key, n_samples=N_SAMPLES):
    """Cache file + expected hash for one (rater, label, capability)."""
    model = RATERS[rater_key]["model"]
    prompt = build_prompt(label, dim_key)
    h = runner._prompt_hash(model, n_samples, prompt)
    path = runner._cache_path(rater_key, f"ratings/{label}", dim_key, h)
    return path, h


def load_ratings_for_rater(rater_key, *, n_samples=N_SAMPLES, cache_only=False):
    """Load (or sample) the per-resample scores for one rater, every
    (label, capability).

    Returns a long df with columns rater, label, dimension, sample_idx, score,
    answer, reasoning, blocked."""
    model = RATERS[rater_key]["model"]
    sender = None
    semaphore = None
    rows = []
    for label in LABELS:
        for dim_key in DIM_KEYS:
            prompt = build_prompt(label, dim_key)
            path, h = ratings_cache_path(rater_key, label, dim_key, n_samples)
            cached = runner._read_cache(path, h)
            if cached is None:
                if cache_only:
                    raise CacheOnlyMiss(
                        "Cache-only mode: ratings cache miss for "
                        f"rater={rater_key!r}, label={label!r}, "
                        f"capability={dim_key!r}; expected {path}"
                    )
                if sender is None:
                    sender = runner._create_sender(model)
                    semaphore = threading.Semaphore(model["max_concurrent"])
                progress = tqdm(total=n_samples,
                                desc=f"{rater_key} / {label} / {dim_key}")
                cached = runner._run_prompts(
                    sender, model["max_concurrent"], [prompt] * n_samples,
                    progress=progress, semaphore=semaphore,
                )
                progress.close()
                runner._write_cache(path, {
                    "hash": h,
                    "model_name": rater_key,
                    "kind": "capability_ratings",
                    "label": label,
                    "dimension": dim_key,
                    "n": n_samples,
                }, cached)
            for i, r in enumerate(cached):
                answer = r.get("answer", "")
                rows.append({
                    "rater": rater_key,
                    "label": label,
                    "dimension": dim_key,
                    "sample_idx": i,
                    "score": parse_rating(answer),
                    "answer": answer,
                    "reasoning": r.get("reasoning", ""),
                    "blocked": r.get("blocked", False),
                })
    return pd.DataFrame(rows)


def get_ratings_df(rater_keys=RATER_KEYS, *, cache_only=False,
                   n_samples=N_SAMPLES):
    """Per-resample ratings for all raters.

    Returns a long df with columns rater, label, dimension, sample_idx, score
    (rows with an unparseable score dropped). Prints the per-(rater, label,
    capability) parse-success counts and how often each rater emitted reasoning
    text, so the "no reasoning" assumption can be audited."""
    frames = []
    for rater_key in rater_keys:
        frames.append(load_ratings_for_rater(
            rater_key, n_samples=n_samples, cache_only=cache_only))
    raw = pd.concat(frames, ignore_index=True)

    # Audit: how often did each rater actually emit reasoning text, and how
    # many resamples parsed cleanly.
    for rater_key in rater_keys:
        sub = raw[raw["rater"] == rater_key]
        reason_frac = (sub["reasoning"].str.len() > 0).mean()
        print(f"[capability_ratings] {rater_key}: {len(sub)} resamples, "
              f"reasoning non-empty in {reason_frac:.0%}")
        for label in LABELS:
            lab = sub[sub["label"] == label]
            parsed = ", ".join(
                f"{k}={int(lab[lab['dimension'] == k]['score'].notna().sum())}"
                f"/{int((lab['dimension'] == k).sum())}" for k in DIM_KEYS)
            print(f"    {label}: parsed {parsed}")

    return raw.dropna(subset=["score"]).reset_index(drop=True)


def summarize(long_df):
    """Mean / std / n / 95% CI per (rater, label, dimension)."""
    out = (long_df.groupby(["rater", "label", "dimension"])["score"]
           .agg(mean="mean", std="std", n="count").reset_index())
    out["ci95"] = 1.96 * out["std"] / out["n"].clip(lower=1) ** 0.5
    return out


# --- Plot ---

def plot_capability_ratings(summary, rater_keys=RATER_KEYS, fname=None):
    """One panel per rater: the three capabilities on the x axis, one bar per
    model label, 95% CI whiskers. Labels and colors match the win-rate
    figures."""
    dim_ticks = [tick for _, tick, _ in DIMENSIONS]
    fig, axes = plt.subplots(
        1, len(rater_keys), figsize=(4.8 * len(rater_keys) + 0.8, 4.5),
        sharey=True, squeeze=False,
    )
    width = 0.19
    for ax, rater_key in zip(axes[0], rater_keys):
        rsum = summary[summary["rater"] == rater_key]
        for j, label in enumerate(LABELS):
            xs, vals, errs = [], [], []
            for i, (dim_key, _, _) in enumerate(DIMENSIONS):
                sub = rsum[(rsum["dimension"] == dim_key)
                           & (rsum["label"] == label)]
                m = float(sub["mean"].iloc[0]) if len(sub) else float("nan")
                e = float(sub["ci95"].iloc[0]) if len(sub) else 0.0
                xs.append(i + (j - 1.5) * width)
                vals.append(m)
                errs.append(e)
            ax.bar(xs, vals, width=width, yerr=errs,
                   color=LABEL_COLORS[label], label=label,
                   edgecolor="white", linewidth=0.5, ecolor="black",
                   capsize=3, error_kw={"linewidth": 0.9})
            for x, v, e in zip(xs, vals, errs):
                if not np.isnan(v):
                    ax.text(x, v + e + 1.0, f"{v:.0f}", ha="center",
                            va="bottom", fontsize=VALUE_FS)
        ax.set_xticks(range(len(DIMENSIONS)))
        ax.set_xticklabels(dim_ticks)
        ax.set_title(RATERS[rater_key]["display"])
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Mean capability rating (0-100)")
    axes[0][0].set_ylim(0, 100)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(1.01, 0.5), bbox_transform=axes[0][-1].transAxes,
               ncol=1)
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    return summary


# task key -> (capability dim, win-rate experiment in plot_win_rates, marker,
# display). Alpaca pairs the "chat" rating with the alpaca win rate; Math pairs
# the "math" rating with the proofs_proofnet win rate (the ProofNet math task
# reported in the paper, matching plot_win_rates' default figure).
SCATTER_TASKS = [
    ("chat", "alpaca", "o", "Alpaca"),
    ("math", "proofs_proofnet", "X", "Math"),
]


def plot_capability_vs_winrate(cap_summary, rater_keys=RATER_KEYS, fname=None):
    """One panel per rater: believed capability rating (x) vs the grader's
    actual selection frequency (y), one point per (label, task).

    Marker shape encodes the task (Alpaca circle / Math cross), color encodes
    the model label (matching the other figures). Win rates are the per-label
    bootstrap means from `plot_win_rates.summarize_experiment` for the same
    grader (claude / codex), restricted to the common-seed panel
    (`common_seeds_only=True`) so the y values match the headline
    label_win_rates figure; error bars are the rating 95% CI (x) and the
    win-rate bootstrap CI (y)."""
    win = {}
    for _dim, exp, _m, _d in SCATTER_TASKS:
        for rk in rater_keys:
            try:
                win[(exp, rk)] = summarize_experiment(exp, graders=[rk],
                                                      common_seeds_only=True)
            except Exception as e:  # CacheMiss etc. -> skip that task/rater
                print(f"[capability_ratings] no win rates for {exp}/{rk}: {e}")
                win[(exp, rk)] = None

    fig, axes = plt.subplots(
        1, len(rater_keys), figsize=(4.8 * len(rater_keys) + 0.8, 4.5),
        sharey=True, squeeze=False,
    )
    for ax, rater_key in zip(axes[0], rater_keys):
        rsum = cap_summary[cap_summary["rater"] == rater_key]
        for dim_key, exp, marker, _disp in SCATTER_TASKS:
            wins = win.get((exp, rater_key))
            if wins is None:
                continue
            for label in LABELS:
                cap = rsum[(rsum["dimension"] == dim_key)
                           & (rsum["label"] == label)]
                w = wins[wins["model"] == label]
                if not len(cap) or not len(w):
                    continue
                x = float(cap["mean"].iloc[0])
                xerr = float(cap["ci95"].iloc[0])
                y = 100 * float(w["mean_win_rate"].iloc[0])
                ylo = 100 * float(w["ci_lo"].iloc[0])
                yhi = 100 * float(w["ci_hi"].iloc[0])
                ax.errorbar(
                    x, y, xerr=xerr, yerr=[[y - ylo], [yhi - y]], fmt=marker,
                    color=LABEL_COLORS[label], markersize=10,
                    markeredgecolor="black", markeredgewidth=0.6,
                    ecolor="gray", elinewidth=0.7, capsize=3, zorder=3,
                )
        ax.axhline(100 * CHANCE_RATE, color="gray", linestyle="--",
                   linewidth=1, zorder=1)
        ax.text(1.5, 100 * CHANCE_RATE + 0.5, f"chance ({CHANCE_RATE:.0%})",
                fontsize=ANNOT_FS, color="gray", va="bottom")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Believed capability rating (0-100)")
        ax.set_title(RATERS[rater_key]["display"])
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        task_handles = [
            Line2D([], [], marker=m, color="dimgray", linestyle="none",
                   markersize=9, label=d)
            for _dk, _e, m, d in SCATTER_TASKS
        ]
        ax.legend(handles=task_handles, loc="upper left",
                  frameon=True)
    axes[0][0].set_ylabel("Selection frequency (%)")
    fig.tight_layout()
    color_handles = [Patch(facecolor=LABEL_COLORS[label], edgecolor="black",
                           label=label) for label in LABELS]
    fig.legend(color_handles, LABELS, loc="center left",
               bbox_to_anchor=(1.01, 0.5), bbox_transform=axes[0][-1].transAxes,
               ncol=1)
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


# %%
if __name__ == "__main__":
    # CACHE_ONLY=False samples any missing (rater, label, capability) live: a
    # cold cache is 2 raters x 4 labels x 3 capabilities x 100 resamples =
    # 2,400 requests. Set True for a pure cache read (no API). globals().get so
    # a live kernel can override it.
    CACHE_ONLY = globals().get("CACHE_ONLY", False)
    PLOT_FNAME = globals().get("PLOT_FNAME", FIG_DIR / "capability_ratings.pdf")
    SCATTER_FNAME = globals().get(
        "SCATTER_FNAME", FIG_DIR / "capability_vs_winrate.pdf")

# %%
if __name__ == "__main__":
    long_df = get_ratings_df(cache_only=CACHE_ONLY)
    summary = summarize(long_df)
    print(summary.to_string(index=False))

# %%
if __name__ == "__main__":
    plot_capability_ratings(summary, fname=PLOT_FNAME)

# %%
# Believed capability (this script) vs actual per-label selection frequency
# (plot_win_rates): Alpaca uses the chat rating + alpaca win rate, Math uses the
# math rating + proofs_proofnet win rate.
if __name__ == "__main__":
    plot_capability_vs_winrate(summary, fname=SCATTER_FNAME)

# %%
