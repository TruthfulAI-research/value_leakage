# %%
"""Agentic answer-grading (rater label-bias) experiments: Alpaca & real-analysis proofs.

Jan Dubinski's rater-bias setup (branch `jd/rate-llm-answers`,
`janekd/rate_llm_answers` + `janekd/rate_llm_math`): per trial (= seed), an
agentic grader gets a CSV file with 10 questions x 4 candidate answers. All
four candidates are written by the same source model (`claude-haiku-4.5`) but
are tagged with four model labels -- `claude-opus-3`, `claude-haiku-3.5`,
`gpt-4o`, `gpt-4o-mini` -- under a fresh random permutation per trial. The
grader reads the file from disk and picks the best answer per question, so
any per-label deviation from the uniform 25% win rate is label bias. Three
agentic graders: Claude Code (`claude-opus-4-7`), Codex (`gpt-5.4`), Qwen
(`qwen/qwen3.6-max-preview`).

Experiments covered here (the two named tasks, three grading harnesses each):

  - **alpaca**: AlpacaEval instruction-following questions.
  - **proofs_opus** / **proofs_gemini**: open-ended undergraduate
    real-analysis proofs (epsilon-delta limits, sequences, sup/inf, ...);
    problems authored by `claude-opus-4.7` resp. `gemini-3.1-pro`, proofs
    written by `claude-haiku-4.5`. No ground truth.
  - **proofs_proofnet**: same proof-grading task, but the problem statements
    are taken from the ProofNet benchmark (`hoskinson-center/proofnet`,
    Rudin + Pugh real-analysis subset) instead of being model-authored;
    proofs still written by `claude-haiku-4.5`. Three graders over 500
    permutations (codex 500, claude 497, qwen 453 completed). This is the
    math task reported in the paper.

All data is loaded from `final_data/answer_grading_cache/`, populated from
Jan's branch by `migrate_from_janekd.py` (same directory; run it once with a
worktree of `origin/jd/rate-llm-answers`); the `proofs_proofnet` entry is
populated by the sibling `build_proofnet_cache.py`. Nothing here calls any API.

Provenance (committed run dirs on the branch; counts = completed trials):

  experiment      grader   source run dir                          trials  transcripts
  alpaca          claude   2026-05-21_10-27-34_lab_x_tier_2024        250  250 rater_log.jsonl
  alpaca          codex    2026-05-24_15-53-01_lab_x_tier_2024        250  250 rater_log.jsonl
  alpaca          qwen     2026-05-24_23-46-17_lab_x_tier_2024        497  497 rater_log.jsonl
  proofs_gemini   claude   full_gemini_claude (interrupted)         92...  none retained
  proofs_gemini   codex    full_gemini_codex                          250  none retained
  proofs_gemini   qwen     full_gemini_qwen (45 failed attempts)      105  105 rater_log.jsonl
  proofs_opus     (all)    runs deleted upstream                        -  extracted rationales only
  proofs_proofnet codex    2026-06-22_proofs_analysis_lab_x_tier_2024    500  none retained
  proofs_proofnet claude   2026-06-22_proofs_analysis_lab_x_tier_2024    497  none retained
  proofs_proofnet qwen     2026-06-22_proofs_analysis_lab_x_tier_2024    453  none retained

Known data gotchas (verified against the branch at migration time):

  - `best_answers.csv` rows are just (question, answer); the picked label is
    NOT in the committed raw alpaca data (the per-seed replicate inputs and
    the alpaca answer cache are gitignored upstream). Per-label numbers
    therefore come from Jan's committed `analysis/per_label_per_seed_wins.csv`
    (fractional credit for byte-identical-answer ties, see
    `janekd/rate_llm_answers/analyze.py`), migrated verbatim.
  - The cross-grader CoT extraction (`rationales.jsonl` here; `trials_*.jsonl`
    upstream) does carry one `picked_label` per question. For the alpaca
    claude/codex graders it covers Jan's local 500-seed runs, a superset of
    the committed 250-seed run dirs.
  - proofs_gemini is unevenly complete: claude was interrupted at 92 trials
    and its committed analysis covers only the first 90; qwen completed
    105/150 attempts. Jan's analysis CSVs are kept as-is (claude n=90).
  - proofs_opus raw run dirs (250/250/217 trials) were deleted upstream; the
    per-seed rationales (with per-pick labels, rater text, and thinking text)
    survive, and the win rates recomputed from them reproduce the numbers in
    Jan's slide deck (`data/slides/make_plots.py`), asserted at migration.
  - Four alpaca trials are contaminated by self-identity leaks (the source
    model wrote "I'm Claude ..." into an answer, which interacts with the
    label): seeds 222367, 222480, 222483, 222541. Jan's `_dedup` sibling runs
    drop/replace exactly these; pass `drop_leak_seeds=True` to the loaders to
    drop them here. Of the qwen run's extra seeds >222582, only 222583-222586
    (the codex-dedup replacements) were leak-screened upstream; the rest were
    never audited.

Usage:

    git worktree add /tmp/giraffes-rate origin/jd/rate-llm-answers
    uv run python -m shared.final_scripts.answer_grading.migrate_from_janekd \
        /tmp/giraffes-rate
    uv run python -m agentic_grading.plot_win_rates
"""
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from shared.plot_style import VALUE_FS  # noqa: F401  applies shared figure sizing on import
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "final_data"
CACHE_ROOT = DATA_ROOT / "answer_grading_cache"
# Figure destination: every plot is written (PDF only) into the answer_grading
# section of the gitignored Overleaf clone -- the single figure output location.
FIG_DIR = DATA_ROOT.parents[1] / "overleaf" / "figures" / "answer_grading"

# x-axis order requested for the headline plot.
GRADERS = ["codex", "claude", "qwen"]
GRADER_DISPLAY = {
    "claude": "Claude Code\n(claude-opus-4.7)",
    "codex": "Codex\n(gpt-5.4)",
    "qwen": "Qwen Agent\n(qwen3.6)",
}

# Label order from the runs' meta.json `model_labels`.
MODEL_LABELS = ["claude-opus-3", "claude-haiku-3.5", "gpt-4o", "gpt-4o-mini"]
LABEL_COLORS = {
    "claude-opus-3": "#C0392B",
    "claude-haiku-3.5": "#E67E22",
    "gpt-4o": "#2471A3",
    "gpt-4o-mini": "#85C1E9",
}
N_QUESTIONS = 10
CHANCE_RATE = 1.0 / len(MODEL_LABELS)

EXPERIMENTS = {
    "alpaca": {
        "title": "Alpaca dataset",
        "kind": "runs",
    },
    "proofs_opus": {
        "title": "Math proof dataset",
        "kind": "rationales_only",
    },
    "proofs_gemini": {
        "title": "Math proof dataset\n(Gemini-authored)",
        "kind": "runs",
    },
    "proofs_proofnet": {
        "title": "ProofNet dataset",
        "kind": "runs",
    },
}

# Self-identity-leak trials (see module docstring). Exhaustive for seeds
# 222333-222582 (the committed claude/codex range); the qwen run's extra
# seeds were never leak-audited upstream.
ALPACA_LEAK_SEEDS = frozenset({222367, 222480, 222483, 222541})

# Jan's bootstrap (janekd/rate_llm_answers/analyze.py): mean of per-trial win
# rates, 10k-resample percentile CI, fixed seed.
BOOT_N = 10_000
BOOT_SEED = 0


class CacheMiss(FileNotFoundError):
    """A required answer_grading_cache file is missing -- run the migration."""


def _boot_ci(values, alpha=0.05, seed=BOOT_SEED):
    """Bootstrap mean + symmetric percentile CI (byte-compatible with Jan's)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(v, size=(BOOT_N, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


# --- Cache access ---

def grader_dir(experiment, grader):
    assert experiment in EXPERIMENTS, f"unknown experiment {experiment!r}"
    assert grader in GRADERS, f"unknown grader {grader!r}"
    return CACHE_ROOT / experiment / grader


def _read_jsonl(path, expect_kind):
    """Read one cache file -> (meta, rows). Validates the meta line."""
    if not path.exists():
        raise CacheMiss(
            f"{path} not found; populate the cache with\n"
            "  uv run python -m "
            "shared.final_scripts.answer_grading.migrate_from_janekd "
            "<worktree of origin/jd/rate-llm-answers>"
        )
    with open(path) as f:
        meta = json.loads(f.readline())
        assert meta.get("kind") == expect_kind, (
            f"{path}: meta kind {meta.get('kind')!r} != {expect_kind!r}"
        )
        rows = [json.loads(line) for line in f]
    assert len(rows) == meta["n_rows"], (
        f"{path}: {len(rows)} rows, meta says {meta['n_rows']}"
    )
    return meta, rows


def load_meta(experiment, grader=None):
    path = (CACHE_ROOT / experiment if grader is None
            else grader_dir(experiment, grader)) / "meta.json"
    if not path.exists():
        raise CacheMiss(f"{path} not found; run the migration first")
    return json.loads(path.read_text())


def _maybe_drop_leak_seeds(df, experiment, drop_leak_seeds):
    if not drop_leak_seeds:
        return df
    assert experiment == "alpaca", (
        "drop_leak_seeds only applies to the alpaca experiment"
    )
    return df[~df["seed"].isin(ALPACA_LEAK_SEEDS)].reset_index(drop=True)


@lru_cache(maxsize=None)
def _common_seeds_cached(experiment, graders, drop_leak_seeds):
    """Seeds present for *every* grader in BOTH the win-rate data and the
    rationale extraction. Forward-refs per_seed_wins / load_rationales, which
    are defined below and resolved at call time; both are read here without the
    common-seed restriction (so there is no recursion)."""
    sets = []
    for grader in graders:
        wins = set(per_seed_wins(experiment, grader,
                                 drop_leak_seeds=drop_leak_seeds)["seed"])
        rats = set(load_rationales(experiment, grader,
                                   drop_leak_seeds=drop_leak_seeds)["seed"])
        sets.append(wins & rats)
    return frozenset(set.intersection(*sets)) if sets else frozenset()


def common_seeds(experiment, graders=GRADERS, drop_leak_seeds=False):
    """The matched seed panel shared by all `graders` of `experiment`.

    Restricting to this set gives every grader the same trials, so no grader
    contributes extra data to a comparison -- e.g. alpaca qwen has 497
    win-trials vs 250 for claude/codex, an asymmetry that vanishes here
    (alpaca -> 250, proofs_opus -> 217 with the default graders)."""
    return _common_seeds_cached(experiment, tuple(graders), drop_leak_seeds)


def _maybe_common_seeds(df, experiment, drop_leak_seeds, common_seeds_only,
                        graders=GRADERS):
    if not common_seeds_only:
        return df
    keep = common_seeds(experiment, graders, drop_leak_seeds=drop_leak_seeds)
    return df[df["seed"].isin(keep)].reset_index(drop=True)


def load_picks(experiment, grader, drop_leak_seeds=False,
               common_seeds_only=False):
    """Grader picks, one row per (trial, question): seed, question, answer.

    `answer` is the full text of the picked candidate. The picked *label* is
    not part of the committed raw data; see `load_rationales`.
    """
    _, rows = _read_jsonl(grader_dir(experiment, grader) / "picks.jsonl",
                          "picks")
    df = _maybe_drop_leak_seeds(pd.DataFrame(rows), experiment,
                                drop_leak_seeds)
    return _maybe_common_seeds(df, experiment, drop_leak_seeds,
                               common_seeds_only)


def load_transcripts(experiment, grader):
    """Full agent transcripts: dict seed -> list of rater_log.jsonl records.

    Empty dict where no transcripts were retained upstream (proofs_gemini
    claude/codex); the file's meta line records why.
    """
    _, rows = _read_jsonl(
        grader_dir(experiment, grader) / "transcripts.jsonl", "transcripts")
    return {r["seed"]: r["lines"] for r in rows}


def load_per_label_per_seed_wins(experiment, grader, drop_leak_seeds=False,
                                 common_seeds_only=False):
    """Jan's per-(trial, label) wins: seed, model, n_wins, n_questions,
    win_rate. Fractional n_wins = byte-identical-answer ties split 1/N.
    Only for `kind == "runs"` experiments."""
    _, rows = _read_jsonl(
        grader_dir(experiment, grader) / "per_label_per_seed_wins.jsonl",
        "per_label_per_seed_wins")
    df = _maybe_drop_leak_seeds(pd.DataFrame(rows), experiment,
                                drop_leak_seeds)
    return _maybe_common_seeds(df, experiment, drop_leak_seeds,
                               common_seeds_only)


def load_jan_summary(experiment, grader):
    """Jan's committed analysis/per_label_summary.csv, verbatim."""
    _, rows = _read_jsonl(
        grader_dir(experiment, grader) / "per_label_summary.jsonl",
        "per_label_summary")
    return pd.DataFrame(rows)


def load_rationales(experiment, grader, drop_leak_seeds=False,
                    common_seeds_only=False):
    """Cross-grader CoT extraction (upstream `trials_<grader>.jsonl`): one row
    per trial with `picks` (per-question `picked_label`), `rater_text`,
    `thinking_text`, `behavior`, etc. For alpaca claude/codex this covers
    Jan's local 500-seed runs (superset of the committed 250 trials)."""
    _, rows = _read_jsonl(
        grader_dir(experiment, grader) / "rationales.jsonl", "rationales")
    df = _maybe_drop_leak_seeds(pd.DataFrame(rows), experiment,
                                drop_leak_seeds)
    return _maybe_common_seeds(df, experiment, drop_leak_seeds,
                               common_seeds_only)


def load_source_answers(experiment):
    """The candidate answers shown to the graders (upstream `answers.csv`):
    model, model_api, sample_idx, question, answer. Not committed upstream
    for alpaca."""
    _, rows = _read_jsonl(CACHE_ROOT / experiment / "source_answers.jsonl",
                          "source_answers")
    return pd.DataFrame(rows)


# --- Win rates ---

def per_seed_wins_from_rationales(rationales_df):
    """Per-(trial, label) wins recomputed from rationale picks.

    Same columns as `load_per_label_per_seed_wins`. Tie semantics differ from
    Jan's analyze.py: when a picked answer is byte-identical across N labeled
    candidates, analyze.py splits the win 1/N, while the upstream rationale
    extraction attributed it whole to one arbitrary (permutation-uniform)
    tied label. Identical on tie-free trials -- ties are common on alpaca
    (short answers) and essentially absent on the proofs."""
    rows = []
    for r in rationales_df.itertuples():
        counts = {}
        for p in r.picks:
            if p.get("match_failed"):
                continue
            counts[p["picked_label"]] = counts.get(p["picked_label"], 0) + 1
        n_q = len(r.picks)
        for label in MODEL_LABELS:
            n_wins = float(counts.get(label, 0))
            rows.append({
                "seed": r.seed,
                "model": label,
                "n_wins": n_wins,
                "n_questions": n_q,
                "win_rate": n_wins / n_q,
            })
    return pd.DataFrame(rows)


def per_seed_wins(experiment, grader, drop_leak_seeds=False,
                  common_seeds_only=False):
    """Canonical per-(trial, label) wins for one (experiment, grader):
    Jan's committed analysis for run-backed experiments, rationale-derived
    for proofs_opus."""
    if EXPERIMENTS[experiment]["kind"] == "runs":
        return load_per_label_per_seed_wins(
            experiment, grader, drop_leak_seeds=drop_leak_seeds,
            common_seeds_only=common_seeds_only)
    return per_seed_wins_from_rationales(
        load_rationales(experiment, grader, drop_leak_seeds=drop_leak_seeds,
                        common_seeds_only=common_seeds_only))


def summarize_per_label(wins_df):
    """Per-label summary over trials (Jan's analyze.py `_summarise`):
    bootstrap mean/CI over the per-trial win rates."""
    rows = []
    for label in MODEL_LABELS:
        sub = wins_df[wins_df["model"] == label]
        mean, lo, hi = _boot_ci(sub["win_rate"].values)
        rows.append({
            "model": label,
            "n_seeds": int(len(sub)),
            "total_wins": round(float(sub["n_wins"].sum()), 3),
            "total_questions": int(sub["n_questions"].sum()),
            "mean_win_rate": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })
    return pd.DataFrame(rows)


def summarize_experiment(experiment, graders=GRADERS, drop_leak_seeds=False,
                         common_seeds_only=False):
    """Per-label summaries for all graders of one experiment, with a `grader`
    column. Graders whose cache is missing are skipped with a warning."""
    drop_leak_seeds = drop_leak_seeds and experiment == "alpaca"
    frames = []
    for grader in graders:
        try:
            wins = per_seed_wins(experiment, grader,
                                 drop_leak_seeds=drop_leak_seeds,
                                 common_seeds_only=common_seeds_only)
        except CacheMiss as e:
            print(f"[{experiment}] skipping {grader}: {e}")
            continue
        frames.append(summarize_per_label(wins).assign(grader=grader))
    if not frames:
        raise CacheMiss(f"no cached graders for {experiment}")
    return pd.concat(frames, ignore_index=True)


# --- Plot ---

def plot_label_win_rates(experiments=("alpaca", "proofs_proofnet"),
                         graders=GRADERS, drop_leak_seeds=False,
                         common_seeds_only=False, fname=None):
    """Grouped bars per experiment: graders on the x axis, one bar per
    attached model label, bootstrap 95% CIs, chance line at 25%."""
    summaries = {e: summarize_experiment(e, graders=graders,
                                         drop_leak_seeds=drop_leak_seeds,
                                         common_seeds_only=common_seeds_only)
                 for e in experiments}
    fig, axes = plt.subplots(
        1, len(experiments), figsize=(4.6 * len(experiments) + 0.8, 4.5),
        sharey=True, squeeze=False,
    )
    width = 0.19
    for ax, experiment in zip(axes[0], experiments):
        summary = summaries[experiment]
        present = [g for g in graders if g in set(summary["grader"])]
        for j, label in enumerate(MODEL_LABELS):
            xs, vals, err_lo, err_hi = [], [], [], []
            for i, g in enumerate(present):
                sub = summary[(summary["grader"] == g)
                              & (summary["model"] == label)]
                m = float(sub["mean_win_rate"].iloc[0])
                xs.append(i + (j - 1.5) * width)
                vals.append(m)
                err_lo.append(m - float(sub["ci_lo"].iloc[0]))
                err_hi.append(float(sub["ci_hi"].iloc[0]) - m)
            ax.bar(xs, vals, width=width, yerr=[err_lo, err_hi],
                   color=LABEL_COLORS[label], label=label,
                   edgecolor="white", linewidth=0.5, ecolor="black",
                   capsize=3, error_kw={"linewidth": 0.9})
            for x, v, e in zip(xs, vals, err_hi):
                ax.text(x, v + e + 0.008, f"{100 * v:.0f}", ha="center",
                        va="bottom", fontsize=VALUE_FS)
        ax.axhline(CHANCE_RATE, color="gray", linestyle="--", linewidth=1,
                   label=f"chance ({CHANCE_RATE:.0%})")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([GRADER_DISPLAY[g] for g in present])
        ax.set_title(EXPERIMENTS[experiment]["title"])
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Mean per-trial win rate")
    ymax = max(s["ci_hi"].max() for s in summaries.values())
    axes[0][0].set_ylim(0, ymax * 1.12 + 0.02)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    # matplotlib floats the chance (axhline) handle to the front; move it last.
    order = ([i for i, l in enumerate(labels) if not l.startswith("chance")]
             + [i for i, l in enumerate(labels) if l.startswith("chance")])
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(1.01, 0.5), bbox_transform=axes[0][-1].transAxes,
               ncol=1)
    if fname is not None:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    return summaries


# %%
if __name__ == "__main__":
    DROP_LEAK_SEEDS = globals().get("DROP_LEAK_SEEDS", False)
    # Restrict every grader to the seeds shared by all graders, so no grader
    # contributes extra trials -- e.g. alpaca qwen has 497 win-trials vs 250
    # for claude/codex (-> 250). A no-op for proofs_proofnet, whose committed
    # cache is already capped to the equal-n 250-seed subset used in the paper.
    # Set False to use each grader's full data.
    COMMON_SEEDS_ONLY = globals().get("COMMON_SEEDS_ONLY", True)
    # The math panel is proofs_proofnet (ProofNet problem statements), the math
    # task reported in the paper. The model-authored proofs_opus / proofs_gemini
    # variants are cached and loadable but off the default figure; swap them in
    # via EXPERIMENTS_TO_PLOT.
    EXPERIMENTS_TO_PLOT = globals().get(
        "EXPERIMENTS_TO_PLOT", ("alpaca", "proofs_proofnet"))
    PLOT_FNAME = globals().get("PLOT_FNAME", FIG_DIR / "label_win_rates.pdf")

# %%
# Per-label win-rate summaries (and, for run-backed experiments, a
# consistency check against Jan's committed per_label_summary.csv).
if __name__ == "__main__":
    for experiment in EXPERIMENTS_TO_PLOT:
        print(f"\n=== {experiment} ===")
        summary = summarize_experiment(experiment,
                                       drop_leak_seeds=DROP_LEAK_SEEDS,
                                       common_seeds_only=COMMON_SEEDS_ONLY)
        print(summary.to_string(index=False))
        # The committed-summary check only holds on each grader's full data;
        # the common-seed subset (esp. qwen) intentionally diverges from it.
        if (EXPERIMENTS[experiment]["kind"] == "runs" and not DROP_LEAK_SEEDS
                and not COMMON_SEEDS_ONLY):
            for grader in summary["grader"].unique():
                ours = summary[summary["grader"] == grader].drop(
                    columns="grader").reset_index(drop=True)
                jans = load_jan_summary(experiment, grader)
                # atol covers bootstrap-CI RNG drift across numpy versions;
                # means and totals match far tighter (see the migration).
                pd.testing.assert_frame_equal(
                    ours, jans[ours.columns], check_exact=False, atol=2e-3,
                )
            print(f"[{experiment}] matches Jan's committed summaries.")

# %%
if __name__ == "__main__":
    plot_label_win_rates(EXPERIMENTS_TO_PLOT, drop_leak_seeds=DROP_LEAK_SEEDS,
                         common_seeds_only=COMMON_SEEDS_ONLY, fname=PLOT_FNAME)

# %%
