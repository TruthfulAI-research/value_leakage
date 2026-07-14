"""Revealed-preference scoring for the activity-preferences experiment.

`score_activities.py` measures *stated* preference: it asks the model to rate
each activity 0-100 in isolation. This script measures *revealed* preference
instead. It reuses the exact pairwise (pair, order) variations that pipeline.py
feeds the random-pick task, but renders them with prompts/preference_template.yaml,
which swaps "pick totally randomly" for "pick whichever you'd genuinely prefer".
Each response goes through the same choice judge, and the per-activity selection
rate under this prompt is the model's revealed preference.

Template wording is sampled too: preference_template.yaml holds 5 semantically
equivalent wordings and pipeline.build_variations draws one per variation (from a
separate seeded RNG stream, so the pairing is unaffected). The 5 wordings are kept
in sync with prompt_template.yaml at the same index, and the template stream is seeded
identically, so a matched variation uses the same wording in both runs and only
the final instruction differs.

Because it shares pipeline.SEED / N_VARIATIONS and the same
pairing logic, every pair+order matches the random-pick run one-to-one — the
only difference is the final instruction. So the random-pick selection rate
(does preference leak into a supposedly random pick?) and this preference-pick
selection rate (what is the preference?) are directly comparable, pair for pair.

Outputs (in the activity_preferences/ folder, alongside the stated-score files
and separate from the random-pick pipeline/ folder):
    results/<MODEL_KEY>/activity_preferences/revealed_preference.jsonl       one row per variation
    results/<MODEL_KEY>/activity_preferences/revealed_preference_summary.csv per-activity selection rate

Run:
    uv run python choosing_activities/score_activities_revealed.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

# Reuse every piece of the random-pick pipeline so the only difference between
# the two runs is the prompt template handed to build_variations.
try:
    from .pipeline import (
        EXPERIMENT_NAME,
        MODEL_KEY,
        N_VARIATIONS,
        SCORE_SUMMARY_PATH,
        SEED,
        assemble,
        build_variations,
        load_score_summary,
        parse_choice_judge_answer,
        pearson_r_ci,
        run_choice_judge,
        run_main,
        save_csv,
        save_jsonl,
        summarize_selection_rates,
    )
except ImportError:  # pragma: no cover - script execution path
    from pipeline import (
        EXPERIMENT_NAME,
        MODEL_KEY,
        N_VARIATIONS,
        SCORE_SUMMARY_PATH,
        SEED,
        assemble,
        build_variations,
        load_score_summary,
        parse_choice_judge_answer,
        pearson_r_ci,
        run_choice_judge,
        run_main,
        save_csv,
        save_jsonl,
        summarize_selection_rates,
    )

load_dotenv(REPO_ROOT / ".env")

PREFERENCE_TEMPLATE_PATH = HERE / "prompts" / "preference_template.yaml"

# One no-tools setting; the dedicated name keeps its main-model cache in its own
# directory (runner_main/<MODEL_KEY>/revealed_preference/) rather than mixing in
# with the random-pick no_tools cache.
TOOL_SETTING = {
    "name": "revealed_preference",
    "label": "Revealed preference",
    "tool_names": [],
}

JUDGE_COL = f"{EXPERIMENT_NAME}_choice_judge"


def run_revealed_preference(
    n_variations: int,
    *,
    seed: int,
    cache_only: bool = False,
) -> pd.DataFrame:
    """Build matched preference prompts, sample, judge, and assemble per-variation rows."""
    print(
        f"Building {n_variations} revealed-preference variations "
        f"(seed={seed}, template={PREFERENCE_TEMPLATE_PATH.name})..."
    )
    variations = build_variations(
        n_variations,
        seed,
        template_path=PREFERENCE_TEMPLATE_PATH,
    )

    print(f"Running main model ({MODEL_KEY}) on the preference prompt...")
    df_main = run_main(variations, TOOL_SETTING, cache_only=cache_only)
    if len(df_main) != len(variations):
        print(f"  warning: {MODEL_KEY} returned {len(df_main)} rows, "
              f"expected {len(variations)}")

    print("Running choice judge...")
    df_judge = run_choice_judge(variations, df_main["answer"].tolist(), cache_only=cache_only)
    judgments = [parse_choice_judge_answer(a) for a in df_judge[JUDGE_COL]]
    choice_judge_reasoning = df_judge[f"{JUDGE_COL}_reasoning"].tolist()

    n = len(variations)
    # The randomness-reasoning judge is irrelevant here (we asked for preference,
    # not randomness), so its columns are filled with None.
    return assemble(
        variations,
        df_main,
        judgments,
        df_judge[JUDGE_COL].tolist(),
        choice_judge_reasoning,
        [None] * n,
        [None] * n,
        [None] * n,
    )


def report(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    n = len(df)
    decisive = df[df["judgment"].isin([1, 2])]
    n_dec = len(decisive)
    print("\n=== Revealed preference: response summary ===")
    print(f"  Total variations: {n}")
    if n:
        print(f"  Decisive picks:   {n_dec} ({100 * n_dec / n:.1f}%)")
        print(f"  Refusals:         {n - n_dec}")

    scored = summary.dropna(subset=["selection_rate"])
    if scored.empty:
        print("\n  No scored activities (no decisive appearances).")
        return

    picks = int(scored["n_picked"].sum())
    apps = int(scored["n_appearances"].sum())
    rate = 100 * picks / apps if apps else 0.0
    print("\n=== Revealed preference (appearance-weighted) ===")
    print(f"  activities: {len(scored)}  picks: {picks}  appearances: {apps}  "
          f"pick rate: {rate:.1f}%")

    ranked = scored.sort_values("selection_rate", ascending=False)
    print("\n=== Top 10 most-preferred activities ===")
    print(f"  {'rate':>6}  {'picks/app':<10}  activity")
    for _, r in ranked.head(10).iterrows():
        print(f"  {100 * r['selection_rate']:>5.1f}%  "
              f"{int(r['n_picked'])}/{int(r['n_appearances']):<7}  {r['activity']}")
    print("\n=== Bottom 10 least-preferred activities ===")
    for _, r in ranked.tail(10).iloc[::-1].iterrows():
        print(f"  {100 * r['selection_rate']:>5.1f}%  "
              f"{int(r['n_picked'])}/{int(r['n_appearances']):<7}  {r['activity']}")


def compare_with_stated(summary: pd.DataFrame) -> None:
    """Pearson r between revealed preference (this run) and stated 0-100 liking."""
    score_summary = load_score_summary()
    if score_summary is None:
        print(f"\nSkipping stated-vs-revealed comparison: missing {SCORE_SUMMARY_PATH}")
        return
    merged = summary.merge(
        score_summary[["activity_ix", "activity", "mean_score"]],
        on=["activity_ix", "activity"],
        how="left",
    ).dropna(subset=["mean_score", "selection_rate"])
    if len(merged) < 2:
        print("\nStated-vs-revealed comparison: too few overlapping activities.")
        return
    r = merged["mean_score"].corr(merged["selection_rate"])
    lo, hi = pearson_r_ci(r, len(merged))
    print("\n=== Stated (0-100 liking) vs revealed (preference pick rate) ===")
    print(f"  stated source: {SCORE_SUMMARY_PATH.name}")
    print(f"  activities: {len(merged)}")
    print(f"  Pearson r:  {r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variations",
        type=int,
        default=N_VARIATIONS,
        help="Number of (pair, order) variations (default: pipeline.N_VARIATIONS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Pairing seed (default: pipeline.SEED; match it for one-to-one "
             "matched pairs against the random-pick run).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Raise instead of sampling if the main/judge cache is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = run_revealed_preference(
        args.variations,
        seed=args.seed,
        cache_only=args.cache_only,
    )

    out = save_jsonl(df, name=f"revealed_preference_{MODEL_KEY}")
    summary = summarize_selection_rates(df)
    summary_out = save_csv(summary, name=f"revealed_preference_{MODEL_KEY}_summary")

    report(df, summary)
    compare_with_stated(summary)

    print(f"\nSaved {len(df)} records -> {out}")
    print(f"Saved {len(summary)} activity selection rows -> {summary_out}")


if __name__ == "__main__":
    main()
