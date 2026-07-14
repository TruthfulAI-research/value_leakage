"""Sample data for one model into an isolated cache tree.

Usage:
    python -m donation_bet.get_data --model <model_key> [--experiment NAME] [--cache-only]

Writes both the model-output cache and the estimate-judge cache under
``<repo>/final_data/``, entirely separate from the committed
``shared/cache/`` and ``shared/estimate_judge_cache/`` trees. Re-running
with the same arguments reuses what's already on disk.
"""
import argparse
from pathlib import Path

import shared.runner as runner
from shared.models import MODELS
from shared.experiments import THRESHOLD_EXPERIMENTS
from shared.get_main_dfs import get_main_dfs


DEFAULT_EXPERIMENT = "main_experiment_accurate"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "final_data"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=sorted(MODELS), required=True,
        help="Model key from shared.models.MODELS",
    )
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        choices=sorted(THRESHOLD_EXPERIMENTS),
        help=f"Experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Raise instead of sampling if any cache is missing.",
    )
    args = parser.parse_args()

    # Redirect both caches away from the committed shared/cache and
    # shared/estimate_judge_cache trees. Both knobs live on shared.runner
    # and are read at call time, so reassignment here propagates through
    # get_main_dfs and run_thresholds_experiment.
    runner.CACHE_DIR = str(DATA_ROOT / "cache")
    runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA_ROOT / "estimate_judge_cache")

    print(f"Model:       {args.model}")
    print(f"Experiment:  {args.experiment}")
    print(f"Model cache: {runner.CACHE_DIR}")
    print(f"Judge cache: {runner.estimate_judge_cache_dir(args.experiment)}")

    out = get_main_dfs(
        args.experiment, [args.model], cache_only=args.cache_only,
    )
    df, thresholds, display_name = out[args.model]

    print(f"\nDone: {display_name}  ({len(df)} rows)")
    print("Thresholds:")
    for pk, t in sorted(thresholds.items()):
        print(f"  {pk}: {t}")


if __name__ == "__main__":
    main()
