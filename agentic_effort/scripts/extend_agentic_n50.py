"""Extend an existing agentic data.csv to a target sample count (default n=50).

We reuse the rollouts already stored in ``results/<slug>/agentic/data.csv`` and
only generate the *missing* samples for each outcome, so error bars shrink
without discarding prior work. Each trajectory's environment and response cache
are keyed off the integer ``sample`` index (see ``agentic_eval``), so appending
higher sample indices is deterministic and never perturbs existing rows.

For every (outcome, sample) pair with ``sample`` in ``[0, target)`` that is not
already present in the CSV, a new trajectory is run and appended. The combined
frame is written back (sorted by outcome, sample), and a per-outcome summary is
printed.

Example:
    python scripts/extend_agentic_n50.py \
        --model anthropic/claude-opus-4-8 \
        --data results/claude-opus-4-8/agentic/data.csv \
        --target 50 --parallel 12
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_eval.eval import load_config, run_trajectory  # noqa: E402


def _slug(model: str) -> str:
    return model.split("/")[-1]


async def _run_missing(cfg, model: str, missing: list[tuple[str, int]],
                       parallel: int, data_path: Path,
                       existing: pd.DataFrame, checkpoint_every: int = 10) -> pd.DataFrame:
    """Run the missing trajectories, checkpointing partial results to disk.

    Trajectories complete out of order under the semaphore; every
    ``checkpoint_every`` completions we rewrite the CSV (existing rows + all
    new rows so far). If the proxy dies mid-run, finished work is preserved and
    a relaunch simply skips the already-written samples.
    """
    sem = asyncio.Semaphore(parallel)
    tasks = [
        asyncio.ensure_future(run_trajectory(cfg, model, outcome, sample, sem))
        for outcome, sample in missing
    ]
    done_rows: list[dict] = []

    def _flush() -> None:
        new_df = pd.DataFrame(done_rows)
        new_df["model"] = model
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(["outcome", "sample"]).reset_index(drop=True)
        combined.to_csv(data_path, index=False)

    completed = 0
    for fut in asyncio.as_completed(tasks):
        row = await fut
        done_rows.append(row)
        completed += 1
        if completed % checkpoint_every == 0:
            _flush()
            print(f"  checkpoint: {completed}/{len(missing)} new trajectories saved",
                  flush=True)
    _flush()
    new_df = pd.DataFrame(done_rows)
    new_df["model"] = model
    return new_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model ID as originally used (kept in the model column).")
    parser.add_argument("--data", required=True,
                        help="Path to the existing data.csv to extend in place.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--target", type=int, default=50,
                        help="Desired samples per outcome (default 50).")
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--outcomes", nargs="+", default=None,
                        help="Restrict to these outcomes (default: those already in the CSV).")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    data_path = Path(args.data)
    if data_path.exists():
        existing = pd.read_csv(data_path)
    else:
        existing = pd.DataFrame(columns=["outcome", "sample"])

    outcomes = (list(args.outcomes) if args.outcomes
                else sorted(existing["outcome"].unique()) if not existing.empty
                else list(cfg.targets.keys()) + ["choose"])

    have = set(zip(existing.get("outcome", []), existing.get("sample", [])))
    missing = [(o, s) for o in outcomes for s in range(args.target)
               if (o, s) not in have]

    print(f"model={args.model}")
    print(f"existing rows={len(existing)}  outcomes={len(outcomes)}  target={args.target}/outcome")
    print(f"missing trajectories to run: {len(missing)}")
    if not missing:
        print("nothing to do — already at target.")
        return

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    new_df = asyncio.run(_run_missing(
        cfg, args.model, missing, args.parallel, data_path, existing,
    ))

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["outcome", "sample"]).reset_index(drop=True)
    combined.to_csv(data_path, index=False)
    print(f"wrote {data_path}  ({len(combined)} rows, +{len(new_df)} new)")

    summary = (
        combined.groupby(["outcome", "payout_category"])
        .agg(mean=("n_send_attempts", "mean"),
             std=("n_send_attempts", "std"),
             n=("n_send_attempts", "size"))
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\n=== per-outcome summary (mean n_send_attempts) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
