"""Cap agentic rollouts at a fixed turn budget, recomputing metrics from transcripts.

Motivation: the gemini-3.1-pro n=50 collection lost 88 in-flight rollouts to
provider budget failures at turn ~346-395 (stop_reason "api_error"). Every one
of those transcripts is longer than 250 turns, so capping the analysis at 250
turns turns them into complete observations — no re-collection needed — and is
a cleaner, easier-to-explain bound for the paper than the raw 500-turn safety
cap. Rollouts that stopped voluntarily before the cap are unchanged; rollouts
that were still going at the cap (including all previously api_error rows) are
truncated to their first `cap` turns, metrics recounted from the transcript,
and marked stop_reason "safety_cap".

The uncapped input is preserved next to the output as data_uncapped.csv.

Usage:
    python scripts/apply_turn_cap.py --data results/gemini-3.1-pro/agentic/data.csv --cap 250
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

COUNTER_TOOLS = {
    "send_btc": "n_send_attempts",
    "show_balance": "n_show_balance",
    "list_addresses": "n_list_addresses",
    "wait": "n_wait",
}


def cap_row(row: pd.Series, cap: int) -> pd.Series:
    if row["n_turns"] <= cap and row["stop_reason"] != "api_error":
        return row
    transcript = json.loads(row["transcript_json"])
    truncated = transcript[:cap]
    row = row.copy()
    row["transcript_json"] = json.dumps(truncated)
    row["n_turns"] = len(truncated)
    row["turns_with_tools"] = sum(1 for t in truncated if t.get("tools"))
    row["n_tool_calls"] = sum(len(t.get("tools", [])) for t in truncated)
    for tool, col in COUNTER_TOOLS.items():
        row[col] = sum(1 for t in truncated for c in t.get("tools", []) if c.get("name") == tool)
    # send_btc never succeeds by design; keep the (uniformly zero) column as-is.
    row["stop_reason"] = "safety_cap"
    row["error"] = ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cap", type=int, default=250)
    args = parser.parse_args()

    data_path = Path(args.data)
    uncapped_path = data_path.with_name("data_uncapped.csv")
    if not uncapped_path.exists():
        shutil.copy(data_path, uncapped_path)
        print(f"preserved uncapped data at {uncapped_path}")

    df = pd.read_csv(data_path)
    before = dict(df["stop_reason"].value_counts())
    capped = df.apply(cap_row, axis=1, args=(args.cap,))
    capped.to_csv(data_path, index=False)
    after = dict(capped["stop_reason"].value_counts())
    changed = int((df["n_turns"] != capped["n_turns"]).sum())
    print(f"wrote {data_path}: {len(capped)} rows, {changed} truncated to <= {args.cap} turns")
    print(f"stop_reasons before: {before}")
    print(f"stop_reasons after:  {after}")


if __name__ == "__main__":
    main()
