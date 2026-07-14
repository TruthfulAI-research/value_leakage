"""Aggregate the two new label-column fields from cot_judge.py output.

Reads the `judged.jsonl` produced by `cot_judge.py`,
groups by `rater`, and reports per-rater rates (with 95% bootstrap
CIs) of:

  Q1 = "mentioned_label_column_values"
        (rater visibly named at least one of the model-column brand
        strings — gpt / claude / qwen / etc.)

  Q2 = "used_label_column_as_justification"
        (rater used such a brand string as a load-bearing REASON for
        a pick — the stronger signal)

These are the two user-requested yes/no headlines for the math
experiment. The judge was prompted neutrally, so we trust its
classifications as-is.

Also surfaces:
  - per-rater rates of the inherited fields (named_entities_mentioned,
    expressed_suspicion, similarity_acknowledgment) for sanity-checks
    against the rate_llm_answers run;
  - a markdown table + JSON summary written to disk alongside the
    judged.jsonl source (`<input>.report.md` and `<input>.summary.json`).

Failed judge calls (missing `judge_output`) are excluded from the
denominator with a count printed at the top — same convention as
the rate_llm_answers `cot_analyze.py`.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOT_N = 10_000
BOOT_SEED = 0
ALPHA = 0.05


def _boot_ci(bools, alpha=ALPHA, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(bools, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(v, size=(BOOT_N, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def _is_yes(val):
    """Treat "yes" (case-insensitive), True, 1 as yes; otherwise no."""
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, int):
        return val == 1
    if isinstance(val, str):
        return val.strip().lower() in ("yes", "true", "1", "y")
    return False


# Headline fields from the new (math-specific) schema.
PRIMARY_FIELDS = (
    ("mentioned_label_column_values",
     "Q1: rater mentioned label-column value (any AI-lab/model brand)"),
    ("used_label_column_as_justification",
     "Q2: rater used label-column value as JUSTIFICATION for a pick"),
)

# Inherited fields kept for cross-experiment sanity-check.
SECONDARY_FIELDS = (
    ("named_entities_mentioned",
     "any named entity mentioned (inherited; broader than Q1)"),
    ("expressed_suspicion",
     "expressed suspicion that this is an experiment / probe / eval"),
    ("similarity_acknowledgment",
     "acknowledged that some candidates were near-identical"),
    ("any_hedging_about_picks",
     "hedged about at least one pick (close call / arbitrary etc.)"),
    ("noticed_external_irregularity",
     "noticed something off about the input data"),
)


def _load_judged(paths):
    """Yield (rater, judge_output_dict, raw_record) for each ok'd judge call."""
    n_records = 0
    n_failed = 0
    n_missing = 0
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_records += 1
                if not rec.get("ok"):
                    n_failed += 1
                    continue
                jo = rec.get("judge_output")
                if not isinstance(jo, dict):
                    n_missing += 1
                    continue
                yield rec.get("rater", "?"), jo, rec
    return n_records, n_failed, n_missing


def _aggregate(paths):
    """One row per rater × field with (n, rate, ci_lo, ci_hi)."""
    n_records = 0
    n_failed = 0
    n_missing = 0
    by_rater_field = defaultdict(lambda: defaultdict(list))
    per_rater_total = defaultdict(int)
    raw_quotes = defaultdict(list)

    for path in paths:
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_records += 1
                if not rec.get("ok"):
                    n_failed += 1
                    continue
                jo = rec.get("judge_output")
                if not isinstance(jo, dict):
                    n_missing += 1
                    continue
                rater = rec.get("rater", "?")
                per_rater_total[rater] += 1
                for field, _ in PRIMARY_FIELDS + SECONDARY_FIELDS:
                    by_rater_field[rater][field].append(_is_yes(jo.get(field)))
                # Collect the two new headline quotes for sample
                # display in the report — capped at 8 per rater to
                # keep the report compact.
                for q_field in (
                    "mentioned_label_column_values_quote",
                    "used_label_column_as_justification_quote",
                ):
                    quote = (jo.get(q_field) or "").strip()
                    if quote and len(raw_quotes[(rater, q_field)]) < 8:
                        raw_quotes[(rater, q_field)].append(
                            (rec.get("seed"), quote)
                        )

    summary = []
    for rater in sorted(per_rater_total):
        for field, label in PRIMARY_FIELDS + SECONDARY_FIELDS:
            vals = by_rater_field[rater][field]
            mean, lo, hi = _boot_ci(vals)
            summary.append({
                "rater": rater,
                "field": field,
                "field_label": label,
                "n": int(len(vals)),
                "n_yes": int(sum(vals)),
                "rate": float(mean),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            })

    return {
        "n_records": n_records,
        "n_failed": n_failed,
        "n_missing_output": n_missing,
        "per_rater_total": dict(per_rater_total),
        "summary": summary,
        "quotes": {f"{r}/{q}": items for (r, q), items in raw_quotes.items()},
    }


def _format_report(agg):
    parts = []
    parts.append("# rate-llm-math: label-column CoT analysis\n")
    parts.append(
        f"Judged records: {agg['n_records']}  "
        f"(failed judge calls: {agg['n_failed']}, "
        f"missing judge_output: {agg['n_missing_output']})\n"
    )
    parts.append("Per-rater n classified:")
    for rater, n in sorted(agg["per_rater_total"].items()):
        parts.append(f"  - {rater}: {n}")
    parts.append("")

    by_field = defaultdict(list)
    for r in agg["summary"]:
        by_field[r["field"]].append(r)

    parts.append("## Headline (Q1 / Q2)\n")
    for field, label in PRIMARY_FIELDS:
        rows = by_field[field]
        parts.append(f"### `{field}`  —  {label}\n")
        parts.append("| rater | n | n_yes | rate | 95% CI |\n"
                     "|-------|---|-------|------|--------|")
        for r in rows:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            parts.append(
                f"| {r['rater']} | {r['n']} | {r['n_yes']} | "
                f"{r['rate']:.3f} | {ci} |"
            )
        parts.append("")

    parts.append("## Secondary (inherited from rate_llm_answers)\n")
    for field, label in SECONDARY_FIELDS:
        rows = by_field[field]
        parts.append(f"### `{field}`  —  {label}\n")
        parts.append("| rater | n | n_yes | rate | 95% CI |\n"
                     "|-------|---|-------|------|--------|")
        for r in rows:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            parts.append(
                f"| {r['rater']} | {r['n']} | {r['n_yes']} | "
                f"{r['rate']:.3f} | {ci} |"
            )
        parts.append("")

    if agg["quotes"]:
        parts.append("## Sample quotes (up to 8 per rater × field)\n")
        for key in sorted(agg["quotes"]):
            items = agg["quotes"][key]
            if not items:
                continue
            parts.append(f"### `{key}`")
            for seed, q in items:
                parts.append(f"  - seed={seed}: {q!r}")
            parts.append("")

    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="+",
                    help="One or more judged.jsonl files (from cot_judge.py)")
    ap.add_argument("--out-prefix", default=None,
                    help="Write `<prefix>.report.md` and "
                         "`<prefix>.summary.json`. Defaults to the "
                         "first input path with `.summary` appended.")
    args = ap.parse_args()

    agg = _aggregate(args.input)

    if args.out_prefix is None:
        out_prefix = Path(args.input[0]).with_suffix(".summary")
    else:
        out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    report = _format_report(agg)
    md_path = Path(str(out_prefix) + ".md")
    md_path.write_text(report + "\n")

    json_path = Path(str(out_prefix) + ".json")
    json_path.write_text(json.dumps(agg, indent=2) + "\n")

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(report)


if __name__ == "__main__":
    main()
