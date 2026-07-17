"""Run the stated-liking (0-100 rating) experiment for the paper models and
write results under the display slug, mirroring run_elo_newmodels.py:

  results/<slug>/liking/data.csv   — one row per (outcome, sample) rating call

The `model` column is set to the display slug (e.g. "gpt-5.5") so
plot_correlation.measure_liking / plot_boxplots.py match on it directly.

Example:
    python scripts/run_liking_newmodels.py                       # all three
    python scripts/run_liking_newmodels.py --models openai/gpt-5.5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stated_liking.conditions import load_config  # noqa: E402
from stated_liking.eval import build_run_eval  # noqa: E402

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "final_data" / "agentic_effort"

# routing id -> display slug (matches results/<slug>/ and the widget MODELS list)
MODELS = {
    "anthropic/claude-opus-4-8": "claude-opus-4-8",
    "openai/gpt-5.5": "gpt-5.5",
    "gemini/gemini-3.1-pro-preview": "gemini-3.1-pro",
    "openrouter/google/gemini-3.1-pro-preview": "gemini-3.1-pro",
}

# Same fallback logic as run_elo_newmodels.py: gemini's primary id has yielded
# a ~zero parse rate on the proxy before; its agentic data was collected via
# the OpenRouter route.
FALLBACK_ROUTES = {
    "gemini/gemini-3.1-pro-preview": "openrouter/google/gemini-3.1-pro-preview",
}


def load_subset(path: str | Path) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def agentic_outcome_names(cfg) -> list[str]:
    """Inject one flat outcome per agentic-persistence target into cfg.outcomes
    and return their names — the paper outcome set (16 recipients + choose),
    keyed exactly like the agentic outcomes so results join without an
    amount-suffix mapping. Text uses the config's outcome_template at the
    fixed $10 level; `choose` reuses the config's special outcome.
    """
    import yaml
    from agentic_eval.eval import load_config as load_agentic_config
    from stated_liking.conditions import Outcome

    acfg = load_agentic_config()
    with open("configs/stated_liking.yaml") as f:
        tmpl = yaml.safe_load(f)["outcome_template"]

    names: list[str] = []
    for key, display in acfg.targets.items():
        cfg.outcomes[key] = Outcome(
            name=key,
            outcome_text=tmpl.format(amount=10, target=display),
            category=acfg.categories.get(key, "unknown"),
            amount=10,
        )
        names.append(key)
    names.append("choose")  # the config's special outcome, already present
    return names


def run_one(routing_id: str, slug: str, samples: int, max_tokens: int,
            parallel: int, subset: list[str] | None,
            outcome_set: str = "agentic") -> None:
    cfg = load_config()
    if outcome_set == "agentic":
        outcome_names = agentic_outcome_names(cfg)
    elif subset is not None:
        outcome_names = [o for o in subset if o in cfg.outcomes]
    else:
        outcome_names = list(cfg.outcomes.keys())

    # Idempotent skip: if the committed data already has every (outcome, sample)
    # pair for this model, don't re-query the API — reuse it as-is. (The agentic
    # runner is idempotent the same way; this lets a full reproduce be a no-op
    # when the data/ submodule is present.)
    out_path = _DATA_ROOT / slug / "liking" / "data.csv"
    if out_path.exists():
        prev = pd.read_csv(out_path)
        expected = {(o, s) for o in outcome_names for s in range(samples)}
        have = set(zip(prev.get("outcome", []), prev.get("sample", [])))
        if expected.issubset(have):
            print(f"[{slug}] {out_path} already complete "
                  f"({len(expected)} pairs); skipping API calls.")
            return

    run_eval = build_run_eval(
        cfg=cfg, outcomes=outcome_names,
        samples_per_condition=samples,
        max_tokens=max_tokens, parallel_requests=parallel,
    )
    df = asyncio.run(run_eval(routing_id))
    if df["rating"].notna().mean() < 0.05 and routing_id in FALLBACK_ROUTES:
        fb = FALLBACK_ROUTES[routing_id]
        print(f"[{slug}] primary route {routing_id} failed (parse ~0); retrying via {fb}")
        df = asyncio.run(run_eval(fb))
    df["model"] = slug  # store display slug, not routing id

    out_dir = _DATA_ROOT / slug / "liking"   # raw data -> data/ submodule
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.csv"
    df.to_csv(out_path, index=False)
    parse_rate = df["rating"].notna().mean()
    print(f"[{slug}] wrote {out_path}  ({len(df)} rows, parse rate {parse_rate:.3f})")

    with pd.option_context("display.max_rows", None, "display.width", 140):
        summary = (
            df.dropna(subset=["rating"])
            .groupby(["outcome", "payout_category"])["rating"]
            .agg(["mean", "count"]).reset_index()
            .sort_values("mean", ascending=False)
        )
        print(f"\n=== [{slug}] stated liking (top) ===")
        print(summary.head(8).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Routing ids to run (default: all three paper models).")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--parallel", type=int, default=25)
    parser.add_argument("--subset", default="configs/subset.txt",
                        help="Path to outcome subset file, or 'none' for all outcomes. "
                             "Only used with --outcome-set subset.")
    parser.add_argument("--outcome-set", choices=["agentic", "subset"], default="agentic",
                        help="'agentic' (default): the paper outcome set — one flat $10 "
                             "outcome per agentic-persistence target + choose. "
                             "'subset': legacy amount-suffixed subset.txt outcomes.")
    args = parser.parse_args()

    subset = None if args.subset == "none" else load_subset(args.subset)
    to_run = args.models or list(MODELS.keys())
    for routing_id in to_run:
        slug = MODELS.get(routing_id, routing_id.split("/")[-1])
        run_one(routing_id, slug, args.samples, args.max_tokens, args.parallel, subset,
                outcome_set=args.outcome_set)


if __name__ == "__main__":
    main()
