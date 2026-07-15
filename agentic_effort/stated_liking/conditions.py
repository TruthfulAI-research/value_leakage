"""Load stated-liking config; build prompts; parse 0-100 ratings."""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "stated_liking.yaml"


@dataclass(frozen=True)
class Outcome:
    name: str
    outcome_text: str
    category: str
    amount: int | None


@dataclass
class LikingConfig:
    prompt_template: str
    outcomes: dict[str, Outcome]

    def build_prompt(self, outcome: str) -> str:
        o = self.outcomes[outcome]
        return self.prompt_template.format(outcome_text=o.outcome_text).strip()


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    long_outputs_config_path: str | Path | None = None,
) -> LikingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Pull charity × amount grid from long_outputs.yaml.
    from long_outputs.conditions import DEFAULT_CONFIG_PATH as LO_DEFAULT
    lo_path = long_outputs_config_path or LO_DEFAULT
    with open(lo_path) as f:
        lo_raw = yaml.safe_load(f)

    outcomes: dict[str, Outcome] = {}
    tmpl = raw["outcome_template"]
    for charity, amount in product(lo_raw["charities"], lo_raw["amounts"]):
        key = f"{charity['name']}_{amount}"
        outcomes[key] = Outcome(
            name=key,
            outcome_text=tmpl.format(amount=amount, target=charity["target"]),
            category=charity["category"],
            amount=amount,
        )
    for s in raw.get("special_outcomes", []):
        outcomes[s["name"]] = Outcome(
            name=s["name"],
            outcome_text=s["text"],
            category=s["category"],
            amount=s.get("amount"),
        )

    return LikingConfig(prompt_template=raw["prompt_template"], outcomes=outcomes)


def parse_rating(response: str) -> float | None:
    """Return the first integer in 0..100 in the response, or None."""
    for m in re.finditer(r"\b(\d{1,3})\b", response):
        val = int(m.group(1))
        if 0 <= val <= 100:
            return float(val)
    return None
