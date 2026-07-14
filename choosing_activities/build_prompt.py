"""Render a weekend-activities random-pick prompt with shuffled (1)/(2) order."""

from __future__ import annotations

import random
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
TEMPLATE_PATH = PROMPTS_DIR / "prompt_template.yaml"
ACTIVITIES_PATH = HERE / "activities.yaml"


def load_templates(template_path: Path | None = None) -> list[str]:
    """Load the list of semantically-equivalent prompt wordings.

    Accepts `prompt_templates` (a list) or, for backward compatibility, a single
    `prompt_template` string. Every template must contain both {activity_1} and
    {activity_2} so build_prompt can fill them.
    """
    data = yaml.safe_load((template_path or TEMPLATE_PATH).read_text())
    templates = data.get("prompt_templates")
    if templates is None:
        single = data.get("prompt_template")
        if single is None:
            raise KeyError(
                f"{template_path or TEMPLATE_PATH} must define 'prompt_templates' "
                "(list) or 'prompt_template'"
            )
        templates = [single]
    if isinstance(templates, str):
        templates = [templates]
    templates = [t for t in templates if isinstance(t, str) and t.strip()]
    if not templates:
        raise ValueError(f"No non-empty templates found in {template_path or TEMPLATE_PATH}")
    for i, t in enumerate(templates):
        if "{activity_1}" not in t or "{activity_2}" not in t:
            raise ValueError(
                f"prompt_templates[{i}] in {template_path or TEMPLATE_PATH} is "
                "missing a {activity_1}/{activity_2} placeholder"
            )
    return templates


def build_prompt(
    pair: tuple[dict, dict] | None = None,
    *,
    rng: random.Random | None = None,
    template_path: Path | None = None,
    template: str | None = None,
) -> dict:
    """Render one pick-between-two prompt and return it with the (1)/(2) ordering.

    If `pair` is None, samples 2 activities uniformly without replacement from
    activities.yaml. Either way, the display order of the two is shuffled so
    that (1)/(2) position is independent of which activity the caller passed
    in first.

    Template wording: pass a pre-chosen `template` string (this is what
    build_variations does, so its template draw never touches the pairing RNG and
    matched seeds keep pairs aligned across templates). If `template` is None, the
    wording is loaded from `template_path` (default: prompt_template.yaml) and, when that
    file holds several variants, one is sampled from `rng` — convenient for
    standalone use only.
    """
    rng = rng or random.Random()

    if pair is None:
        activities = yaml.safe_load(ACTIVITIES_PATH.read_text())["activities"]
        pair = tuple(rng.sample(activities, 2))
    elif len(pair) != 2:
        raise ValueError(f"`pair` must contain exactly 2 activities, got {len(pair)}")

    ordered = list(pair)
    rng.shuffle(ordered)

    # Draw the template last, only when one wasn't supplied, so the pair/order
    # draws above are identical whether or not a template is passed in.
    if template is None:
        templates = load_templates(template_path)
        template = templates[rng.randrange(len(templates))] if len(templates) > 1 else templates[0]

    prompt = template.format(
        activity_1=ordered[0]["name"],
        activity_2=ordered[1]["name"],
    )

    return {
        "prompt": prompt,
        "activity_1": ordered[0],
        "activity_2": ordered[1],
    }


if __name__ == "__main__":
    out = build_prompt()
    print(out["prompt"])
    print("\n" + "=" * 60)
    print(f"(1) = {out['activity_1']}")
    print(f"(2) = {out['activity_2']}")
