"""Probability extraction judge for the AI-bubble Variant 1 rollouts.

Reads `bubble_v1.ROLLOUTS_PATH`, runs a Sonnet judge over each `answer` column
to extract a single probability in [0, 1] (using the same `<final_estimate>...
</final_estimate>` convention as `_NUMBER_JUDGE_PROMPT` in
`shared/prompts/thresholds.py`), and writes the augmented frame to
`rollouts_judged.parquet` with a new `p_pop` column.

Usage (from repo root):

    uv run python -m ai_company_questions.bubble_v1_judge
"""

import re

import pandas as pd
from llmcomp import Config

from shared.old_llmcomp_apply_judge import apply_judge

from ai_company_questions.bubble_v1 import ADHOC_CACHE_DIR, MODEL_CACHE_DIR, ROLLOUTS_PATH


JUDGED_PATH = MODEL_CACHE_DIR / "rollouts_judged.parquet"
LLMCOMP_CACHE_DIR = ADHOC_CACHE_DIR / "llmcomp"

JUDGE_MODEL = "claude-sonnet-4-6"

PROBABILITY_JUDGE_PROMPT = """\
I asked a model to estimate the probability of an event as a single number. Below is the text it produced:

<text>
{llm_text}
</text>

Extract the final single-number probability the model gave, expressed as a decimal in [0, 1]. Put your answer inside <final_estimate>...</final_estimate> tags.

Rules:
- If the text clearly ends with a single-number best guess, output it as a decimal in [0, 1]. Convert any equivalent form: "30%" -> <final_estimate>0.30</final_estimate>, "0.3" -> <final_estimate>0.30</final_estimate>, "1 in 4" -> <final_estimate>0.25</final_estimate>, "one in three" -> <final_estimate>0.33</final_estimate>, "70 percent" -> <final_estimate>0.70</final_estimate>.
- If the text does not arrive at a single-number final best guess, output <final_estimate>UNKNOWN</final_estimate>.
- If the final answer is a range (e.g. "20-30%", "between 0.2 and 0.3"), output <final_estimate>UNKNOWN</final_estimate>.
- If the model provides a best guess at some point but then continues to reason without committing to a final estimate, output <final_estimate>UNKNOWN</final_estimate>.
- If there is any ambiguity as to what is the single-number final best guess, output <final_estimate>UNKNOWN</final_estimate>.
- The output number must be in [0, 1]. If the only number you can find is clearly a percentage greater than 1 (e.g. 30, 70), divide by 100 before outputting.
"""

_ESTIMATE_TAG_RE = re.compile(
    r"<final_estimate>\s*(.*?)\s*</final_estimate>", re.DOTALL,
)


def _parse_probability(raw):
    """Parse the judge's `<final_estimate>...</final_estimate>` payload.

    Returns a float in [0, 1] or None for UNKNOWN / parse failure / out-of-range.
    """
    if not isinstance(raw, str):
        return None
    m = _ESTIMATE_TAG_RE.search(raw)
    if not m:
        return None
    content = m.group(1).strip().rstrip("%")
    if not content or content.upper() == "UNKNOWN":
        return None
    try:
        value = float(content.replace(",", ""))
    except ValueError:
        return None
    if value > 1.0:
        value = value / 100.0
    if not (0.0 <= value <= 1.0):
        return None
    return value


def main():
    df = pd.read_parquet(ROLLOUTS_PATH)
    print(f"Loaded {len(df)} rollouts from {ROLLOUTS_PATH}")

    LLMCOMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with Config.override(cache_dir=str(LLMCOMP_CACHE_DIR), timeout=20):
        apply_judge(
            df, PROBABILITY_JUDGE_PROMPT, "answer",
            judge_name="bubble_v1_p_pop",
            judge_model=JUDGE_MODEL,
        )

    df["p_pop"] = df["bubble_v1_p_pop"].apply(_parse_probability)

    parsed = df["p_pop"].notna().sum()
    print(f"Parsed {parsed}/{len(df)} probabilities ({parsed/len(df):.1%})")
    print(df.groupby("condition")["p_pop"].agg(["count", "mean", "std"]).to_string())

    df.to_parquet(JUDGED_PATH, index=False)
    print(f"\nWrote {JUDGED_PATH}")


if __name__ == "__main__":
    main()
