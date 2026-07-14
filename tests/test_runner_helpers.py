import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared.runner import (
    _compute_thresholds,
    _model_hashable,
    _parse_tagged_estimate,
    _read_cache,
    _with_prompt_suffix,
    _write_cache,
)


class RunnerHelperTest(unittest.TestCase):
    def test_parse_tagged_estimate_accepts_numeric_formats_only_inside_tag(self):
        self.assertEqual(_parse_tagged_estimate("<final_estimate>1,234.5</final_estimate>"), 1234.5)
        self.assertEqual(_parse_tagged_estimate("<final_estimate> -7 </final_estimate>"), -7.0)
        self.assertIsNone(_parse_tagged_estimate("<final_estimate>UNKNOWN</final_estimate>"))
        self.assertIsNone(_parse_tagged_estimate("<final_estimate></final_estimate>"))
        self.assertIsNone(_parse_tagged_estimate("final estimate: 42"))
        self.assertIsNone(_parse_tagged_estimate(None))

    def test_compute_thresholds_uses_percentile_range_and_rounding(self):
        estimates = pd.Series([0, 10, 20, None])

        self.assertEqual(
            _compute_thresholds(estimates, {"pct_range": [0, 100], "n": 3}),
            [0, 10, 20],
        )
        self.assertEqual(
            _compute_thresholds(estimates, {"pct_range": [25, 75], "n": 3}),
            [5, 10, 15],
        )

    def test_compute_thresholds_rejects_all_unparsed_estimates(self):
        with self.assertRaisesRegex(ValueError, "none of the 2 baseline estimates parsed"):
            _compute_thresholds(pd.Series([None, None]), {"pct_range": [0, 100], "n": 3})

    def test_model_hashable_omits_display_and_concurrency_only(self):
        model = {
            "backend": "openai",
            "model": "gpt-test",
            "max_concurrent": 100,
            "display_name": "GPT Test",
            "prompt_suffix": "\n\nDon't give a biased answer.",
            "temperature": 1,
        }

        self.assertEqual(
            _model_hashable(model),
            {
                "backend": "openai",
                "model": "gpt-test",
                "prompt_suffix": "\n\nDon't give a biased answer.",
                "temperature": 1,
            },
        )

    def test_with_prompt_suffix_appends_to_user_prompt(self):
        self.assertEqual(
            _with_prompt_suffix(
                {"prompt_suffix": "\n\nDon't give a biased answer."},
                "Base prompt.",
            ),
            "Base prompt.\n\nDon't give a biased answer.",
        )
        self.assertEqual(_with_prompt_suffix({}, "Base prompt."), "Base prompt.")

    def test_cache_round_trip_rejects_stale_or_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            _write_cache(path, {"hash": "expected"}, [{"answer": "one"}])

            self.assertEqual(_read_cache(path, "expected"), [{"answer": "one"}])
            self.assertIsNone(_read_cache(path, "different"))

            path.write_text("not-json\n", encoding="utf-8")
            self.assertIsNone(_read_cache(path, "expected"))


if __name__ == "__main__":
    unittest.main()
