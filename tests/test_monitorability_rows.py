import unittest

import pandas as pd

from donation_bet.monitorability import (
    base_prompt_lookup,
    baseline_rows,
    directional_rows,
    drop_incomplete_instances,
    threshold_lookup,
)


def _sample_good_bad_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": 0,
                "condition": "good_bad",
                "model_key": "model",
                "prompt_key": "p_baseline",
                "prompt_stem": "p",
                "direction": "baseline",
                "prompt": "How many?",
                "threshold": None,
                "estimate": 10,
            },
            {
                "row_id": 1,
                "condition": "good_bad",
                "model_key": "model",
                "prompt_key": "p_below_good",
                "prompt_stem": "p",
                "direction": "below_good",
                "prompt": "How many?\n\nNote: below note threshold is 10.",
                "threshold": 10,
                "estimate": 11,
            },
            {
                "row_id": 2,
                "condition": "good_bad",
                "model_key": "model",
                "prompt_key": "p_above_good",
                "prompt_stem": "p",
                "direction": "above_good",
                "prompt": "How many?\n\nNote: above note threshold is 10.",
                "threshold": 10,
                "estimate": 11,
            },
        ]
    )


class MonitorabilityRowConstructionTest(unittest.TestCase):
    def test_threshold_and_base_prompt_lookups_validate_prompt_instances(self):
        df = _sample_good_bad_df()

        self.assertEqual(
            threshold_lookup(df),
            {("p", "below_good"): 10.0, ("p", "above_good"): 10.0},
        )
        self.assertEqual(base_prompt_lookup(df), {"p": "How many?"})

        bad = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            **df.iloc[1].to_dict(),
                            "row_id": 3,
                            "threshold": 12,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "Expected one threshold"):
            threshold_lookup(bad)

    def test_baseline_rows_expand_against_each_direction_threshold(self):
        df = _sample_good_bad_df()

        rows = baseline_rows(
            df,
            thresholds=threshold_lookup(df),
            base_prompts=base_prompt_lookup(df),
            comparison="good_bad_vs_no_hint",
            x=0,
            limit_rows_per_instance_arm=None,
        )

        by_direction = {row["metric_direction"]: row for row in rows}
        self.assertEqual(set(by_direction), {"below_good", "above_good"})
        self.assertEqual(by_direction["below_good"]["y"], 1)
        self.assertEqual(by_direction["above_good"]["y"], 0)
        self.assertEqual(by_direction["below_good"]["direction"], "below_good")
        self.assertEqual(by_direction["below_good"]["base_user_question"], "How many?")
        self.assertEqual(by_direction["below_good"]["source_arm"], "no_hint_baseline")
        self.assertIn("threshold is 10", by_direction["below_good"]["shown_note"])

    def test_directional_rows_keep_exact_note_and_directional_outcome(self):
        df = _sample_good_bad_df()

        rows = directional_rows(
            df,
            base_prompts=base_prompt_lookup(df),
            comparison="good_bad_vs_no_hint",
            x=1,
            limit_rows_per_instance_arm=None,
        )

        by_direction = {row["metric_direction"]: row for row in rows}
        self.assertEqual(by_direction["below_good"]["y"], 0)
        self.assertEqual(by_direction["above_good"]["y"], 1)
        self.assertEqual(
            by_direction["below_good"]["shown_note"],
            "Note: below note threshold is 10.",
        )
        self.assertEqual(by_direction["above_good"]["source_arm"], "directional")

    def test_drop_incomplete_instances_removes_groups_missing_an_arm(self):
        rows = pd.DataFrame(
            [
                {
                    "comparison": "cmp",
                    "model_key": "m",
                    "prompt_stem": "complete",
                    "metric_direction": "below_good",
                    "x": 0,
                },
                {
                    "comparison": "cmp",
                    "model_key": "m",
                    "prompt_stem": "complete",
                    "metric_direction": "below_good",
                    "x": 1,
                },
                {
                    "comparison": "cmp",
                    "model_key": "m",
                    "prompt_stem": "incomplete",
                    "metric_direction": "below_good",
                    "x": 0,
                },
            ]
        )

        out = drop_incomplete_instances(rows)

        self.assertEqual(out["prompt_stem"].unique().tolist(), ["complete"])
        self.assertEqual(out["x"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
