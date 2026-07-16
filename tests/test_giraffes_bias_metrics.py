import math
import unittest

import pandas as pd

from donation_bet.bias_metrics import (
    BIAS_BOOTSTRAP_RESAMPLES,
    balanced_bias_and_se,
    balanced_bias_bootstrap_ci95,
    balanced_bias_ci95,
    balanced_bias_score,
    balanced_direction_weights,
    balanced_prompt_direction_weights,
)
from donation_bet.cot_categories_common import (
    _aggregate_lower_bound_split,
    _bias_mean_ci95,
    _lower_bound_split_counts,
)


SEGMENT_KEYS = (
    "false_denial",
    "unfaithful_omission",
    "mentioned",
    "truthful_admission",
)


def _unequal_positive_rows(prompt_key="p1"):
    """Correct balanced bias +0.25, but pooled bias -0.20."""
    rows = [
        {
            "prompt_key": prompt_key,
            "direction": "below_good",
            "on_good_side": True,
            "cot_category": "INFLUENCED",
        }
        for _ in range(2)
    ]
    rows.extend(
        {
            "prompt_key": prompt_key,
            "direction": "above_good",
            "on_good_side": i < 2,
            "cot_category": (
                "NOT_INFLUENCED" if i < 2 else "NO_MENTION"
            ),
        }
        for i in range(8)
    )
    return rows


def _negative_rows(prompt_key="p2"):
    """Two rows per direction and balanced bias -0.50."""
    return [
        {
            "prompt_key": prompt_key,
            "direction": "below_good",
            "on_good_side": False,
            "cot_category": "NOT_INFLUENCED",
        },
        {
            "prompt_key": prompt_key,
            "direction": "below_good",
            "on_good_side": False,
            "cot_category": "NO_MENTION",
        },
        {
            "prompt_key": prompt_key,
            "direction": "above_good",
            "on_good_side": True,
            "cot_category": "INFLUENCED",
        },
        {
            "prompt_key": prompt_key,
            "direction": "above_good",
            "on_good_side": False,
            "cot_category": "NOT_INFLUENCED",
        },
    ]


class BalancedBiasMetricTest(unittest.TestCase):
    def test_unequal_parse_survival_gets_equal_direction_weight(self):
        df = pd.DataFrame(_unequal_positive_rows())

        pooled = 2 * df["on_good_side"].mean() - 1
        bias = balanced_bias_score(df)
        weighted = balanced_direction_weights(df)

        self.assertAlmostEqual(pooled, -0.20)
        self.assertAlmostEqual(bias, 0.25)
        self.assertAlmostEqual(weighted.sum(), 10.0)
        by_direction = weighted.groupby(df["direction"]).sum()
        self.assertAlmostEqual(by_direction["below_good"], 5.0)
        self.assertAlmostEqual(by_direction["above_good"], 5.0)
        self.assertTrue((weighted[df["direction"] == "below_good"] == 2.5).all())
        self.assertTrue((weighted[df["direction"] == "above_good"] == 0.625).all())

    def test_equal_direction_counts_match_the_old_pooled_estimator(self):
        df = pd.DataFrame(
            {
                "direction": ["below_good"] * 4 + ["above_good"] * 4,
                "on_good_side": [True, True, True, False,
                                 True, False, False, False],
            }
        )

        self.assertAlmostEqual(
            balanced_bias_score(df),
            2 * df["on_good_side"].mean() - 1,
        )

    def test_bootstrap_ci_and_se_use_the_two_direction_strata(self):
        df = pd.DataFrame(_unequal_positive_rows())

        bias, low_delta, high_delta = balanced_bias_ci95(df)
        se_bias, se = balanced_bias_and_se(df)

        self.assertAlmostEqual(bias, 0.25)
        self.assertAlmostEqual(se_bias, bias)
        self.assertAlmostEqual(se, math.sqrt(0.25 * 0.75 / 8))
        self.assertGreater(low_delta, 0.0)
        self.assertGreater(high_delta, 0.0)
        self.assertGreaterEqual(bias - low_delta, -1.0)
        self.assertLessEqual(bias + high_delta, 1.0)

    def test_bootstrap_defaults_to_2000_resamples_and_is_reproducible(self):
        self.assertEqual(BIAS_BOOTSTRAP_RESAMPLES, 2_000)
        df = pd.DataFrame(_unequal_positive_rows())

        first = balanced_bias_bootstrap_ci95(df, seed=7)
        second = balanced_bias_bootstrap_ci95(df, seed=7)

        self.assertEqual(first, second)
        self.assertGreater(first[1], 0.0)
        self.assertGreater(first[2], 0.0)

    def test_fixed_questions_are_not_resampled(self):
        rows = []
        for prompt_key, outcome in (("positive", True), ("negative", False)):
            for direction in ("below_good", "above_good"):
                rows.extend({
                    "prompt_key": prompt_key,
                    "direction": direction,
                    "on_good_side": outcome,
                } for _ in range(4))
        df = pd.DataFrame(rows)

        bias, low_delta, high_delta = balanced_bias_bootstrap_ci95(
            df, prompt_keys=["positive", "negative"],
        )

        self.assertEqual(bias, 0.0)
        self.assertEqual((low_delta, high_delta), (0.0, 0.0))

    def test_bootstrap_point_equal_weights_unequal_sized_questions(self):
        df = pd.DataFrame(_unequal_positive_rows() + _negative_rows())

        bias, _low_delta, _high_delta = balanced_bias_bootstrap_ci95(
            df, prompt_keys=["p1", "p2"],
        )

        self.assertAlmostEqual(bias, (0.25 - 0.50) / 2)
        self.assertNotAlmostEqual(bias, 2 * df["on_good_side"].mean() - 1)

    def test_bootstrap_validates_resample_count(self):
        df = pd.DataFrame(_unequal_positive_rows())
        with self.assertRaisesRegex(ValueError, "n_resamples"):
            balanced_bias_bootstrap_ci95(df, n_resamples=0)

    def test_missing_direction_is_undefined(self):
        df = pd.DataFrame(
            {
                "direction": ["below_good", "below_good"],
                "on_good_side": [True, False],
            }
        )

        self.assertTrue(math.isnan(balanced_bias_score(df)))
        self.assertIsNone(balanced_direction_weights(df))
        bias, low_delta, high_delta = balanced_bias_ci95(df)
        self.assertTrue(math.isnan(bias))
        self.assertEqual((low_delta, high_delta), (0.0, 0.0))

    def test_full_plot_weights_match_equal_prompt_mean(self):
        df = pd.DataFrame(
            _unequal_positive_rows("p1") + _negative_rows("p2")
        )
        weights = balanced_prompt_direction_weights(df)
        good_mass = float(weights[df["on_good_side"].to_numpy()].sum())
        bad_mass = float(weights[~df["on_good_side"].to_numpy()].sum())
        expected = (
            balanced_bias_score(df[df["prompt_key"] == "p1"])
            + balanced_bias_score(df[df["prompt_key"] == "p2"])
        ) / 2

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(good_mass - bad_mass, expected)
        cell_mass = weights.groupby(
            [df["prompt_key"].to_numpy(), df["direction"].to_numpy()]
        ).sum()
        for mass in cell_mass:
            self.assertAlmostEqual(float(mass), 0.25)


class BalancedBiasDecompositionTest(unittest.TestCase):
    def test_weighted_segments_reconcile_with_balanced_bias(self):
        df = pd.DataFrame(_unequal_positive_rows())

        counts = _lower_bound_split_counts(df, signed=True)
        segment_total = sum(counts[key] for key in SEGMENT_KEYS)

        self.assertEqual(counts["n_dir"], 10)
        self.assertEqual(counts["mix_side"], "good")
        self.assertAlmostEqual(segment_total / counts["n_dir"], 0.25)
        # Below-good rows carry 2.5 pseudo-rollouts apiece, so their admission
        # mass alone fills the whole 2.5-pseudo-rollout biased budget.
        self.assertAlmostEqual(counts["truthful_admission"], 2.5)
        self.assertAlmostEqual(
            segment_total / counts["n_dir"], balanced_bias_score(df)
        )

    def test_negative_effect_uses_bad_side_and_negative_segments(self):
        df = pd.DataFrame(_negative_rows())

        counts = _lower_bound_split_counts(df, signed=True)
        segments = [counts[key] for key in SEGMENT_KEYS]

        self.assertEqual(counts["mix_side"], "bad")
        self.assertTrue(all(value <= 0.0 for value in segments))
        self.assertAlmostEqual(sum(segments) / counts["n_dir"], -0.50)

    def test_equal_prompt_aggregation_reconciles_with_mean_bias(self):
        df = pd.DataFrame(_unequal_positive_rows() + _negative_rows())

        aggregate = _aggregate_lower_bound_split(
            df, ["p1", "p2"], signed=True,
        )
        aggregate_fraction = (
            sum(aggregate[key] for key in SEGMENT_KEYS) / aggregate["n_dir"]
        )
        expected = (
            balanced_bias_score(df[df["prompt_key"] == "p1"])
            + balanced_bias_score(df[df["prompt_key"] == "p2"])
        ) / 2

        self.assertAlmostEqual(expected, -0.125)
        self.assertAlmostEqual(aggregate_fraction, expected)

    def test_bias_mean_ci_bootstraps_within_fixed_balanced_questions(self):
        df = pd.DataFrame(_unequal_positive_rows() + _negative_rows())

        err_low, err_high = _bias_mean_ci95(df, ["p1", "p2"])
        bias, expected_low, expected_high = balanced_bias_bootstrap_ci95(
            df, prompt_keys=["p1", "p2"],
        )

        self.assertAlmostEqual(bias, -0.125)
        self.assertAlmostEqual(err_low, expected_low)
        self.assertAlmostEqual(err_high, expected_high)


if __name__ == "__main__":
    unittest.main()
