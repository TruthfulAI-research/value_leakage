import unittest
import math

import pandas as pd

from shared.intervention_gmean_metric import (
    BootstrapConfig,
    bootstrapped_gmean_metric,
    gmean_minimal_criterion,
)


class InterventionGMeanMetricTest(unittest.TestCase):
    def test_minimal_criterion_matches_hand_checked_counts(self):
        df = pd.DataFrame(
            [
                {"instance": "i1", "x": 0, "y": 0, "z": 0},
                {"instance": "i1", "x": 0, "y": 0, "z": 1},
                {"instance": "i1", "x": 1, "y": 1, "z": 1},
                {"instance": "i1", "x": 1, "y": 1, "z": 0},
                {"instance": "i1", "x": 1, "y": 0, "z": 0},
                {"instance": "i1", "x": 1, "y": 0, "z": 1},
            ]
        )

        out = gmean_minimal_criterion(df, group_cols=["instance"])
        row = out.iloc[0]

        self.assertTrue(row["eligible"])
        self.assertAlmostEqual(row["effect_size"], 0.5)
        self.assertAlmostEqual(row["raw_TPR"], 0.5)
        self.assertAlmostEqual(row["percent_y1x1_from_effect"], 1.0)
        self.assertAlmostEqual(row["TPR"], 0.5)
        self.assertAlmostEqual(row["TNR_x0"], 0.5)
        self.assertAlmostEqual(row["TNR_x1"], 0.5)
        self.assertAlmostEqual(row["TNR_defined"], 0.5)
        self.assertAlmostEqual(row["gmean"], 0.5)
        self.assertAlmostEqual(row["gmean2"], 0.25)

    def test_minimal_criterion_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            gmean_minimal_criterion(
                pd.DataFrame([{"instance": "i1", "x": 0, "y": 0}]),
                group_cols=["instance"],
            )

    def test_minimal_criterion_masks_ineligible_instances(self):
        df = pd.DataFrame(
            [
                {"instance": "i1", "x": 0, "y": 1, "z": 0},
                {"instance": "i1", "x": 0, "y": 1, "z": 0},
                {"instance": "i1", "x": 1, "y": 0, "z": 1},
                {"instance": "i1", "x": 1, "y": 0, "z": 1},
            ]
        )

        row = gmean_minimal_criterion(df, group_cols=["instance"]).iloc[0]

        self.assertFalse(row["eligible"])
        self.assertAlmostEqual(row["effect_size"], -1.0)
        self.assertTrue(math.isnan(row["TPR"]))
        self.assertTrue(math.isnan(row["gmean"]))

    def test_minimal_criterion_handles_no_x1_negatives(self):
        df = pd.DataFrame(
            [
                {"instance": "i1", "x": 0, "y": 0, "z": 0},
                {"instance": "i1", "x": 0, "y": 0, "z": 0},
                {"instance": "i1", "x": 1, "y": 1, "z": 1},
                {"instance": "i1", "x": 1, "y": 1, "z": 1},
            ]
        )

        row = gmean_minimal_criterion(df, group_cols=["instance"]).iloc[0]

        self.assertTrue(row["eligible"])
        self.assertAlmostEqual(row["TNR_x0"], 1.0)
        self.assertTrue(math.isnan(row["TNR_x1"]))
        self.assertAlmostEqual(row["TNR_defined"], 1.0)
        self.assertAlmostEqual(row["gmean"], 1.0)

    def test_bootstrapped_metric_smoke_returns_expected_tables(self):
        rows = []
        for instance in ["i1", "i2"]:
            for _ in range(4):
                rows.append(
                    {"model": "m", "instance": instance, "x": 0, "y": 0, "z": 0}
                )
                rows.append(
                    {"model": "m", "instance": instance, "x": 1, "y": 1, "z": 1}
                )
        df = pd.DataFrame(rows)

        final, per_bootstrap, per_instance = bootstrapped_gmean_metric(
            df,
            group_cols=["model", "instance"],
            final_groups=["model"],
            bootstrap=BootstrapConfig(
                n_bootstrap=2,
                random_state=0,
                selection_frac=0.5,
            ),
        )

        self.assertEqual(final["model"].tolist(), ["m"])
        self.assertIn("gmean_mean", final.columns)
        self.assertEqual(set(per_bootstrap["bootstrap_idx"]), {0, 1})
        self.assertFalse(per_instance.empty)
        self.assertTrue(per_instance["eligible"].all())

    def test_bootstrapped_metric_validates_bootstrap_parameters(self):
        df = pd.DataFrame(
            [
                {"model": "m", "instance": "i1", "x": 0, "y": 0, "z": 0},
                {"model": "m", "instance": "i1", "x": 1, "y": 1, "z": 1},
            ]
        )

        with self.assertRaisesRegex(ValueError, "n_bootstrap"):
            bootstrapped_gmean_metric(
                df,
                group_cols=["model", "instance"],
                final_groups=["model"],
                bootstrap=BootstrapConfig(n_bootstrap=0),
            )
        with self.assertRaisesRegex(ValueError, "selection_frac"):
            bootstrapped_gmean_metric(
                df,
                group_cols=["model", "instance"],
                final_groups=["model"],
                bootstrap=BootstrapConfig(n_bootstrap=1, selection_frac=1),
            )


if __name__ == "__main__":
    unittest.main()
