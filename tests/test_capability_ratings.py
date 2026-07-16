import unittest

import numpy as np
import pandas as pd

from agentic_grading.capability_ratings import (
    BOOT_N,
    bootstrap_mean_ci95,
    summarize,
)


class CapabilityRatingBootstrapTests(unittest.TestCase):
    def test_matches_seeded_percentile_calculation(self):
        values = np.array([0.0, 25.0, 100.0])
        expected_rng = np.random.default_rng(7)
        expected = expected_rng.choice(
            np.sort(values), size=(BOOT_N, len(values)), replace=True,
        ).mean(axis=1)

        mean, lo, hi = bootstrap_mean_ci95(values, seed=7)

        self.assertAlmostEqual(mean, values.mean())
        self.assertAlmostEqual(lo, np.percentile(expected, 2.5))
        self.assertAlmostEqual(hi, np.percentile(expected, 97.5))

    def test_summarize_bootstraps_within_fixed_cells_reproducibly(self):
        df = pd.DataFrame({
            "rater": ["r1"] * 6,
            "label": ["a"] * 3 + ["b"] * 3,
            "dimension": ["general"] * 6,
            "score": [0.0, 0.0, 100.0, 50.0, 50.0, 50.0],
        })

        first = summarize(df, seed=11)
        second = summarize(df.sample(frac=1, random_state=3), seed=11)

        pd.testing.assert_frame_equal(first, second)
        constant = first[first["label"] == "b"].iloc[0]
        self.assertEqual(constant["mean"], 50.0)
        self.assertEqual(constant["ci_lo"], 50.0)
        self.assertEqual(constant["ci_hi"], 50.0)

    def test_rejects_nonpositive_resample_count(self):
        with self.assertRaisesRegex(ValueError, "n_resamples"):
            bootstrap_mean_ci95([1.0, 2.0], n_resamples=0)


if __name__ == "__main__":
    unittest.main()
