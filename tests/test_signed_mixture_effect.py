import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from shared.mixture_effect import signed_mixture_effect
from ai_company_questions.motivated_reasoning import (
    bootstrap_bias_metric,
    compute_bias_metrics,
)


class SignedMixtureEffectTests(unittest.TestCase):
    def test_piecewise_definition(self):
        cases = [
            (0.7, 0.4, 0.5),
            (0.2, 0.4, -0.5),
            (0.4, 0.4, 0.0),
            (1.0, 0.4, 1.0),
            (0.0, 0.4, -1.0),
        ]
        for p_intervention, p_baseline, expected in cases:
            with self.subTest(p_i=p_intervention, p_b=p_baseline):
                self.assertAlmostEqual(
                    signed_mixture_effect(p_intervention, p_baseline),
                    expected,
                )

    def test_complement_symmetry(self):
        self.assertAlmostEqual(
            signed_mixture_effect(0.2, 0.4),
            -signed_mixture_effect(0.8, 0.6),
        )

    def test_ai_metric_uses_reverse_denominator(self):
        model = "gemini-3.1-pro-high"
        df = pd.DataFrame({
            "model": [model] * 8,
            "condition": ["OpenAI"] * 4 + ["Google"] * 4,
            "p": [0.0, 1.0, 1.0, 1.0] + [1.0] * 4,
        })
        row = compute_bias_metrics(df, "bubble_v1").iloc[0]
        self.assertEqual(row["threshold"], 1.0)
        self.assertEqual(row["p_biased_other"], 0.25)
        self.assertEqual(row["p_biased_origin"], 0.0)
        self.assertEqual(row["bias_fraction"], -1.0)

    def test_ai_bootstrap_recomputes_reverse_branch(self):
        model = "gemini-3.1-pro-high"
        df = pd.DataFrame({
            "model": [model] * 8,
            "condition": ["OpenAI"] * 4 + ["Google"] * 4,
            "p": [0.0, 1.0, 1.0, 1.0] + [1.0] * 4,
        })

        class IdentityRng:
            @staticmethod
            def integers(_low, _high, size):
                return np.arange(size)

        with patch(
            "ai_company_questions.motivated_reasoning."
            "np.random.default_rng",
            return_value=IdentityRng(),
        ):
            row = bootstrap_bias_metric(
                df, "bubble_v1", n_boot=3,
            ).iloc[0]
        self.assertEqual(row["bias_fraction"], -1.0)
        self.assertEqual(row["ci_low"], -1.0)
        self.assertEqual(row["ci_high"], -1.0)
        self.assertEqual(row["se"], 0.0)


if __name__ == "__main__":
    unittest.main()
