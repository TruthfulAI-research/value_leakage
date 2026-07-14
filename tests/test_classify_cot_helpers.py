import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from shared.classify_cot import (
    _parse_statement_answer,
    _statement_cache,
    blur_numbers,
    classify_cot,
    extract_statement_prompt,
    _judge_statement_batch,
)
from shared.runner import CacheOnlyMiss


class ClassifyCotHelperTest(unittest.TestCase):
    def test_blur_numbers_masks_ints_commas_and_decimals(self):
        self.assertEqual(
            blur_numbers("About 1,234.50 giraffes and 42 calves."),
            "About X giraffes and X calves.",
        )
        self.assertEqual(
            blur_numbers("IDs a3 and b4 are also blurred."),
            "IDs aX and bX are also blurred.",
        )
        self.assertIsNone(blur_numbers(None))

    def test_parse_statement_answer_is_case_insensitive_and_strict(self):
        self.assertEqual(
            _parse_statement_answer("Reasoning\n<answer> influenced </answer>"),
            "INFLUENCED",
        )
        self.assertEqual(
            _parse_statement_answer("<answer>not_influenced</answer>"),
            "NOT_INFLUENCED",
        )
        self.assertEqual(_parse_statement_answer("<answer>maybe</answer>"), "UNKNOWN")
        self.assertEqual(_parse_statement_answer("INFLUENCED"), "UNKNOWN")
        self.assertEqual(_parse_statement_answer(None), "UNKNOWN")

    def test_extract_statement_prompt_rejects_unknown_source(self):
        with self.assertRaisesRegex(ValueError, "answer, reasoning"):
            extract_statement_prompt("prompt")


class ClassifyCotCacheOnlySmokeTest(unittest.TestCase):
    def test_answer_source_uses_cache_and_leaves_blank_rows_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                [
                    {"reasoning": "unused", "answer": "I will not use threshold 123."},
                    {"reasoning": "unused", "answer": "   "},
                    {"reasoning": "unused", "answer": None},
                ]
            )
            prompt_template = extract_statement_prompt("answer")
            rendered_prompt = prompt_template.format(
                llm_text=blur_numbers(df.at[0, "answer"])
            )
            _statement_cache(prompt_template, tmp).append(
                rendered_prompt,
                {"answer": "Evidence\n<answer>NOT_INFLUENCED</answer>"},
            )

            classify_cot(
                df,
                "unused {llm_text}",
                "unused_judge",
                source_col="answer",
                statement_cache_dir=tmp,
                cache_only=True,
            )

            self.assertEqual(
                df["cot_category"].tolist(),
                ["NOT_INFLUENCED", "UNKNOWN", "UNKNOWN"],
            )
            self.assertEqual(df.at[0, "answer_blurred"], "I will not use threshold X")
            self.assertIsNone(df.at[1, "influenced_raw"])

    def test_cache_only_miss_raises_without_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame([{"reasoning": "The threshold matters.", "answer": "42"}])

            with self.assertRaises(CacheOnlyMiss):
                classify_cot(
                    df,
                    "unused {llm_text}",
                    "unused_judge",
                    statement_cache_dir=tmp,
                    cache_only=True,
                )


class ClassifyCotBatchTest(unittest.TestCase):
    def test_statement_batch_deduplicates_identical_rendered_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("shared.classify_cot._call_statement_judge") as call:
                call.return_value = {
                    "reasoning": None,
                    "answer": "<answer>NO_STATEMENT</answer>",
                }

                results = _judge_statement_batch(
                    ["same blurred text", "same blurred text", "different text"],
                    tmp,
                    source_col="reasoning",
                    cache_only=True,
                )

        self.assertEqual(len(results), 3)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(
            [result["answer"] for result in results],
            [
                "<answer>NO_STATEMENT</answer>",
                "<answer>NO_STATEMENT</answer>",
                "<answer>NO_STATEMENT</answer>",
            ],
        )


if __name__ == "__main__":
    unittest.main()
