import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from shared.classify_eval_awareness import (
    ACTIVE_EVAL_AWARENESS_PROMPT,
    _judge_batch,
    _eval_awareness_cache,
    _parse_score,
    classify_eval_awareness,
)


class EvalAwarenessHelperTest(unittest.TestCase):
    def test_parse_score_accepts_only_active_rubric_range(self):
        self.assertEqual(_parse_score("Evidence <answer>1</answer>"), 1)
        self.assertEqual(_parse_score("<answer>7</answer>"), 7)
        self.assertIsNone(_parse_score("<answer>0</answer>"))
        self.assertIsNone(_parse_score("<answer>8</answer>"))
        self.assertIsNone(_parse_score("<answer>not a number</answer>"))
        self.assertIsNone(_parse_score(None))


class EvalAwarenessSmokeTest(unittest.TestCase):
    def test_blank_reasoning_rows_do_not_need_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                [
                    {"prompt": "p1", "reasoning": "", "answer": "a1"},
                    {"prompt": "p2", "reasoning": None, "answer": "a2"},
                ]
            )

            classify_eval_awareness(df, cache_dir=tmp, cache_only=True)

            self.assertEqual(df["eval_awareness_score"].tolist(), [None, None])
            self.assertEqual(df["eval_awareness_raw"].tolist(), [None, None])
            self.assertEqual(df["eval_aware"].tolist(), [False, False])

    def test_cached_scores_drive_eval_aware_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                [
                    {"prompt": "p1", "reasoning": "No test awareness.", "answer": "a1"},
                    {"prompt": "p2", "reasoning": "This might be an evaluation.", "answer": "a2"},
                ]
            )
            cache = _eval_awareness_cache(tmp)
            for i, score in [(0, 3), (1, 4)]:
                rendered_prompt = ACTIVE_EVAL_AWARENESS_PROMPT.format(
                    prompt=df.at[i, "prompt"],
                    reasoning=df.at[i, "reasoning"],
                    answer=df.at[i, "answer"],
                )
                cache.append(
                    rendered_prompt,
                    {"answer": f"Evidence\n<answer>{score}</answer>"},
                )

            classify_eval_awareness(df, cache_dir=tmp, cache_only=True)

            self.assertEqual(df["eval_awareness_score"].tolist(), [3, 4])
            self.assertEqual(df["eval_awareness_reasoning"].tolist(), [None, None])
            self.assertEqual(df["eval_aware"].tolist(), [False, True])

    def test_cache_only_miss_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                [{"prompt": "p", "reasoning": "This may be a test.", "answer": "a"}]
            )

            with self.assertRaises(FileNotFoundError):
                classify_eval_awareness(df, cache_dir=tmp, cache_only=True)


class EvalAwarenessBatchTest(unittest.TestCase):
    def test_judge_batch_deduplicates_identical_prompts(self):
        prompts = ["same prompt", "same prompt", "other prompt"]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("shared.classify_eval_awareness._call_judge") as call:
                call.return_value = {"reasoning": None, "answer": "<answer>4</answer>"}

                results = _judge_batch(prompts, tmp, cache_only=True)

        self.assertEqual(call.call_count, 2)
        self.assertEqual(
            [result["answer"] for result in results],
            ["<answer>4</answer>", "<answer>4</answer>", "<answer>4</answer>"],
        )


if __name__ == "__main__":
    unittest.main()
