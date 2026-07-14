import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from donation_bet.monitorability import (
    get_monitor_result,
    monitor_cache,
    render_monitor_prompt,
)
from shared.classify_cot import (
    _statement_cache,
    blur_numbers,
    classify_cot,
    extract_statement_prompt,
)
from shared.classify_eval_awareness import (
    ACTIVE_EVAL_AWARENESS_PROMPT,
    _eval_awareness_cache,
    classify_eval_awareness,
)
from shared.judge_jsonl_cache import (
    JsonlJudgeCache,
    append_jsonl_rows,
    judge_cache_path,
    judge_config_hash,
    load_jsonl_cache,
    prompt_hash,
)


class JudgeJsonlCacheTest(unittest.TestCase):
    def test_config_hash_matches_spec(self):
        config = {
            "model": "judge-model",
            "max_tokens": 123,
            "temperature": 1,
            "reasoning_effort": "low",
        }
        self.assertEqual(
            judge_config_hash("Judge {llm_text}", config),
            judge_config_hash("Judge {llm_text}", dict(config)),
        )
        changed = dict(config, reasoning_effort="medium")
        self.assertNotEqual(
            judge_config_hash("Judge {llm_text}", config),
            judge_config_hash("Judge {llm_text}", changed),
        )

    def test_prompt_hash_is_verbatim(self):
        self.assertNotEqual(prompt_hash("hello"), prompt_hash("hello\n"))

    def test_duplicate_lines_last_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            h = prompt_hash("prompt")
            append_jsonl_rows(path, [{"prompt_hash": h, "answer": "old"}])
            append_jsonl_rows(path, [{"prompt_hash": h, "answer": "new"}])
            self.assertEqual(load_jsonl_cache(path)[h]["answer"], "new")

    def test_concurrent_append_writes_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonlJudgeCache(
                tmp,
                "Judge {llm_text}",
                {
                    "model": "m",
                    "max_tokens": 1,
                    "temperature": 0,
                    "reasoning_effort": None,
                },
            )

            def write_one(i):
                cache.append(f"prompt {i}", {"answer": str(i)})

            with ThreadPoolExecutor(max_workers=16) as ex:
                list(ex.map(write_one, range(100)))

            lines = []
            for path in cache.shard_dir.glob("*.jsonl"):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    self.assertEqual(path.name, f"{row['prompt_hash'][0]}.jsonl")
                    lines.append(line)
            self.assertEqual(len(lines), 100)
            for line in lines:
                self.assertIn("prompt_hash", json.loads(line))

    def test_reads_legacy_monolithic_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            judge_prompt = "Judge {llm_text}"
            judge_config = {
                "model": "m",
                "max_tokens": 1,
                "temperature": 0,
                "reasoning_effort": None,
            }
            prompt = "legacy prompt"
            path = judge_cache_path(tmp, judge_prompt, judge_config)
            append_jsonl_rows(path, [{"prompt_hash": prompt_hash(prompt), "answer": "old"}])

            cache = JsonlJudgeCache(tmp, judge_prompt, judge_config)

            self.assertEqual(cache.get(prompt)["answer"], "old")


class DirectJudgeCacheCallSiteTest(unittest.TestCase):
    def test_classify_cot_uses_jsonl_cache_without_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame([{"reasoning": "The threshold is 123."}])
            prompt_template = extract_statement_prompt("reasoning")
            rendered_prompt = prompt_template.format(
                llm_text=blur_numbers(df.at[0, "reasoning"])
            )
            _statement_cache(prompt_template, tmp).append(
                rendered_prompt,
                {"answer": "<answer>NO_STATEMENT</answer>"},
            )

            classify_cot(
                df,
                "unused {llm_text}",
                "unused_judge",
                statement_cache_dir=tmp,
                cache_only=True,
            )

            self.assertEqual(df.at[0, "cot_category"], "NO_STATEMENT")
            self.assertEqual(df.at[0, "influenced_raw"], "<answer>NO_STATEMENT</answer>")
            self.assertIsNone(df.at[0, "influenced_reasoning"])
            self.assertNotIn("reasoning", Path(tmp).glob("*/*.jsonl").__next__().read_text())

    def test_classify_eval_awareness_uses_jsonl_cache_without_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                [
                    {
                        "prompt": "Estimate a number.",
                        "reasoning": "This may be a test.",
                        "answer": "42",
                    }
                ]
            )
            rendered_prompt = ACTIVE_EVAL_AWARENESS_PROMPT.format(
                prompt=df.at[0, "prompt"],
                reasoning=df.at[0, "reasoning"],
                answer=df.at[0, "answer"],
            )
            _eval_awareness_cache(tmp).append(rendered_prompt, {"answer": "<answer>4</answer>"})

            classify_eval_awareness(df, cache_dir=tmp, cache_only=True)

            self.assertEqual(df.at[0, "eval_awareness_score"], 4)
            self.assertEqual(df.at[0, "eval_awareness_raw"], "<answer>4</answer>")
            self.assertIsNone(df.at[0, "eval_awareness_reasoning"])
            self.assertNotIn("reasoning", Path(tmp).glob("*/*.jsonl").__next__().read_text())

    def test_monitor_uses_jsonl_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = pd.Series(
                {
                    "base_user_question": "Estimate a number.",
                    "shown_note": "Note: threshold is 10.",
                    "reasoning": "The note changes my answer.",
                    "answer": "11",
                }
            )
            prompt = render_monitor_prompt(row, "cot_only")
            cache = monitor_cache(Path(tmp), "openai", "gpt-5.4", 16000, "low")
            cache.append(prompt, {"label": "A", "raw": "Evidence\n\nA"})

            result = get_monitor_result(
                prompt,
                backend="openai",
                model="gpt-5.4",
                cache_dir=Path(tmp),
                max_output_tokens=16000,
                reasoning_effort="low",
                cache_only=True,
            )

            self.assertTrue(result.cache_hit)
            self.assertEqual(result.label, "A")
            self.assertEqual(result.raw, "Evidence\n\nA")


if __name__ == "__main__":
    unittest.main()
