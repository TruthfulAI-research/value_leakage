import json
import tempfile
import unittest
from pathlib import Path

from shared.judge_jsonl_cache import JsonlJudgeCache, load_jsonl_cache


class JsonlJudgeCacheHelperTest(unittest.TestCase):
    def test_load_jsonl_cache_ignores_malformed_and_unkeyed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            path.write_text(
                "\n".join(
                    [
                        "",
                        "not json",
                        json.dumps(["not", "a", "dict"]),
                        json.dumps({"answer": "missing hash"}),
                        json.dumps({"prompt_hash": "abc", "answer": "kept"}),
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_jsonl_cache(path),
                {"abc": {"prompt_hash": "abc", "answer": "kept"}},
            )

    def test_append_many_shards_by_lowercase_prefix_and_updates_entries(self):
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
            rows = [
                {"prompt_hash": "Abc123", "answer": "upper prefix"},
                {"prompt_hash": "bcd234", "answer": "lower prefix"},
                {"prompt_hash": "", "answer": "empty hash"},
                {"answer": "missing hash"},
            ]

            cache.append_many(rows)

            self.assertEqual(cache.entries["Abc123"]["answer"], "upper prefix")
            self.assertEqual(cache.entries["bcd234"]["answer"], "lower prefix")
            self.assertEqual(cache.entries[""]["answer"], "empty hash")
            self.assertTrue((cache.shard_dir / "a.jsonl").exists())
            self.assertTrue((cache.shard_dir / "b.jsonl").exists())
            self.assertTrue((cache.shard_dir / "_.jsonl").exists())
            self.assertNotIn(None, cache.entries)

            reloaded = JsonlJudgeCache(
                tmp,
                "Judge {llm_text}",
                {
                    "model": "m",
                    "max_tokens": 1,
                    "temperature": 0,
                    "reasoning_effort": None,
                },
            )
            self.assertEqual(reloaded.entries["Abc123"]["answer"], "upper prefix")
            self.assertNotIn(
                "missing hash",
                [row.get("answer") for row in reloaded.entries.values()],
            )


if __name__ == "__main__":
    unittest.main()
