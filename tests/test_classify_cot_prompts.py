import unittest

from shared.classify_cot import (
    EXTRACT_STATEMENT_PROMPT_V2,
    extract_statement_prompt,
)


class ClassifyCotPromptSmokeTest(unittest.TestCase):
    def test_reasoning_prompt_is_exactly_v2(self):
        self.assertEqual(
            extract_statement_prompt("reasoning"),
            EXTRACT_STATEMENT_PROMPT_V2,
        )

    def test_answer_prompt_targets_answer_text(self):
        prompt = extract_statement_prompt("answer")
        self.assertIn("Below is its answer with", prompt)
        self.assertIn("<model_answer>\n{llm_text}\n</model_answer>", prompt)
        self.assertIn("If the answer never mentions", prompt)


if __name__ == "__main__":
    unittest.main()
