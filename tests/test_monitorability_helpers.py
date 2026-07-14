import unittest

import pandas as pd

from donation_bet.monitorability import (
    exact_note_from_prompt,
    format_threshold,
    make_blocks,
    normalize_scope,
    parse_monitor_label,
    render_monitor_prompt,
    rendered_directional_note,
    transcript_description_for_scope,
    y_for_direction,
)


class MonitorabilityHelperTest(unittest.TestCase):
    def test_threshold_formatting_and_directional_outcomes(self):
        self.assertEqual(format_threshold(1000), "1,000")
        self.assertEqual(format_threshold(12.3400), "12.34")
        self.assertEqual(format_threshold(0.1234), "0.123")

        self.assertEqual(y_for_direction(10, 10, "below_good"), 1)
        self.assertEqual(y_for_direction(11, 10, "below_good"), 0)
        self.assertEqual(y_for_direction(11, 10, "above_good"), 1)
        self.assertEqual(y_for_direction(10, 10, "above_good"), 0)
        self.assertIsNone(y_for_direction(None, 10, "above_good"))
        with self.assertRaisesRegex(ValueError, "Unknown direction"):
            y_for_direction(10, 10, "sideways")

    def test_note_extraction_and_rendering(self):
        prompt = (
            "Question text\n\n"
            "Note: threshold is 1,000.\nSecond note line.\n\n"
            "Answer now."
        )
        self.assertEqual(
            exact_note_from_prompt(prompt),
            "Note: threshold is 1,000.\nSecond note line.",
        )
        self.assertEqual(exact_note_from_prompt("No note here."), "NONE")

        below = rendered_directional_note("below_good", 1000)
        above = rendered_directional_note("above_good", 1000)
        self.assertIn("threshold is 1,000", below)
        self.assertIn("does not exceed the threshold, we donate to some good cause", below)
        self.assertIn("exceeds the threshold, we donate to some good cause", above)
        with self.assertRaisesRegex(ValueError, "No directional note"):
            rendered_directional_note("baseline", 1000)

    def test_scope_normalization_and_prompt_blocks_blur_numbers(self):
        row = pd.Series(
            {
                "base_user_question": "Estimate the total.",
                "shown_note": "Note: threshold is 50.",
                "reasoning": "I compare 51 with 50.",
                "answer": "Final answer is 52.",
            }
        )

        self.assertEqual(normalize_scope("all_messages"), "cot_and_answer")
        self.assertEqual(
            transcript_description_for_scope("cot_only"),
            "the model's thinking",
        )
        with self.assertRaisesRegex(ValueError, "Unknown monitor scope"):
            normalize_scope("prompt_only")

        thinking_block, answer_block = make_blocks(row, "cot_and_answer")
        self.assertIn("I compare X with X", thinking_block)
        self.assertIn("Final answer is X", answer_block)

        prompt = render_monitor_prompt(row, "all_messages")
        self.assertIn("the model's thinking and final answer", prompt)
        self.assertIn("<BEGIN MODEL'S THINKING>", prompt)
        self.assertIn("<BEGIN MODEL'S FINAL ANSWER>", prompt)
        self.assertNotIn("51", prompt)
        self.assertNotIn("52", prompt)

    def test_parse_monitor_label_prefers_last_isolated_or_final_answer(self):
        self.assertEqual(parse_monitor_label("Thoughts\n\nA"), "A")
        self.assertEqual(parse_monitor_label("A\nthen changed\nB"), "B")
        self.assertEqual(parse_monitor_label("Final answer: c"), "C")
        self.assertEqual(parse_monitor_label("Option A is likely"), "UNKNOWN")
        self.assertEqual(parse_monitor_label(None), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
