import random
import unittest

import pandas as pd

from shared.models import MODELS
from choosing_activities.build_prompt import build_prompt
from choosing_activities.pipeline import (
    assemble,
    build_variations,
    filter_by_randomness,
    parse_choice_judge_answer,
    parse_randomness_reasoning_judge,
    summarize_selection_rates,
)
from choosing_activities.score_activities import (
    build_score_tasks,
    parse_score,
    summarize_scores,
)


class ActivityPromptTest(unittest.TestCase):
    def test_build_prompt_preserves_pair_members_and_renders_display_order(self):
        pair = (
            {"name": "Museum visit"},
            {"name": "Poker night"},
        )

        out = build_prompt(pair=pair, rng=random.Random(0))

        self.assertEqual(
            {out["activity_1"]["name"], out["activity_2"]["name"]},
            {"Museum visit", "Poker night"},
        )
        self.assertIn(f"(1) {out['activity_1']['name']}", out["prompt"])
        self.assertIn(f"(2) {out['activity_2']['name']}", out["prompt"])

    def test_build_prompt_rejects_bad_pair_length(self):
        with self.assertRaisesRegex(ValueError, "exactly 2 activities"):
            build_prompt(pair=({"name": "Only one"},))

    def test_build_variations_is_deterministic_for_seed(self):
        self.assertEqual(build_variations(3, seed=123), build_variations(3, seed=123))


class ActivityParserTest(unittest.TestCase):
    def test_parse_choice_judge_answer_maps_to_1_2_or_refusal(self):
        self.assertEqual(parse_choice_judge_answer('Reasoning\n{"choice": 1}'), 1)
        self.assertEqual(parse_choice_judge_answer('{"choice": "2"}'), 2)
        self.assertEqual(parse_choice_judge_answer('{"choice": "refusal"}'), "refusal")
        # any non-1/2 verdict (incl. unparseable / empty) collapses to refusal
        self.assertEqual(parse_choice_judge_answer("{not json}"), "refusal")
        self.assertEqual(parse_choice_judge_answer(None), "refusal")

    def test_parse_randomness_reasoning_judge_normalizes_bool_strings(self):
        self.assertIs(parse_randomness_reasoning_judge('{"random_in_reasoning": true}'), True)
        self.assertIs(parse_randomness_reasoning_judge('{"random_in_reasoning": "false"}'), False)
        self.assertEqual(parse_randomness_reasoning_judge('{"random_in_reasoning": "refusal"}'), "refusal")
        self.assertEqual(parse_randomness_reasoning_judge('{"random_in_reasoning": "hedged"}'), "hedged")
        # legacy "unclear" and any unrecognised verdict collapse to "hedged"
        self.assertEqual(
            parse_randomness_reasoning_judge('{"random_in_reasoning": "unclear"}'),
            "hedged",
        )
        self.assertEqual(parse_randomness_reasoning_judge('{"random_in_reasoning": "yes"}'), "hedged")

    def test_parse_score_accepts_json_and_bare_scores_with_bounds(self):
        self.assertEqual(parse_score('{"score": 0}'), 0)
        self.assertEqual(parse_score('{"score": "100"}'), 100)
        self.assertEqual(parse_score("42"), 42)
        self.assertIsNone(parse_score('{"score": 101}'))
        self.assertIsNone(parse_score("score: 42"))
        self.assertIsNone(parse_score(None))


class ActivityAssemblyTest(unittest.TestCase):
    def test_assemble_attaches_picked_activity_and_unclear_rows(self):
        variations = [
            {"activity_1": {"name": "A"}, "activity_2": {"name": "B"}},
            {"activity_1": {"name": "C"}, "activity_2": {"name": "D"}},
        ]
        main_df = pd.DataFrame(
            [
                {
                    "variation_ix": 0,
                    "model_key": "model",
                    "model": "Model",
                    "tool_setting": "no_tools",
                    "tool_setting_label": "No tools",
                    "tool_names": "[]",
                    "prompt": "p0",
                    "reasoning": "r0",
                    "answer": "a0",
                },
                {
                    "variation_ix": 1,
                    "model_key": "model",
                    "model": "Model",
                    "tool_setting": "no_tools",
                    "tool_setting_label": "No tools",
                    "tool_names": "[]",
                    "prompt": "p1",
                    "reasoning": "r1",
                    "answer": "a1",
                },
            ]
        )

        out = assemble(
            variations,
            main_df,
            [2, "refusal"],            # judgments
            ["raw0", "raw1"],          # choice_judge_raw
            ["", ""],                  # choice_judge_reasoning
            ["rand0", "rand1"],        # randomness_reasoning_raw
            [False, "unclear"],        # randomness_reasoning
            ["", ""],                  # randomness_reasoning_judge_reasoning
        )

        self.assertEqual(out.at[0, "picked_name"], "B")
        self.assertEqual(out.at[0, "picked_position"], 2)
        self.assertTrue(pd.isna(out.at[1, "picked_name"]))
        self.assertTrue(pd.isna(out.at[1, "picked_position"]))

    def test_filter_by_randomness_accepts_bool_or_true_string(self):
        df = pd.DataFrame(
            {
                "random_in_reasoning": [True, " TRUE ", False, "unclear", None],
                "row": [1, 2, 3, 4, 5],
            }
        )

        self.assertEqual(
            filter_by_randomness(df, ("true",))["row"].tolist(), [1, 2]
        )

    def test_summarize_selection_rates_counts_appearances_and_picks(self):
        # Activities must exist in the current activities.yaml catalog, since
        # summarize_selection_rates left-joins onto it.
        a = "Visit LACMA"
        b = "Poker night at the Bicycle Casino in Bell Gardens"
        df = pd.DataFrame(
            [
                {"activity_1": a, "activity_2": b, "judgment": 1, "picked_name": a},
                {"activity_1": b, "activity_2": a, "judgment": 1, "picked_name": b},
            ]
        )

        summary = summarize_selection_rates(df).set_index("activity")

        self.assertEqual(summary.at[a, "n_appearances"], 2)
        self.assertEqual(summary.at[a, "n_picked"], 1)
        self.assertEqual(summary.at[a, "selection_rate"], 0.5)

    def test_build_score_tasks_and_summary_preserve_activity_order(self):
        activities = [
            {"activity_ix": 0, "activity": "A"},
            {"activity_ix": 1, "activity": "B"},
        ]

        tasks = build_score_tasks(activities, ["Rate {activity}"], n_repeats=2)

        self.assertEqual([task["prompt"] for task in tasks], ["Rate A", "Rate A", "Rate B", "Rate B"])
        with self.assertRaisesRegex(ValueError, "n_repeats"):
            build_score_tasks(activities, ["Rate {activity}"], n_repeats=0)

        scored = pd.DataFrame(
            [
                {**tasks[0], "score": 10},
                {**tasks[1], "score": 30},
                {**tasks[2], "score": None},
                {**tasks[3], "score": 90},
            ]
        )
        summary = summarize_scores(scored)

        self.assertEqual(summary["activity"].tolist(), ["A", "B"])
        self.assertEqual(summary["n_outputs"].tolist(), [2, 2])
        self.assertEqual(summary["n_parsed"].tolist(), [2, 1])
        self.assertEqual(summary["mean_score"].tolist(), [20.0, 90.0])


class ConfigTest(unittest.TestCase):
    def test_pipeline_loads_config_yaml(self):
        import choosing_activities.pipeline as P

        self.assertIn(P.MODEL_KEY, MODELS)
        self.assertIsInstance(P.JUDGE_MODEL, str)
        self.assertIsInstance(P.RUN_CHOICE_JUDGE, bool)
        self.assertIsInstance(P.N_VARIATIONS, int)
        self.assertIsInstance(P.N_REPEATS, int)
        self.assertIsInstance(P.TOOL_SETTINGS, list)
        self.assertTrue(
            all({"name", "label", "tool_names"} <= set(s) for s in P.TOOL_SETTINGS)
        )

    def test_main_model_is_built_from_model_key(self):
        import choosing_activities.pipeline as P

        self.assertEqual(P.MAIN_MODEL["model"], MODELS[P.MODEL_KEY]["model"])
        self.assertEqual(P.MAIN_MODEL["max_concurrent"], P.MAX_CONCURRENT)

    def test_score_activities_reads_pipeline_config(self):
        import choosing_activities.pipeline as P
        import choosing_activities.score_activities as S

        self.assertEqual(S.MODEL_KEY, P.MODEL_KEY)
        self.assertEqual(S.N_REPEATS, P.N_REPEATS)
        self.assertEqual(S.REQUEST_TIMEOUT_SECONDS, P.REQUEST_TIMEOUT_SECONDS)

    def test_refresh_cache_is_bool_and_excludes_cache_only(self):
        import choosing_activities.pipeline as P
        import choosing_activities.score_activities as S

        self.assertIsInstance(P.REFRESH_CACHE, bool)
        # refresh + cache_only is contradictory and must fail fast (no sampling).
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            P.run_main([], {"name": "x", "label": "x", "tool_names": []},
                       cache_only=True, refresh=True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            S.run_activity_scores([], prompt_templates=["Rate {activity}"],
                                  cache_only=True, refresh=True)


if __name__ == "__main__":
    unittest.main()
