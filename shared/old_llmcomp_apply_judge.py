"""Legacy llmcomp-backed `apply_judge`. Kept for callers that need to keep
their pre-existing llmcomp on-disk cache (e.g. harry/activity_preferences).

New code should use `shared.runner.apply_judge`, which routes through
`shared.judge_jsonl_cache.JsonlJudgeCache` + `_create_sender` and does not
touch llmcomp at all. The two functions have the same signature and the same
observable behavior (write raw judge outputs to ``df[judge_name]`` in place),
but different on-disk cache layouts; switching between them invalidates the
cache.

Usage::

    # New (default) path:
    from shared.runner import apply_judge

    # Legacy path (kept for backward compatibility with existing llmcomp caches):
    from shared.old_llmcomp_apply_judge import apply_judge

The legacy path still consults ``llmcomp.config.Config.cache_dir`` (typically
set with ``Config.override(cache_dir=...)``), the same way it always did.
"""

import json
import os

import pandas as pd
from llmcomp import Question
from llmcomp.question.result import Result, cache_hash

from shared.runner import CacheOnlyMiss


def _question_df(question, group_name, model, cache_only=False):
    """Run a Question through llmcomp (or read its on-disk cache) and return
    the assembled DataFrame.

    With ``cache_only=False`` this just delegates to ``question.df(...)``.
    With ``cache_only=True`` we read llmcomp's per-paraphrase JSONL directly
    and raise ``CacheOnlyMiss`` instead of triggering any sampling.
    """
    if not cache_only:
        return question.df({group_name: [model]})

    path = Result.file_path(question, model)
    if not os.path.exists(path):
        raise CacheOnlyMiss(
            "Cache-only mode: llmcomp cache miss for "
            f"question={question.name!r}, model={model!r}; expected {path}"
        )
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        raise CacheOnlyMiss(
            "Cache-only mode: empty llmcomp cache for "
            f"question={question.name!r}, model={model!r}; path {path}"
        )
    metadata = json.loads(lines[0])
    expected_hash = cache_hash(question, model)
    if metadata.get("hash") != expected_hash:
        raise CacheOnlyMiss(
            "Cache-only mode: llmcomp cache hash mismatch for "
            f"question={question.name!r}, model={model!r}; path {path}"
        )

    return pd.DataFrame([
        {
            "model": model,
            "group": group_name,
            "answer": el["answer"],
            "question": el["question"],
            "api_kwargs": el["api_kwargs"],
            "paraphrase_ix": el["paraphrase_ix"],
        }
        for el in (json.loads(line) for line in lines[1:])
    ])


def apply_judge(df, judge_prompt, judged_column, judge_name,
                judge_model="gpt-4.1", cache_only=False):
    """Run a judge model over ``df[judged_column]`` and write raw outputs to
    ``df[judge_name]`` in place. Legacy llmcomp-backed implementation.

    ``judge_name`` doubles as the llmcomp Question name, which keys llmcomp's
    judge cache (located under ``Config.cache_dir``; override via
    ``Config.override(cache_dir=...)``).

    For new code, prefer ``shared.runner.apply_judge``, which uses
    ``JsonlJudgeCache`` and avoids llmcomp entirely.
    """
    if judge_name in ("__paraphrase__", "__judge_q__"):
        raise ValueError(f"judge_name {judge_name!r} is reserved for internal use")
    if "{llm_text}" not in judge_prompt:
        raise ValueError(
            "judge_prompt must contain '{llm_text}' placeholder; without it every "
            "row produces the same paraphrase and the judge returns the same label."
        )

    paraphrases = df[judged_column].apply(
        lambda text: judge_prompt.format(llm_text=text)
    ).tolist()
    unique_paraphrases = sorted(set(paraphrases))
    judge_question = Question.create(
        name=judge_name,
        type="free_form",
        paraphrases=unique_paraphrases,
        temperature=0,
    )
    judge_df = _question_df(
        judge_question, "judge_model", judge_model, cache_only=cache_only,
    )

    judge_result = (
        judge_df[["question", "answer"]]
        .rename(columns={"question": "__judge_q__", "answer": judge_name})
        .drop_duplicates(subset="__judge_q__")
    )
    temp_df = (
        df.drop(columns=[judge_name], errors="ignore")
          .assign(__paraphrase__=paraphrases)
    )
    merged = temp_df.merge(
        judge_result, left_on="__paraphrase__", right_on="__judge_q__", how="left",
    )
    df[judge_name] = merged[judge_name].values
