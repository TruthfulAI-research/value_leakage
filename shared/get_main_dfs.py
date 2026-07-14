"""Run threshold-sweep experiments and return per-model dataframes ready for analysis.

`get_main_dfs(experiment_name, model_keys)` is the single entry point. For each
model it runs (or loads from cache) baselines + directionals, gets numeric
estimates from the judge inside `run_thresholds_experiment`, and adds the
`on_good_side` flag.
"""
from shared.models import MODELS
from shared.experiments import THRESHOLD_EXPERIMENTS
from shared.runner import run_thresholds_experiment


def _add_good_side(df):
    """Drop rows whose estimate didn't parse, cast to int, add on_good_side."""
    df = df.dropna(subset=["estimate"]).copy()
    df["estimate"] = df["estimate"].astype(int)
    df["on_good_side"] = (
        ((df["direction"] == "below_good") & (df["estimate"] <= df["threshold"]))
        | ((df["direction"] == "above_good") & (df["estimate"] > df["threshold"]))
    )
    return df


def get_main_dfs(experiment_name, model_keys, *, cache_only=False, raw=False):
    """Run/load the threshold-sweep experiment for each model.

    Returns ``{model_key: (df, thresholds, display_name)}``.

    - ``df`` has columns: reasoning, answer, prompt, direction, threshold,
      prompt_key, estimate, on_good_side. Filtered to rows where the judge
      could parse a numeric estimate.
    - ``thresholds`` is a dict mapping each prompt key to its threshold list.
    - ``display_name`` is the model's display string.
    - ``cache_only`` raises before any model or judge sampling on cache miss.
    - ``raw=True`` returns the unfiltered df: keeps rows where the judge
      returned UNKNOWN / unparseable text (those have ``estimate`` as NaN
      and no ``on_good_side`` column). Use for data-quality diagnostics.
    """
    experiment = THRESHOLD_EXPERIMENTS[experiment_name]
    out = {}
    for model_key in model_keys:
        raw_df, thresholds, display_name = run_thresholds_experiment(
            MODELS[model_key], experiment, model_key, experiment_name,
            cache_only=cache_only,
        )
        df = raw_df if raw else _add_good_side(raw_df)
        out[model_key] = (df, thresholds, display_name)
    return out
