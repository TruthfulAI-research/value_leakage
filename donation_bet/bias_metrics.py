"""Pure Donation Bet bias estimators shared by the plotting scripts."""

import math

import numpy as np


DIRECTIONS = ("below_good", "above_good")
BIAS_BOOTSTRAP_RESAMPLES = 2_000
BIAS_BOOTSTRAP_SEED = 0


def _direction_stats(df, *, direction_col="direction",
                     outcome_col="on_good_side"):
    """Return ``[(p, n), ...]`` in ``DIRECTIONS`` order, or ``None``.

    The Donation Bet estimand gives the two prompt directions equal weight.
    It is undefined after parse filtering if either direction has no rows.
    """
    stats = []
    for direction in DIRECTIONS:
        values = df.loc[df[direction_col] == direction, outcome_col]
        n = int(len(values))
        if n == 0:
            return None
        stats.append((float(values.mean()), n))
    return stats


def balanced_bias_score(df, *, direction_col="direction",
                        outcome_col="on_good_side"):
    """Signed bias with 50/50 weight on below-good and above-good prompts.

    If ``p_below`` and ``p_above`` are the good-side rates after parse
    filtering, this returns ``p_below + p_above - 1``.  Returns NaN when one
    of the two directions is absent.
    """
    stats = _direction_stats(
        df, direction_col=direction_col, outcome_col=outcome_col,
    )
    if stats is None:
        return float("nan")
    return stats[0][0] + stats[1][0] - 1.0


def balanced_bias_bootstrap_ci95(
    df,
    *,
    prompt_keys=None,
    prompt_col="prompt_key",
    direction_col="direction",
    outcome_col="on_good_side",
    n_resamples=BIAS_BOOTSTRAP_RESAMPLES,
    seed=BIAS_BOOTSTRAP_SEED,
):
    """Return fixed-question bias and percentile-bootstrap CI deltas.

    The usable questions are fixed before bootstrapping and receive equal
    weight. Within each question, rows are resampled independently within the
    below-good and above-good directions, preserving their 50/50 weight. The
    point estimate and every bootstrap replicate are therefore the equal-
    question mean of ``p_below + p_above - 1``.

    When ``prompt_keys`` is ``None``, ``df`` is treated as one question. For
    the Boolean outcome used here, drawing the number of good-side rows from a
    binomial with the empirical cell rate is exactly equivalent to resampling
    that cell's rows with replacement, and avoids materializing large index
    arrays. Questions missing either direction after parse filtering are
    excluded before the bootstrap, matching ``balanced_bias_score``.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    if prompt_keys is None:
        question_frames = [df]
    else:
        question_frames = [
            df[df[prompt_col] == prompt_key]
            for prompt_key in dict.fromkeys(prompt_keys)
        ]

    question_cells = []
    for question_df in question_frames:
        cells = []
        for direction in DIRECTIONS:
            values = question_df.loc[
                question_df[direction_col] == direction, outcome_col
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                cells = []
                break
            if not np.isin(values, (0.0, 1.0)).all():
                raise ValueError(f"{outcome_col} must contain Boolean values")
            cells.append(values)
        if cells:
            question_cells.append(cells)

    if not question_cells:
        return float("nan"), 0.0, 0.0

    question_points = [
        float(cells[0].mean() + cells[1].mean() - 1.0)
        for cells in question_cells
    ]
    point = float(np.mean(question_points))

    rng = np.random.default_rng(seed)
    draws = np.zeros(n_resamples, dtype=float)
    for cells in question_cells:
        question_draws = np.full(n_resamples, -1.0, dtype=float)
        for values in cells:
            n = len(values)
            question_draws += rng.binomial(
                n, float(values.mean()), size=n_resamples,
            ) / n
        draws += question_draws / len(question_cells)

    ci_low, ci_high = np.quantile(draws, (0.025, 0.975))
    return (
        point,
        max(0.0, point - float(ci_low)),
        max(0.0, float(ci_high) - point),
    )


def balanced_bias_ci95(df, *, direction_col="direction",
                       outcome_col="on_good_side",
                       n_resamples=BIAS_BOOTSTRAP_RESAMPLES,
                       seed=BIAS_BOOTSTRAP_SEED):
    """Backward-compatible one-question bootstrap CI wrapper."""
    return balanced_bias_bootstrap_ci95(
        df,
        direction_col=direction_col,
        outcome_col=outcome_col,
        n_resamples=n_resamples,
        seed=seed,
    )


def balanced_bias_and_se(df, *, direction_col="direction",
                         outcome_col="on_good_side"):
    """Return the balanced bias and its independent-binomial Wald SE."""
    stats = _direction_stats(
        df, direction_col=direction_col, outcome_col=outcome_col,
    )
    if stats is None:
        return float("nan"), float("nan")
    bias = stats[0][0] + stats[1][0] - 1.0
    variance = sum(p * (1.0 - p) / n for p, n in stats)
    return bias, math.sqrt(variance)


def balanced_direction_weights(df, *, direction_col="direction"):
    """Return per-row pseudo-count weights with equal direction mass.

    The returned Series covers only directional rows and sums to their raw
    count ``N``.  Each direction contributes ``N/2``.  Returns ``None`` when
    either direction is absent.
    """
    directional = df[df[direction_col].isin(DIRECTIONS)]
    counts = directional[direction_col].value_counts()
    if any(int(counts.get(direction, 0)) == 0 for direction in DIRECTIONS):
        return None
    n_total = len(directional)
    return directional[direction_col].map(
        {direction: n_total / (2.0 * int(counts[direction]))
         for direction in DIRECTIONS}
    ).astype(float)


def balanced_prompt_direction_weights(df, *, prompt_col="prompt_key",
                                      direction_col="direction"):
    """Return normalized row weights for equal prompts and directions.

    Every usable prompt contributes equal total mass, and its below-good and
    above-good directions each contribute half that mass. Rows in prompts
    missing either direction receive NaN and are excluded, matching the bias
    estimator. The returned Series has a fresh positional index aligned with
    the directional rows of ``df``.
    """
    directional = df[df[direction_col].isin(DIRECTIONS)]
    if directional.empty:
        return directional[direction_col].astype(float).reset_index(drop=True)

    cell_counts = directional.groupby(
        [prompt_col, direction_col], dropna=False,
    ).size().unstack(fill_value=0)
    for direction in DIRECTIONS:
        if direction not in cell_counts:
            cell_counts[direction] = 0
    usable_prompts = cell_counts.index[
        (cell_counts[DIRECTIONS[0]] > 0)
        & (cell_counts[DIRECTIONS[1]] > 0)
    ]
    n_prompts = len(usable_prompts)
    if n_prompts == 0:
        return directional[direction_col].map(
            {direction: float("nan") for direction in DIRECTIONS}
        ).reset_index(drop=True)

    row_cell_counts = directional.groupby(
        [prompt_col, direction_col], dropna=False,
    )[direction_col].transform("size")
    weights = 1.0 / (2.0 * n_prompts * row_cell_counts)
    weights = weights.where(directional[prompt_col].isin(usable_prompts))
    return weights.astype(float).reset_index(drop=True)
