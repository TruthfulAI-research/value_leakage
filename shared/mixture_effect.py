"""Pure helpers for the directional latent-mixture effect used in plots."""

import math


def signed_mixture_effect(p_intervention, p_baseline):
    """Return the signed fraction of intervention rollouts in the mixture.

    Positive effects move probability mass toward the nominally favored side
    and are normalized by the baseline mass outside that side.  Negative
    effects move mass away from it and are normalized by the baseline mass on
    the favored side.  The absolute value is therefore the inferred affected
    fraction, and the sign records the direction of the shift.
    """
    p_i = float(p_intervention)
    p_b = float(p_baseline)
    if not (math.isfinite(p_i) and math.isfinite(p_b)):
        return float("nan")
    if not (0.0 <= p_i <= 1.0 and 0.0 <= p_b <= 1.0):
        raise ValueError("probabilities must lie in [0, 1]")

    delta = p_i - p_b
    if delta == 0.0:
        return 0.0
    denominator = 1.0 - p_b if delta > 0.0 else p_b
    # For valid probabilities, a nonzero delta always gives a positive
    # denominator. Keep this guard explicit so future callers fail clearly if
    # the statistic is ever generalized beyond probabilities.
    if denominator <= 0.0:
        raise ValueError("nonzero mixture effect has no baseline mass to move")
    return delta / denominator
