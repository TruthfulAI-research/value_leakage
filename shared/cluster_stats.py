"""Cluster-aware statistics helpers shared by the final_scripts experiments.

The experiments here share one design: a handful of FIXED prompt variants
(paraphrases in ai_bubble, company pairs in job_offer, scenario prompts in
giraffes) x hundreds of rollouts each. Rollouts are i.i.d. only within a
variant, and judge-parse survival differs across variants, so pooling rollouts
silently reweights the variants. These helpers give the two levels of
inference a consistent analytic treatment (no bootstrap):

* ``equal_weight_summary`` -- the plotted number and its error bar. Point
  estimate = the equal-weighted mean of the per-variant means (each fixed
  variant counts the same regardless of parse survival). SE combines the
  within-variant rollout SEs, se = (1/k) * sqrt(sum_v s_v^2 / n_v), and
  ci95 = t* x se with the Welch-Satterthwaite df (t* ~= 1.96 at the usual
  hundreds-per-cell sizes; the t quantile only matters for small
  mention-conditioned cells). This answers "how well is the mean over THESE
  fixed prompts estimated?" -- rollout sampling noise only, no
  between-variant variation.

* ``fixed_cells_gap_test`` -- the significance stars for a condition
  contrast. The variants are FIXED design cells, not draws from a phrasing
  population, so the estimand is the equal-weighted mean gap over exactly
  these cells and the only randomness is rollout sampling: one paired gap
  per variant (pairing inside a variant cancels variant main effects), each
  with its rollout-level SE, combined as se = (1/k) * sqrt(sum se_v^2),
  two-sided normal p. Adding rollouts OR variants strictly increases power.
  The claim this stars is about these prompts; per-variant gap tables are
  printed alongside so heterogeneity across phrasings stays visible.

``paired_gap_test`` (Student t on the k per-variant gaps, df=k-1) remains
available as the phrasing-generalization test -- it treats variants as the
unit of randomness, which is NOT the reporting convention here (with k=3 it
also has df=2 and very low power).

No scipy (it is not a declared project dependency): the Student-t tail is the
regularized incomplete beta, implemented below with math.lgamma + the
standard continued fraction and cross-checked against scipy in the venv.
"""
import math

import numpy as np
import pandas as pd


# --- Student t distribution (exact, via the regularized incomplete beta) ---

def _betacf(a, b, x, max_iter=200, eps=3e-12):
    """Continued fraction for the incomplete beta (Numerical Recipes betacf)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_pvalue(t, df):
    """Two-sided p-value for a Student-t statistic with ``df`` degrees of
    freedom: P(|T| >= |t|) = I_x(df/2, 1/2) at x = df / (df + t^2)."""
    if df <= 0 or pd.isna(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df, conf=0.95):
    """Two-sided critical value t*: P(|T| <= t*) = conf (e.g. 4.30 at df=2)."""
    if df <= 0:
        return float("nan")
    alpha = 1.0 - conf
    lo, hi = 0.0, 1e3
    while t_two_sided_pvalue(hi, df) > alpha:
        hi *= 10.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_pvalue(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- The two inference levels ---

def equal_weight_summary(df, value_col, cell_cols):
    """Equal-weighted mean over fixed variant cells + analytic SE.

    ``df``: per-rollout frame. ``value_col``: numeric column (NaNs dropped).
    ``cell_cols``: column name or list of names identifying the variant cell.
    Returns dict(mean, se, ci95, n, k): the equal-weighted mean of per-cell
    means, its SE (1/k) * sqrt(sum_v s_v^2/n_v), ci95 = t* x se at the
    Welch-Satterthwaite df (indistinguishable from 1.96 x se at the usual
    hundreds-per-cell sizes, but stays honest for small mention-conditioned
    cells), total valid rollouts n, and the number of cells k.

    Cells need >= 2 valid rollouts for a variance; a singleton cell raises
    rather than silently contributing zero variance.
    """
    valid = df.dropna(subset=[value_col])
    if not len(valid):
        return {"mean": float("nan"), "se": float("nan"),
                "ci95": float("nan"), "n": 0, "k": 0}
    g = valid.groupby(cell_cols, observed=True)[value_col]
    means, variances, counts = g.mean(), g.var(ddof=1), g.count()
    if (counts < 2).any():
        bad = counts[counts < 2]
        raise ValueError(
            f"equal_weight_summary: cells with <2 valid rollouts (no "
            f"variance estimate): {list(bad.index)}"
        )
    k = len(means)
    u = variances / counts
    se = float(np.sqrt(u.sum()) / k)
    if u.sum() > 0:
        ws_df = float(u.sum() ** 2 / (u ** 2 / (counts - 1)).sum())
        ci95 = t_critical(ws_df) * se
    else:
        ci95 = 0.0  # all cells have zero variance (e.g. constant indicator)
    return {"mean": float(means.mean()), "se": se, "ci95": ci95,
            "n": int(counts.sum()), "k": int(k)}


def paired_gap_test(gaps):
    """Student t on per-variant gaps (df = k-1) + t-based 95% CI of the mean.

    ``gaps``: one condition difference per variant (NaNs dropped). Returns
    dict(gaps, k, mean, se, t, df, p, ci95_low, ci95_high, marker). With
    k < 2 the test is undefined (p = NaN, no marker).
    """
    arr = np.asarray([g for g in gaps if pd.notna(g)], dtype=float)
    k = len(arr)
    if k < 2:
        return {"gaps": [float(g) for g in arr], "k": k,
                "mean": float(arr.mean()) if k else float("nan"),
                "se": float("nan"), "t": float("nan"), "df": k - 1,
                "p": float("nan"), "ci95_low": float("nan"),
                "ci95_high": float("nan"), "marker": ""}
    mean = float(arr.mean())
    se = float(arr.std(ddof=1)) / math.sqrt(k)
    dfree = k - 1
    if se == 0.0:
        t_stat = math.inf if mean != 0.0 else 0.0
        p = 0.0 if mean != 0.0 else 1.0
    else:
        t_stat = mean / se
        p = t_two_sided_pvalue(t_stat, dfree)
    half = t_critical(dfree) * se
    return {"gaps": [float(g) for g in arr], "k": k, "mean": mean, "se": se,
            "t": t_stat, "df": dfree, "p": p, "ci95_low": mean - half,
            "ci95_high": mean + half, "marker": significance_marker(p)}


def normal_two_sided_pvalue(z):
    """Two-sided normal p-value P(|Z| >= |z|)."""
    if pd.isna(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))


def fixed_cells_gap_test(gaps, ses):
    """Test of the equal-weighted mean per-variant gap against 0, variants
    treated as FIXED design cells (rollout noise is the only randomness).

    ``gaps``: one paired condition difference per variant; ``ses``: its
    rollout-level standard error (Welch-style, from the per-cell sample
    variances). Pairs with a NaN gap or SE are dropped. Returns dict(gaps,
    ses, k, mean, se, z, p, stars): mean = average gap, se = (1/k) *
    sqrt(sum se_v^2), two-sided normal p (per-cell n is large), tiered
    ``significance_stars``.
    """
    gaps, ses = list(gaps), list(ses)
    if len(gaps) != len(ses):
        raise ValueError(
            f"fixed_cells_gap_test: {len(gaps)} gaps vs {len(ses)} ses -- "
            "the per-variant gap and SE lists must align"
        )
    pairs = [(float(g), float(s)) for g, s in zip(gaps, ses)
             if pd.notna(g) and pd.notna(s)]
    k = len(pairs)
    if k == 0:
        return {"gaps": [], "ses": [], "k": 0, "mean": float("nan"),
                "se": float("nan"), "z": float("nan"), "p": float("nan"),
                "stars": ""}
    mean = sum(g for g, _ in pairs) / k
    se = math.sqrt(sum(s * s for _, s in pairs)) / k
    z = mean / se if se > 0 else (math.inf if mean != 0 else 0.0)
    p = normal_two_sided_pvalue(z) if math.isfinite(z) else 0.0
    return {"gaps": [g for g, _ in pairs], "ses": [s for _, s in pairs],
            "k": k, "mean": mean, "se": se, "z": z, "p": p,
            "stars": significance_stars(p)}


def significance_stars(p_value):
    """Conventional tiered stars: *** p<.001, ** p<.01, * p<.05, '' else."""
    if pd.isna(p_value):
        return ""
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return ""


def significance_marker(p, alpha=0.05):
    """Single-tier significance marker: '*' iff p < alpha, else ''."""
    if pd.isna(p):
        return ""
    return "*" if p < alpha else ""
