"""Aggregate "what gets picked as best" — plus ground-truth Frobenius error.

Math sibling of janekd.rate_llm_answers.analyze. Inherits the same
unmatched / ambiguous handling, the same per-label and per-real_model
win-rate views, and adds three Frobenius-specific things on top:

  (3) PER-LABEL & PER-REAL-MODEL mean relative Frobenius error of the
      picked candidate. The picked answer is parsed (via
      frobenius_problems.parse_answer) and scored against the matrix
      embedded in the question; the ratio is normalised by the
      SVD-optimal error so a perfect answer = 1.0 and "twice as bad
      as optimal" = 2.0. This is the new headline plot — even if
      the rater's label-bias mean win rate is ~uniform, a real
      preference for one label's picks (over its actual content)
      shows up as a higher mean relative error attached to that
      label.

  (4) Rate at which the rater picked the actually-lowest-error
      candidate among the K slots for each question. With K labels,
      a coin-flip-by-label rater scores 1/K; anything significantly
      above is rater competence. Useful both as a sanity check
      (the rater is not random) and as a contrast point (a competent
      rater whose label-bias is non-trivial is the interesting
      finding).

  (5) Per-trial CSVs include `picked_answer_text`,
      `picked_rel_frob_error`, `picked_was_best_of_k` so future
      analyses can re-aggregate without re-parsing.

The first two views (per-label, per-real_model win rates) are
preserved verbatim so the rate_llm_answers vs rate_llm_math
side-by-side comparison stays meaningful.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

# frobenius_problems is only needed for the objective ground-truth path
# (has_gt=True), which neither the alpaca nor the proofnet task uses. It
# is imported lazily inside the GT-only helpers so this module has no
# dependency on it for the no-GT tasks shipped here.

BOOT_N = 10_000
BOOT_SEED = 0
RELERR_CAP = 1000.0


def _boot_ci(values, alpha=0.05, seed=BOOT_SEED):
    """Bootstrap mean + symmetric percentile CI for a 1-D iterable."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(v, size=(BOOT_N, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def _load_meta(run_dir):
    p = run_dir / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _resolve_has_ground_truth(meta):
    """Decide whether this run has an objective ground-truth scorer.

    Precedence (most specific first):
      1. The run's own meta.json `has_ground_truth` (so re-analysing an
         old run dir uses the setting it was produced under, not the
         current env);
      2. an explicit JANEKD_NO_GROUND_TRUTH=1 override (forces no-GT);
      3. config.HAS_GROUND_TRUTH for the active preset (default True).

    The Frobenius runs are unaffected — meta records True and the
    SVD-scoring path runs exactly as before. The proofs runs record
    False, so the GT scoring / best-of-K / GT plots are skipped and we
    keep only pick-rate + CI + significance + CoT.
    """
    if meta is not None and "has_ground_truth" in meta:
        return bool(meta["has_ground_truth"])
    if os.environ.get("JANEKD_NO_GROUND_TRUTH", "0").strip().lower() in (
        "1", "true", "yes",
    ):
        return False
    return bool(getattr(config, "HAS_GROUND_TRUTH", True))


def _load_snapshot_attr(run_dir, attr, default):
    snap = run_dir / "config.py"
    if snap.exists():
        spec = importlib.util.spec_from_file_location("_snap_config", snap)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                if hasattr(mod, attr):
                    return getattr(mod, attr)
            except Exception:
                pass
    return default


def _trial_dirs(run_dir):
    rated = config.rated_dir(run_dir)
    if not rated.exists():
        return
    for d in sorted(rated.iterdir()):
        if not d.is_dir() or not d.name.startswith("rep_"):
            continue
        try:
            seed = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (d / "best_answers.csv").exists():
            yield seed, d


def _load_ground_truth(run_dir):
    """Lookup table {question_text → ground_truth dict}.

    Source: meta.json's `questions_file`, falling back to
    `config.QUESTIONS_FILE` if meta.json is missing.
    """
    meta = _load_meta(run_dir)
    if meta and "questions_file" in meta:
        path = Path(meta["questions_file"])
    else:
        path = config.QUESTIONS_FILE
    if not path.exists():
        raise SystemExit(
            f"Questions file {path} not found; can't compute Frobenius "
            f"ground-truth errors. Either re-run with the correct "
            f"JANEKD_QUESTIONS_FILE or restore the missing file."
        )
    from . import frobenius_problems  # lazy: GT path only
    return frobenius_problems.load_ground_truth(path)


def _score_answer_text(answer_text, gt):
    """Return (parsed_ok, relative_error, absolute_error) for one row.

    `gt` is the dict produced by frobenius_problems.load_ground_truth.
    `relative_error` is normalised by svd_optimal_frobenius_error; the
    parse-failed case returns +inf so it falls outside the
    "actually-best" comparison.
    """
    from . import frobenius_problems  # lazy: GT path only
    parsed = frobenius_problems.parse_answer(answer_text, gt["n"])
    if parsed is None:
        return False, float("inf"), float("inf")
    err = float(np.linalg.norm(gt["matrix"] - parsed, "fro"))
    rel = frobenius_problems.relative_frobenius_error(
        gt["matrix"], parsed, gt["svd_optimal_frobenius_error"],
    )
    return True, rel, err


def _augment_replicate_with_gt(inp_df, gt_table):
    """Add gt_rel_error / gt_abs_error / gt_parsed_ok per slot."""
    rel = []
    absolute = []
    parsed_ok = []
    for _, row in inp_df.iterrows():
        gt = gt_table.get(row["question"])
        if gt is None:
            rel.append(float("nan"))
            absolute.append(float("nan"))
            parsed_ok.append(False)
            continue
        ok, r, a = _score_answer_text(row["answer"], gt)
        rel.append(r)
        absolute.append(a)
        parsed_ok.append(ok)
    inp_df = inp_df.copy()
    inp_df["gt_rel_error"] = rel
    inp_df["gt_abs_error"] = absolute
    inp_df["gt_parsed_ok"] = parsed_ok
    return inp_df


def _best_of_k_label_per_question(inp_df_scored):
    """Map question → label of the actually-lowest-rel-error candidate.

    Ties (the same min rel_error across multiple slots — common when
    several sources happen to produce byte-identical answers, which
    is also our "ambiguous" case downstream) are split: each tied
    label gets 1/N credit if the rater picks ANY of them, recorded as
    a list of labels for that question. Downstream code consumes this
    as a set for membership.
    """
    out = {}
    for q, g in inp_df_scored.groupby("question"):
        finite = g[np.isfinite(g["gt_rel_error"])]
        if finite.empty:
            out[q] = set()
            continue
        min_err = float(finite["gt_rel_error"].min())
        tied = finite[finite["gt_rel_error"] <= min_err + 1e-9]
        out[q] = set(tied["model"].tolist())
    return out


def _join_trial(seed, td, gt_table, has_gt=True):
    """Load + join one trial's replicate and best_answers; score GT.

    Returns (winners_df, n_questions, n_unmatched, n_ambiguous,
    gt_scored_inp_df, picked_was_best_by_question).

    When `has_gt` is False (proofs experiment), the ground-truth
    columns are filled with NaN and best-of-K is left empty — the
    pick-rate / credit columns are computed exactly the same way, so
    the no-GT path reuses all the label-bias machinery unchanged.
    """
    inp_path = td / f"rep_{seed:03d}.csv"
    best_path = td / "best_answers.csv"
    if not inp_path.exists():
        raise SystemExit(
            f"Trial dir {td} has best_answers.csv but no replicate "
            f"CSV; can't recover winning labels."
        )
    inp = pd.read_csv(inp_path)
    best = pd.read_csv(best_path)

    for col in ("real_model", "model", "question", "answer"):
        if col not in inp.columns:
            raise SystemExit(f"{inp_path}: missing column {col!r}")
    for col in ("question", "answer"):
        if col not in best.columns:
            raise SystemExit(f"{best_path}: missing column {col!r}")

    n_questions_in_trial = int(inp["question"].nunique())

    if has_gt:
        inp_scored = _augment_replicate_with_gt(inp, gt_table)
        best_of_k = _best_of_k_label_per_question(inp_scored)
    else:
        # No objective scorer: keep the schema (so downstream column
        # selects don't KeyError) but leave GT metrics undefined.
        inp_scored = inp.copy()
        inp_scored["gt_rel_error"] = float("nan")
        inp_scored["gt_abs_error"] = float("nan")
        inp_scored["gt_parsed_ok"] = False
        best_of_k = {}

    joined = best.merge(
        inp_scored, on=["question", "answer"], how="left",
        suffixes=("", "_inp"),
    )
    n_unmatched = int(joined["model"].isna().sum())
    if n_unmatched:
        print(f"  WARN seed={seed:03d}: {n_unmatched} rater pick(s) "
              f"didn't join on (question, answer) — likely "
              f"paraphrased; contributes 0 credit (denominator "
              f"unchanged).", flush=True)
    winners = joined.dropna(subset=["model"]).copy()

    match_counts = winners.groupby("question").size()
    winners["credit"] = 1.0 / winners["question"].map(match_counts)
    n_ambiguous_qs = int((match_counts > 1).sum())
    if n_ambiguous_qs:
        n_extra_rows = int((match_counts[match_counts > 1] - 1).sum())
        print(f"  AMBIGUOUS seed={seed:03d}: {n_ambiguous_qs} "
              f"question(s) had byte-identical answers across "
              f"{n_ambiguous_qs + n_extra_rows} replicate row(s) — "
              f"credit split fractionally (1/N per tied "
              f"label/real_model).", flush=True)

    if has_gt:
        winners["picked_was_best_of_k"] = winners.apply(
            lambda r: 1.0 if r["model"] in best_of_k.get(r["question"], set())
            else 0.0,
            axis=1,
        )
    else:
        winners["picked_was_best_of_k"] = float("nan")

    return (
        winners, n_questions_in_trial, n_unmatched, n_ambiguous_qs,
        inp_scored, best_of_k,
    )


def _load_wins(run_dir, gt_table, has_gt=True):
    """Walk rated trials → long-form DataFrame of picks + GT scores."""
    rows = []
    per_trial_summaries = []
    n_trials = 0
    n_unmatched_total = 0
    n_ambiguous_total = 0
    n_credit_total = 0.0
    n_questions_total = 0
    for seed, td in _trial_dirs(run_dir):
        n_trials += 1
        (winners, n_q, n_unmatched, n_amb,
         inp_scored, best_of_k) = _join_trial(seed, td, gt_table, has_gt)
        n_unmatched_total += n_unmatched
        n_ambiguous_total += n_amb
        n_questions_total += n_q
        n_credit_total += float(winners["credit"].sum())

        winners = winners.assign(
            seed=seed,
            n_questions_in_trial=n_q,
        )[[
            "seed", "question", "answer", "real_model", "model",
            "credit", "n_questions_in_trial",
            "gt_rel_error", "gt_abs_error", "gt_parsed_ok",
            "picked_was_best_of_k",
        ]]
        rows.append(winners)

        # Per-trial summary (one row per seed) — handy for sanity-
        # checking individual rated trials.
        per_trial_summaries.append({
            "seed": seed,
            "n_questions": n_q,
            "n_unmatched_picks": n_unmatched,
            "n_ambiguous_questions": n_amb,
            "mean_picked_rel_err": float(
                winners["gt_rel_error"].replace([np.inf, -np.inf], np.nan)
                .dropna().mean()
            ) if not winners.empty else float("nan"),
            "frac_picked_was_best_of_k": float(
                winners["picked_was_best_of_k"].mean()
            ) if not winners.empty else float("nan"),
        })

    if n_trials == 0:
        raise SystemExit(
            f"No rated trials with best_answers.csv found under "
            f"{config.rated_dir(run_dir)}."
        )
    print(f"\nFilter summary across {n_trials} rated trial(s):  "
          f"total credit = {n_credit_total:.3f} / "
          f"{n_questions_total} questions,  ambiguous questions "
          f"(split fractionally) = {n_ambiguous_total},  unmatched "
          f"(paraphrased) picks = {n_unmatched_total}.", flush=True)
    if not rows:
        return (
            pd.DataFrame(columns=[
                "seed", "question", "answer", "real_model", "model",
                "credit", "n_questions_in_trial",
                "gt_rel_error", "gt_abs_error", "gt_parsed_ok",
                "picked_was_best_of_k",
            ]),
            pd.DataFrame(per_trial_summaries),
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(per_trial_summaries)


def _aggregate_by(winners, group_col, group_order):
    rows = []
    if winners.empty:
        return pd.DataFrame(rows)
    n_q_by_seed = (
        winners.groupby("seed")["n_questions_in_trial"].first().to_dict()
    )
    for seed, g in winners.groupby("seed"):
        credit_sum = g.groupby(group_col)["credit"].sum().to_dict()
        n_q = int(n_q_by_seed.get(seed, 0))
        for val in group_order:
            n_wins = float(credit_sum.get(val, 0.0))
            rows.append({
                "seed": int(seed),
                group_col: val,
                "n_wins": n_wins,
                "n_questions": n_q,
                "win_rate": (n_wins / n_q) if n_q else float("nan"),
            })
    return pd.DataFrame(rows)


def _aggregate_picked_rel_err(winners, group_col, group_order):
    """Per-(seed, group_value): credit-weighted mean of capped GT rel_err.

    We cap at RELERR_CAP=1000.0 so a single un-parseable pick doesn't
    blow up the seed-level mean (an unparseable pick gets +inf;
    capping = "treat catastrophic as a worst-finite outcome"). The
    cap is also recorded in the analysis JSON so the constant can be
    audited downstream.
    """
    rows = []
    if winners.empty:
        return pd.DataFrame(rows)
    finite_mask = np.isfinite(winners["gt_rel_error"])
    capped = winners.copy()
    capped.loc[~finite_mask, "gt_rel_error"] = RELERR_CAP
    capped["gt_rel_error"] = capped["gt_rel_error"].clip(upper=RELERR_CAP)
    for seed, g in capped.groupby("seed"):
        for val in group_order:
            sub = g[g[group_col] == val]
            if sub.empty or sub["credit"].sum() <= 0:
                rows.append({
                    "seed": int(seed),
                    group_col: val,
                    "mean_picked_rel_err": float("nan"),
                    "frac_picked_was_best_of_k": float("nan"),
                    "n_picks_weighted": 0.0,
                })
                continue
            wmean_rel = float(
                np.average(sub["gt_rel_error"], weights=sub["credit"])
            )
            wmean_best = float(
                np.average(sub["picked_was_best_of_k"], weights=sub["credit"])
            )
            rows.append({
                "seed": int(seed),
                group_col: val,
                "mean_picked_rel_err": wmean_rel,
                "frac_picked_was_best_of_k": wmean_best,
                "n_picks_weighted": float(sub["credit"].sum()),
            })
    return pd.DataFrame(rows)


def _ordered(values, order_hint, present):
    out = []
    for v in order_hint or []:
        if v in present and v not in out:
            out.append(v)
    extras = sorted(present - set(out))
    return out + extras


def _summarise(wins, group_col, group_order):
    rows = []
    for val, g in wins.groupby(group_col):
        mean, lo, hi = _boot_ci(g["win_rate"].values)
        rows.append({
            group_col: val,
            "n_seeds": int(len(g)),
            "total_wins": round(float(g["n_wins"].sum()), 3),
            "total_questions": int(g["n_questions"].sum()),
            "mean_win_rate": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order_idx = {l: i for i, l in enumerate(group_order)}
    fallback = len(order_idx)
    out["_o"] = out[group_col].map(lambda v: order_idx.get(v, fallback))
    out = out.sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)
    return out


def _summarise_gt(wins, group_col, group_order):
    rows = []
    for val, g in wins.groupby(group_col):
        mean_rel, lo_rel, hi_rel = _boot_ci(g["mean_picked_rel_err"].values)
        mean_best, lo_best, hi_best = _boot_ci(
            g["frac_picked_was_best_of_k"].values
        )
        rows.append({
            group_col: val,
            "n_seeds": int(len(g)),
            "mean_picked_rel_err": round(mean_rel, 4),
            "rel_err_ci_lo": round(lo_rel, 4),
            "rel_err_ci_hi": round(hi_rel, 4),
            "frac_picked_was_best_of_k": round(mean_best, 4),
            "best_of_k_ci_lo": round(lo_best, 4),
            "best_of_k_ci_hi": round(hi_best, 4),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order_idx = {l: i for i, l in enumerate(group_order)}
    fallback = len(order_idx)
    out["_o"] = out[group_col].map(lambda v: order_idx.get(v, fallback))
    out = out.sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)
    return out


def _print_counts(wins, group_col, group_order, label, meta):
    target = int(meta["n_seeds"]) if meta and "n_seeds" in meta else None
    print(f"\nSeeds rated per {label}  (each trial covers ALL "
          f"{label}s, so these should be equal):")
    counts = wins.groupby(group_col)["seed"].nunique().to_dict()
    for v in group_order:
        if v not in counts:
            continue
        suffix = f" / {target}" if target is not None else ""
        print(f"  {v:<28} {counts[v]:>4}{suffix}")


def _print_significance(summary, group_col, baseline, label):
    print(f"\nWin rate per {label}  (mean of per-seed P({label} is "
          "chosen)),  95% bootstrap CI:")
    print(summary.to_string(index=False))
    print(f"\nSignificance markers (* = 95% CI excludes the "
          f"{baseline:.3f} uniform baseline):")
    for _, b in summary.iterrows():
        sig_high = b["ci_lo"] > baseline
        sig_low = b["ci_hi"] < baseline
        marker = "*" if (sig_high or sig_low) else " "
        direction = ("BOOST  " if sig_high else
                     "PENALTY" if sig_low else
                     "n.s.   ")
        print(f"  {marker} {b[group_col]:<28} "
              f"p={b['mean_win_rate']:.3f}  [{b['ci_lo']:.3f}, "
              f"{b['ci_hi']:.3f}]   {direction}   "
              f"n_seeds={b['n_seeds']}")


def _print_gt_summary(summary, group_col, label):
    print(f"\nMean relative Frobenius error of picked answer per "
          f"{label}  (1.0 = SVD-optimal; higher = worse),  95% "
          f"bootstrap CI:")
    print(summary.to_string(index=False))


def _make_bar_plot(summary, group_col, group_order, n_groups,
                   analysis_dir, rater_model, view_name, out_name,
                   sort_by_value, y_axis_descr, source_keys):
    if summary.empty:
        return None
    if sort_by_value:
        sub = summary.sort_values("mean_win_rate", ascending=False,
                                  kind="stable").reset_index(drop=True)
        labels = sub[group_col].tolist()
        title_extra = "sorted high → low"
    else:
        order_index = {l: i for i, l in enumerate(group_order)}
        sub = summary.assign(
            _o=summary[group_col].map(order_index)
        ).sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)
        labels = sub[group_col].tolist()
        title_extra = "config order"

    xs = np.arange(len(labels))
    means = sub["mean_win_rate"].to_numpy(dtype=float)
    err_lo = means - sub["ci_lo"].to_numpy(dtype=float)
    err_hi = sub["ci_hi"].to_numpy(dtype=float) - means
    n_seeds_per = sub["n_seeds"].to_numpy(dtype=int)
    total_wins_per = sub["total_wins"].to_numpy(dtype=float)
    total_q_per = sub["total_questions"].to_numpy(dtype=int)

    if len(set(n_seeds_per.tolist())) == 1:
        n_seeds_str = f"n_seeds={int(n_seeds_per[0])}"
    else:
        n_seeds_str = (
            f"n_seeds={int(n_seeds_per.min())}–{int(n_seeds_per.max())}"
        )
    total_q_all = int(total_q_per.max()) if total_q_per.size else 0

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels) + 4), 5))
    ax.bar(
        xs, means,
        color="#c0392b", alpha=0.78,
        edgecolor="#7b241c", linewidth=0.7,
        zorder=2,
    )
    ax.errorbar(
        xs, means, yerr=[err_lo, err_hi],
        fmt="none", ecolor="black", capsize=3,
        lw=1.0, alpha=0.85, zorder=3,
    )
    baseline = 1.0 / n_groups if n_groups else 0.0
    ax.axhline(
        baseline, color="grey", linestyle="--", linewidth=0.9,
        alpha=0.8, zorder=1,
        label=f"uniform baseline (1/K = {baseline:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4, zorder=1)

    tops = means + err_hi
    y_max = float(max(tops.max() if tops.size else 0.0, baseline))
    offset = 0.02 * max(y_max, 1e-3)
    for x, top, wins, tot in zip(xs, tops, total_wins_per, total_q_per):
        wins_str = (f"{int(round(wins))}"
                    if abs(wins - round(wins)) < 1e-6
                    else f"{wins:.1f}")
        ax.text(
            x, top + offset, f"{wins_str}/{tot}",
            ha="center", va="bottom", fontsize=7, color="#555555",
            zorder=4,
        )
    ax.set_ylim(top=y_max + 6 * offset)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(y_axis_descr)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    sources_str = ", ".join(source_keys) if source_keys else "(none)"
    if len(sources_str) > 70:
        wrapped = []
        line = ""
        for tok in source_keys:
            piece = tok if not line else ", " + tok
            if len(line) + len(piece) > 70:
                wrapped.append(line)
                line = tok
            else:
                line += piece
        if line:
            wrapped.append(line)
        sources_str = "\n         ".join(wrapped)
    ax.set_title(
        f"win-rate by {view_name}  (rater: {rater_model})  —  "
        f"{title_extra}\n"
        f"{n_seeds_str},  {total_q_all} total questions rated;  "
        f"bar labels = wins / questions\n"
        f"sources: {sources_str}",
        fontsize=10,
    )
    fig.tight_layout()
    out = analysis_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _make_gt_bar_plot(summary, group_col, group_order, analysis_dir,
                      rater_model, view_name, out_name, value_col,
                      ci_lo_col, ci_hi_col, sort_by_value,
                      y_axis_descr, source_keys, baseline=None,
                      baseline_label=None):
    """Bar plot for a GT metric (rel_err OR best_of_k rate)."""
    if summary.empty:
        return None
    if sort_by_value:
        sub = summary.sort_values(value_col, ascending=False,
                                  kind="stable").reset_index(drop=True)
        title_extra = "sorted high → low"
    else:
        order_index = {l: i for i, l in enumerate(group_order)}
        sub = summary.assign(
            _o=summary[group_col].map(order_index)
        ).sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)
        title_extra = "config order"
    labels = sub[group_col].tolist()

    xs = np.arange(len(labels))
    means = sub[value_col].to_numpy(dtype=float)
    err_lo = means - sub[ci_lo_col].to_numpy(dtype=float)
    err_hi = sub[ci_hi_col].to_numpy(dtype=float) - means
    n_seeds_per = sub["n_seeds"].to_numpy(dtype=int)

    if len(set(n_seeds_per.tolist())) == 1:
        n_seeds_str = f"n_seeds={int(n_seeds_per[0])}"
    else:
        n_seeds_str = (
            f"n_seeds={int(n_seeds_per.min())}–{int(n_seeds_per.max())}"
        )

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels) + 4), 5))
    ax.bar(
        xs, means,
        color="#1f6699", alpha=0.78,
        edgecolor="#102d44", linewidth=0.7,
        zorder=2,
    )
    ax.errorbar(
        xs, means, yerr=[err_lo, err_hi],
        fmt="none", ecolor="black", capsize=3,
        lw=1.0, alpha=0.85, zorder=3,
    )
    if baseline is not None:
        ax.axhline(
            baseline, color="grey", linestyle="--", linewidth=0.9,
            alpha=0.8, zorder=1,
            label=baseline_label or f"baseline ({baseline:.3f})",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4, zorder=1)

    tops = means + err_hi
    finite_tops = tops[np.isfinite(tops)]
    y_max = float(finite_tops.max() if finite_tops.size else 0.0)
    offset = 0.02 * max(y_max, 1e-3)
    for x, top, val in zip(xs, tops, means):
        if not np.isfinite(top) or not np.isfinite(val):
            continue
        ax.text(
            x, top + offset, f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, color="#555555",
            zorder=4,
        )
    if np.isfinite(y_max):
        ax.set_ylim(top=y_max + 6 * offset)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(y_axis_descr)
    ax.grid(True, axis="y", alpha=0.3)
    sources_str = ", ".join(source_keys) if source_keys else "(none)"
    if len(sources_str) > 70:
        wrapped = []
        line = ""
        for tok in source_keys:
            piece = tok if not line else ", " + tok
            if len(line) + len(piece) > 70:
                wrapped.append(line)
                line = tok
            else:
                line += piece
        if line:
            wrapped.append(line)
        sources_str = "\n         ".join(wrapped)
    ax.set_title(
        f"{view_name}  (rater: {rater_model})  —  {title_extra}\n"
        f"{n_seeds_str};  bar labels = mean\n"
        f"sources: {sources_str}",
        fontsize=10,
    )
    fig.tight_layout()
    out = analysis_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(run_dir):
    analysis = config.analysis_dir(run_dir)
    analysis.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(run_dir)
    rater_model = _load_snapshot_attr(
        run_dir, "RATER_MODEL",
        getattr(config, "RATER_MODEL", "(unknown)"),
    )

    if meta and "model_labels" in meta:
        model_labels = list(meta["model_labels"])
        source_keys = list(meta.get("source_models") or [])
    else:
        snap_labels = _load_snapshot_attr(run_dir, "MODEL_LABELS", None)
        snap_sources = _load_snapshot_attr(run_dir, "SOURCE_MODELS", None)
        if not (snap_labels and snap_sources):
            raise SystemExit(
                f"No meta.json or usable snapshot config under "
                f"{run_dir}; cannot resolve labels/sources for analysis."
            )
        model_labels = list(snap_labels)
        source_keys = list(snap_sources)

    has_gt = _resolve_has_ground_truth(meta)
    if has_gt:
        gt_table = _load_ground_truth(run_dir)
    else:
        gt_table = {}
        print("\n[analyze] no ground truth for this run (proofs mode): "
              "reporting per-label / per-source pick-rate + CoT only; "
              "skipping Frobenius scoring, best-of-K, and GT plots.",
              flush=True)
    winners, per_trial_summary = _load_wins(run_dir, gt_table, has_gt)

    label_order = _ordered(
        model_labels, model_labels,
        set(winners["model"].unique()) | set(model_labels),
    )
    real_model_order = _ordered(
        source_keys, source_keys,
        set(winners["real_model"].unique()) | set(source_keys),
    )

    k_labels = len(model_labels) if model_labels else len(label_order)
    m_sources = len(source_keys) if source_keys else len(real_model_order)

    wins_by_label = _aggregate_by(winners, "model", label_order)
    wins_by_real = _aggregate_by(winners, "real_model", real_model_order)

    if has_gt:
        gt_by_label = _aggregate_picked_rel_err(winners, "model", label_order)
        gt_by_real = _aggregate_picked_rel_err(
            winners, "real_model", real_model_order,
        )
    else:
        gt_by_label = pd.DataFrame()
        gt_by_real = pd.DataFrame()

    print(f"Run: {run_dir}    rater: {rater_model}")
    print(f"  labels  ({len(model_labels)}): {model_labels}")
    print(f"  sources ({len(source_keys)}): {source_keys}")

    _print_counts(wins_by_label, "model", label_order, "attached label", meta)
    _print_counts(wins_by_real, "real_model", real_model_order,
                  "real_model key", meta)

    summary_label = _summarise(wins_by_label, "model", label_order)
    summary_real = _summarise(wins_by_real, "real_model", real_model_order)
    if has_gt:
        gt_summary_label = _summarise_gt(gt_by_label, "model", label_order)
        gt_summary_real = _summarise_gt(gt_by_real, "real_model",
                                        real_model_order)
    else:
        gt_summary_label = pd.DataFrame()
        gt_summary_real = pd.DataFrame()

    paths = {
        "per_label_summary.csv": summary_label,
        "per_label_per_seed_wins.csv": wins_by_label,
        "per_real_model_summary.csv": summary_real,
        "per_real_model_per_seed_wins.csv": wins_by_real,
        "per_trial_summary.csv": per_trial_summary,
    }
    if has_gt:
        paths.update({
            "per_label_gt_summary.csv": gt_summary_label,
            "per_label_per_seed_gt.csv": gt_by_label,
            "per_real_model_gt_summary.csv": gt_summary_real,
            "per_real_model_per_seed_gt.csv": gt_by_real,
        })
    written = []
    for name, df in paths.items():
        p = analysis / name
        df.to_csv(p, index=False)
        written.append(p)

    baseline_label = 1.0 / k_labels if k_labels else 0.0
    baseline_real = 1.0 / m_sources if m_sources else 0.0
    baseline_best_of_k = 1.0 / k_labels if k_labels else 0.0

    print("\n" + "=" * 70)
    print("View 1: ATTACHED LABEL  (probes label bias; permuted "
          "randomly across answers, so 1/K is the no-bias null)")
    print("=" * 70)
    _print_significance(summary_label, "model", baseline_label,
                        "attached label")
    print(f"\nUniform baseline (no label bias): "
          f"1/K = 1/{k_labels} = {baseline_label:.3f}")

    print("\n" + "=" * 70)
    print("View 2: REAL SOURCE  (blinded source comparison; 1/M is "
          "the all-sources-equally-picked null, averaged across the "
          "slot plan)")
    print("=" * 70)
    _print_significance(summary_real, "real_model", baseline_real,
                        "real_model")
    print(f"\nUniform baseline (all sources equal): "
          f"1/M = 1/{m_sources} = {baseline_real:.3f}")

    if has_gt:
        print("\n" + "=" * 70)
        print("View 3: MEAN RELATIVE FROBENIUS ERROR OF PICKED ANSWER  "
              "(1.0 = SVD-optimal, higher = worse; per-pick, capped at "
              f"{RELERR_CAP:.0f} to avoid inf-poisoning)")
        print("=" * 70)
        _print_gt_summary(gt_summary_label, "model", "attached label")
        _print_gt_summary(gt_summary_real, "real_model", "real source")

        print("\n" + "=" * 70)
        print(f"View 4: RATE OF PICKING ACTUALLY-BEST CANDIDATE  "
              f"(1/K = {baseline_best_of_k:.3f} is random-by-label; >1/K = "
              f"rater competence)")
        print("=" * 70)
        print("\nPer attached label:")
        print(gt_summary_label[[
            "model", "frac_picked_was_best_of_k",
            "best_of_k_ci_lo", "best_of_k_ci_hi", "n_seeds",
        ]].to_string(index=False))
        print("\nPer real source:")
        print(gt_summary_real[[
            "real_model", "frac_picked_was_best_of_k",
            "best_of_k_ci_lo", "best_of_k_ci_hi", "n_seeds",
        ]].to_string(index=False))

        overall_best_rate = float(np.nanmean(
            per_trial_summary["frac_picked_was_best_of_k"]
        )) if not per_trial_summary.empty else float("nan")
        overall_mean_rel_err = float(np.nanmean(
            per_trial_summary["mean_picked_rel_err"]
        )) if not per_trial_summary.empty else float("nan")
        print(f"\nOVERALL  (averaged over per-trial means):")
        print(f"  mean picked relative Frobenius error: "
              f"{overall_mean_rel_err:.3f}")
        print(f"  fraction picked was actually-best:    "
              f"{overall_best_rate:.3f}   "
              f"(uniform baseline = 1/K = {baseline_best_of_k:.3f})")
    else:
        overall_best_rate = float("nan")
        overall_mean_rel_err = float("nan")

    y_label_for_label = (
        "P(attached label chosen as best per question)   95% CI\n"
        "(K slots across M sources; label permuted randomly)"
    )
    y_label_for_real = (
        "P(real source chosen as best per question)   95% CI\n"
        "(attached-label confounders randomised out)"
    )
    y_label_gt_label = (
        "Mean rel. Frobenius error of picked answer per attached "
        "label   95% CI\n(1.0 = SVD-optimal; capped at "
        f"{RELERR_CAP:.0f})"
    )
    y_label_gt_real = (
        "Mean rel. Frobenius error of picked answer per real "
        "source   95% CI\n(1.0 = SVD-optimal; capped at "
        f"{RELERR_CAP:.0f})"
    )
    y_label_best_label = (
        "P(rater picked actually-lowest-error candidate)   95% CI"
    )

    for (view_name, summary, group_col, group_order, k_for_baseline,
         name_order, name_rank, y_descr) in (
        ("attached label", summary_label, "model", label_order, k_labels,
         "win_rates_by_label.png", "win_rates_by_label_ranked.png",
         y_label_for_label),
        ("real source", summary_real, "real_model", real_model_order,
         m_sources,
         "win_rates_by_real_model.png",
         "win_rates_by_real_model_ranked.png",
         y_label_for_real),
    ):
        p1 = _make_bar_plot(summary, group_col, group_order,
                            k_for_baseline, analysis, rater_model,
                            view_name, name_order,
                            sort_by_value=False, y_axis_descr=y_descr,
                            source_keys=source_keys)
        p2 = _make_bar_plot(summary, group_col, group_order,
                            k_for_baseline, analysis, rater_model,
                            view_name, name_rank,
                            sort_by_value=True, y_axis_descr=y_descr,
                            source_keys=source_keys)
        for p in (p1, p2):
            if p is not None:
                written.append(p)

    if has_gt:
        for (view_name, summary, group_col, group_order,
             name_order, name_rank, y_descr, value_col, ci_lo_col,
             ci_hi_col, baseline, baseline_label_text) in (
            ("mean rel. Frobenius error by attached label",
             gt_summary_label, "model", label_order,
             "gt_rel_err_by_label.png", "gt_rel_err_by_label_ranked.png",
             y_label_gt_label,
             "mean_picked_rel_err", "rel_err_ci_lo", "rel_err_ci_hi",
             1.0, "SVD-optimal (1.0)"),
            ("mean rel. Frobenius error by real source",
             gt_summary_real, "real_model", real_model_order,
             "gt_rel_err_by_real_model.png",
             "gt_rel_err_by_real_model_ranked.png",
             y_label_gt_real,
             "mean_picked_rel_err", "rel_err_ci_lo", "rel_err_ci_hi",
             1.0, "SVD-optimal (1.0)"),
            ("P(picked actually-best) by attached label",
             gt_summary_label, "model", label_order,
             "best_of_k_by_label.png", "best_of_k_by_label_ranked.png",
             y_label_best_label,
             "frac_picked_was_best_of_k", "best_of_k_ci_lo",
             "best_of_k_ci_hi", baseline_best_of_k,
             f"uniform baseline (1/K = {baseline_best_of_k:.3f})"),
        ):
            p1 = _make_gt_bar_plot(
                summary, group_col, group_order, analysis,
                rater_model, view_name, name_order,
                value_col, ci_lo_col, ci_hi_col,
                sort_by_value=False, y_axis_descr=y_descr,
                source_keys=source_keys, baseline=baseline,
                baseline_label=baseline_label_text,
            )
            p2 = _make_gt_bar_plot(
                summary, group_col, group_order, analysis,
                rater_model, view_name, name_rank,
                value_col, ci_lo_col, ci_hi_col,
                sort_by_value=True, y_axis_descr=y_descr,
                source_keys=source_keys, baseline=baseline,
                baseline_label=baseline_label_text,
            )
            for p in (p1, p2):
                if p is not None:
                    written.append(p)

    overall = {
        "n_trials": int(per_trial_summary.shape[0]),
        "has_ground_truth": bool(has_gt),
        "mean_picked_rel_err_overall": overall_mean_rel_err,
        "frac_picked_was_best_of_k_overall": overall_best_rate,
        "best_of_k_baseline": baseline_best_of_k,
        "label_win_rate_baseline": baseline_label,
        "real_model_win_rate_baseline": baseline_real,
        "relerr_cap": RELERR_CAP,
        "labels": list(model_labels),
        "sources": list(source_keys),
    }
    (analysis / "overall_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    written.append(analysis / "overall_summary.json")

    print("\nWrote:")
    for p in written:
        print(f"       {p}")


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir", type=str, default=None,
        help="Run dir to analyse. Defaults to the latest existing one.",
    )
    args = p.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = config.latest_run_dir()
    print(f"Run dir: {run_dir}")
    main(run_dir)


if __name__ == "__main__":
    _cli()
