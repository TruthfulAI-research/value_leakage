"""Driver: blinded K-label best-of-K sweep across multiple seeds.

Runs the generate -> permute -> rate -> analyze pipeline for one grader
backend and one task (alpaca / proofnet), driving the raters inside a
Docker container (bind-mounted rate script + scratch /work).

A hard startup assertion guards that `_RATER_VISIBLE_COLUMNS` contains
`model` (the label column is the bias-test surface; stripping it would
silently turn the whole experiment into a null result) and that it does
NOT contain `real_model` (the grader must never see the true source).
"""

import os
import sys


def _bootstrap_experiment_env():
    if "JANEKD_EXPERIMENT" in os.environ:
        return
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--experiment" and i + 1 < len(argv):
            os.environ["JANEKD_EXPERIMENT"] = argv[i + 1]
            return
        if arg.startswith("--experiment="):
            os.environ["JANEKD_EXPERIMENT"] = arg.split("=", 1)[1]
            return


_bootstrap_experiment_env()

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from shared.models import MODELS

from . import analyze, config, generate
from .config import (
    DOCKER_IMAGE,
    DOCKERFILE,
    EXPERIMENT,
    HERE,
    LATEST_LINK,
    LIVE_PREVIEW_EVERY,
    MODEL_LABELS,
    N_SEEDS,
    RATE_SCRIPT,
    RATER_API_KEY_ENV,
    RATER_BACKEND,
    RATER_EXTRA_CONTAINER_ENV,
    RATER_HOST_API_KEY_ENV,
    RATER_MODEL,
    RATER_REASONING_EFFORT,
    RATER_REASONING_SUMMARY,
    RATING_PARALLELISM,
    SOURCE_MODELS,
    START_SEED,
)

# The rater MUST see the `model` column — that's the label whose
# permutation we're using to measure bias. Stripping it would silently
# turn the whole experiment into a null result. `real_model` and
# `sample_idx` are CORRECTLY hidden so the rater can't learn the
# constant source.
_RATER_VISIBLE_COLUMNS = ["model", "question", "answer"]
assert "model" in _RATER_VISIBLE_COLUMNS, (
    "rate_llm_answers: the `model` column MUST be visible to the grader "
    "for the bias measurement to make sense. _RATER_VISIBLE_COLUMNS "
    "was edited to remove it; this is a critical regression."
)
assert "real_model" not in _RATER_VISIBLE_COLUMNS, (
    "rate_llm_answers: `real_model` MUST be hidden from the grader. "
    "_RATER_VISIBLE_COLUMNS was edited to include it; the experiment "
    "becomes invalid if the rater can see the true source."
)


def _permute_one(df_base, seed, labels):
    """Per-(seed, question) random permutation of labels onto slots.

    Verbatim from rate_llm_answers/run.py:_permute_one — see that
    docstring for the full positional-bias rationale.
    """
    k = len(labels)
    rng = random.Random(int(seed))
    df = df_base.copy()
    df["model"] = ""

    out_indices = []
    for question, group in df.sort_values(
        ["question", "real_model", "sample_idx"]
    ).groupby("question", sort=True):
        if len(group) != k:
            raise RuntimeError(
                f"seed={seed}: question {question!r} has {len(group)} "
                f"rows but K={k} labels — base CSV is inconsistent."
            )
        perm = list(labels)
        rng.shuffle(perm)
        group_indices = list(group.index)
        for i, df_idx in enumerate(group_indices):
            df.loc[df_idx, "model"] = perm[i]

        rng.shuffle(group_indices)
        out_indices.extend(group_indices)

    return (
        df.loc[out_indices][
            ["real_model", "sample_idx", "model", "question", "answer"]
        ].reset_index(drop=True)
    )


def _write_bases(run_dir):
    bases = config.bases_dir(run_dir)
    bases.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in range(START_SEED, START_SEED + N_SEEDS):
        out = bases / f"rep_{seed:03d}.csv"
        if not out.exists():
            pending.append((seed, out))
    have = N_SEEDS - len(pending)
    if not pending:
        print(f"\n=== Bases: all {have} already on disk ===")
        return

    seed_qs = {}
    union, seen = [], set()
    for seed, _out in pending:
        qs = generate.select_questions(generate.question_seed(seed))
        seed_qs[seed] = qs
        for q in qs:
            if q not in seen:
                seen.add(q)
                union.append(q)

    print(f"\n=== Generating {len(pending)} base CSV(s) "
          f"({have} already on disk;  union of {len(union)} unique "
          f"question(s) across pending seeds) ===", flush=True)

    print(f"\n--- Warming answer cache: {len(union)} questions × "
          f"{len(MODEL_LABELS)} slot(s) from "
          f"{len(SOURCE_MODELS)} source model(s) ---", flush=True)
    generate.warm_cache(union)

    for seed, out in pending:
        print(f"\n--- seed={seed:03d}: writing base CSV "
              f"({len(seed_qs[seed])} questions × {len(MODEL_LABELS)} "
              f"slot(s), rng-seed={generate.question_seed(seed)!r}) ---",
              flush=True)
        generate.generate_to(out, seed_qs[seed])


def _write_replicates(run_dir):
    bases = config.bases_dir(run_dir)
    reps = config.replicates_dir(run_dir)
    reps.mkdir(parents=True, exist_ok=True)
    wrote = 0
    skipped = 0
    for seed in range(START_SEED, START_SEED + N_SEEDS):
        base_path = bases / f"rep_{seed:03d}.csv"
        out_path = reps / f"rep_{seed:03d}.csv"
        if not base_path.exists():
            raise RuntimeError(
                f"Missing base CSV for seed {seed}: {base_path} — "
                f"_write_bases should have created it. Did the generate "
                f"stage fail?"
            )
        if out_path.exists():
            skipped += 1
            continue
        base_df = pd.read_csv(base_path)
        permuted = _permute_one(base_df, seed, MODEL_LABELS)
        permuted.to_csv(out_path, index=False)
        wrote += 1
    print(f"Replicates: wrote {wrote} new, {skipped} already on disk "
          f"(target {N_SEEDS} seed(s)).")


def build_image():
    print(f"\n=== Building Docker image: {DOCKER_IMAGE} "
          f"(backend={RATER_BACKEND}, file={DOCKERFILE}) ===", flush=True)
    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE,
         "-f", str(HERE / DOCKERFILE), str(HERE)],
        check=True,
    )


def _write_stripped(src_full, dst):
    with open(src_full, newline="") as fin, \
            open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        missing = [c for c in _RATER_VISIBLE_COLUMNS
                   if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"Replicate {src_full} missing rater columns {missing}; "
                f"have {reader.fieldnames}."
            )
        writer = csv.DictWriter(fout, fieldnames=_RATER_VISIBLE_COLUMNS)
        writer.writeheader()
        for row in reader:
            writer.writerow({c: row[c] for c in _RATER_VISIBLE_COLUMNS})


def _rate_one(src, rated_dir, api_key):
    stem = src.stem
    out_dir = rated_dir / stem
    log_path = rated_dir / f"{stem}.log"
    scratch = rated_dir / f".tmp_{stem}_{uuid.uuid4().hex[:8]}"

    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    _write_stripped(src, scratch / src.name)

    input_csv = scratch / src.name
    input_hash_before = hashlib.sha256(input_csv.read_bytes()).hexdigest()

    start = time.time()
    container_name = f"eval-rater-{stem}-{uuid.uuid4().hex[:8]}"
    rate_script = (HERE / RATE_SCRIPT).resolve()
    container_rate_script = f"/opt/rater/{RATE_SCRIPT}"

    docker_env_args = [
        "-e", f"{RATER_API_KEY_ENV}={api_key}",
        "-e", f"RATER_MODEL={RATER_MODEL}",
        "-e", f"RATER_REASONING_EFFORT={RATER_REASONING_EFFORT}",
        "-e", f"RATER_REASONING_SUMMARY={RATER_REASONING_SUMMARY}",
    ]
    for name, value in RATER_EXTRA_CONTAINER_ENV:
        docker_env_args.extend(["-e", f"{name}={value}"])
    # Forward JANEKD_NO_BASH into the container so rate.py /
    # rate_codex.py can flip into no-compute mode. Without this,
    # the orchestrator's `--no-bash` setting only affected run.py's
    # meta.json but not the actual rater container, making the
    # no-bash run silently equivalent to a bash-equipped run.
    no_bash_env = os.environ.get("JANEKD_NO_BASH")
    if no_bash_env:
        docker_env_args.extend(["-e", f"JANEKD_NO_BASH={no_bash_env}"])
    # Forward JANEKD_TASK so the in-container rater (rate_codex.py)
    # selects the proofs rating prompt. Without this the container
    # always defaults to the Frobenius prompt regardless of the host
    # setting (same forwarding contract as JANEKD_NO_BASH above).
    task_env = os.environ.get("JANEKD_TASK")
    if task_env:
        docker_env_args.extend(["-e", f"JANEKD_TASK={task_env}"])
    # Forward JANEKD_QWEN_ENABLE_COMPUTE into the container so
    # rate_qwen.py can add the `run_python` tool to its function_list
    # and switch to the compute-enabled system prompt. Same reasoning
    # as JANEKD_NO_BASH above — without explicit forwarding the rater
    # would silently run in no-compute mode regardless of host setting.
    qwen_compute_env = os.environ.get("JANEKD_QWEN_ENABLE_COMPUTE")
    if qwen_compute_env:
        docker_env_args.extend(
            ["-e", f"JANEKD_QWEN_ENABLE_COMPUTE={qwen_compute_env}"]
        )

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--name", container_name,
                *docker_env_args,
                "-v", f"{scratch.resolve()}:/work",
                "-v", f"{rate_script}:{container_rate_script}:ro",
                DOCKER_IMAGE,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.time() - start
        log_path.write_text(
            f"=== exit_code: {result.returncode} ===\n"
            f"=== stdout ===\n{result.stdout}\n"
            f"=== stderr ===\n{result.stderr}\n"
        )
        if result.returncode != 0:
            return {
                "name": src.name, "ok": False, "elapsed": elapsed,
                "log_path": log_path,
                "status": f"docker exit={result.returncode}",
                "out_dir": None,
            }
        if not input_csv.exists():
            return {
                "name": src.name, "ok": False, "elapsed": elapsed,
                "log_path": log_path,
                "status": "rater deleted input CSV",
                "out_dir": None,
            }
        input_hash_after = hashlib.sha256(input_csv.read_bytes()).hexdigest()
        if input_hash_after != input_hash_before:
            return {
                "name": src.name, "ok": False, "elapsed": elapsed,
                "log_path": log_path,
                "status": (
                    f"rater mutated input CSV "
                    f"(sha256 {input_hash_before[:12]}..→"
                    f"{input_hash_after[:12]}..)"
                ),
                "out_dir": None,
            }
        best = scratch / "best_answers.csv"
        if not best.exists():
            return {
                "name": src.name, "ok": False, "elapsed": elapsed,
                "log_path": log_path,
                "status": "rater produced no best_answers.csv",
                "out_dir": None,
            }
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        shutil.copy(str(src), str(out_dir / src.name))
        shutil.move(str(best), str(out_dir / "best_answers.csv"))
        rater_log = scratch / "rater_log.jsonl"
        if rater_log.exists():
            shutil.move(str(rater_log), str(out_dir / "rater_log.jsonl"))
        rater_stderr = scratch / "rate_codex.stderr"
        if rater_stderr.exists():
            shutil.move(str(rater_stderr), str(out_dir / "rate_codex.stderr"))
        return {
            "name": src.name, "ok": True, "elapsed": elapsed,
            "log_path": log_path, "status": "ok", "out_dir": out_dir,
        }
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


def _trial_order(run_dir):
    reps = config.replicates_dir(run_dir)
    rated = config.rated_dir(run_dir)
    for seed in range(START_SEED, START_SEED + N_SEEDS):
        src = reps / f"rep_{seed:03d}.csv"
        if not src.exists():
            continue
        dst = rated / f"rep_{seed:03d}"
        yield seed, src, dst


def _live_preview(run_dir, success_count):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(buf):
            analyze.main(run_dir)
    except SystemExit as e:
        print(f"  · preview skipped ({e})", flush=True)
        return
    except Exception as e:  # noqa: BLE001
        print(f"  · preview FAILED: {e!r}", flush=True)
        return
    log_path = run_dir / "preview.log"
    with log_path.open("a") as f:
        f.write(f"\n\n=== preview @ {success_count} rated trial(s), "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(buf.getvalue())
    plot_rel = (run_dir / "analysis" / "win_rates_by_label.png"
                ).relative_to(run_dir)
    print(f"  · preview refreshed @ {success_count} rated  "
          f"→ {plot_rel}  (text → preview.log)", flush=True)


def _rate_all(run_dir):
    api_key = os.environ.get(RATER_HOST_API_KEY_ENV)
    if not api_key:
        sys.exit(
            f"{RATER_HOST_API_KEY_ENV} is not set in the host "
            f"environment. RATER_BACKEND={RATER_BACKEND!r} requires it "
            f"(passed into the container as {RATER_API_KEY_ENV})."
        )

    rated = config.rated_dir(run_dir)
    rated.mkdir(parents=True, exist_ok=True)

    trials = list(_trial_order(run_dir))
    pending = [t for t in trials if not (t[2] / "best_answers.csv").exists()]
    already = len(trials) - len(pending)

    if already:
        print(f"\n=== Skipping {already} already-rated trial(s) ===",
              flush=True)
    if not pending:
        print("All trials already rated.")
        return

    workers = min(RATING_PARALLELISM, len(pending))
    print(f"\n=== Rating {len(pending)} trial(s) with {workers} parallel "
          f"container(s) ===", flush=True)
    if LIVE_PREVIEW_EVERY > 0:
        print(f"    (live preview every {LIVE_PREVIEW_EVERY} successful "
              f"trial(s) → analysis/win_rates_by_label.png)", flush=True)
    print("", flush=True)

    overall_start = time.time()
    results = []
    success_count = already
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for seed, src, _dst in pending:
            fut = ex.submit(_rate_one, src, rated, api_key)
            futures[fut] = seed
        try:
            for completed, fut in enumerate(as_completed(futures), start=1):
                seed = futures[fut]
                r = fut.result()
                results.append((seed, r))
                mark = "OK " if r["ok"] else "ERR"
                print(
                    f"  [{completed}/{len(pending)}] {mark}  "
                    f"seed={seed:03d}  ({r['elapsed']:.0f}s)  "
                    f"{r['status']}  → {r['log_path']}",
                    flush=True,
                )
                if r["ok"]:
                    success_count += 1
                    if (LIVE_PREVIEW_EVERY > 0
                            and success_count % LIVE_PREVIEW_EVERY == 0):
                        _live_preview(run_dir, success_count)
        except KeyboardInterrupt:
            print("\nInterrupted — killing in-flight eval-rater containers...",
                  flush=True)
            for f in futures:
                f.cancel()
            for c in subprocess.run(
                ["docker", "ps", "-q", "-f", "name=eval-rater-"],
                capture_output=True, text=True, check=False,
            ).stdout.split():
                subprocess.run(["docker", "kill", c],
                               capture_output=True, check=False)
            raise

    failed = [(s, r) for s, r in results if not r["ok"]]
    total = time.time() - overall_start
    print(f"\nRating done in {total:.0f}s "
          f"({len(results) - len(failed)} ok, {len(failed)} failed).",
          flush=True)
    if failed:
        print(f"\n{len(failed)} trial(s) failed; re-run to retry:")
        for s, r in failed:
            print(f"  seed={s:03d}  → {r['log_path']}")


def _snapshot_config(run_dir):
    for name in ("config.py", RATE_SCRIPT, DOCKERFILE):
        src = HERE / name
        if not src.exists():
            continue
        dst = run_dir / name
        if dst.exists():
            dst.chmod(0o644)
            dst.unlink()
        shutil.copy(src, dst)
        os.chmod(dst, 0o444)


def _write_meta(run_dir):
    no_bash_env = os.environ.get("JANEKD_NO_BASH", "0").strip().lower()
    qwen_compute_env = os.environ.get(
        "JANEKD_QWEN_ENABLE_COMPUTE", "0",
    ).strip().lower()
    task_env = os.environ.get("JANEKD_TASK", "alpaca").strip().lower()
    (run_dir / "meta.json").write_text(json.dumps({
        "experiment": EXPERIMENT,
        "task": task_env,
        "has_ground_truth": bool(getattr(config, "HAS_GROUND_TRUTH", True)),
        "model_labels": list(MODEL_LABELS),
        "source_models": list(SOURCE_MODELS),
        "source_display_names": [
            MODELS[k]["display_name"] for k in SOURCE_MODELS
        ],
        "rater_backend": RATER_BACKEND,
        "rater_model": RATER_MODEL,
        "rater_reasoning_effort": RATER_REASONING_EFFORT,
        "no_bash": no_bash_env in ("1", "true", "yes"),
        "qwen_compute_enabled": qwen_compute_env in ("1", "true", "yes"),
        "docker_image": DOCKER_IMAGE,
        "start_seed": int(START_SEED),
        "n_seeds": int(N_SEEDS),
        "questions_file": str(config.QUESTIONS_FILE),
    }, indent=2) + "\n")


def _update_latest_symlink(run_dir):
    if os.name != "posix":
        return
    try:
        tmp = LATEST_LINK.with_name(".latest.tmp")
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        tmp.symlink_to(run_dir.name, target_is_directory=True)
        os.replace(tmp, LATEST_LINK)
    except OSError as e:
        print(f"(note: couldn't update {LATEST_LINK}: {e})", flush=True)


def _resolve_run_dir(args):
    if args.run_dir:
        return Path(args.run_dir)
    if args.resume:
        latest = config.latest_run_dir(required=False)
        if latest:
            print(f"Resuming latest run: {latest}")
            return latest
        print("(--resume requested but no existing run; "
              "starting a fresh one)")
    return config.new_run_dir()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(config.EXPERIMENTS),
        default=None,
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-rate", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    if N_SEEDS <= 0:
        sys.exit(f"config.N_SEEDS must be positive (got {N_SEEDS}).")
    if len(MODEL_LABELS) < 2:
        sys.exit(f"config.MODEL_LABELS needs at least 2 entries (got "
                 f"{len(MODEL_LABELS)}).")
    if len(set(MODEL_LABELS)) != len(MODEL_LABELS):
        sys.exit(f"config.MODEL_LABELS must be unique; got {MODEL_LABELS}.")
    if len(SOURCE_MODELS) < 1:
        sys.exit(f"config.SOURCE_MODELS needs at least 1 entry.")
    if len(set(SOURCE_MODELS)) != len(SOURCE_MODELS):
        sys.exit(f"config.SOURCE_MODELS must be unique; got {SOURCE_MODELS}.")
    for key in SOURCE_MODELS:
        if key not in MODELS:
            sys.exit(
                f"config.SOURCE_MODELS references {key!r} which is not "
                f"in shared.models.MODELS. Available keys: {sorted(MODELS)}"
            )

    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_config(run_dir)
    _write_meta(run_dir)
    _update_latest_symlink(run_dir)
    end_seed = START_SEED + N_SEEDS
    print(f"\n>>> Run dir: {run_dir}", flush=True)
    print(f"    Experiment: {EXPERIMENT}", flush=True)
    print(f"    Backend   : {RATER_BACKEND}  "
          f"(image={DOCKER_IMAGE}, rate script={RATE_SCRIPT})", flush=True)
    print(f"    Rater     : {RATER_MODEL}  "
          f"(reasoning_effort={RATER_REASONING_EFFORT})", flush=True)
    print(f"    Labels  ({len(MODEL_LABELS)}): {list(MODEL_LABELS)}",
          flush=True)
    print(f"    Sources ({len(SOURCE_MODELS)}): {list(SOURCE_MODELS)}",
          flush=True)
    print(f"    Seeds   : [{START_SEED}, {end_seed})   "
          f"= {N_SEEDS} trials total", flush=True)
    print(f"    Questions file: {config.QUESTIONS_FILE}", flush=True)

    _write_bases(run_dir)
    _write_replicates(run_dir)

    if not args.skip_rate:
        if not args.skip_build:
            build_image()
        _rate_all(run_dir)

    if not args.skip_analyze:
        try:
            analyze.main(run_dir)
        except SystemExit as e:
            print(f"(skipped analyze: {e})")


if __name__ == "__main__":
    main()
