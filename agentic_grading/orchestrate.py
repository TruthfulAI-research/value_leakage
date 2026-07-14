"""End-to-end orchestrator for the agentic answer-grading bias experiment.

A single Python process that runs the blinded best-of-K rating pipeline
across up to three agentic grader backends (Claude Code, Codex,
Qwen-Agent) on one of two tasks:

  --task alpaca    : subjective helpfulness grading of Alpaca instructions
  --task proofnet  : rigor grading of undergraduate proofs (statements
                     from the ProofNet benchmark; candidate proofs by
                     claude-haiku-4.5)

Both tasks use a committed question pool (no procedural generation) and
have NO objective ground truth, so analysis reports per-label /
per-source pick-rate + CI + significance vs 1/K plus CoT.

Stages:
  1. smoke test (2 seeds × backends) — a cheap end-to-end gate; bail on
     any failure before spending on the full run.
  2. full rating run (N_SEEDS × backends) in parallel, each via
     `python -m <pkg>.run`, stdout streamed to data/logs/rate_<backend>.log.
  3. per-backend CoT extraction (cot_extract_rationales) → data/cot/.
  4. CoT judge (Sonnet by default) on all trials files → data/cot/judged.jsonl.
  5. label-column aggregator → data/cot/judged.summary.{json,md}.

Chain-of-thought capture is enabled by default:
  - claude: extended thinking via MAX_THINKING_TOKENS
    (--claude-thinking-tokens, forwarded into the container),
  - codex : reasoning summaries via model_reasoning_summary
    (--reasoning-summary),
  - qwen  : reasoning_content when the provider returns it (best-effort).

Cost guardrails: the smoke gate is mandatory by default; the orchestrator
prints a cost estimate and prompts unless --yes is passed. --smoke-only
runs just the smoke gate.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config as cfg

# Resolve our own package name so the -m subprocess invocations work no
# matter what the package is called on disk (rate_llm_answers here).
_PKG = __package__ or "agentic_grading"

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
LOG_DIR = DATA_DIR / "logs"
COT_DIR = DATA_DIR / "cot"
SMOKE_DIR = DATA_DIR / "smoke"

BACKENDS = ("claude", "codex", "qwen")
TASKS = ("alpaca", "proofnet")
# Committed question pool per task (relative to this package dir).
TASK_QUESTIONS = {
    "alpaca": HERE / "alpaca_questions.jsonl",
    "proofnet": HERE / "proof_questions_proofnet.jsonl",
}
RUNDIR_RE = re.compile(r">>>\s*Run dir:\s*(\S+)")


@dataclass
class BackendOutcome:
    backend: str
    run_dir: Optional[Path]
    log_path: Path
    returncode: int
    elapsed_sec: float


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stream_subprocess(cmd, env, log_path, label):
    """Run `cmd` to completion, streaming stdout to log + prefixed echo."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                logf.write(line + "\n")
                logf.flush()
                lines.append(line)
                print(f"[{label}] {line}", flush=True)
        except KeyboardInterrupt:
            print(f"\n[{label}] Ctrl-C received — terminating subprocess",
                  flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        rc = proc.wait()
    return rc, lines


def _run_subprocess_parallel(jobs):
    """Launch `jobs` (label, cmd, env, log_path) in parallel; return
    [(label, returncode, output_lines)] in completion order."""
    procs: dict = {}
    log_handles: dict = {}
    out_lines: dict[str, list[str]] = {}
    threads: list[threading.Thread] = []
    completed: list[tuple[str, int, list[str]]] = []
    completed_lock = threading.Lock()

    for label, cmd, env, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = log_path.open("w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        procs[label] = proc
        log_handles[label] = logf
        out_lines[label] = []

        def reader(lbl=label, p=proc, lf=logf):
            assert p.stdout is not None
            for raw in p.stdout:
                line = raw.rstrip("\n")
                lf.write(line + "\n")
                lf.flush()
                out_lines[lbl].append(line)
                print(f"[{lbl}] {line}", flush=True)
            rc = p.wait()
            lf.close()
            with completed_lock:
                completed.append((lbl, rc, out_lines[lbl]))

        t = threading.Thread(target=reader, name=f"reader-{label}",
                             daemon=True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[orchestrate] Ctrl-C received — terminating all "
              "rater subprocesses", flush=True)
        for label, p in procs.items():
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 15
        for label, p in procs.items():
            while time.time() < deadline and p.poll() is None:
                time.sleep(0.1)
            if p.poll() is None:
                p.kill()
        for t in threads:
            t.join(timeout=5)
        raise

    return completed


def _parse_run_dir(lines):
    for line in lines:
        m = RUNDIR_RE.search(line)
        if m:
            return Path(m.group(1))
    return None


def _require_api_keys(backends, qwen_provider):
    """Bail out before spending anything if a required key is missing."""
    needed = {"ANTHROPIC_API_KEY"}  # source model + Claude grader + judge
    if "codex" in backends:
        needed.add("OPENAI_API_KEY")
    if "qwen" in backends:
        needed.add("OPENROUTER_API_KEY" if qwen_provider == "openrouter"
                   else "DASHSCOPE_API_KEY")
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"Missing required API key(s) in the host env: "
            f"{', '.join(missing)}. Backends needed: {backends}; "
            f"qwen provider: {qwen_provider}. Aborting before any "
            f"work is done."
        )


def _backend_env(args, backend, *, n_seeds, questions_file, parallelism):
    """Common per-backend env for a run.py subprocess."""
    env = os.environ.copy()
    env["JANEKD_EXPERIMENT"] = args.task          # preset name == task name
    env["JANEKD_RATER_BACKEND"] = backend
    env["JANEKD_TASK"] = args.task
    env["JANEKD_NO_GROUND_TRUTH"] = "1"           # both tasks are no-GT
    env["JANEKD_N_SEEDS"] = str(n_seeds)
    env["JANEKD_QUESTIONS_FILE"] = str(questions_file)
    env["JANEKD_RATING_PARALLELISM"] = str(parallelism)
    env["PYTHONUNBUFFERED"] = "1"
    if backend == "qwen":
        env["JANEKD_QWEN_PROVIDER"] = args.qwen_provider
    if backend == "claude" and args.claude_thinking_tokens:
        # Enable + budget Claude extended thinking so ThinkingBlocks
        # carry real CoT (config._claude_extra_container_env forwards
        # this into the container as MAX_THINKING_TOKENS).
        env["JANEKD_CLAUDE_THINKING_TOKENS"] = str(args.claude_thinking_tokens)
    env["JANEKD_RATER_REASONING_SUMMARY"] = args.reasoning_summary
    return env


# ----------------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------------

def stage_smoke_test(args) -> None:
    if args.no_smoke:
        print("\n[smoke] --no-smoke set — skipping smoke gate "
              "(NOT recommended for the full run)", flush=True)
        return

    print(f"\n=== Smoke test (2 seeds × {len(args.backends)} backend(s)) "
          f"({_now_str()}) ===", flush=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # Both tasks reuse the committed pool; run.py samples N_QUESTIONS per
    # seed, so 2 seeds gives a tiny but real smoke.
    smoke_questions = TASK_QUESTIONS[args.task]

    jobs = []
    for backend in args.backends:
        env = _backend_env(args, backend, n_seeds=2,
                           questions_file=smoke_questions, parallelism=2)
        cmd = [sys.executable, "-u", "-m", f"{_PKG}.run",
               "--experiment", args.task]
        if args.skip_build:
            cmd.append("--skip-build")
        log_path = LOG_DIR / "smoke" / f"rate_{backend}.log"
        jobs.append((f"smoke-{backend}", cmd, env, log_path))

    results = _run_subprocess_parallel(jobs)
    failed = [(lbl, rc) for lbl, rc, _ in results if rc != 0]
    if failed:
        raise SystemExit(
            f"Smoke test FAILED in {len(failed)} backend(s): {failed}. "
            f"Check {LOG_DIR / 'smoke'}/*.log. Aborting before the "
            f"full launch."
        )
    print(f"\n[smoke] all {len(args.backends)} backends passed.", flush=True)

    smoke_run_dirs = {}
    for lbl, _rc, lines in results:
        backend = lbl.replace("smoke-", "")
        run_dir = _parse_run_dir(lines)
        if run_dir is None:
            raise SystemExit(
                f"Smoke: could not find a `Run dir:` line in "
                f"backend={backend} log."
            )
        smoke_run_dirs[backend] = run_dir

    smoke_cot_dir = COT_DIR / "smoke"
    smoke_cot_dir.mkdir(parents=True, exist_ok=True)
    trials_paths = []
    for backend, run_dir in smoke_run_dirs.items():
        out = smoke_cot_dir / f"trials_{backend}.jsonl"
        cmd = [sys.executable, "-u", "-m",
               f"{_PKG}.cot_extract_rationales",
               backend, str(run_dir), str(out)]
        log_path = LOG_DIR / "smoke" / f"cot_extract_{backend}.log"
        rc, _ = _stream_subprocess(cmd, os.environ.copy(), log_path,
                                   f"smoke-cot-extract-{backend}")
        if rc != 0 or not out.exists() or out.stat().st_size == 0:
            raise SystemExit(
                f"Smoke cot_extract failed for backend={backend} "
                f"(rc={rc}); see {log_path}"
            )
        trials_paths.append(out)

    judged_path = smoke_cot_dir / "judged.jsonl"
    cmd = [sys.executable, "-u", "-m", f"{_PKG}.cot_judge",
           "--output", str(judged_path), "--n-per-rater", "2",
           "--workers", "3"]
    for t in trials_paths:
        cmd.extend(["--input", str(t)])
    rc, _ = _stream_subprocess(cmd, os.environ.copy(),
                               LOG_DIR / "smoke" / "cot_judge.log",
                               "smoke-cot-judge")
    if rc != 0:
        raise SystemExit(f"Smoke cot_judge failed rc={rc}")

    cmd = [sys.executable, "-u", "-m",
           f"{_PKG}.cot_aggregate_label_column", str(judged_path),
           "--out-prefix", str(smoke_cot_dir / "judged.summary")]
    rc, _ = _stream_subprocess(cmd, os.environ.copy(),
                               LOG_DIR / "smoke" / "cot_aggregate.log",
                               "smoke-cot-aggregate")
    if rc != 0:
        raise SystemExit(f"Smoke cot_aggregate failed rc={rc}")

    print(f"\n[smoke] OK. End-to-end pipeline works. Run dirs:", flush=True)
    for backend, run_dir in smoke_run_dirs.items():
        print(f"        {backend:6s} → {run_dir}", flush=True)
    print(f"        cot   → {smoke_cot_dir}", flush=True)


def stage_full_run(args) -> dict[str, BackendOutcome]:
    print(f"\n=== Full rating run ({_now_str()}) ===  "
          f"{len(args.backends)} backend(s) × N_SEEDS={args.n_seeds} ===",
          flush=True)
    questions_file = TASK_QUESTIONS[args.task]
    jobs = []
    log_paths = {}
    for backend in args.backends:
        env = _backend_env(args, backend, n_seeds=args.n_seeds,
                           questions_file=questions_file,
                           parallelism=args.rating_parallelism)
        cmd = [sys.executable, "-u", "-m", f"{_PKG}.run",
               "--experiment", args.task]
        if args.skip_build:
            cmd.append("--skip-build")
        log_path = LOG_DIR / f"rate_{backend}.log"
        log_paths[backend] = log_path
        jobs.append((f"rate-{backend}", cmd, env, log_path))

    results = _run_subprocess_parallel(jobs)
    outcomes = {}
    for lbl, rc, lines in results:
        backend = lbl.replace("rate-", "")
        outcomes[backend] = BackendOutcome(
            backend=backend, run_dir=_parse_run_dir(lines),
            log_path=log_paths[backend], returncode=rc, elapsed_sec=0.0,
        )
    failed = [b for b, o in outcomes.items() if o.returncode != 0]
    if failed:
        for b in failed:
            print(f"[rate-{b}] FAILED rc={outcomes[b].returncode}; "
                  f"see {outcomes[b].log_path}", flush=True)
        print(f"\n[orchestrate] {len(failed)}/{len(outcomes)} backend(s) "
              f"failed; continuing into CoT analysis for the survivors.",
              flush=True)
    return outcomes


def stage_cot_extract(outcomes) -> list[Path]:
    print(f"\n=== cot_extract_rationales ({_now_str()}) ===", flush=True)
    COT_DIR.mkdir(parents=True, exist_ok=True)
    trials = []
    for backend, outcome in outcomes.items():
        if outcome.run_dir is None or outcome.returncode != 0:
            print(f"[cot-extract] skipping backend={backend} "
                  f"(run_dir={outcome.run_dir}, rc={outcome.returncode})",
                  flush=True)
            continue
        out = COT_DIR / f"trials_{backend}.jsonl"
        cmd = [sys.executable, "-u", "-m",
               f"{_PKG}.cot_extract_rationales",
               backend, str(outcome.run_dir), str(out)]
        log_path = LOG_DIR / f"cot_extract_{backend}.log"
        rc, _ = _stream_subprocess(cmd, os.environ.copy(), log_path,
                                   f"cot-extract-{backend}")
        if rc != 0 or not out.exists() or out.stat().st_size == 0:
            print(f"[cot-extract] backend={backend} produced no trials "
                  f"file or rc={rc}; see {log_path}", flush=True)
            continue
        trials.append(out)
    return trials


def stage_cot_judge(trials_paths, args) -> Optional[Path]:
    if not trials_paths:
        print("\n[cot-judge] no trials files; skipping", flush=True)
        return None
    print(f"\n=== cot_judge ({_now_str()}) ===  "
          f"inputs: {[p.name for p in trials_paths]} ===", flush=True)
    judged_path = COT_DIR / "judged.jsonl"
    cmd = [sys.executable, "-u", "-m", f"{_PKG}.cot_judge",
           "--output", str(judged_path),
           "--n-per-rater", str(args.judge_n_per_rater),
           "--workers", str(args.judge_workers)]
    for t in trials_paths:
        cmd.extend(["--input", str(t)])
    rc, _ = _stream_subprocess(cmd, os.environ.copy(),
                               LOG_DIR / "cot_judge.log", "cot-judge")
    if rc != 0:
        raise SystemExit(f"cot_judge failed rc={rc}")
    return judged_path


def stage_cot_aggregate(judged_path) -> Optional[Path]:
    if judged_path is None or not judged_path.exists():
        print("\n[cot-aggregate] no judged.jsonl; skipping", flush=True)
        return None
    print(f"\n=== cot_aggregate_label_column ({_now_str()}) ===", flush=True)
    summary_prefix = COT_DIR / "judged.summary"
    cmd = [sys.executable, "-u", "-m",
           f"{_PKG}.cot_aggregate_label_column", str(judged_path),
           "--out-prefix", str(summary_prefix)]
    rc, _ = _stream_subprocess(cmd, os.environ.copy(),
                               LOG_DIR / "cot_aggregate.log", "cot-aggregate")
    if rc != 0:
        raise SystemExit(f"cot_aggregate failed rc={rc}")
    return Path(str(summary_prefix) + ".md")


def stage_final_summary(outcomes, judged_summary_md) -> None:
    print(f"\n=== Final summary ({_now_str()}) ===", flush=True)
    print(f"Run dirs (per backend):")
    for backend, outcome in outcomes.items():
        ok = "OK " if outcome.returncode == 0 else f"ERR rc={outcome.returncode}"
        print(f"  {backend:6s} {ok:10s}  {outcome.run_dir}", flush=True)
    if judged_summary_md and judged_summary_md.exists():
        print(f"\nCoT label-column report: {judged_summary_md}", flush=True)
        body = judged_summary_md.read_text()
        head = body.split("## Secondary")[0]
        print(textwrap.indent(head, "  "), flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", default="alpaca", choices=TASKS,
                    help="Which task to grade (default: alpaca). Selects "
                         "the question pool, rating prompt, experiment "
                         "preset, and cache dir.")
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS),
                    choices=BACKENDS,
                    help="Which grader backends to run (default: all 3).")
    ap.add_argument("--qwen-provider", default="openrouter",
                    choices=("dashscope", "openrouter"),
                    help="Qwen sub-provider (default: openrouter).")
    ap.add_argument("--n-seeds", type=int, default=250,
                    help="Seeds per grader backend (default: 250).")
    ap.add_argument("--rating-parallelism", type=int, default=20,
                    help="Docker containers per grader backend.")
    ap.add_argument("--claude-thinking-tokens", type=int, default=8000,
                    help="Claude extended-thinking budget (tokens). >0 "
                         "turns on extended thinking so CoT is captured; "
                         "set 0 to disable. Default 8000.")
    ap.add_argument("--reasoning-summary", default="detailed",
                    choices=("auto", "concise", "detailed", "none"),
                    help="Codex model_reasoning_summary verbosity "
                         "(default: detailed — 'auto' often emits no "
                         "summary on easy picks, losing CoT). 'none' "
                         "suppresses CoT capture.")
    ap.add_argument("--judge-n-per-rater", type=int, default=100,
                    help="How many rationales per grader to send to the "
                         "CoT judge (default: 100).")
    ap.add_argument("--judge-workers", type=int, default=8,
                    help="Parallel calls to the CoT judge.")
    ap.add_argument("--no-smoke", action="store_true",
                    help="Skip the smoke gate before the full launch. "
                         "NOT recommended.")
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip docker build in each grader subprocess "
                         "(use only when the eval-rater* images are "
                         "already tagged locally and up to date).")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Run only the smoke gate and stop (no full "
                         "launch, no CoT pipeline). Cheap dry-run.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the cost-estimate confirmation prompt.")
    args = ap.parse_args()

    if not TASK_QUESTIONS[args.task].exists():
        raise SystemExit(
            f"--task {args.task} but {TASK_QUESTIONS[args.task]} does not "
            f"exist. The committed question pool is missing."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _require_api_keys(args.backends, args.qwen_provider)

    print(f"=== orchestrate.py start ({_now_str()}) ===")
    print(f"  package         : {_PKG}")
    print(f"  task            : {args.task}")
    print(f"  question pool   : {TASK_QUESTIONS[args.task]}")
    print(f"  backends        : {args.backends}")
    print(f"  qwen provider   : {args.qwen_provider}")
    print(f"  n_seeds         : {args.n_seeds}")
    print(f"  rating parallel : {args.rating_parallelism}")
    print(f"  claude thinking : {args.claude_thinking_tokens} tokens")
    print(f"  reasoning summ. : {args.reasoning_summary}")
    print(f"  skip smoke?     : {args.no_smoke}")
    print(f"  smoke only?     : {args.smoke_only}")

    if not args.yes and not args.smoke_only:
        est_low = 100 * len(args.backends)
        est_high = 250 * len(args.backends)
        print(f"\n!! Cost estimate: ${est_low} – ${est_high} for the "
              f"full rating run (N={args.n_seeds} × {len(args.backends)} "
              f"backends).  Set --yes to skip this prompt.", flush=True)
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted by user.", flush=True)
            return

    stage_smoke_test(args)

    if args.smoke_only:
        print("\n--smoke-only: stopping after smoke gate.", flush=True)
        return

    outcomes = stage_full_run(args)
    trials_paths = stage_cot_extract(outcomes)
    judged_path = stage_cot_judge(trials_paths, args)
    judged_summary = stage_cot_aggregate(judged_path)
    stage_final_summary(outcomes, judged_summary)

    print(f"\n=== orchestrate.py done ({_now_str()}) ===", flush=True)


if __name__ == "__main__":
    _cli()
