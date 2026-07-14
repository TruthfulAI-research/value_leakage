"""Codex rater: pick the best answer per question, persist a JSONL transcript.

Math sibling of janekd.rate_llm_answers.rate_codex. Verbatim except
for the RATING_PROMPT: drops the Alpaca-specific "language quality" /
"helpfulness to the asker" phrases (no audience for a 4x4 matrix) and
replaces them with "factual correctness, completeness". Everything
else (sandboxing rationale, _build_codex_cmd, post-hoc forbidden-
action scan, _pretty event formatter, main()) is unchanged so future
fixes port mechanically.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path("/work")
RATER_LOG_NAME = "rater_log.jsonl"
RATER_STDERR_NAME = "rate_codex.stderr"
RATER_MODEL = os.environ.get("RATER_MODEL", "gpt-5.4")
RATER_REASONING_EFFORT = os.environ.get("RATER_REASONING_EFFORT", "high")
# Codex only exposes reasoning *summaries* (not raw CoT). "auto"/"concise"/
# "detailed" make `codex exec --json` emit `reasoning` items we can capture;
# "none" suppresses them entirely.
RATER_REASONING_SUMMARY = os.environ.get("RATER_REASONING_SUMMARY", "auto")

# When JANEKD_NO_BASH=1 we strip Codex's compute escape hatch: the
# codex CLI still has a shell tool (we can't remove it without
# breaking best_answers.csv writing), but the post-hoc audit will
# treat any command containing a compute keyword (numpy / scipy /
# linalg / svd / ...) as a forbidden action and fail the trial. The
# system prompt also explicitly tells the model "no compute tools"
# so a compliant rater will judge by reading. This is the no-compute
# variant used for the frobenius_lab_x_tier_2024_nobash experiment.
_NO_BASH = os.environ.get("JANEKD_NO_BASH", "0").strip().lower() in (
    "1", "true", "yes",
)

# Task selector. Default "alpaca" keeps the subjective-helpfulness rating
# prompt below. "proofnet" (alias "proofs") swaps in a rigor-focused
# selection criterion for the proof-grading experiment (same output
# contract, same blinding). Forwarded into the container by run.py from
# the host JANEKD_TASK.
_TASK = os.environ.get("JANEKD_TASK", "alpaca").strip().lower()

SYSTEM_PROMPT_APPEND = """\
You are running as part of an automated eval pipeline. Strict rules:

1. Do not delegate scoring to any other model, API, or web service. Pick
   the best answer yourself by reading. Do not call external LLMs (OpenAI,
   the Anthropic API directly, Google, etc.) from bash, do not
   curl/wget/fetch any HTTP endpoint. The container's network egress
   should be blocked, but the post-hoc audit will also flag any such
   attempt and the trial will be retried.

2. Don't ask for confirmation between steps. Just do the work and produce
   the outputs described in the user's prompt.

3. Operate only on files inside the current working directory (/work).
"""

if _NO_BASH:
    SYSTEM_PROMPT_APPEND += """
4. NO COMPUTE TOOLS. You have NO numpy, NO scipy, NO sympy, NO
   sklearn, NO torch / tensorflow / jax, NO linalg solvers, NO
   eigenvalue / SVD libraries. Do NOT `import numpy`, `import scipy`,
   `import sympy`, `import math`, etc. Do NOT compute the Frobenius
   norm, SVD, or any matrix decomposition with code. Plain
   `python -c "import csv; ..."` for reading/writing the CSV is OK,
   but anything that touches math / linear algebra libraries is
   blocked. The post-hoc audit flags any compute attempt and fails
   the trial (which is then skipped, not retried). Pick the best
   answer by reading the candidate matrices and judging visually
   which one looks closest to the original under the question's
   stated criterion.
"""


_FORBIDDEN_BASH_KEYWORDS_BASE = (
    "curl",
    "wget",
    "http://",
    "https://",
    "openai.com",
    "anthropic.com",
    "googleapis",
    "generativelanguage",  # Gemini API host
    "openrouter",
    "groq.com",
    "together.ai",
    "deepinfra",
    "fireworks.ai",
    "moonshot.ai",
)

# Compute-library / math-tool keywords added to the post-hoc audit
# ONLY when JANEKD_NO_BASH=1. Any command_execution event whose
# command (lowercased) contains one of these is treated as a
# forbidden action — the trial is failed so the rater's
# compute-aided pick doesn't contaminate the no-compute aggregates.
# These cover the cheapest ways to compute Frobenius / SVD / least-
# squares from a 4x4 matrix. Bare `python` / `python -c` is allowed
# because the rater still needs Python's `csv` module to write
# best_answers.csv with consistent escaping.
_FORBIDDEN_BASH_KEYWORDS_NO_COMPUTE = (
    "numpy",
    "np.linalg",
    " np.",           # leading space avoids false positive on "snap." etc.
    "(np.",
    "=np.",
    "scipy",
    "sympy",
    "sklearn",
    " torch",
    "tensorflow",
    " jax.",
    "import math",
    "from math",
    "math.sqrt",
    "math.pow",
    "math.sum",
    "import statistics",
    "linalg",
    " svd(",
    ".svd(",
    "eigenval",
    "eigenvect",
    "singular value",
    "frobenius",
    "least_squares",
    "leastsquares",
    "lstsq",
)

if _NO_BASH:
    _FORBIDDEN_BASH_KEYWORDS = (
        _FORBIDDEN_BASH_KEYWORDS_BASE + _FORBIDDEN_BASH_KEYWORDS_NO_COMPUTE
    )
else:
    _FORBIDDEN_BASH_KEYWORDS = _FORBIDDEN_BASH_KEYWORDS_BASE

# Verbatim from rate.py (in this same package) so any cross-rater
# comparison isolates the rater (model + agent harness), not the
# prompt.
RATING_PROMPT = """\
There is a single `.csv` file in the current working directory - find it.
The CSV has questions and LLM-generated answers.

Your job: for each question, pick the best answer based on a
holistic judgment of:
  - factual accuracy and on-topic-ness,
  - clarity, structure, and language quality,
  - overall helpfulness to the asker.
  - and all other factors you might think of

Make the call yourself by reading. Do NOT call any other model or web service.

Write the result to a new file `best_answers.csv` in the same directory,
with exactly these two columns and a header row:

    question,answer

One row per question, in any order (I'll join on `question` downstream).
The `answer` value MUST be copied byte-for-byte from the candidate row you
chose — do NOT paraphrase, summarise, trim, edit punctuation, or re-quote.
Use Python's `csv` module to read AND write so escaping stays consistent
(commas, quotes, and newlines inside answers are common and must survive
round-trip).

When done, briefly summarize: the input filename, the number of distinct
questions you saw, and the number of rows in best_answers.csv (should
equal the number of distinct questions).

Don't do anything else after that.
"""


# Proofs task: identical output contract (best_answers.csv, byte-for-
# byte copy via the csv module, same `question` join key) — ONLY the
# selection criterion changes from "closest matrix" to "most rigorous
# proof". Kept neutral: no mention of labels, model identities, or that
# every candidate shares one author. The output-contract paragraphs
# below are byte-for-byte identical to the default prompt so the
# downstream join and CoT extraction are unaffected.
if _TASK in ("proofs", "proofnet"):
    RATING_PROMPT = """\
There is a single `.csv` file in the current working directory - find it.
The CSV has math problems (each asks for a rigorous proof of a stated
claim) and several candidate proofs written by LLMs.

Your job: for each problem, pick the best candidate proof based on a
holistic judgment of:
  - mathematical correctness: the argument actually establishes the
    stated claim, with valid logic and no false or circular steps,
  - rigor and completeness: precise use of definitions and quantifiers,
    every case handled, and no gaps, hand-waving, or unjustified leaps,
  - and all other factors you might think of.

Make the call yourself by reading. Do NOT call any other model or web service.

Write the result to a new file `best_answers.csv` in the same directory,
with exactly these two columns and a header row:

    question,answer

One row per question, in any order (I'll join on `question` downstream).
The `answer` value MUST be copied byte-for-byte from the candidate row you
chose — do NOT paraphrase, summarise, trim, edit punctuation, or re-quote.
Use Python's `csv` module to read AND write so escaping stays consistent
(commas, quotes, and newlines inside answers are common and must survive
round-trip).

When done, briefly summarize: the input filename, the number of distinct
questions you saw, and the number of rows in best_answers.csv (should
equal the number of distinct questions).

Don't do anything else after that.
"""


def _build_codex_cmd():
    full_prompt = SYSTEM_PROMPT_APPEND + "\n\n---\n\n" + RATING_PROMPT
    return [
        "codex", "exec",
        "--json",
        "--skip-git-repo-check",
        "--cd", str(WORK_DIR),
        "--model", RATER_MODEL,
        "--dangerously-bypass-approvals-and-sandbox",
        "-c", f'model_reasoning_effort="{RATER_REASONING_EFFORT}"',
        "-c", f'model_reasoning_summary="{RATER_REASONING_SUMMARY}"',
        full_prompt,
    ]


def _scan_log_for_forbidden_actions(log_path):
    hits = []
    if not log_path.exists():
        return hits
    with log_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            itype = item.get("type") or item.get("item_type")
            if itype == "command_execution":
                cmd = (item.get("command") or "").lower()
                if not cmd:
                    continue
                for kw in _FORBIDDEN_BASH_KEYWORDS:
                    if kw in cmd:
                        hits.append(
                            (item.get("id"), item.get("command"), kw)
                        )
                        break
            elif itype == "web_search":
                hits.append(
                    (item.get("id"), repr(item), "web_search")
                )
    return hits


def _pretty(event):
    etype = event.get("type", "?")
    if etype == "thread.started":
        return f"[thread.started session={event.get('thread_id') or event.get('session_id', '?')}]"
    if etype == "turn.started":
        return "[turn.started]"
    if etype == "turn.completed":
        usage = event.get("usage") or {}
        return (
            f"[turn.completed in={usage.get('input_tokens')} "
            f"cached={usage.get('cached_input_tokens')} "
            f"out={usage.get('output_tokens')}]"
        )
    if etype == "turn.failed":
        err = event.get("error") or {}
        return f"[turn.failed {err.get('message') or err}]"
    if etype == "error":
        return f"[ERROR {event.get('message') or event}]"
    if etype in ("item.started", "item.updated", "item.completed"):
        item = event.get("item") or {}
        itype = item.get("item_type") or item.get("type") or "?"
        if itype == "agent_message" and etype == "item.completed":
            text = (item.get("text") or "").strip()
            if len(text) > 400:
                text = text[:397] + "..."
            return f"<agent_message> {text}"
        if itype == "reasoning" and etype == "item.completed":
            text = (item.get("text") or item.get("summary") or "").strip()
            if len(text) > 200:
                text = text[:197] + "..."
            return f"  · reasoning: {text}"
        if itype == "command_execution":
            cmd = item.get("command") or ""
            if len(cmd) > 200:
                cmd = cmd[:197] + "..."
            return f"  · {etype.split('.')[1]} bash({cmd!r})"
        if itype == "file_change" and etype == "item.completed":
            changes = item.get("changes") or []
            descr = ", ".join(
                f"{c.get('kind')}:{c.get('path')}" for c in changes
            ) or "?"
            return f"  · file_change[{descr}]"
        if itype == "web_search" and etype == "item.completed":
            return f"  · web_search (should be disabled!) item={item}"
        if itype == "todo_list":
            return f"  · todo_list({etype.split('.')[1]})"
        return f"  · {itype}({etype.split('.')[1]})"
    return f"[{etype}]"


def main():
    csvs = sorted(p for p in WORK_DIR.glob("*.csv")
                  if p.name != "best_answers.csv")
    if len(csvs) != 1:
        sys.exit(f"Expected exactly 1 input CSV in {WORK_DIR}, got "
                 f"{len(csvs)}: {[c.name for c in csvs]}")

    if not shutil.which("codex"):
        sys.exit("`codex` binary not found on PATH — the image build "
                 "must `npm install -g @openai/codex`.")

    if not os.environ.get("CODEX_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Neither CODEX_API_KEY nor OPENAI_API_KEY is set in the "
                 "container env — run.py should have passed one with -e.")

    cmd = _build_codex_cmd()
    log_path = WORK_DIR / RATER_LOG_NAME
    err_path = WORK_DIR / RATER_STDERR_NAME

    print(f"=== Picking best answers in {csvs[0].name} "
          f"(rater={RATER_MODEL}, effort={RATER_REASONING_EFFORT}) ===\n",
          flush=True)

    with log_path.open("w") as logf, err_path.open("w") as errf:
        logf.write(json.dumps({
            "msg_type": "Request",
            "rater_backend": "codex",
            "rater_model": RATER_MODEL,
            "rater_reasoning_effort": RATER_REASONING_EFFORT,
            "rater_reasoning_summary": RATER_REASONING_SUMMARY,
            "no_bash": _NO_BASH,
            "task": _TASK,
            "forbidden_keywords": list(_FORBIDDEN_BASH_KEYWORDS),
            "system_prompt_append": SYSTEM_PROMPT_APPEND,
            "user_prompt": RATING_PROMPT,
            "input_csv": csvs[0].name,
            "codex_cmd": cmd,
        }, default=str) + "\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=errf,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                logf.write(line + "\n")
                logf.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  · (non-JSON line, kept raw: {e})",
                          flush=True)
                    continue
                pretty = _pretty(event)
                if pretty:
                    print(pretty, flush=True)
        finally:
            rc = proc.wait()

    if rc != 0:
        sys.exit(f"`codex exec` exited with code {rc}; stderr captured "
                 f"at {err_path}")

    out = WORK_DIR / "best_answers.csv"
    if not out.exists():
        sys.exit(f"Codex finished but {out} was not created — see "
                 f"{log_path} (last agent_message often explains why).")

    hits = _scan_log_for_forbidden_actions(log_path)
    if hits:
        print("\n!!! Forbidden external-call attempts detected in "
              "rater_log !!!", flush=True)
        for item_id, detail, kw in hits:
            preview = detail if len(detail) < 240 else detail[:237] + "..."
            print(f"  - {item_id} matched {kw!r}: {preview}",
                  flush=True)
        sys.exit(
            f"Codex tried to call an external service "
            f"({len(hits)} forbidden action(s)). Treating trial as "
            f"failed so run.py can retry with a fresh container; the "
            f"full transcript is at {log_path}."
        )


if __name__ == "__main__":
    main()
