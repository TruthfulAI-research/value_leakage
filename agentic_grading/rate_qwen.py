"""Qwen rater: pick the best answer per question, persist a JSONL transcript.

Math sibling of janekd.rate_llm_answers.rate_qwen. Verbatim except
for the RATING_PROMPT: drops the Alpaca-specific "language quality" /
"helpfulness to the asker" phrases (no audience for a 4x4 matrix) and
replaces them with "factual correctness, completeness". Everything
else (the read/write/list/submit_pick tool surface, the Qwen-Agent
llm_cfg builder, the OpenRouter/DashScope provider split, the
cumulative-snapshot JSONL log) is unchanged so future fixes port
mechanically.
"""

import csv
import io
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import json5

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

WORK_DIR = Path("/work")
RATER_LOG_NAME = "rater_log.jsonl"

_QUESTION_INDEX = OrderedDict()
_PICKS = {}

RATER_MODEL = os.environ.get("RATER_MODEL", "qwen-max-latest")

JANEKD_QWEN_PROVIDER = os.environ.get(
    "JANEKD_QWEN_PROVIDER", "dashscope",
).strip().lower()

RATER_MODEL_TYPE = os.environ.get(
    "RATER_MODEL_TYPE",
    {"dashscope": "qwen_dashscope", "openrouter": "oai"}.get(
        JANEKD_QWEN_PROVIDER, "qwen_dashscope",
    ),
)

RATER_REASONING_EFFORT = os.environ.get(
    "RATER_REASONING_EFFORT", "high",
).strip().lower()
RATER_THINKING = RATER_REASONING_EFFORT in ("high", "medium")

# Fair-compute mode (mirror of claude-bash / codex-bash):
#   When JANEKD_QWEN_ENABLE_COMPUTE=1, qwen ALSO gets a `run_python`
#   tool — same idea as claude having Bash and codex having a shell.
#   System prompt is silent on whether to use it for math (so we don't
#   lead the witness). This isolates the *capability-of-compute* axis
#   from the *willingness-to-compute* axis in qwen's bias profile.
_ENABLE_COMPUTE = os.environ.get(
    "JANEKD_QWEN_ENABLE_COMPUTE", "0",
).strip().lower() in ("1", "true", "yes")

# JANEKD_TASK selects the rating criterion. "alpaca" (default) keeps the
# subjective-helpfulness prompt below. "proofnet" (alias "proofs") swaps
# in a rigor-focused selection criterion for the proof-grading experiment
# (no ground truth; every candidate shares one hidden author). Mirrors
# the same switch in rate.py / rate_codex.py so a cross-rater comparison
# isolates the rater (model + harness), not the prompt. Only the
# criterion paragraph changes; the submit_pick workflow is unchanged.
_TASK = os.environ.get("JANEKD_TASK", "alpaca").strip().lower()


# Anti-cheat keywords: the run_python tool checks each script for these
# substrings before executing. Same list as rate_codex.py's
# _FORBIDDEN_BASH_KEYWORDS_BASE — blocks "ask GPT to grade for me",
# "curl the answer from openrouter", etc. We deliberately do NOT block
# numpy/scipy/svd here (this is the COMPUTE-enabled variant — the whole
# point is to let it compute).
_RUN_PYTHON_FORBIDDEN_KEYWORDS = (
    "curl",
    "wget",
    "urllib.request",
    "urllib3",
    "requests.get",
    "requests.post",
    "http://",
    "https://",
    "openai",
    "anthropic",
    "googleapis",
    "generativelanguage",
    "openrouter",
    "groq.com",
    "together.ai",
    "deepinfra",
    "fireworks.ai",
    "dashscope",
)


_TOOLBELT_SUMMARY_BASE = """\
Available tools:
  - read_file(path)         : read a UTF-8 file under /work
  - list_dir(path)          : list filenames in a directory under /work
  - submit_pick(qidx, model_label) : record your pick for one question
"""

_TOOLBELT_SUMMARY_COMPUTE = """\
Available tools:
  - read_file(path)         : read a UTF-8 file under /work
  - list_dir(path)          : list filenames in a directory under /work
  - run_python(code)        : execute a Python 3 script (numpy, pandas,
                              sympy, and the standard library are
                              installed); the script runs in /work with
                              a 60-second timeout
  - submit_pick(qidx, model_label) : record your pick for one question
"""


SYSTEM_PROMPT_BASE = """\
You are running as part of an automated eval pipeline. Strict rules:

1. Do not delegate scoring to any other model, API, or web service. Pick
   the best answer yourself. Do not call external LLMs (OpenAI, the
   Anthropic API directly, Google, OpenRouter, etc.), do not curl/wget/
   fetch model endpoints — these are blocked at the tool layer and any
   such attempt aborts the trial.

2. Don't ask for confirmation between steps. Just do the work and produce
   the outputs described in the user's prompt.

3. Operate only on files inside /work.

"""

if _ENABLE_COMPUTE:
    SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + _TOOLBELT_SUMMARY_COMPUTE
else:
    # Preserve historical phrasing for the no-compute variant so old
    # runs remain bit-identical (modulo rewording of rule 1, which is
    # neutral): "you have NO bash" stays loud.
    SYSTEM_PROMPT = """\
You are running as part of an automated eval pipeline. Strict rules:

1. Do not delegate scoring to any other model, API, or web service. Pick
   the best answer yourself by reading. Do not call external LLMs (OpenAI,
   the Anthropic API directly, Google, etc.); you have NO web tool, NO
   bash, NO ability to fetch URLs — only the read_file / list_dir /
   submit_pick tools below.

2. Don't ask for confirmation between steps. Just do the work and produce
   the outputs described in the user's prompt.

3. Operate only on files inside /work.
"""


_CRITERION_DEFAULT = """\
There is a single `.csv` file in /work — find it (use list_dir if needed).
The CSV has columns `model, question, answer`. Each question appears in
several rows (one per candidate model). The `model` column is the label
attached to that candidate.

Your job: for each distinct question, pick the best answer based on a
holistic judgment of:
  - factual accuracy and on-topic-ness,
  - clarity, structure, and language quality,
  - overall helpfulness to the asker,
  - and any other factors you think relevant.

"""

# Proofs criterion: same CSV framing and same submit_pick workflow; only
# the judgment criterion changes from "best answer" to "most rigorous
# proof" (mirrors rate.py / rate_codex.py). Neutral: no labels, no model
# identities, no hint that all candidates share one author.
_CRITERION_PROOFS = """\
There is a single `.csv` file in /work — find it (use list_dir if needed).
The CSV has columns `model, question, answer`. Each question is a math
problem asking for a rigorous proof of a stated claim; it appears in
several rows (one per candidate proof). The `model` column is the label
attached to that candidate.

Your job: for each distinct question, pick the best candidate proof based
on a holistic judgment of:
  - mathematical correctness: the argument actually establishes the
    stated claim, with valid logic and no false or circular steps,
  - rigor and completeness: precise use of definitions and quantifiers,
    every case handled, and no gaps, hand-waving, or unjustified leaps,
  - and any other factors you think relevant.

"""

RATING_PROMPT = (
    (_CRITERION_PROOFS if _TASK in ("proofs", "proofnet")
     else _CRITERION_DEFAULT)
    + """\
Make the call yourself by reading. Do NOT call any other model or web service.

Workflow:
  1. read_file the input CSV.
  2. For each distinct question (in the order they first appear in the
     CSV — call that question's index `qidx`, starting at 1), briefly
     note your reasoning about which candidate's answer is best.
  3. Submit each pick via:
         submit_pick(qidx=<int 1..N>, model_label=<string>)
     where `model_label` is the exact `model` value from the row whose
     answer you're picking (e.g. "gpt-4o" or "claude-opus-3"). The
     answer text itself is recorded server-side from the CSV — you do
     NOT need to copy it. Do NOT call write_file; the harness writes
     best_answers.csv from your submit_pick calls.
  4. Call submit_pick exactly ONCE per distinct question. After all
     questions have a pick, briefly summarize (input filename, number
     of distinct questions, number of picks recorded) and stop.

Don't do anything else after that.
"""
)


def _resolve_under_work(raw_path):
    p = Path(raw_path)
    if not p.is_absolute():
        p = WORK_DIR / p
    p = p.resolve()
    work_root = WORK_DIR.resolve()
    try:
        p.relative_to(work_root)
    except ValueError:
        return None, f"path {p} is outside {work_root}"
    return p, None


@register_tool("read_file")
class ReadFile(BaseTool):
    description = (
        "Read a UTF-8 text file from /work and return its full "
        "contents as a string."
    )
    parameters = [{
        "name": "path",
        "type": "string",
        "description": (
            "Filename relative to /work, or an absolute path under "
            "/work. Anything outside /work is rejected."
        ),
        "required": True,
    }]

    def call(self, params, **kwargs):
        path, err = _resolve_under_work(json5.loads(params)["path"])
        if err:
            return f"ERROR: {err}"
        if not path.exists():
            return f"ERROR: {path} does not exist"
        if path.is_dir():
            return f"ERROR: {path} is a directory; use list_dir"
        return path.read_text()


@register_tool("submit_pick")
class SubmitPick(BaseTool):
    description = (
        "Record your pick for ONE distinct question, identified by its "
        "1-based index in the order questions first appear in the input "
        "CSV. The harness looks up the candidate row whose `model` "
        "column equals `model_label` for that question and stores its "
        "answer verbatim. Call exactly once per distinct question. "
        "Returns 'ok' on success, an error string on failure (e.g. "
        "qidx out of range, model_label doesn't match any candidate "
        "for that question, or duplicate submission)."
    )
    parameters = [
        {
            "name": "qidx",
            "type": "integer",
            "description": (
                "1-based index of the distinct question in the input "
                "CSV (1..N where N = number of distinct questions)."
            ),
            "required": True,
        },
        {
            "name": "model_label",
            "type": "string",
            "description": (
                "The exact value of the `model` column from the row "
                "whose answer you want to pick. E.g. 'gpt-4o' or "
                "'claude-opus-3' — case-sensitive, copied verbatim "
                "from the CSV."
            ),
            "required": True,
        },
    ]

    def call(self, params, **kwargs):
        try:
            p = json5.loads(params) if isinstance(params, str) else params
        except Exception as e:  # noqa: BLE001
            return f"ERROR: could not parse args ({e!r}); call shape: " \
                   f"submit_pick(qidx=int, model_label=str)"
        try:
            qidx = int(p.get("qidx"))
        except Exception:
            return "ERROR: qidx must be an integer (1-based)"
        label = p.get("model_label")
        if not isinstance(label, str):
            return "ERROR: model_label must be a string"
        if qidx not in _QUESTION_INDEX:
            return (
                f"ERROR: qidx={qidx} out of range. "
                f"Valid: 1..{len(_QUESTION_INDEX)}."
            )
        cands = _QUESTION_INDEX[qidx]["candidates"]
        if label not in cands:
            return (
                f"ERROR: model_label={label!r} doesn't match any "
                f"candidate for qidx={qidx}. Valid labels for that "
                f"question: {sorted(cands)}."
            )
        if qidx in _PICKS:
            return (
                f"ERROR: qidx={qidx} already picked "
                f"({_PICKS[qidx]['model_label']!r}). One pick per "
                f"question."
            )
        _PICKS[qidx] = {"model_label": label, "answer": cands[label]}
        return f"ok: recorded pick for qidx={qidx} (model_label={label!r})"


@register_tool("list_dir")
class ListDir(BaseTool):
    description = "List filenames in a directory under /work."
    parameters = [{
        "name": "path",
        "type": "string",
        "description": (
            "Directory relative to /work, or an absolute path under "
            "/work. Defaults to /work itself."
        ),
        "required": False,
    }]

    def call(self, params, **kwargs):
        raw = json5.loads(params or "{}").get("path") or "/work"
        path, err = _resolve_under_work(raw)
        if err:
            return f"ERROR: {err}"
        if not path.is_dir():
            return f"ERROR: {path} is not a directory"
        return "\n".join(p.name for p in sorted(path.iterdir()))


@register_tool("run_python")
class RunPython(BaseTool):
    # Description is INTENTIONALLY task-neutral — we describe the
    # mechanics of the tool but not when to call it. This mirrors how
    # Bash is exposed to claude-bash and how the shell is exposed to
    # codex-bash: the model discovers the tool exists from the
    # function-calling schema and decides for itself whether (and how)
    # to use it. Do NOT add language like "use this to compute the
    # Frobenius norm" — that would bias the experiment.
    description = (
        "Execute a Python 3 script and return its stdout, stderr, and "
        "return code. The script runs in /work with a 60-second timeout. "
        "Available packages: numpy, pandas, sympy, json5, python-dateutil, "
        "tabulate, soundfile, plus the standard library. The script does "
        "NOT have network access for LLM/web APIs (attempts to call "
        "openai/anthropic/openrouter/curl/etc. are blocked at the tool "
        "layer)."
    )
    parameters = [{
        "name": "code",
        "type": "string",
        "description": (
            "Python 3 source code to execute. Use print() to surface "
            "anything you want returned (stdout is truncated to the "
            "last 4000 characters)."
        ),
        "required": True,
    }]

    def call(self, params, **kwargs):
        try:
            p = json5.loads(params) if isinstance(params, str) else params
        except Exception as e:  # noqa: BLE001
            return f"ERROR: could not parse args ({e!r}); call shape: " \
                   f"run_python(code=str)"
        code = p.get("code")
        if not isinstance(code, str) or not code.strip():
            return "ERROR: code must be a non-empty Python source string"

        code_lc = code.lower()
        for kw in _RUN_PYTHON_FORBIDDEN_KEYWORDS:
            if kw in code_lc:
                return (
                    f"ERROR: code contains the forbidden keyword "
                    f"{kw!r}. You may not call any external LLM or web "
                    f"service from inside run_python — pick the best "
                    f"answer yourself."
                )

        import subprocess
        import tempfile

        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=str(WORK_DIR),
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code)
                tmp = Path(f.name)

            try:
                proc = subprocess.run(
                    ["python3", str(tmp)],
                    cwd=str(WORK_DIR),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "stdout": "",
                    "stderr": "timeout after 60s",
                    "returncode": -1,
                    "timed_out": True,
                })

            return json.dumps({
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-1000:],
                "returncode": proc.returncode,
            }, default=str)
        finally:
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def _build_llm_cfg():
    api_key = (
        os.environ.get("JANEKD_QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or "EMPTY"
    )
    cfg = {
        "model": RATER_MODEL,
        "model_type": RATER_MODEL_TYPE,
        "api_key": api_key,
        "generate_cfg": {},
    }

    # Raise qwen-agent's input truncation budget (default 58k) so long
    # inputs — e.g. proofnet's 10 long proofs × K candidate proofs —
    # aren't silently truncated, which otherwise drops questions and
    # makes the rater fail to submit all picks. Respected at runtime by
    # qwen_agent.llm.base (generate_cfg.pop('max_input_tokens', ...)).
    _max_in = int(os.environ.get("JANEKD_QWEN_MAX_INPUT_TOKENS", "0") or 0)
    if _max_in > 0:
        cfg["generate_cfg"]["max_input_tokens"] = _max_in

    if RATER_MODEL_TYPE in ("oai", "openai"):
        server = os.environ.get("JANEKD_QWEN_MODEL_SERVER")
        if not server:
            sys.exit(
                "RATER_MODEL_TYPE=oai but JANEKD_QWEN_MODEL_SERVER is "
                "not set in the container env."
            )
        cfg["model_server"] = server

    if RATER_THINKING and RATER_MODEL_TYPE == "qwen_dashscope":
        cfg["generate_cfg"]["enable_thinking"] = True

    return cfg


def _pretty_print_chunk(chunk):
    if not chunk:
        return
    msg = chunk[-1]
    role = msg.get("role", "?")
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    fc = msg.get("function_call")

    if fc:
        name = fc.get("name") if isinstance(fc, dict) else "?"
        args = (fc.get("arguments", "") if isinstance(fc, dict) else "") or ""
        preview = args if len(args) < 200 else args[:197] + "..."
        print(f"  · {role}: function_call {name}({preview})", flush=True)
        return

    if role == "function":
        preview = (content or "")
        if len(preview) > 200:
            preview = preview[:197] + "..."
        print(f"  · function_result[{msg.get('name')}]: {preview}",
              flush=True)
        return

    if reasoning:
        preview = reasoning if len(reasoning) < 200 else reasoning[:197] + "..."
        print(f"  · reasoning: {preview}", flush=True)

    if content:
        text = (content or "").strip()
        if text:
            preview = text if len(text) < 400 else text[:397] + "..."
            print(preview, flush=True)


def _load_question_index(csv_path: Path) -> None:
    _QUESTION_INDEX.clear()
    _PICKS.clear()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"]
            label = row["model"]
            ans = row["answer"]
            qidx = None
            for i, entry in _QUESTION_INDEX.items():
                if entry["question"] == q:
                    qidx = i
                    break
            if qidx is None:
                qidx = len(_QUESTION_INDEX) + 1
                _QUESTION_INDEX[qidx] = {"question": q, "candidates": {}}
            _QUESTION_INDEX[qidx]["candidates"][label] = ans


def _write_best_answers(out_path: Path) -> tuple[int, int]:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        for qidx, entry in _QUESTION_INDEX.items():
            pick = _PICKS.get(qidx)
            if pick is None:
                continue
            writer.writerow([entry["question"], pick["answer"]])
    return len(_PICKS), len(_QUESTION_INDEX)


def main():
    csvs = sorted(p for p in WORK_DIR.glob("*.csv")
                  if p.name != "best_answers.csv")
    if len(csvs) != 1:
        sys.exit(f"Expected exactly 1 input CSV in {WORK_DIR}, got "
                 f"{len(csvs)}: {[c.name for c in csvs]}")
    _load_question_index(csvs[0])
    print(f"Loaded {len(_QUESTION_INDEX)} distinct questions × "
          f"{len(next(iter(_QUESTION_INDEX.values()))['candidates'])} "
          f"candidates from {csvs[0].name}", flush=True)

    if not (os.environ.get("JANEKD_QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")):
        sys.exit(
            "Neither JANEKD_QWEN_API_KEY nor DASHSCOPE_API_KEY is set "
            "in the container env — run.py should have passed one with "
            "-e (mapped from the host's DASHSCOPE_API_KEY for the "
            "dashscope provider, or OPENROUTER_API_KEY for openrouter)."
        )

    llm_cfg = _build_llm_cfg()
    # Conditionally expose run_python — see _ENABLE_COMPUTE / the
    # SYSTEM_PROMPT branch above. With compute enabled the toolbelt
    # mirrors claude-bash / codex-bash (a generic compute tool the
    # model is free to use or ignore).
    function_list = ["read_file", "list_dir", "submit_pick"]
    if _ENABLE_COMPUTE:
        function_list.append("run_python")

    bot = Assistant(
        llm=llm_cfg,
        system_message=SYSTEM_PROMPT,
        function_list=function_list,
    )

    log_path = WORK_DIR / RATER_LOG_NAME
    print(
        f"=== Picking best answers in {csvs[0].name} "
        f"(rater={RATER_MODEL}, provider={JANEKD_QWEN_PROVIDER}, "
        f"type={RATER_MODEL_TYPE}, thinking={RATER_THINKING}) ===\n",
        flush=True,
    )

    safe_llm_cfg = {k: v for k, v in llm_cfg.items() if k != "api_key"}

    with log_path.open("w") as logf:
        logf.write(json.dumps({
            "msg_type": "Request",
            "rater_backend": "qwen",
            "rater_model": RATER_MODEL,
            "rater_model_type": RATER_MODEL_TYPE,
            "rater_provider": JANEKD_QWEN_PROVIDER,
            "rater_thinking": RATER_THINKING,
            "rater_reasoning_effort": RATER_REASONING_EFFORT,
            "compute_enabled": _ENABLE_COMPUTE,
            "task": _TASK,
            "function_list": function_list,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": RATING_PROMPT,
            "input_csv": csvs[0].name,
            "llm_cfg": safe_llm_cfg,
        }, default=str) + "\n")
        logf.flush()

        messages = [{"role": "user", "content": RATING_PROMPT}]
        chunk_count = 0
        last_chunk = None
        for chunk in bot.run(messages=messages):
            chunk_count += 1
            last_chunk = chunk
            _pretty_print_chunk(chunk)

        if last_chunk is not None:
            logf.write(json.dumps({
                "msg_type": "Final",
                "chunk_count": chunk_count,
                "messages": last_chunk,
            }, default=str) + "\n")

    out = WORK_DIR / "best_answers.csv"
    n_picks, n_qs = _write_best_answers(out)
    print(f"Wrote {out.name}: {n_picks} pick(s) for "
          f"{n_qs} distinct question(s)", flush=True)
    if n_picks < n_qs:
        missing = [i for i in _QUESTION_INDEX if i not in _PICKS]
        sys.exit(f"Rater finished but only {n_picks}/{n_qs} questions "
                 f"have picks (missing qidx: {missing}).")


if __name__ == "__main__":
    main()
