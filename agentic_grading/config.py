"""Per-experiment knobs for the agentic answer-grading bias sweep.

This module drives a blinded best-of-K rating experiment used to probe
label bias in *agentic* graders (Claude Code, Codex, Qwen-Agent). The
design is the same for every task:

  - K = len(MODEL_LABELS) candidate answer slots per question,
  - all candidates produced by SOURCE_MODELS (held constant per
    experiment — claude-haiku-4.5), so the ONLY thing that varies
    across slots is the (fake) `model` label the grader sees,
  - labels are permuted randomly across slots per (seed, question),
  - the grader sees the `model` column but never `real_model`.

Two tasks are supported (selected with JANEKD_TASK / --task):

  - `alpaca`   : subjective helpfulness grading of Alpaca instructions
                 (question pool: alpaca_questions.jsonl).
  - `proofnet` : rigor grading of undergraduate proofs whose statements
                 come from the ProofNet benchmark, with candidate proofs
                 written by claude-haiku-4.5
                 (question pool: proof_questions_proofnet.jsonl).

Neither task has an objective ground-truth scorer, so analyze.py reports
per-label / per-source pick-rate + CI + significance vs 1/K plus CoT.

Grader backend selection: JANEKD_RATER_BACKEND in {claude, codex, qwen};
the qwen sub-provider knob is JANEKD_QWEN_PROVIDER in {dashscope,
openrouter}. Reasoning effort defaults to high.

Chain-of-thought capture:
  - claude: MAX_THINKING_TOKENS (a positive budget turns on extended
    thinking so ThinkingBlocks carry real CoT). Opt-in via the host env
    JANEKD_CLAUDE_THINKING_TOKENS (forwarded by run.py into the claude
    container).
  - codex : model_reasoning_summary (JANEKD_RATER_REASONING_SUMMARY,
    default "auto") makes `codex exec --json` emit reasoning summaries.
  - qwen  : reasoning_content is captured when the provider returns it
    (best-effort; provider-dependent).
"""

import os
from pathlib import Path

HERE = Path(__file__).parent

# Global fallback questions file. A preset overrides this with its own
# DEFAULT_QUESTIONS_FILE; the runtime override JANEKD_QUESTIONS_FILE
# still wins (used by the smoke stage). QUESTIONS_FILE is resolved AFTER
# the preset is selected — see below the EXPERIMENTS block.
_GLOBAL_DEFAULT_QUESTIONS_FILE = "alpaca_questions.jsonl"

# ---------------------------------------------------------------------------
# Experiment presets. One per task; both share the same blinded design.
# ---------------------------------------------------------------------------
EXPERIMENT = os.environ.get("JANEKD_EXPERIMENT", "alpaca")

# Shared label set (6 labels = 3 labs × {default, high-effort}). The
# labels are fictional identifiers shown to the grader; they need NOT be
# real models in shared.models — only SOURCE_MODELS are validated there.
_LAB_X_EFFORT_LABELS = [
    "gpt-6",
    "gpt-6-high",
    "claude-opus-5",
    "claude-opus-5-high",
    "qwen-4",
    "qwen-4-high",
]

EXPERIMENTS = {
    # Subjective helpfulness grading of Alpaca instructions. Candidate
    # answers are all authored by claude-haiku-4.5; the grader picks the
    # "best" answer per question. No objective ground truth.
    "alpaca": {
        "MODEL_LABELS": list(_LAB_X_EFFORT_LABELS),
        "SOURCE_MODELS": ["claude-haiku-4.5"],
        "RATER_MODEL": "claude-opus-4-7",
        "HAS_GROUND_TRUTH": False,
        "DEFAULT_QUESTIONS_FILE": "alpaca_questions.jsonl",
        "CACHE_DIR": "cache_alpaca",
    },

    # Rigor grading of undergraduate-level proofs. Statements are drawn
    # from the ProofNet benchmark; candidate proofs are all authored by
    # claude-haiku-4.5. Same 6 labels + single source model as `alpaca`
    # so the per-label pick-rate plot is directly comparable. No
    # objective ground truth (a proof has no cheap closed-form score).
    "proofnet": {
        "MODEL_LABELS": list(_LAB_X_EFFORT_LABELS),
        "SOURCE_MODELS": ["claude-haiku-4.5"],
        "RATER_MODEL": "claude-opus-4-7",
        "HAS_GROUND_TRUTH": False,
        "DEFAULT_QUESTIONS_FILE": "proof_questions_proofnet.jsonl",
        "CACHE_DIR": "cache_proofnet",
    },
}

if EXPERIMENT not in EXPERIMENTS:
    raise SystemExit(
        f"config.EXPERIMENT={EXPERIMENT!r} is not in EXPERIMENTS "
        f"({sorted(EXPERIMENTS)})."
    )

_preset = EXPERIMENTS[EXPERIMENT]
MODEL_LABELS = _preset["MODEL_LABELS"]
SOURCE_MODELS = _preset["SOURCE_MODELS"]

# Whether this experiment has an objective ground-truth scorer. Both
# tasks here are subjective (helpfulness / rigor) so this is False;
# analyze.py keys its no-GT path off this flag (an explicit
# JANEKD_NO_GROUND_TRUTH=1 also forces it).
HAS_GROUND_TRUTH = bool(_preset.get("HAS_GROUND_TRUTH", False))

# Resolve the question pool now that the preset is known: runtime
# JANEKD_QUESTIONS_FILE wins (smoke stage), else the preset's
# DEFAULT_QUESTIONS_FILE, else the global default.
QUESTIONS_FILE = HERE / os.environ.get(
    "JANEKD_QUESTIONS_FILE",
    _preset.get("DEFAULT_QUESTIONS_FILE", _GLOBAL_DEFAULT_QUESTIONS_FILE),
)

# ---------------------------------------------------------------------------
# Grader backend selection.
# ---------------------------------------------------------------------------
RATER_BACKEND = os.environ.get("JANEKD_RATER_BACKEND", "claude").strip().lower()
_VALID_BACKENDS = ("claude", "codex", "qwen")
if RATER_BACKEND not in _VALID_BACKENDS:
    raise SystemExit(
        f"JANEKD_RATER_BACKEND={RATER_BACKEND!r} must be one of "
        f"{_VALID_BACKENDS}."
    )

QWEN_PROVIDER = os.environ.get(
    "JANEKD_QWEN_PROVIDER", "openrouter",
).strip().lower()
_VALID_QWEN_PROVIDERS = ("dashscope", "openrouter")
if RATER_BACKEND == "qwen" and QWEN_PROVIDER not in _VALID_QWEN_PROVIDERS:
    raise SystemExit(
        f"JANEKD_QWEN_PROVIDER={QWEN_PROVIDER!r} must be one of "
        f"{_VALID_QWEN_PROVIDERS}."
    )

_BACKEND_DEFAULT_MODEL = {
    "claude": "claude-opus-4-7",
    "codex": "gpt-5.4",
    "qwen": {
        "dashscope": "qwen-max-latest",
        "openrouter": "qwen/qwen3.6-max-preview",
    },
}


def _default_model_for(backend, qwen_provider):
    val = _BACKEND_DEFAULT_MODEL[backend]
    if isinstance(val, dict):
        return val[qwen_provider]
    return val


_preset_rater = _preset.get(
    "RATER_MODEL", _default_model_for(RATER_BACKEND, QWEN_PROVIDER),
)


def _preset_rater_matches_backend(name, backend):
    if backend == "claude":
        return name.startswith("claude-")
    if backend == "codex":
        return name.startswith("gpt-") or name.startswith("o")
    if backend == "qwen":
        return (name.startswith("qwen") or name.startswith("qwq")
                or name.startswith("qwen/") or name.startswith("qwq/"))
    return False


if not _preset_rater_matches_backend(_preset_rater, RATER_BACKEND):
    _preset_rater = _default_model_for(RATER_BACKEND, QWEN_PROVIDER)
RATER_MODEL = os.environ.get("JANEKD_RATER_MODEL", _preset_rater)

RATER_REASONING_EFFORT = os.environ.get(
    "JANEKD_RATER_REASONING_EFFORT", "high",
)
# Codex reasoning-summary verbosity. Codex exposes reasoning *summaries*
# (not raw CoT); "auto"/"concise"/"detailed" make `codex exec --json` emit
# `reasoning` items we can capture, "none" suppresses them. Forwarded into
# the codex container as RATER_REASONING_SUMMARY (see run.py).
RATER_REASONING_SUMMARY = os.environ.get(
    "JANEKD_RATER_REASONING_SUMMARY", "detailed",
)

# ---------------------------------------------------------------------------
# Sweep knobs (shared across experiments).
# ---------------------------------------------------------------------------
N_QUESTIONS = 10
N_SEEDS = int(os.environ.get("JANEKD_N_SEEDS", 250))
SAMPLE_SEED = 111332
START_SEED = 222333
RATING_PARALLELISM = int(os.environ.get("JANEKD_RATING_PARALLELISM", "20"))

# ---------------------------------------------------------------------------
# Image / file plumbing. One docker image tag per backend (the rate
# scripts are bind-mounted at runtime, so the image is rate-agnostic).
# ---------------------------------------------------------------------------
DOCKER_IMAGE_BY_BACKEND = {
    "claude": "eval-rater:latest",
    "codex": "eval-rater-codex:latest",
    "qwen": "eval-rater-qwen:latest",
}
DOCKER_IMAGE = DOCKER_IMAGE_BY_BACKEND[RATER_BACKEND]
DOCKERFILE_BY_BACKEND = {
    "claude": "Dockerfile",
    "codex": "Dockerfile.codex",
    "qwen": "Dockerfile.qwen",
}
DOCKERFILE = DOCKERFILE_BY_BACKEND[RATER_BACKEND]
RATE_SCRIPT_BY_BACKEND = {
    "claude": "rate.py",
    "codex": "rate_codex.py",
    "qwen": "rate_qwen.py",
}
RATE_SCRIPT = RATE_SCRIPT_BY_BACKEND[RATER_BACKEND]
RATER_API_KEY_ENV_BY_BACKEND = {
    "claude": "ANTHROPIC_API_KEY",
    "codex": "CODEX_API_KEY",
    "qwen": "JANEKD_QWEN_API_KEY",
}
RATER_API_KEY_ENV = RATER_API_KEY_ENV_BY_BACKEND[RATER_BACKEND]


def _qwen_host_api_key_env(provider):
    return {
        "dashscope": "DASHSCOPE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }[provider]


RATER_HOST_API_KEY_ENV_BY_BACKEND = {
    "claude": "ANTHROPIC_API_KEY",
    "codex": "OPENAI_API_KEY",
    "qwen": (_qwen_host_api_key_env(QWEN_PROVIDER)
             if RATER_BACKEND == "qwen" else "DASHSCOPE_API_KEY"),
}
RATER_HOST_API_KEY_ENV = RATER_HOST_API_KEY_ENV_BY_BACKEND[RATER_BACKEND]


def _qwen_extra_container_env():
    # qwen-agent truncates the model input to DEFAULT_MAX_INPUT_TOKENS
    # (58k in 0.0.34). Proofnet inputs (10 long proofs × K candidates)
    # blow past that, so qwen would drop questions and fail to submit
    # all picks. Raise the budget (rate_qwen forwards it into the llm
    # generate_cfg as max_input_tokens) so long-proof trials fit.
    max_in = os.environ.get("JANEKD_QWEN_MAX_INPUT_TOKENS", "200000")
    base = [("JANEKD_QWEN_MAX_INPUT_TOKENS", max_in)]
    if QWEN_PROVIDER == "openrouter":
        return base + [
            ("JANEKD_QWEN_PROVIDER", "openrouter"),
            ("RATER_MODEL_TYPE", "oai"),
            ("JANEKD_QWEN_MODEL_SERVER", "https://openrouter.ai/api/v1"),
        ]
    return base + [
        ("JANEKD_QWEN_PROVIDER", "dashscope"),
        ("RATER_MODEL_TYPE", "qwen_dashscope"),
    ]


def _claude_extra_container_env():
    # Claude Code honours MAX_THINKING_TOKENS: a positive budget turns on
    # extended thinking, so AssistantMessage ThinkingBlocks carry real CoT
    # text (otherwise they arrive empty). Opt-in via the host env var.
    tk = os.environ.get("JANEKD_CLAUDE_THINKING_TOKENS")
    return [("MAX_THINKING_TOKENS", tk)] if tk else []


RATER_EXTRA_CONTAINER_ENV = (
    _qwen_extra_container_env() if RATER_BACKEND == "qwen"
    else _claude_extra_container_env() if RATER_BACKEND == "claude"
    else []
)

LIVE_PREVIEW_EVERY = 5

DATA_DIR = HERE / "data"
RUNS_DIR = DATA_DIR / "runs"
LATEST_LINK = RUNS_DIR / "latest"

# Per-experiment answer cache so the two tasks never mingle rows (their
# question strings are disjoint anyway; the dir split is a hygienic
# guard). Same key shape as the generate._load_cache / _append_cache
# helpers expect.
CACHE_DIR = DATA_DIR / _preset.get("CACHE_DIR", "cache")
CACHE_FILE = CACHE_DIR / "answers.csv"


def new_run_dir():
    """Make a fresh timestamped run directory path (not yet on disk)."""
    from datetime import datetime
    return RUNS_DIR / (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
        f"{EXPERIMENT}_{RATER_BACKEND}"
    )


def latest_run_dir(*, required=True):
    if RUNS_DIR.exists():
        dirs = sorted(
            d for d in RUNS_DIR.iterdir()
            if d.is_dir() and not d.is_symlink()
        )
        if dirs:
            return dirs[-1]
    if required:
        raise SystemExit(
            f"No runs found under {RUNS_DIR}. "
            f"Run `python -m agentic_grading.run` to make one."
        )
    return None


def bases_dir(run_dir):
    return run_dir / "bases"


def replicates_dir(run_dir):
    return run_dir / "replicates"


def rated_dir(run_dir):
    return run_dir / "rated"


def analysis_dir(run_dir):
    return run_dir / "analysis"
