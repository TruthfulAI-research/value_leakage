
from pathlib import Path

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "alpaca_questions.jsonl"

# 10 questions seems like a reasonable number. 
# Too many -> rater will start doing weird things to avoid reading all.
N_QUESTIONS = 10

# Model labels. This will be shown to the rater.
MODEL_LABELS = [
    "gpt-6",
    "gpt-6-high",
    "claude-opus-5",
    "claude-opus-5-high",
    "qwen-4",
    "qwen-4-high",
]

# Source models. This will be used to sample the answers.
SOURCE_MODELS = [
    "claude-haiku-4.5",
]

# The model that will run in Claude Code
RATER_MODEL = "claude-opus-4-7"

# How many seeds to run. Each seed means rating select best answers to 
# randomly sampled N_QUESTIONS. N_SEEDS = 100 seems to be giving quite solid results,
# but for the final results we should have like 500 or so.
N_SEEDS = 100

# --- Sweep knobs ----------------------------------------------------------
# Global salt for question sampling: change it to reshuffle which questions
# each trial seed draws (and invalidate the answer cache for those draws).
SAMPLE_SEED = 111332
# First trial seed; the sweep covers [START_SEED, START_SEED + N_SEEDS).
# Bump it to add fresh trials without disturbing existing rated runs.
START_SEED = 222333

# How many rater containers run concurrently. Each is independent the shared 
# resource is the API quota and RAM. 
# With 18GB RAM, 20 works well and some more probably works too.
RATING_PARALLELISM = 20

# ------------------------------------------------------
# NO NEED TO CHANGE ANYTHING BELOW THIS EVER HOPEFULLY
# ------------------------------------------------------
DOCKER_IMAGE = "eval-rater:latest"

# How often analysis.py is run
LIVE_PREVIEW_EVERY = 5

DATA_DIR = HERE / "data"
RUNS_DIR = DATA_DIR / "runs"
LATEST_LINK = RUNS_DIR / "latest"

# Append-only answer cache. Keyed by (display_name, api_id, sample_idx,
# question). sample_idx > 0 only when a single source has to fill multiple
# slots per question (M < K in the round-robin slot plan); each idx is an
# independent temperature-1 draw from the same source.
CACHE_DIR = DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "answers.csv"


def new_run_dir():
    """Make a fresh timestamped run directory path (not yet on disk)."""
    from datetime import datetime
    return RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def latest_run_dir(*, required=True):
    """Return the most recent run dir, or exit / None if there are none."""
    if RUNS_DIR.exists():
        dirs = sorted(
            d for d in RUNS_DIR.iterdir()
            if d.is_dir() and not d.is_symlink()
        )
        if dirs:
            return dirs[-1]
    if required:
        raise SystemExit(
            f"No runs found under {RUNS_DIR}. Run `python run.py` to make one."
        )
    return None


def bases_dir(run_dir):
    return run_dir / "bases"


def replicates_dir(run_dir):
    return run_dir / "replicates"


def rated_dir(run_dir):
    """Per-seed rated subdirs live under here; each contains the FULL
    replicate CSV (with real_model column intact) + best_answers.csv
    produced by the rater. The rater itself sees a stripped version
    (without real_model) — see run._rate_one."""
    return run_dir / "rated"


def analysis_dir(run_dir):
    return run_dir / "analysis"
