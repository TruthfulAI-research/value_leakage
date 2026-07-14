"""Stage 1: produce K answer slots per question from M source models.

Identical logic to janekd.rate_llm_answers.generate (round-robin slot
plan, append-only on-disk cache, per-source batched fetch), with one
behavioural difference at the I/O boundary:

  - load_questions() reads `question` from each JSONL line as before,
    but tolerates extra fields (here: `ground_truth`, used by
    analyze.py to compute per-pick Frobenius errors). The Alpaca pool
    has only `question` per line; the Frobenius pool has `question` +
    `ground_truth`. The two formats are interchangeable from the
    cache's point of view because the cache is keyed on the `question`
    text alone.

Cache layout: see config.CACHE_FILE — a per-experiment cache dir so the
two tasks can't accidentally share answer rows that happen to have
matching keys (in practice they never can, because the question strings
are disjoint, but the dir split is a hygienic guard).

Wipe `data/cache/answers.csv` to force fresh API calls — e.g. after
fiddling with temperature or max_tokens in shared/models.py.
"""

import csv
import hashlib
import json
import random
import threading

import pandas as pd
from tqdm import tqdm

from shared.models import MODELS
from shared.runner import _create_sender, _run_prompts

from .config import (
    CACHE_FILE,
    MODEL_LABELS,
    N_QUESTIONS,
    QUESTIONS_FILE,
    SAMPLE_SEED,
    SOURCE_MODELS,
)

CACHE_COLUMNS = ["model", "model_api", "sample_idx", "question", "answer"]

# Frobenius prompts ask for free-form math; no model in our suite
# reliably trips a content filter on these, so the banned-question set
# is empty (kept for parity with the Alpaca pipeline).
_BANNED_QUESTIONS = frozenset()


def load_questions(path):
    """Load JSONL questions (one record per line).

    Each line must be a JSON object with at least a `question` key.
    Extra keys (e.g. `ground_truth`) are ignored here — they're
    consumed downstream by analyze.py.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["question"])
    return out


def _cache_key(display, api, sample_idx, question):
    return (display, api, int(sample_idx), question)


def _load_cache(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"sample_idx": int})
    missing = [c for c in CACHE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Cache file {path} is missing columns {missing}. "
            f"Either fix the file or delete it to start over."
        )
    out = {}
    for row in df.itertuples(index=False):
        out[_cache_key(row.model, row.model_api, row.sample_idx,
                       row.question)] = row.answer
    return out


def _append_cache(path, label, api, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_COLUMNS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({
                "model": label,
                "model_api": api,
                "sample_idx": int(r["sample_idx"]),
                "question": r["question"],
                "answer": r["answer"],
            })


def question_seed(seed):
    return f"questions:{SAMPLE_SEED}:{int(seed)}"


def select_questions(seed):
    pool = [q for q in load_questions(QUESTIONS_FILE) if q not in _BANNED_QUESTIONS]
    if N_QUESTIONS > len(pool):
        raise ValueError(
            f"Asked for {N_QUESTIONS} questions, only {len(pool)} "
            f"available in {QUESTIONS_FILE} after excluding "
            f"{len(_BANNED_QUESTIONS)} banned."
        )
    return random.Random(seed).sample(pool, N_QUESTIONS)


def _iter_sources():
    for key in SOURCE_MODELS:
        if key not in MODELS:
            raise SystemExit(
                f"config.SOURCE_MODELS references {key!r} which is not "
                f"in shared.models.MODELS. Available: {sorted(MODELS)}"
            )
        cfg = MODELS[key]
        yield key, cfg, cfg["display_name"], cfg["model"]


def _question_offset(question, m):
    if m <= 0:
        raise ValueError("source count must be positive")
    h = hashlib.sha256(question.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % m


def slot_plan_for_question(question, source_keys=None, k=None):
    if source_keys is None:
        source_keys = SOURCE_MODELS
    if k is None:
        k = len(MODEL_LABELS)
    m = len(source_keys)
    offset = _question_offset(question, m)
    plan = []
    counter = {s: 0 for s in source_keys}
    for i in range(k):
        s = source_keys[(offset + i) % m]
        plan.append((s, counter[s]))
        counter[s] += 1
    return plan


def _per_source_needs(questions, source_keys=None, k=None):
    if source_keys is None:
        source_keys = SOURCE_MODELS
    if k is None:
        k = len(MODEL_LABELS)
    needs = {s: {} for s in source_keys}
    for q in questions:
        for s, _idx in slot_plan_for_question(q, source_keys, k):
            needs[s][q] = needs[s].get(q, 0) + 1
    return needs


def warm_cache(questions):
    if not questions:
        print("warm_cache: nothing to do (empty question set).")
        return

    cache = _load_cache(CACHE_FILE)
    if cache:
        print(f"Loaded {len(cache)} cached answers from {CACHE_FILE}")

    needs = _per_source_needs(questions)

    total_hits = 0
    total_misses = 0
    total_new = 0

    for _key, cfg, label, api in _iter_sources():
        source_needs = needs[_key]
        if not source_needs:
            print(f"  {label}: 0 needed for this batch")
            continue

        to_fetch = []
        next_sample_idx = {}
        cached_for_source = 0
        for q, n_needed in source_needs.items():
            n_cached = 0
            while _cache_key(label, api, n_cached, q) in cache:
                n_cached += 1
            cached_for_source += min(n_cached, n_needed)
            next_sample_idx[q] = n_cached
            if n_cached < n_needed:
                to_fetch.extend([q] * (n_needed - n_cached))

        total_hits += cached_for_source
        total_misses += len(to_fetch)

        if not to_fetch:
            print(f"  {label}: all {cached_for_source} cached")
            continue
        print(f"  {label}: {cached_for_source} cached, "
              f"{len(to_fetch)} to fetch ...", flush=True)

        sender = _create_sender(cfg)
        progress = tqdm(total=len(to_fetch), desc=label)
        semaphore = threading.Semaphore(cfg["max_concurrent"])
        try:
            results = _run_prompts(
                sender, cfg["max_concurrent"], to_fetch,
                progress=progress, semaphore=semaphore,
            )
        finally:
            progress.close()

        rows = []
        idx_for = dict(next_sample_idx)
        for r in results:
            q = r["prompt"]
            idx = idx_for[q]
            rows.append({
                "question": q,
                "answer": r["answer"],
                "sample_idx": idx,
            })
            idx_for[q] = idx + 1

        _append_cache(CACHE_FILE, label, api, rows)
        total_new += len(rows)
        print(f"    flushed {len(rows)} answer(s) to cache")

    print(f"\nCache warming done: {total_hits} hit, {total_misses} miss, "
          f"{total_new} new row(s) written.")


def generate_to(base_csv_path, selected_questions):
    base_csv_path.parent.mkdir(parents=True, exist_ok=True)
    warm_cache(selected_questions)
    cache = _load_cache(CACHE_FILE)

    k = len(MODEL_LABELS)
    rows = []
    for q in selected_questions:
        plan = slot_plan_for_question(q)
        for source_key, sample_idx in plan:
            cfg = MODELS[source_key]
            label, api = cfg["display_name"], cfg["model"]
            ck = _cache_key(label, api, sample_idx, q)
            if ck not in cache:
                raise RuntimeError(
                    f"missing cache entry after warm_cache: "
                    f"source={source_key!r} sample_idx={sample_idx} "
                    f"question={q!r}"
                )
            rows.append({
                "real_model": source_key,
                "sample_idx": int(sample_idx),
                "question": q,
                "answer": cache[ck],
            })

    with open(base_csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["real_model", "sample_idx", "question", "answer"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {base_csv_path}  "
          f"({N_QUESTIONS} questions × {k} slot(s)  from "
          f"{len(SOURCE_MODELS)} source model(s))")
