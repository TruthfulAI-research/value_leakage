"""Small JSONL cache for direct judge calls.

The cache layout is one directory per judge config, sharded by the first
character of the rendered prompt hash. Each line stores the hash of the exact
rendered prompt sent to the judge plus the small judge output fields the caller
needs. Prompt text and judge reasoning are intentionally not stored.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_PATH_LOCKS: dict[Path, threading.Lock] = {}
_PATH_LOCKS_LOCK = threading.Lock()


def judge_config_hash(judge_prompt: str, judge_config: dict[str, Any]) -> str:
    """Hash everything that affects the judge's output cache file."""
    payload = {
        "model": judge_config["model"],
        "max_tokens": judge_config["max_tokens"],
        "temperature": judge_config["temperature"],
        "reasoning_effort": judge_config.get("reasoning_effort"),
        "prompt": judge_prompt,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def prompt_hash(rendered_prompt: str, n: int = 64) -> str:
    """Hash the exact prompt string sent to the judge."""
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()[:n]


def judge_cache_path(
    cache_dir: str | Path,
    judge_prompt: str,
    judge_config: dict[str, Any],
) -> Path:
    """Legacy monolithic cache file path for one judge config."""
    return (
        Path(cache_dir).expanduser().resolve()
        / f"{judge_config_hash(judge_prompt, judge_config)}.jsonl"
    )


def judge_cache_dir(
    cache_dir: str | Path,
    judge_prompt: str,
    judge_config: dict[str, Any],
) -> Path:
    """Sharded cache directory path for one judge config."""
    return (
        Path(cache_dir).expanduser().resolve()
        / judge_config_hash(judge_prompt, judge_config)
    )


def _shard_prefix(prompt_hash_value: str) -> str:
    if not prompt_hash_value:
        return "_"
    return prompt_hash_value[0].lower()


def _path_lock(path: Path) -> threading.Lock:
    path = path.expanduser().resolve()
    with _PATH_LOCKS_LOCK:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[path] = lock
        return lock


@contextmanager
def _locked_append(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _path_lock(path)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield f
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_jsonl_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load prompt_hash -> row. Duplicate hashes are resolved last-line-wins."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            h = row.get("prompt_hash")
            if isinstance(h, str):
                out[h] = row
    return out


def load_sharded_jsonl_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load all JSONL shard files from a cache directory."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for shard_path in sorted(path.glob("*.jsonl")):
        out.update(load_jsonl_cache(shard_path))
    return out


def append_jsonl_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path).expanduser().resolve()
    with _locked_append(path) as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


class JsonlJudgeCache:
    """Thread-safe view of one config-scoped sharded judge cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        judge_prompt: str,
        judge_config: dict[str, Any],
    ) -> None:
        self.legacy_path = judge_cache_path(cache_dir, judge_prompt, judge_config)
        self.shard_dir = judge_cache_dir(cache_dir, judge_prompt, judge_config)
        self.path = self.shard_dir
        self.entries = load_jsonl_cache(self.legacy_path)
        self.entries.update(load_sharded_jsonl_cache(self.shard_dir))
        self._entries_lock = threading.Lock()

    def key(self, rendered_prompt: str) -> str:
        return prompt_hash(rendered_prompt)

    def shard_path(self, prompt_hash_value: str) -> Path:
        return self.shard_dir / f"{_shard_prefix(prompt_hash_value)}.jsonl"

    def get(self, rendered_prompt: str) -> dict[str, Any] | None:
        key = self.key(rendered_prompt)
        with self._entries_lock:
            return self.entries.get(key)

    def append(self, rendered_prompt: str, data: dict[str, Any]) -> dict[str, Any]:
        row = {**data, "prompt_hash": self.key(rendered_prompt)}
        append_jsonl_rows(self.shard_path(row["prompt_hash"]), [row])
        with self._entries_lock:
            self.entries[row["prompt_hash"]] = row
        return row

    def append_many(self, rows: list[dict[str, Any]]) -> None:
        rows_by_path: dict[Path, list[dict[str, Any]]] = {}
        for row in rows:
            h = row.get("prompt_hash")
            path = self.shard_path(h if isinstance(h, str) else "")
            rows_by_path.setdefault(path, []).append(row)
        for path, path_rows in rows_by_path.items():
            append_jsonl_rows(path, path_rows)
        with self._entries_lock:
            for row in rows:
                h = row.get("prompt_hash")
                if isinstance(h, str):
                    self.entries[h] = row
