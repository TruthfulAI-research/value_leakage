"""Repo-local Python startup customization."""

from __future__ import annotations

import os
from pathlib import Path


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError:
        return False
    return True


if "MPLCONFIGDIR" not in os.environ:
    candidates = [
        Path(__file__).resolve().parent / ".cache" / "matplotlib",
        Path("/private/tmp/giraffes-matplotlib-cache"),
    ]
    for mpl_config_dir in candidates:
        if _is_writable_dir(mpl_config_dir):
            os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
            break
