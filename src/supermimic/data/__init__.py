"""Data locations, resolved from the editable source tree rather than a home dir."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(
  os.environ.get("SUPERMIMIC_ROOT") or Path(__file__).resolve().parents[3]
)
MOTIONS_DIR = REPO_ROOT / "data" / "motions"
LAFAN1_CSV_DIR = REPO_ROOT / "data" / "lafan1_g1"
RUNS_DIR = REPO_ROOT / "runs"
