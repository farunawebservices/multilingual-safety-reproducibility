#!/usr/bin/env python3
"""Mint a run ID and create its results/ and logs/ directories.

Runs no model. This exists so that a run's identity, config hash, git state, and
split hash are fixed *before* any inference starts, rather than reconstructed
afterwards from whatever happens to be on disk.

Usage
-----
    python scripts/new_run.py --config configs/experiment.example.yaml --dry-run
    python scripts/new_run.py --config configs/my_experiment.yaml --split dev
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.run_context import _cli

if __name__ == "__main__":
    raise SystemExit(_cli())
