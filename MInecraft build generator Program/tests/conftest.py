"""Pytest bootstrap: make the application modules importable from any CWD.

Without this, ``pytest "<program dir>/tests"`` fails with
``ModuleNotFoundError: No module named 'dataset'`` when the repo root is the
working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
