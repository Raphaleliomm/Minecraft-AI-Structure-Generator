"""Shared constants for GUI, training and generation.

Grid sizes used to be defined three times (train.py, generate.py, app/main_app.py)
which allowed the allowed sizes to drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from app.paths import RUNS_DIR

# model size key -> grid dimensions (must stay in sync with the GUI labels)
ALLOWED_GRID_SIZES: Dict[str, Tuple[int, int, int]] = {
    "16": (16, 16, 16),
    "32": (32, 32, 32),
    "48": (48, 48, 48),
    "64": (64, 64, 64),
    "96": (96, 96, 96),
    "128": (128, 128, 128),
    "256": (256, 256, 256),
}

# Sizes above 64³ are marked as experimental in the UI
_EXPERIMENTAL_SIZES = ("96", "128", "256")
_EXPERIMENTAL_SUFFIX = "（experimentell）"

# Label -> dimensions, in the order shown in the GUI dropdowns
GRID_SIZE_MAP: Dict[str, Tuple[int, int, int]] = {}
for _size, _dims in ALLOWED_GRID_SIZES.items():
    _suffix = _EXPERIMENTAL_SUFFIX if _size in _EXPERIMENTAL_SIZES else ""
    GRID_SIZE_MAP[f"{_dims[0]}×{_dims[1]}×{_dims[2]}{_suffix}"] = _dims

GRID_SIZE_OPTIONS = list(GRID_SIZE_MAP)

# Default label (16³)
DEFAULT_GRID_SIZE_LABEL = GRID_SIZE_OPTIONS[0]
DEFAULT_GRID_SIZE = GRID_SIZE_MAP[DEFAULT_GRID_SIZE_LABEL]


def grid_size_for_label(label: str) -> Tuple[int, int, int]:
    """Resolve a GUI label to grid dimensions (falls back to 16³)."""
    return GRID_SIZE_MAP.get(label, DEFAULT_GRID_SIZE)


def grid_size_label(dims) -> str | None:
    """Reverse lookup of :func:`grid_size_for_label`.

    Used when a checkpoint dictates the grid size (Model Manager -> Train More)
    and the GUI dropdown has to show it again. Returns None for sizes that are
    not offered by the dropdown (e.g. an old checkpoint with a custom grid).
    """
    target = tuple(int(d) for d in dims)
    for label, mapped in GRID_SIZE_MAP.items():
        if mapped == target:
            return label
    return None


def default_run_dir(model_size: str) -> Path:
    """Default run directory for a CLI transformer training/generation."""
    return RUNS_DIR / f"voxel_transformer_{model_size}"
