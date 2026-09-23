"""Central path resolution.

Every runtime path (config, runs, exports, projects, textures, data sets) is
anchored to this file instead of the current working directory. Without this,
starting the app or the CLI scripts from another directory silently created or
searched the wrong folders.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

# .../<program directory>  (contains app/, dataset.py, train.py, ...)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Repository root (one level above the program directory, holds Trainingsdaten/)
REPO_ROOT = PROJECT_ROOT.parent

# ─── App-owned folders / files ───
CONFIG_PATH = PROJECT_ROOT / "config.json"
RUNS_DIR = PROJECT_ROOT / "runs"
EXPORTS_DIR = PROJECT_ROOT / "exports"
PROJECTS_DIR = PROJECT_ROOT / "projects"
TMP_DIR = PROJECT_ROOT / ".tmp"
TEXTURE_DIR = PROJECT_ROOT / "Blocktextures" / "block"
CACHE_DIR = RUNS_DIR / "hidden_states_cache"

# ─── Training data ───
TRAINING_DATA_ROOT = REPO_ROOT / "Trainingsdaten"

# Preferred order for the auto-detection (everything else below
# TRAINING_DATA_ROOT follows alphabetically).
_PREFERRED_DATA_DIR_NAMES = (
    "Trainingsdaten good thoroughly analyzed",
    "phantommarket",
)

# Legacy: a local copy inside the program directory.
_LEGACY_DATA_DIR = PROJECT_ROOT / "Trainingsdaten good thoroughly analyzed"

# Fallback shown in messages when no data set is present locally.
FALLBACK_DATA_DIR = TRAINING_DATA_ROOT / "Trainingsdaten good thoroughly analyzed"

_STRUCTURE_SUFFIXES = {".schem", ".schematic"}


def _contains_prompt_data(folder: Path, limit: int = 500) -> bool:
    """Cheap check: does this folder contain at least one structure with a prompt?

    A prompt is either a ``.txt`` sidecar next to the structure or a
    ``prompts.json`` (in that folder or in a parent of this subtree).
    """
    if (folder / "prompts.json").is_file():
        return True
    checked = 0
    try:
        for path in folder.rglob("*"):
            if path.suffix.lower() not in _STRUCTURE_SUFFIXES:
                continue
            if path.with_suffix(".txt").is_file():
                return True
            checked += 1
            if checked >= limit:
                break
            if (path.parent / "prompts.json").is_file():
                return True
    except OSError:
        return False
    return False


def default_data_dirs() -> List[Path]:
    """Return the training data directories that actually exist on disk.

    Everything below ``<repo>/Trainingsdaten`` is searched recursively, so any
    nesting/category layout works. Folders without prompts (structures without a
    ``.txt`` sidecar or ``prompts.json``) are skipped - they could not produce a
    single training example.
    """
    found: List[Path] = []
    if TRAINING_DATA_ROOT.is_dir():
        children = [path for path in sorted(TRAINING_DATA_ROOT.iterdir()) if path.is_dir()]
        for name in _PREFERRED_DATA_DIR_NAMES:
            preferred = TRAINING_DATA_ROOT / name
            if preferred in children:
                children.remove(preferred)
                found.append(preferred)
        for child in children:
            if _contains_prompt_data(child):
                found.append(child)
        if not found and _contains_prompt_data(TRAINING_DATA_ROOT):
            found.append(TRAINING_DATA_ROOT)

    if _LEGACY_DATA_DIR.is_dir() and _contains_prompt_data(_LEGACY_DATA_DIR):
        found.append(_LEGACY_DATA_DIR)

    if found:
        return found
    return [FALLBACK_DATA_DIR]


def default_data_dir_strings() -> List[str]:
    """Same as :func:`default_data_dirs` but as strings (CLI / export helpers)."""
    return [str(path) for path in default_data_dirs()]
