"""Application configuration and settings management."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Optional, Tuple

from app.constants import DEFAULT_GRID_SIZE, GRID_SIZE_MAP
from app.paths import CONFIG_PATH

logger = logging.getLogger(__name__)

VALID_MODEL_TYPES = ("transformer", "diffusion", "transformer_diffusion")
VALID_LANGUAGES = ("en", "de")


@dataclass
class AppConfig:
    theme: str = "dark"
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE
    default_model: str = "transformer_diffusion"
    default_transformer_name: Optional[str] = "voxel_transformer_scaled"
    default_diffusion_name: Optional[str] = "diffusion_model"
    default_tf_diffusion_name: Optional[str] = None
    temperature: float = 0.2
    top_k: int = 40
    diffusion_steps: int = 50
    noise_block_prob: float = 0.20
    # Nur informativ: die tatsaechlichen Pfade liefert app/paths.py (absolut,
    # unabhaengig vom Arbeitsverzeichnis). Hier bleiben relative Namen stehen,
    # damit config.json portabel bleibt.
    export_directory: str = "exports"
    projects_directory: str = "projects"
    language: str = "en"
    gpu_enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return cls.from_dict(data)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("config.json konnte nicht gelesen werden (%s) - nutze Standards", exc)
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Build a config from raw JSON data, ignoring unknown/invalid values.

        Unknown keys are dropped and values are coerced/validated so a hand
        edited config.json can never crash the app.
        """
        defaults = cls()
        known = {field.name for field in fields(cls)}
        values = {}
        for name in known:
            if name in data:
                values[name] = data[name]

        grid = values.get("grid_size")
        if grid is not None:
            coerced = _coerce_grid_size(grid)
            if coerced is None:
                logger.warning("Ungueltige grid_size in config.json: %r - nutze Standard", grid)
                values.pop("grid_size")
            else:
                values["grid_size"] = coerced

        for name, validator in (
            ("temperature", lambda v: max(0.0, min(5.0, float(v)))),
            ("diffusion_steps", lambda v: max(1, int(v))),
            ("top_k", lambda v: max(0, int(v))),
            ("noise_block_prob", lambda v: max(0.0, min(1.0, float(v)))),
        ):
            if name in values:
                try:
                    values[name] = validator(values[name])
                except (TypeError, ValueError):
                    values.pop(name)

        if values.get("default_model") not in VALID_MODEL_TYPES:
            values["default_model"] = defaults.default_model
        if values.get("language") not in VALID_LANGUAGES:
            values["language"] = defaults.language
        if not isinstance(values.get("gpu_enabled", True), bool):
            values["gpu_enabled"] = bool(values.get("gpu_enabled", True))

        try:
            return cls(**values)
        except TypeError as exc:
            logger.warning("config.json ungueltig (%s) - nutze Standards", exc)
            return defaults


def _coerce_grid_size(value) -> Optional[Tuple[int, int, int]]:
    """Accept [16,16,16], "16,16,16", "16x16x16" or a known GUI label."""
    if isinstance(value, str):
        label = value.strip()
        if label in GRID_SIZE_MAP:
            return GRID_SIZE_MAP[label]
        value = label.lower().replace("x", ",").split(",")
    try:
        parts = tuple(int(str(part).strip()) for part in value)
    except (TypeError, ValueError):
        return None
    if len(parts) != 3 or any(part <= 0 for part in parts):
        return None
    return parts
