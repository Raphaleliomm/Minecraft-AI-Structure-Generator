"""Application configuration and settings management."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "theme": "dark",
    "grid_size": [16, 16, 16],
    "default_model": "transformer",
    "default_transformer_name": "voxel_transformer_scaled",
    "default_diffusion_name": "diffusion_model",
    "default_tf_diffusion_name": None,
    "temperature": 0.85,
    "top_k": 40,
    "diffusion_steps": 50,
    "noise_block_prob": 0.20,
    "export_directory": "exports",
    "projects_directory": "projects",
    "language": "de",
    "gpu_enabled": True,
}


@dataclass
class AppConfig:
    theme: str = "dark"
    grid_size: list = (16, 16, 16)
    default_model: str = "transformer"
    default_transformer_name: str = "voxel_transformer_scaled"
    default_diffusion_name: str = "diffusion_model"
    default_tf_diffusion_name: Optional[str] = None
    temperature: float = 0.85
    top_k: int = 40
    diffusion_steps: int = 50
    noise_block_prob: float = 0.20
    export_directory: str = "exports"
    projects_directory: str = "projects"
    language: str = "de"
    gpu_enabled: bool = True

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> AppConfig:
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                return cls(**{k: data.get(k, v) for k, v in DEFAULT_CONFIG.items()})
            except Exception:
                pass
        return cls()