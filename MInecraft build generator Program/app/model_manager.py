"""Model Manager - discovers, loads, manages multiple Transformer, Diffusion,
and Transformer Diffusion models.

Each model is stored as a checkpoint directory under runs/ with:
  runs/
    my_transformer/
      model.pt
      prompt_vocab.json
      block_vocab.json
    my_diffusion/
      model.pt
      prompt_vocab.json
      block_vocab.json
    my_tf_diffusion/
      model.pt
      prompt_vocab.json
      block_vocab.json
      encoder_config.json

The ModelRegistry discovers all models, tracks metadata, and supports
load/delete/rename/set-default operations.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from app.paths import RUNS_DIR

logger = logging.getLogger(__name__)

MODELS_DIR = RUNS_DIR

# Name der Modellordner unter runs/: keine Pfadtrenner, kein ".."
VALID_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-\.]{0,63}$")

# Checkpoint-Metadaten werden pro (Pfad, mtime, Groesse) zwischengespeichert:
# discover() laeuft bei jedem Tab-Refresh und hat vorher jedes Modell zweimal
# komplett von der Platte gelesen.
_METADATA_CACHE: Dict[Tuple[str, float, int], Dict] = {}


def is_valid_model_name(name: str) -> bool:
    """Reject names that could escape runs/ or break the filesystem."""
    candidate = str(name)
    if candidate in (".", "..") or candidate.strip() != candidate:
        return False
    if not VALID_MODEL_NAME.match(candidate):
        return False
    return Path(candidate).name == candidate


def _checkpoint_signature(checkpoint_path: Path) -> Tuple[str, float, int]:
    stat = checkpoint_path.stat()
    return (str(checkpoint_path), stat.st_mtime, stat.st_size)


def _extract_metadata(ckpt: dict) -> Dict:
    """Reduce a checkpoint dict to plain scalar metadata (no tensors kept)."""
    encoder_config = ckpt.get("encoder_config", {}) or {}
    metadata: Dict = {
        "grid_size": tuple(ckpt.get("grid_size", (16, 16, 16))),
        "block_vocab_size": int(ckpt.get("block_vocab_size", ckpt.get("num_blocks", 0))),
        "text_vocab_size": int(ckpt.get("text_vocab_size", 0)),
        "epoch": int(ckpt.get("epoch", 0)),
        "loss": float(ckpt.get("loss", 0.0)),
        "d_model": int(ckpt.get("d_model", 0)),
        "nhead": int(ckpt.get("nhead", 0)),
        "layers": int(ckpt.get("layers", 0)),
        "dim_feedforward": int(ckpt.get("dim_feedforward", 0)),
        "channels": int(ckpt.get("channels", 0)),
        "num_timesteps": int(ckpt.get("num_timesteps", 0)),
        "context_dim": int(ckpt.get("context_dim", 0)),
        "cross_attn_heads": int(ckpt.get("cross_attn_heads", 0)),
        "encoder_config": encoder_config,
        "encoder_name": str(encoder_config.get("display_name", "")),
        "has_encoder_config": "encoder_config" in ckpt,
        "has_channels": "channels" in ckpt or "num_timesteps" in ckpt,
    }
    return metadata


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    """Load (and cache) the scalar metadata of a checkpoint file."""
    try:
        signature = _checkpoint_signature(checkpoint_path)
    except OSError as exc:
        logger.warning("Checkpoint nicht lesbar (%s): %s", checkpoint_path, exc)
        return {}

    cached = _METADATA_CACHE.get(signature)
    if cached is not None:
        return cached

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - defekte Checkpoints duerfen die App nicht killen
        logger.warning("Checkpoint %s konnte nicht geladen werden: %s", checkpoint_path, exc)
        return {}

    metadata = _extract_metadata(ckpt)
    del ckpt  # Gewichte sofort freigeben, nur die Skalare bleiben
    _METADATA_CACHE[signature] = metadata
    return metadata


def classify_model_type(metadata: Dict) -> str:
    """transformer / diffusion / transformer_diffusion from checkpoint metadata."""
    if metadata.get("has_encoder_config"):
        return "transformer_diffusion"
    if metadata.get("has_channels") and not metadata.get("layers"):
        return "diffusion"
    return "transformer"


@dataclass
class ModelEntry:
    """A discovered model checkpoint."""
    name: str               # Directory name (e.g. "my_transformer")
    model_type: str         # "transformer", "diffusion", or "transformer_diffusion"
    path: Path              # Full path to model directory
    checkpoint_path: Path   # path / model.pt
    
    # Metadata loaded from checkpoint
    grid_size: Tuple[int, int, int] = (16, 16, 16)
    block_vocab_size: int = 0
    text_vocab_size: int = 0
    d_model: int = 0
    nhead: int = 0
    num_layers: int = 0
    dim_feedforward: int = 0
    channels: int = 0       # Diffusion / TF-Diffusion only
    num_timesteps: int = 0  # Diffusion / TF-Diffusion only
    context_dim: int = 0        # TF-Diffusion only: hidden size of the text encoder
    cross_attn_heads: int = 0   # TF-Diffusion only
    num_params: int = 0
    num_params_m: float = 0.0
    encoder_name: str = ""      # Transformer Diffusion: name of the frozen encoder
    epochs_trained: int = 0
    last_loss: float = 0.0
    is_loaded: bool = False
    is_default: bool = False
    
    def apply_metadata(self, metadata: Dict) -> None:
        """Fill the metadata fields from an extracted checkpoint metadata dict."""
        if not metadata:
            return
        self.grid_size = tuple(metadata.get("grid_size", (16, 16, 16)))
        self.block_vocab_size = int(metadata.get("block_vocab_size", 0))
        self.text_vocab_size = int(metadata.get("text_vocab_size", 0))
        self.epochs_trained = int(metadata.get("epoch", 0))
        self.last_loss = float(metadata.get("loss", 0.0))
        self.d_model = int(metadata.get("d_model", 0))
        self.channels = int(metadata.get("channels", 0))
        self.num_timesteps = int(metadata.get("num_timesteps", 0))
        self.context_dim = int(metadata.get("context_dim", 0))
        self.cross_attn_heads = int(metadata.get("cross_attn_heads", 0))
        self.encoder_name = str(metadata.get("encoder_name", "")) or "Unknown"

        if self.model_type == "transformer":
            self.nhead = int(metadata.get("nhead", 0))
            self.num_layers = int(metadata.get("layers", 0))
            self.dim_feedforward = int(metadata.get("dim_feedforward", 0))

    def load_metadata(self) -> None:
        """Load metadata from checkpoint without loading the full model.

        Values are cached per (path, mtime, size) so repeated discover() calls do
        not read every checkpoint from disk again.
        """
        if not self.checkpoint_path.exists():
            return
        self.apply_metadata(load_checkpoint_metadata(self.checkpoint_path))


class ModelRegistry:
    """Scans runs/ for all models and provides management operations."""
    
    def __init__(self):
        self.models: Dict[str, ModelEntry] = {}   # name -> ModelEntry
        self.default_transformer: Optional[str] = None
        self.default_diffusion: Optional[str] = None
        self.default_tf_diffusion: Optional[str] = None
    
    def discover(self) -> List[ModelEntry]:
        """Scan runs/ directory for all model checkpoints."""
        self.models.clear()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for model_dir in sorted(MODELS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            self._register_directory(model_dir)
        
        return list(self.models.values())
    
    def _register_directory(self, model_dir: Path) -> None:
        """Try to register a directory as a model. Returns the type or None."""
        ckpt_path = model_dir / "model.pt"
        if not ckpt_path.exists():
            return

        name = model_dir.name

        # Single load per checkpoint (cached by mtime/size): determine the model
        # type and fill the metadata from the same metadata dict.
        metadata = load_checkpoint_metadata(ckpt_path)
        if metadata:
            model_type = classify_model_type(metadata)
        else:
            logger.warning("Modell '%s' uebersprungen: Checkpoint nicht lesbar", name)
            model_type = "transformer"

        entry = ModelEntry(
            name=name,
            model_type=model_type,
            path=model_dir,
            checkpoint_path=ckpt_path,
        )
        entry.apply_metadata(metadata)
        self.models[name] = entry
    
    def get_by_type(self, model_type: str) -> List[ModelEntry]:
        """Get all models of a given type."""
        return [m for m in self.models.values() if m.model_type == model_type]
    
    def get(self, name: str) -> Optional[ModelEntry]:
        return self.models.get(name)
    
    def delete(self, name: str) -> bool:
        """Delete a model directory entirely."""
        entry = self.models.get(name)
        if entry is None:
            return False
        try:
            shutil.rmtree(entry.path)
        except OSError as exc:
            logger.warning("Modell '%s' konnte nicht geloescht werden: %s", name, exc)
            return False
        del self.models[name]
        if self.default_transformer == name:
            self.default_transformer = None
        if self.default_diffusion == name:
            self.default_diffusion = None
        if self.default_tf_diffusion == name:
            self.default_tf_diffusion = None
        return True
    
    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a model directory (validated - never escapes runs/)."""
        entry = self.models.get(old_name)
        candidate = (new_name or "").strip()
        if entry is None or candidate in self.models or candidate == old_name:
            return False
        if not is_valid_model_name(candidate):
            logger.warning("Ungueltiger Modellname: %r", new_name)
            return False
        new_path = entry.path.parent / candidate
        try:
            entry.path.rename(new_path)
            entry.name = candidate
            entry.path = new_path
            entry.checkpoint_path = new_path / "model.pt"
            self.models[candidate] = entry
            del self.models[old_name]

            if self.default_transformer == old_name:
                self.default_transformer = candidate
            if self.default_diffusion == old_name:
                self.default_diffusion = candidate
            if self.default_tf_diffusion == old_name:
                self.default_tf_diffusion = candidate
            return True
        except OSError as exc:
            logger.warning("Umbenennen von %s nach %s fehlgeschlagen: %s", old_name, candidate, exc)
            return False
    
    def set_default(self, name: str) -> bool:
        """Set a model as default for its type."""
        entry = self.models.get(name)
        if entry is None:
            return False
        if entry.model_type == "transformer":
            self.default_transformer = name
        elif entry.model_type == "transformer_diffusion":
            self.default_tf_diffusion = name
        else:
            self.default_diffusion = name
        return True
    
    def load_config_defaults(self, transformer_name: Optional[str], 
                              diffusion_name: Optional[str],
                              tf_diffusion_name: Optional[str] = None) -> None:
        """Load default model names from config."""
        self.default_transformer = transformer_name
        self.default_diffusion = diffusion_name
        self.default_tf_diffusion = tf_diffusion_name
    
    def to_config_dict(self) -> dict:
        return {
            "default_transformer": self.default_transformer,
            "default_diffusion": self.default_diffusion,
            "default_tf_diffusion": self.default_tf_diffusion,
        }
    
    @classmethod
    def from_config_dict(cls, data: dict) -> ModelRegistry:
        registry = cls()
        registry.default_transformer = data.get("default_transformer")
        registry.default_diffusion = data.get("default_diffusion")
        registry.default_tf_diffusion = data.get("default_tf_diffusion")
        return registry