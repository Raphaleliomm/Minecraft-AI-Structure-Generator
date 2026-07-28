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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


MODELS_DIR = Path("runs")


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
    num_params: int = 0
    num_params_m: float = 0.0
    encoder_name: str = ""      # Transformer Diffusion: name of the frozen encoder
    epochs_trained: int = 0
    last_loss: float = 0.0
    is_loaded: bool = False
    is_default: bool = False
    
    def load_metadata(self) -> None:
        """Load metadata from checkpoint without loading the full model."""
        if not self.checkpoint_path.exists():
            return
        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            self.grid_size = tuple(ckpt.get("grid_size", (16, 16, 16)))
            self.block_vocab_size = int(ckpt.get("block_vocab_size", ckpt.get("num_blocks", 0)))
            self.text_vocab_size = int(ckpt.get("text_vocab_size", 0))
            self.epochs_trained = int(ckpt.get("epoch", 0))
            self.last_loss = float(ckpt.get("loss", 0.0))
            self.d_model = int(ckpt.get("d_model", 0))
            
            if self.model_type == "transformer":
                self.nhead = int(ckpt.get("nhead", 0))
                self.num_layers = int(ckpt.get("layers", 0))
                self.dim_feedforward = int(ckpt.get("dim_feedforward", 0))
            elif self.model_type == "transformer_diffusion":
                self.channels = int(ckpt.get("channels", 0))
                self.num_timesteps = int(ckpt.get("num_timesteps", 0))
                # Load encoder name
                encoder_config = ckpt.get("encoder_config", {})
                self.encoder_name = encoder_config.get("display_name", "Unknown")
            else:  # diffusion
                self.channels = int(ckpt.get("channels", 0))
                self.num_timesteps = int(ckpt.get("num_timesteps", 0))
        except Exception:
            pass


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
        
        # Determine model type from checkpoint contents
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Check if it's a transformer diffusion model (has encoder_config key)
            has_encoder_config = "encoder_config" in ckpt
            has_channels = "channels" in ckpt or "num_timesteps" in ckpt
            has_layers = "layers" in ckpt
            
            # Transformer models have "layers", diffusion models have "channels"/"num_timesteps"
            if has_encoder_config:
                model_type = "transformer_diffusion"
            elif has_channels and not has_layers:
                model_type = "diffusion"
            else:
                model_type = "transformer"
            
            entry = ModelEntry(
                name=name,
                model_type=model_type,
                path=model_dir,
                checkpoint_path=ckpt_path,
            )
            entry.load_metadata()
            self.models[name] = entry
            
        except Exception:
            pass  # Corrupted checkpoint, skip
    
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
        shutil.rmtree(entry.path)
        del self.models[name]
        if self.default_transformer == name:
            self.default_transformer = None
        if self.default_diffusion == name:
            self.default_diffusion = None
        if self.default_tf_diffusion == name:
            self.default_tf_diffusion = None
        return True
    
    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a model directory."""
        entry = self.models.get(old_name)
        if entry is None or new_name in self.models:
            return False
        new_path = entry.path.parent / new_name
        try:
            entry.path.rename(new_path)
            entry.name = new_name
            entry.path = new_path
            entry.checkpoint_path = new_path / "model.pt"
            self.models[new_name] = entry
            del self.models[old_name]
            
            if self.default_transformer == old_name:
                self.default_transformer = new_name
            if self.default_diffusion == old_name:
                self.default_diffusion = new_name
            if self.default_tf_diffusion == old_name:
                self.default_tf_diffusion = new_name
            return True
        except Exception:
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