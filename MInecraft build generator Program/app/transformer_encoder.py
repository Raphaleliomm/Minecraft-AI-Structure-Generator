"""Pre-trained Transformer text encoders for diffusion models.
Supports Phi-3.5, Gemma 2, Gemma 3, and T5/Flan-T5 models.
All weights are frozen during training; only the last hidden state is returned."""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5EncoderModel,
)

logger = logging.getLogger(__name__)

# ─── Supported models ───
# Each entry: (display_name, hf_model_id, model_type, max_length)
SUPPORTED_MODELS = [
    # Phi-3.5
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "decoder", 512),
    # Gemma 2
    ("Gemma-2-2B", "google/gemma-2-2b", "decoder", 512),
    ("Gemma-2-9B", "google/gemma-2-9b", "decoder", 512),
    ("Gemma-2-27B", "google/gemma-2-27b-it", "decoder", 512),
    # Gemma 3
    ("Gemma-3-1B", "google/gemma-3-1b-it", "decoder", 512),
    ("Gemma-3-4B", "google/gemma-3-4b-it", "decoder", 512),
    ("Gemma-3-12B", "google/gemma-3-12b-it", "decoder", 512),
    ("Gemma-3-27B", "google/gemma-3-27b-it", "decoder", 512),
    # T5 / Flan-T5 (encoder-only)
    ("Flan-T5-small", "google/flan-t5-small", "encoder", 512),
    ("Flan-T5-base", "google/flan-t5-base", "encoder", 512),
    ("Flan-T5-large", "google/flan-t5-large", "encoder", 512),
    ("Flan-T5-XL", "google/flan-t5-xl", "encoder", 512),
    ("Flan-T5-XXL", "google/flan-t5-xxl", "encoder", 512),
]

MODEL_NAMES = [entry[0] for entry in SUPPORTED_MODELS]
MODEL_TO_ID = {entry[0]: entry[1] for entry in SUPPORTED_MODELS}
MODEL_TO_TYPE = {entry[0]: entry[2] for entry in SUPPORTED_MODELS}
MODEL_MAX_LENGTH = {entry[0]: entry[3] for entry in SUPPORTED_MODELS}

# Hidden dimension per encoder - used when a model is not loaded yet
# (Kaggle export, UI hints). Kept next to SUPPORTED_MODELS so it cannot drift.
ENCODER_HIDDEN_DIMS = {
    "Phi-3.5-mini": 3072,
    "Gemma-2-2B": 2304,
    "Gemma-2-9B": 3584,
    "Gemma-2-27B": 4608,
    "Gemma-3-1B": 768,
    "Gemma-3-4B": 2560,
    "Gemma-3-12B": 3840,
    "Gemma-3-27B": 5120,
    "Flan-T5-small": 512,
    "Flan-T5-base": 768,
    "Flan-T5-large": 1024,
    "Flan-T5-XL": 2048,
    "Flan-T5-XXL": 4096,
}
DEFAULT_ENCODER_HIDDEN_DIM = ENCODER_HIDDEN_DIMS["Phi-3.5-mini"]

# Ungefaehre Parameterzahl in Milliarden. Nur fuer die Frage "Encoder in den
# VRAM oder in den System-RAM?" und fuer Hinweistexte gedacht - die genaue
# Groesse liefert erst ``from_pretrained``, deshalb wird der Wert mit
# Sicherheitsabstand verwendet. Fuer T5 zaehlt nur der Encoder (den laedt die
# App), fuer die Decoder-Modelle ist der LM-Kopf bereits abgezogen.
ENCODER_PARAMS_B = {
    "Phi-3.5-mini": 3.8,
    "Gemma-2-2B": 2.6,
    "Gemma-2-9B": 9.2,
    "Gemma-2-27B": 27.2,
    "Gemma-3-1B": 1.0,
    "Gemma-3-4B": 4.3,
    "Gemma-3-12B": 12.2,
    "Gemma-3-27B": 27.4,
    "Flan-T5-small": 0.035,
    "Flan-T5-base": 0.11,
    "Flan-T5-large": 0.34,
    "Flan-T5-XL": 1.4,
    "Flan-T5-XXL": 4.8,
}


def encoder_working_set_bytes(model_name: str, item_bytes: int = 2) -> int:
    """Geschaetzter Speicherbedarf der Encoder-Gewichte in Byte (0 = unbekannt).

    ``item_bytes=2`` entspricht fp16 (VRAM- und RAM-Fallback), ``4`` fp32.
    """
    params_b = ENCODER_PARAMS_B.get(model_name)
    if not params_b:
        return 0
    return int(float(params_b) * 1e9 * max(1, int(item_bytes)))


class FrozenTransformerEncoder(nn.Module):
    """Wraps a pre-trained transformer, freezes all weights, tokenizes text,
    and returns the last hidden state [batch, seq_len, hidden_dim].

    For decoder models (Phi-3.5, Gemma): uses the last hidden state of the final layer.
    For encoder models (T5): uses the encoder output.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        compile_model: bool = False,
    ):
        super().__init__()
        if model_name not in MODEL_TO_ID:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_NAMES)}"
            )
        self.display_name = model_name
        self.hf_id = MODEL_TO_ID[model_name]
        self.model_type = MODEL_TO_TYPE[model_name]
        self.max_length = MODEL_MAX_LENGTH[model_name]
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info(f"Loading {model_name} ({self.hf_id})...")

        # Load tokenizer (with padding)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            use_fast=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        # Load model with fp16 to save RAM, then move to target device.
        # Der angeforderte dtype gilt auch auf der CPU: grosse Encoder passen nur
        # in fp16 in den RAM (Phi-3.5-mini: ~15 GB in fp32, ~7,6 GB in fp16).
        # Aufrufer, die fp32 wollen (kleine Encoder, GPU aus), uebergeben es
        # weiterhin explizit - siehe app/main_app.py.
        load_dtype = dtype
        common_kwargs = dict(
            device_map=None,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            output_hidden_states=False,
        )

        if self.model_type == "encoder":
            self.model = T5EncoderModel.from_pretrained(self.hf_id, **common_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **common_kwargs)
            # For decoder models we only need the base transformer, not the LM head.
            if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                self.transformer = self.model.model
                # Drop the LM head (~100M params for Phi-3.5): we only read hidden
                # states, so the logits projection is pure waste of VRAM/time.
                if hasattr(self.model, "lm_head"):
                    try:
                        self.model.lm_head = None
                    except Exception:
                        logger.debug("LM-Head konnte nicht entfernt werden", exc_info=True)
            else:
                self.transformer = self.model

        # Ensure model is on the correct device
        self.model = self.model.to(device)
        if dtype != torch.float32 and getattr(self.model, "dtype", None) != dtype:
            self.model = self.model.to(dtype)

        # Freeze ALL weights - these should NEVER be trained
        self._freeze()

        # Store hidden dimension
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                self.hidden_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, "d_model"):
                self.hidden_dim = self.model.config.d_model
            else:
                # Infer from forward pass
                self.hidden_dim = self._infer_hidden_dim()
        else:
            self.hidden_dim = self._infer_hidden_dim()

        logger.info(
            f"Loaded {model_name}: hidden_dim={self.hidden_dim}, "
            f"max_length={self.max_length}, frozen=True"
        )

    def _freeze(self):
        """Freeze all parameters - no gradients."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def _infer_hidden_dim(self) -> int:
        """Run a tiny forward pass to determine hidden dimension."""
        dummy = self.tokenizer("test", return_tensors="pt", padding=True)
        dummy = {k: v.to(self.device) for k, v in dummy.items()}
        with torch.no_grad():
            out = self._base_model()(**dummy)
            return out.last_hidden_state.shape[-1]

    def _base_model(self):
        """The module that produces hidden states (never the LM head)."""
        return self.transformer if self.model_type != "encoder" else self.model

    @torch.no_grad()
    def forward(self, text_prompts: list[str]) -> dict:
        """Encode text prompts and return the last hidden state.

        Args:
            text_prompts: List of text strings (batch_size).

        Returns:
            dict with:
                - "last_hidden_state": [batch, seq_len, hidden_dim] tensor
                - "attention_mask": [batch, seq_len] tensor (for cross-attention masking)
        """
        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(self.device, non_blocking=True)

        if self.model_type == "encoder":
            # T5 encoder models
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            last_hidden = outputs.last_hidden_state
        else:
            # Decoder models (Phi-3.5, Gemma): use the base transformer without the
            # LM head. output_hidden_states=False + last_hidden_state is the final
            # layer (hidden_states[-1]) without materialising every layer.
            outputs = self._base_model()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            last_hidden = outputs.last_hidden_state

        return {
            "last_hidden_state": last_hidden.to(self.dtype),
            "attention_mask": attention_mask,
        }

    def get_config(self) -> dict:
        """Return config dict for saving."""
        return {
            "display_name": self.display_name,
            "hf_id": self.hf_id,
            "model_type": self.model_type,
            "max_length": self.max_length,
            "hidden_dim": self.hidden_dim,
        }


def list_supported_models() -> list[str]:
    return list(MODEL_NAMES)