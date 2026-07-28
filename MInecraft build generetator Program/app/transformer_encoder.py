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

        # Load model with fp16 to save RAM, then move to target device
        # Use the requested dtype (typically float16 for CUDA, float32 for CPU)
        load_dtype = dtype if device.type == "cuda" else torch.float32
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
            else:
                self.transformer = self.model

        # Ensure model is on the correct device
        self.model = self.model.to(device)
        if device.type == "cuda" and dtype != torch.float32:
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
            if self.model_type == "encoder":
                out = self.model(**dummy)
                return out.last_hidden_state.shape[-1]
            else:
                out = self.model(**dummy, output_hidden_states=True)
                # For decoder models, get the last hidden state of the last layer
                return out.hidden_states[-1].shape[-1]

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
            # Decoder models (Phi-3.5, Gemma)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Last hidden state from the final decoder layer
            last_hidden = outputs.hidden_states[-1]

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