from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


def estimate_transformer_params(
    text_vocab_size: int,
    block_vocab_size: int,
    grid_size: Tuple[int, int, int] = (16, 16, 16),
    d_model: int = 192,
    nhead: int = 6,
    num_layers: int = 5,
    dim_feedforward: int = 768,
) -> int:
    """Calculate total trainable parameters for VoxelTransformer (single pass)."""
    gx, gy, gz = grid_size
    num_voxels = gx * gy * gz

    # Embeddings
    text_emb = text_vocab_size * d_model          # text_embedding
    block_emb = (block_vocab_size + 1) * d_model   # block_embedding (start token +1)
    pos_emb = num_voxels * d_model                 # position_embedding
    total_emb = text_emb + block_emb + pos_emb

    # LayerNorms (2 * d_model each: weight + bias)
    text_norm = 2 * d_model
    target_norm = 2 * d_model

    # One TransformerDecoderLayer breakdown:
    #   self_attn:  QKV = 3*d_model*d_model  +  output_proj = d_model*d_model  → 4*d_model²
    #   cross_attn: Q from target, K/V from memory  → 3*d_model*d_model
    #   FFN:        linear1 = d_model*dim_ff  +  linear2 = dim_ff*d_model  → 2*d_model*dim_ff
    #   LayerNorms: 3 * 2*d_model
    layer = 0
    layer += 4 * d_model * d_model          # self-attention
    layer += 3 * d_model * d_model          # cross-attention
    layer += 2 * d_model * dim_feedforward  # FFN
    layer += 6 * d_model                    # layer norms
    total_decoder = layer * num_layers

    # Output head: LayerNorm(d_model) + Linear(d_model, d_model//2) + Linear(d_model//2, block_vocab)
    output_head = 0
    output_head += 2 * d_model                                         # output LayerNorm
    output_head += d_model * (d_model // 2) + (d_model // 2)           # first Linear
    output_head += (d_model // 2) * block_vocab_size + block_vocab_size # second Linear

    return total_emb + text_norm + target_norm + total_decoder + output_head


def suggest_architecture(
    target_params_m: float,
    text_vocab_size: int = 129,
    block_vocab_size: int = 253,
    grid_size: Tuple[int, int, int] = (16, 16, 16),
) -> dict:
    """Given a target parameter count in millions, suggest a good architecture
    (d_model, nhead, layers, dim_feedforward) that gets close to the target."""
    target = target_params_m * 1_000_000

    # Architecture candidates (d_model → typical nhead, ff_ratio)
    candidates = [
        (64,   4, 3,  2, 8),
        (96,   4, 3,  2, 10),
        (128,  4, 4,  2, 12),
        (192,  6, 4,  2, 12),
        (256,  8, 4,  3, 14),
        (320,  8, 4,  4, 16),
        (384,  8, 4,  4, 16),
        (512,  8, 4,  6, 20),
        (640,  10, 4, 6, 20),
        (768,  12, 4, 8, 24),
    ]

    best = None
    best_diff = float("inf")

    for d_model, nhead, ff_ratio, min_l, max_l in candidates:
        dim_ff = d_model * ff_ratio
        for layers in range(min_l, max_l + 1, 2):  # even steps
            params = estimate_transformer_params(
                text_vocab_size, block_vocab_size, grid_size,
                d_model, nhead, layers, dim_ff,
            )
            diff = abs(params - target)
            if diff < best_diff:
                best_diff = diff
                best = {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": layers,
                    "dim_feedforward": dim_ff,
                    "params": params,
                    "params_m": params / 1_000_000,
                }

    return best


class VoxelTransformer(nn.Module):
    """Single-pass voxel transformer for Minecraft structure generation.

    Takes a text prompt and directly generates the full voxel grid in one pass
    through a Transformer Decoder (with self-attention + cross-attention over
    the encoded prompt).
    """

    def __init__(
        self,
        text_vocab_size: int,
        block_vocab_size: int,
        grid_size: tuple[int, int, int] = (16, 16, 16),
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
        self.block_vocab_size = block_vocab_size
        self.d_model = d_model

        self.text_embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=0)
        # +1 for the start token (block_vocab_size)
        self.block_embedding = nn.Embedding(block_vocab_size + 1, d_model)
        self.position_embedding = nn.Embedding(self.num_voxels, d_model)

        self.text_norm = nn.LayerNorm(d_model)
        self.target_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, block_vocab_size),
        )
        self.start_token_id = block_vocab_size

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def safe_clamp_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Clamp target block IDs to valid range [0, block_vocab_size-1].
        Prevents CUDA out-of-bounds in cross-entropy when tokenizer vocab > model vocab."""
        return torch.clamp(target_ids, 0, self.block_vocab_size - 1)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Single pass: encode prompt, decode all voxel positions at once."""
        return self.output(self.decoded(prompt_ids))

    def decoded(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Decoder output ``[batch, Voxel, d_model]`` (shared by forward/generate)."""
        batch = prompt_ids.shape[0]
        device = prompt_ids.device

        # Encode text prompt
        memory = self.text_norm(self.text_embedding(prompt_ids))

        # Prepare target sequence: all start tokens + position embeddings
        positions = torch.arange(self.num_voxels, device=device).unsqueeze(0).expand(batch, -1)
        token_ids = torch.full((batch, self.num_voxels), self.start_token_id, device=device)
        target = self.block_embedding(token_ids) + self.position_embedding(positions)
        target = self.target_norm(target)

        return self.decoder(target, memory)

    def iter_output_chunks(self, prompt_ids: torch.Tensor,
                           chunk_voxels: int | None = None):
        """Logits haeppchenweise: ``(start, end, logits_chunk)``.

        Der Ausgabekopf (LayerNorm + zwei Lineare Schichten) ist punktweise und
        damit beliebig entlang der Voxelachse aufteilbar; ``chunk_voxels=None``
        liefert genau einen Chunk (identisch zum ungechunkten ``forward()``).
        Ungechunkt kostet er ``Voxel * block_vocab_size`` Werte - bei Grid 64^3
        und 253 Bloecken sind das 265 MB, bei den 17.690 Blockzustaenden des
        augmentierten Datensatzes 18,5 GiB.
        """
        decoded = self.decoded(prompt_ids)
        voxels = self.num_voxels
        step = voxels if not chunk_voxels else max(1, min(voxels, int(chunk_voxels)))
        for start in range(0, voxels, step):
            end = min(start + step, voxels)
            yield start, end, self.output(decoded[:, start:end, :])

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        temperature: float = 0.9,
        top_k: int = 40,
        chunk_voxels: int | None = None,
    ) -> torch.Tensor:
        """Generate in a single pass; ``chunk_voxels`` limits the logits memory."""
        self.eval()
        device = prompt_ids.device
        batch = prompt_ids.shape[0]
        tokens = torch.empty((batch, self.num_voxels), dtype=torch.long, device=device)

        for start, end, logits in self.iter_output_chunks(prompt_ids, chunk_voxels):
            effective_top_k = min(top_k, logits.shape[-1]) if top_k > 0 else 0
            if temperature > 0:
                logits = logits / temperature
                if effective_top_k > 0:
                    top_k_vals, _ = torch.topk(logits, effective_top_k, dim=-1)
                    min_top_k = top_k_vals[..., -1, None]
                    logits[logits < min_top_k] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(
                    probs.reshape(-1, probs.shape[-1]), 1).reshape(batch, end - start)
            else:
                sampled = logits.argmax(dim=-1)
            tokens[:, start:end] = sampled

        # IDs auf gültigen Bereich begrenzen (0 bis block_vocab_size-1)
        tokens = torch.clamp(tokens, 0, self.block_vocab_size - 1)

        return tokens.view(batch, self.grid_size[0], self.grid_size[1], self.grid_size[2])


# Alias for backward compatibility
SharedWeightVoxelTransformer = VoxelTransformer