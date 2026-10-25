from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _partition_local_windows(x: torch.Tensor, window_size: int, shifted: bool):
    """Pad and partition [B, X, Y, Z, C] into non-wrapping 3D windows."""
    batch, sx, sy, sz, channels = x.shape
    win = max(1, int(window_size))
    shifts = (win // 2 if shifted else 0,) * 3
    left = shifts
    right = tuple((-(size + pad)) % win for size, pad in zip((sx, sy, sz), left))
    # Conv-style padding order for [B, C, X, Y, Z]. No cyclic roll means
    # opposite sides of the structure never become artificial neighbours.
    pad = (left[2], right[2], left[1], right[1], left[0], right[0])
    features = F.pad(x.permute(0, 4, 1, 2, 3), pad)
    valid = F.pad(torch.ones((batch, 1, sx, sy, sz), dtype=torch.bool,
                             device=x.device), pad, value=False)
    px, py, pz = features.shape[2:]
    nx, ny, nz = px // win, py // win, pz // win

    def split(tensor: torch.Tensor) -> torch.Tensor:
        c = tensor.shape[1]
        tensor = tensor.reshape(batch, c, nx, win, ny, win, nz, win)
        tensor = tensor.permute(0, 2, 4, 6, 3, 5, 7, 1)
        return tensor.reshape(batch * nx * ny * nz, win ** 3, c)

    windows = split(features)
    padding_mask = ~split(valid).squeeze(-1)
    meta = (batch, sx, sy, sz, channels, nx, ny, nz, win, left)
    return windows, padding_mask, meta


def _merge_local_windows(windows: torch.Tensor, meta) -> torch.Tensor:
    batch, sx, sy, sz, channels, nx, ny, nz, win, left = meta
    features = windows.reshape(batch, nx, ny, nz, win, win, win, channels)
    features = features.permute(0, 7, 1, 4, 2, 5, 3, 6)
    features = features.reshape(batch, channels, nx * win, ny * win, nz * win)
    x, y, z = left
    features = features[:, :, x:x + sx, y:y + sy, z:z + sz]
    return features.permute(0, 2, 3, 4, 1)


class LocalWindowDecoderLayer(nn.Module):
    """Transformer decoder layer with 3D local self-attention and text cross-attention.

    Parameter names intentionally match ``nn.TransformerDecoderLayer`` so
    existing checkpoints can reuse their learned projections.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, window_size: int, tread_ratio: float = 0.25,
                 ssa_topk: int = 16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self.window_size = max(0, int(window_size))
        self.tread_ratio = min(0.75, max(0.0, float(tread_ratio)))
        self.ssa_topk = max(0, int(ssa_topk))
        self.norm_first = True

    def _run_window_tokens(self, windows: torch.Tensor, memory: torch.Tensor,
                           memory_mask: torch.Tensor | None,
                           window_padding: torch.Tensor | None = None) -> torch.Tensor:
        """Run a TREAD-style bypass route on a random subset of window tokens.

        Routed tokens skip this block and are merged back unchanged. They are
        eligible again in the next block, so every voxel remains represented.
        Gathering before attention/MLP makes the route reduce actual work.
        """
        routed = self.training and self.tread_ratio > 0.0 and windows.shape[1] > 1
        if routed:
            keep = max(1, min(windows.shape[1],
                              int(round(windows.shape[1] * (1.0 - self.tread_ratio)))))
            indices = torch.rand(windows.shape[:2], device=windows.device).topk(
                keep, dim=1, sorted=False).indices
            active = windows.gather(1, indices.unsqueeze(-1).expand(-1, -1, windows.shape[-1]))
            active_padding = (window_padding.gather(1, indices)
                              if window_padding is not None else None)
        else:
            indices = None
            active = windows
            active_padding = window_padding

        q = self.norm1(active)
        attended = self.self_attn(q, q, q, key_padding_mask=active_padding,
                                  need_weights=False)[0]
        active = active + self.dropout1(attended)
        attended = self.multihead_attn(
            self.norm2(active), memory, memory,
            key_padding_mask=memory_mask, need_weights=False,
        )[0]
        active = active + self.dropout2(attended)
        feedforward = self.linear2(self.dropout(
            self.activation(self.linear1(self.norm3(active)))))
        active = active + self.dropout3(feedforward)
        if indices is not None:
            return windows.scatter(1, indices.unsqueeze(-1).expand_as(active), active)
        return active

    def _spatial_sparse_attention(self, windows: torch.Tensor, meta) -> torch.Tensor:
        """Lightweight SSA-style global context over spatial window summaries.

        Content similarity selects a small set of other 3D windows per window;
        a distance penalty preserves spatial locality. The selected summaries
        are broadcast back to their voxels as global context.
        """
        batch, _sx, _sy, _sz, channels, nx, ny, nz, win, _left = meta
        count = nx * ny * nz
        if self.ssa_topk <= 0 or count <= 1:
            return windows
        tokens = windows.reshape(batch, count, win ** 3, channels)
        summary = tokens.mean(dim=2)
        normalized = F.normalize(summary.float(), dim=-1)
        scores = torch.matmul(normalized, normalized.transpose(1, 2))
        coords = torch.stack(torch.meshgrid(
            torch.arange(nx, device=windows.device),
            torch.arange(ny, device=windows.device),
            torch.arange(nz, device=windows.device), indexing="ij"), dim=-1).reshape(count, 3)
        coords = coords.float()
        distance = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        distance = distance / max(1.0, math.sqrt(nx * nx + ny * ny + nz * nz))
        scores = scores - 0.25 * distance.unsqueeze(0)
        k = min(self.ssa_topk, count)
        neighbors = scores.topk(k, dim=-1).indices
        expanded = summary.unsqueeze(1).expand(-1, count, -1, -1)
        selected = expanded.gather(2, neighbors.unsqueeze(-1).expand(-1, -1, -1, channels)).float()
        weights = torch.softmax(
            (summary.float().unsqueeze(2) * selected).sum(dim=-1) / math.sqrt(channels), dim=-1)
        context = (weights.unsqueeze(-1) * selected).sum(dim=2)
        return (tokens + context.to(tokens.dtype).unsqueeze(2)).reshape_as(windows)

    def forward(self, target: torch.Tensor, memory: torch.Tensor,
                grid_size: tuple[int, int, int],
                memory_key_padding_mask: torch.Tensor | None = None,
                shifted: bool = False) -> torch.Tensor:
        batch, voxels, channels = target.shape
        if self.window_size == 0:
            return self._run_window_tokens(target, memory, memory_key_padding_mask)

        spatial = target.reshape(batch, *grid_size, channels)
        windows, window_padding, meta = _partition_local_windows(
            spatial, self.window_size, shifted)

        memory_per_window = memory.unsqueeze(1).expand(
            batch, meta[5] * meta[6] * meta[7], *memory.shape[1:]
        ).reshape(windows.shape[0], *memory.shape[1:])
        memory_mask = None
        if memory_key_padding_mask is not None:
            memory_mask = memory_key_padding_mask.unsqueeze(1).expand(
                batch, meta[5] * meta[6] * meta[7], -1
            ).reshape(windows.shape[0], -1)
        windows = self._run_window_tokens(windows, memory_per_window, memory_mask,
                                          window_padding)
        windows = self._spatial_sparse_attention(windows, meta)
        result = _merge_local_windows(windows, meta).reshape(batch, voxels, channels)
        return result


class LocalWindowDecoder(nn.Module):
    """Drop-in decoder container retaining PyTorch's ``decoder.layers`` keys."""

    def __init__(self, layer: LocalWindowDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            layer if i == 0 else LocalWindowDecoderLayer(
                layer.self_attn.embed_dim, layer.self_attn.num_heads,
                layer.linear1.out_features, layer.dropout.p, layer.window_size,
                layer.tread_ratio, layer.ssa_topk
            ) for i in range(max(1, int(num_layers)))
        ])

    def forward(self, target: torch.Tensor, memory: torch.Tensor,
                grid_size: tuple[int, int, int],
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = target
        for index, layer in enumerate(self.layers):
            output = layer(output, memory, grid_size, memory_key_padding_mask,
                           shifted=(index % 2 == 1))
        return output


def estimate_transformer_params(
    text_vocab_size: int,
    block_vocab_size: int,
    grid_size: Tuple[int, int, int] = (16, 16, 16),
    d_model: int = 192,
    nhead: int = 6,
    num_layers: int = 5,
    dim_feedforward: int = 768,
    recurrent_depth: int = 1,
    moe_experts: int = 0,
    text_input_dim: int = 0,
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
    total_decoder = layer * (1 if recurrent_depth > 1 else num_layers)
    # A shared recurrent block has a learned per-token continue/halting router.
    recurrence_router = d_model + 1 if recurrent_depth > 1 else 0
    text_projection = d_model * text_input_dim + d_model if text_input_dim else 0

    # Output head: LayerNorm(d_model) + Linear(d_model, d_model//2) + Linear(d_model//2, block_vocab)
    output_head = 0
    output_head += 2 * d_model                                         # output LayerNorm
    output_head += d_model * (d_model // 2) + (d_model // 2)           # first Linear
    output_head += (d_model // 2) * block_vocab_size + block_vocab_size # second Linear

    moe = 0
    if moe_experts > 0:
        moe += d_model * moe_experts + moe_experts  # router
        moe += moe_experts * (2 * d_model * dim_feedforward + dim_feedforward + d_model)
    if text_input_dim:
        total_emb -= text_vocab_size * d_model
    return total_emb + text_norm + target_norm + total_decoder + recurrence_router + text_projection + output_head + moe


def estimate_active_transformer_params(
    text_vocab_size: int, block_vocab_size: int,
    grid_size: Tuple[int, int, int] = (16, 16, 16), d_model: int = 192,
    nhead: int = 6, num_layers: int = 5, dim_feedforward: int = 768,
    recurrent_depth: int = 1, moe_experts: int = 0, moe_active: int = 1,
    text_input_dim: int = 0,
) -> int:
    total = estimate_transformer_params(text_vocab_size, block_vocab_size, grid_size,
        d_model, nhead, num_layers, dim_feedforward, recurrent_depth, moe_experts,
        text_input_dim)
    if not moe_experts:
        return total
    expert_params = 2 * d_model * dim_feedforward + dim_feedforward + d_model
    return total - (moe_experts - min(max(1, moe_active), moe_experts)) * expert_params


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
        recurrent_depth: int = 1,
        moe_experts: int = 0,
        moe_active: int = 1,
        text_input_dim: int = 0,
        text_encoder_name: str | None = None,
        window_size: int = 8,
        tread_ratio: float = 0.25,
        ssa_topk: int = 16,
        voxel_mask_ratio: float = 0.65,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
        self.block_vocab_size = block_vocab_size
        self.d_model = d_model
        self.window_size = max(0, int(window_size))
        self.tread_ratio = min(0.75, max(0.0, float(tread_ratio)))
        self.ssa_topk = max(0, int(ssa_topk))
        self.voxel_mask_ratio = min(0.95, max(0.0, float(voxel_mask_ratio)))

        self.text_encoder_name = text_encoder_name or ""
        self.text_input_dim = max(0, int(text_input_dim))
        self.text_embedding = (None if self.text_input_dim else
                               nn.Embedding(text_vocab_size, d_model, padding_idx=0))
        self.text_projection = (nn.Linear(self.text_input_dim, d_model)
                                if self.text_input_dim else None)
        # +1 for the start token (block_vocab_size)
        self.block_embedding = nn.Embedding(block_vocab_size + 1, d_model)
        self.position_embedding = nn.Embedding(self.num_voxels, d_model)

        self.text_norm = nn.LayerNorm(d_model)
        self.target_norm = nn.LayerNorm(d_model)

        layer = LocalWindowDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, window_size=self.window_size,
            tread_ratio=self.tread_ratio, ssa_topk=self.ssa_topk,
        )
        self.recurrent_depth = max(1, int(recurrent_depth))
        self.decoder = LocalWindowDecoder(layer, num_layers=num_layers) if self.recurrent_depth == 1 else None
        self.decoder_layer = layer if self.recurrent_depth > 1 else None
        self.recurrence_router = nn.Linear(d_model, 1) if self.recurrent_depth > 1 else None
        if self.recurrence_router is not None:
            # Start by using the available depth; training learns which tokens
            # can halt earlier.
            nn.init.constant_(self.recurrence_router.bias, 2.0)
        self.moe_experts = max(0, int(moe_experts))
        self.moe_active = min(max(1, int(moe_active)), self.moe_experts) if self.moe_experts else 0
        self.moe_router = nn.Linear(d_model, self.moe_experts) if self.moe_experts else None
        self.moe = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
            for _ in range(self.moe_experts)
        ])
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

    @property
    def active_parameters(self) -> int:
        total = self.num_parameters
        if not self.moe_experts:
            return total
        per_expert = sum(p.numel() for p in self.moe[0].parameters())
        return total - (self.moe_experts - self.moe_active) * per_expert

    def safe_clamp_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Clamp target block IDs to valid range [0, block_vocab_size-1].
        Prevents CUDA out-of-bounds in cross-entropy when tokenizer vocab > model vocab."""
        return torch.clamp(target_ids, 0, self.block_vocab_size - 1)

    def forward(self, prompt_ids: torch.Tensor | None = None,
                prompt_embeddings: torch.Tensor | None = None,
                prompt_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Generate voxel logits from learned token IDs or external text features."""
        return self.output(self.decoded(prompt_ids, prompt_embeddings, prompt_attention_mask))

    def backward_weighted_loss(
        self, target_ids: torch.Tensor, weight_per_token: torch.Tensor,
        chunk_voxels: int | None = None, grad_scale: float = 1.0,
        prompt_ids: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> float:
        """Backpropagate weighted voxel CE in chunks to bound output-logit memory."""
        decoded = self.decoded(prompt_ids, prompt_embeddings, prompt_attention_mask)
        batch, voxels, _ = decoded.shape
        targets = self.safe_clamp_target(target_ids.reshape(batch, voxels))
        weights = weight_per_token.reshape(batch, voxels)
        denominator = weights.sum().clamp_min(1.0)
        # Voxel-aware masking: retain every non-air target and randomly mask a
        # fraction of air targets. Importance correction keeps the weighted
        # loss unbiased while the large air-only output head work is reduced.
        mask_ratio = self.voxel_mask_ratio if self.training else 0.0
        if mask_ratio > 0.0:
            air = targets.eq(0)
            keep_air = torch.rand_like(weights) >= mask_ratio
            weights = torch.where(air, weights * keep_air / (1.0 - mask_ratio), weights)
        step = voxels if not chunk_voxels else max(1, min(voxels, int(chunk_voxels)))
        total_loss = decoded.new_zeros(())

        for start in range(0, voxels, step):
            end = min(start + step, voxels)
            chunk_weights = weights[:, start:end]
            selected = chunk_weights.ne(0).reshape(-1)
            if not bool(selected.any()):
                continue
            chunk_features = decoded[:, start:end, :].reshape(-1, decoded.shape[-1])[selected]
            logits = self.output(chunk_features)
            nll = F.cross_entropy(
                logits, targets[:, start:end].reshape(-1)[selected], reduction="none",
            )
            chunk_weights = chunk_weights.reshape(-1)[selected]
            loss = (nll * chunk_weights).sum() / denominator
            (loss * float(grad_scale)).backward(retain_graph=end < voxels)
            total_loss = total_loss + loss.detach()
        return float(total_loss)

    def decoded(self, prompt_ids: torch.Tensor | None = None,
                prompt_embeddings: torch.Tensor | None = None,
                prompt_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Decoder output ``[batch, Voxel, d_model]`` (shared by forward/generate)."""
        if (prompt_ids is None) == (prompt_embeddings is None):
            raise ValueError("Provide exactly one of prompt_ids or prompt_embeddings")
        batch = prompt_ids.shape[0] if prompt_ids is not None else prompt_embeddings.shape[0]
        device = prompt_ids.device if prompt_ids is not None else prompt_embeddings.device

        # Encode text prompt
        if prompt_embeddings is not None:
            if self.text_projection is None:
                raise ValueError("This model was trained with learned prompt embeddings")
            memory = self.text_norm(self.text_projection(prompt_embeddings))
            if prompt_attention_mask is not None:
                memory_padding_mask = ~prompt_attention_mask.to(device=device, dtype=torch.bool)
            else:
                memory_padding_mask = None
        else:
            if self.text_embedding is None:
                raise ValueError("This model requires a pretrained text encoder")
            memory = self.text_norm(self.text_embedding(prompt_ids))
            memory_padding_mask = prompt_ids.eq(0)

        # Prepare target sequence: all start tokens + position embeddings
        positions = torch.arange(self.num_voxels, device=device).unsqueeze(0).expand(batch, -1)
        token_ids = torch.full((batch, self.num_voxels), self.start_token_id, device=device)
        target = self.block_embedding(token_ids) + self.position_embedding(positions)
        target = self.target_norm(target)

        if self.decoder is not None:
            decoded = self.decoder(target, memory, self.grid_size,
                                  memory_key_padding_mask=memory_padding_mask)
        else:
            decoded = target
            active = torch.ones((*decoded.shape[:2], 1), dtype=decoded.dtype, device=device)
            for depth in range(self.recurrent_depth):
                candidate = self.decoder_layer(
                    decoded, memory, self.grid_size,
                    memory_key_padding_mask=memory_padding_mask,
                    shifted=(depth % 2 == 1))
                decoded = active * candidate + (1.0 - active) * decoded
                if depth + 1 < self.recurrent_depth:
                    # Straight-through per-token routing: hard continue/stop
                    # decisions at inference, with a differentiable probability
                    # path while training.
                    continue_prob = torch.sigmoid(self.recurrence_router(decoded))
                    continue_hard = (continue_prob >= 0.5).to(continue_prob.dtype)
                    continue_gate = continue_prob + (continue_hard - continue_prob).detach()
                    active = active * continue_gate
        if self.moe_router is not None:
            scores = self.moe_router(decoded)
            weights, indices = scores.topk(self.moe_active, dim=-1)
            weights = torch.softmax(weights, dim=-1)
            flat_x = decoded.reshape(-1, self.d_model)
            flat_indices = indices.reshape(-1, self.moe_active)
            flat_weights = weights.reshape(-1, self.moe_active)
            routed = torch.zeros_like(flat_x)
            for expert_id, expert in enumerate(self.moe):
                token_ids, slots = (flat_indices == expert_id).nonzero(as_tuple=True)
                if token_ids.numel():
                    expert_out = expert(flat_x.index_select(0, token_ids))
                    contribution = expert_out * flat_weights[token_ids, slots].unsqueeze(-1)
                    routed = routed.index_add(0, token_ids, contribution)
            decoded = decoded + routed.view_as(decoded)
        return decoded

    def iter_output_chunks(self, prompt_ids: torch.Tensor | None = None,
                           chunk_voxels: int | None = None,
                           prompt_embeddings: torch.Tensor | None = None,
                           prompt_attention_mask: torch.Tensor | None = None):
        """Logits haeppchenweise: ``(start, end, logits_chunk)``.

        Der Ausgabekopf (LayerNorm + zwei Lineare Schichten) ist punktweise und
        damit beliebig entlang der Voxelachse aufteilbar; ``chunk_voxels=None``
        liefert genau einen Chunk (identisch zum ungechunkten ``forward()``).
        Ungechunkt kostet er ``Voxel * block_vocab_size`` Werte - bei Grid 64^3
        und 253 Bloecken sind das 265 MB, bei den 17.690 Blockzustaenden des
        augmentierten Datensatzes 18,5 GiB.
        """
        decoded = self.decoded(prompt_ids, prompt_embeddings, prompt_attention_mask)
        voxels = self.num_voxels
        step = voxels if not chunk_voxels else max(1, min(voxels, int(chunk_voxels)))
        for start in range(0, voxels, step):
            end = min(start + step, voxels)
            yield start, end, self.output(decoded[:, start:end, :])

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor | None = None,
        temperature: float = 0.9,
        top_k: int = 40,
        chunk_voxels: int | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate in a single pass; ``chunk_voxels`` limits the logits memory."""
        self.eval()
        if (prompt_ids is None) == (prompt_embeddings is None):
            raise ValueError("Provide exactly one of prompt_ids or prompt_embeddings")
        device = prompt_ids.device if prompt_ids is not None else prompt_embeddings.device
        batch = prompt_ids.shape[0] if prompt_ids is not None else prompt_embeddings.shape[0]
        tokens = torch.empty((batch, self.num_voxels), dtype=torch.long, device=device)

        for start, end, logits in self.iter_output_chunks(
                prompt_ids, chunk_voxels, prompt_embeddings, prompt_attention_mask):
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
