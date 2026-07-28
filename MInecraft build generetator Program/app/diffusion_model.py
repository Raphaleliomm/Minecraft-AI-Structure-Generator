"""3D Voxel Diffusion Model - works like Stable Diffusion but on Minecraft blocks.
Instead of adding/subtracting values, it directly sets blocks. Air is a valid block.
Uses a 3D UNet with discrete denoising diffusion on block IDs.
Optimized for GTX 1060 6GB (no Tensor Cores, 6GB VRAM).

Includes a variant that uses a frozen pre-trained transformer (Phi-3.5, Gemma, T5)
as text encoder with cross-attention instead of average pooling."""
from __future__ import annotations

import math
import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class BlockEmbedding(nn.Module):
    def __init__(self, num_blocks: int, d_embed: int):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed = nn.Embedding(num_blocks, d_embed)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(torch.clamp(x, 0, self.num_blocks - 1))


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, time_dim)
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple:
        h = self.res(x, t_emb)
        return self.down(h), h


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock3D(out_ch + out_ch, out_ch, time_dim)
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        if h.shape[2:] != skip.shape[2:]:
            h = torch.nn.functional.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.res(torch.cat([h, skip], dim=1), t_emb)


class MiddleBlock3D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.res1 = ResBlock3D(channels, channels, time_dim)
        self.res2 = ResBlock3D(channels, channels, time_dim)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(x, t_emb), t_emb)


class TextConditioning(nn.Module):
    def __init__(self, text_vocab: int, d_text: int, d_model: int):
        super().__init__()
        self.text_vocab = text_vocab
        self.embed = nn.Embedding(text_vocab, d_text, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(d_text, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        text_feat = self.proj(self.embed(torch.clamp(text_ids, 0, self.text_vocab - 1)))
        mask = (text_ids != 0).float().unsqueeze(-1)
        return (text_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class CrossAttention(nn.Module):
    """Cross-attention layer: query from voxel features, key/value from text features."""
    def __init__(self, query_dim: int, context_dim: int, nheads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.nheads = nheads
        self.head_dim = query_dim // nheads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, N, C] (voxel features), context: [B, S, D] (text features),
           mask: [B, S] (1=valid, 0=padded)"""
        B, N, C = x.shape
        S = context.shape[1]

        q = self.to_q(x).reshape(B, N, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, S, self.nheads, self.head_dim).permute(0, 2, 3, 1)
        v = self.to_v(context).reshape(B, S, self.nheads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale  # [B, nheads, N, S]

        if mask is not None:
            # mask: [B, S] -> [B, 1, 1, S]
            attn_mask = mask[:, None, None, :].float()
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)


class CrossAttnResBlock3D(nn.Module):
    """ResBlock3D with cross-attention over text features injected between convs."""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int,
                 context_dim: int, nheads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.ca = CrossAttention(out_channels, context_dim, nheads=nheads)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None, None]

        # Cross-attention: [B, C, D, H, W] -> [B, D*H*W, C]
        B, C, D, H, W = h.shape
        h_flat = h.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        h_flat = self.ca(h_flat, context, context_mask)
        h = h_flat.permute(0, 2, 1).reshape(B, C, D, H, W)

        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class CrossAttnDownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, context_dim: int, nheads: int = 4):
        super().__init__()
        self.res = CrossAttnResBlock3D(in_ch, out_ch, time_dim, context_dim, nheads=nheads)
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor | None = None) -> tuple:
        h = self.res(x, t_emb, context, context_mask)
        return self.down(h), h


class CrossAttnUpBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, context_dim: int, nheads: int = 4):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = CrossAttnResBlock3D(out_ch + out_ch, out_ch, time_dim, context_dim, nheads=nheads)
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.up(x)
        if h.shape[2:] != skip.shape[2:]:
            h = torch.nn.functional.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.res(torch.cat([h, skip], dim=1), t_emb, context, context_mask)


class CrossAttnMiddleBlock3D(nn.Module):
    def __init__(self, channels: int, time_dim: int, context_dim: int, nheads: int = 4):
        super().__init__()
        self.res1 = CrossAttnResBlock3D(channels, channels, time_dim, context_dim, nheads=nheads)
        self.res2 = CrossAttnResBlock3D(channels, channels, time_dim, context_dim, nheads=nheads)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.res2(self.res1(x, t_emb, context, context_mask), t_emb, context, context_mask)


class VoxelDiffusionModel(nn.Module):
    def __init__(self, num_blocks: int, text_vocab_size: int, grid_size=(16, 16, 16),
                 d_model: int = 64, d_text: int = 32, channels: int = 32,
                 channel_multipliers=(1, 2, 2), num_timesteps: int = 50):
        super().__init__()
        self.grid_size = grid_size
        self.num_blocks = num_blocks
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        self.d_text = d_text
        self.channels = channels
        self.channel_multipliers = channel_multipliers
        self.block_embed = BlockEmbedding(num_blocks, channels)
        self.time_embed = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model))
        self.text_cond = TextConditioning(text_vocab_size, d_text, d_model)
        self.text_to_ch = nn.Linear(d_model, channels)
        in_ch = channels
        self.encoder = nn.ModuleList()
        for mult in channel_multipliers:
            out_ch = channels * mult
            self.encoder.append(DownBlock3D(in_ch, out_ch, d_model))
            in_ch = out_ch
        self.middle = MiddleBlock3D(in_ch, d_model)
        self.decoder = nn.ModuleList()
        for mult in reversed(channel_multipliers):
            out_ch = channels * mult
            self.decoder.append(UpBlock3D(in_ch, out_ch, d_model))
            in_ch = out_ch
        self.out_norm = nn.GroupNorm(min(8, channels), channels)
        self.out_proj = nn.Conv3d(channels, num_blocks, 1)

    def forward(self, noisy_blocks, timesteps, text_ids):
        x = self.block_embed(torch.clamp(noisy_blocks, 0, self.num_blocks - 1)).permute(0, 4, 1, 2, 3)
        t_emb = self.time_embed(sinusoidal_embedding(timesteps, self.d_model))
        x = x + self.text_to_ch(self.text_cond(text_ids))[:, :, None, None, None]
        skips = []
        for down in self.encoder:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.middle(x, t_emb)
        for up in self.decoder:
            x = up(x, skips.pop(), t_emb)
        return self.out_proj(self.out_norm(x))

    @torch.no_grad()
    def sample(self, text_ids, num_steps=None, temperature=1.0, top_k=0):
        self.eval()
        device = next(self.parameters()).device
        B = text_ids.shape[0]
        steps = num_steps or self.num_timesteps
        GX, GY, GZ = self.grid_size
        x = torch.randint(0, self.num_blocks, (B, GX, GY, GZ), device=device)
        for t in range(steps - 1, -1, -1):
            logits = self.forward(x, torch.full((B,), t, device=device, dtype=torch.long), text_ids)
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.shape[1])
                min_k = torch.topk(logits, k, dim=1)[0][:, -1, :, :, :].unsqueeze(1)
                logits[logits < min_k] = float("-inf")
            probs = torch.softmax(logits, dim=1)
            flat_probs = probs.permute(0, 2, 3, 4, 1).reshape(-1, self.num_blocks)
            if t > 0:
                pred = torch.multinomial(flat_probs, 1).reshape(B, GX, GY, GZ)
                mask = torch.rand_like(pred.float()) < ((t - 1) / steps)
                x = torch.where(mask, torch.randint(0, self.num_blocks, pred.shape, device=device), pred)
            else:
                x = torch.multinomial(flat_probs, 1).reshape(B, GX, GY, GZ)
            x = torch.clamp(x, 0, self.num_blocks - 1)
        return x


class TransformerDiffusionModel(nn.Module):
    """Diffusion model that uses a frozen pre-trained transformer (Phi-3.5, Gemma, T5)
    as text encoder with cross-attention instead of average pooling.

    The transformer is loaded separately and passed in. Its weights are frozen.
    The UNet uses CrossAttnResBlock3D blocks that attend to the transformer's
    last hidden state [batch, seq_len, hidden_dim].
    """

    def __init__(
        self,
        num_blocks: int,
        grid_size: tuple[int, int, int] = (16, 16, 16),
        d_model: int = 64,
        channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 2),
        num_timesteps: int = 50,
        context_dim: int = 768,  # hidden_dim of the transformer
        cross_attn_heads: int = 4,
        context_proj_dim: int | None = None,  # if set, project context to this dim first
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_blocks = num_blocks
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        self.channels = channels
        self.channel_multipliers = channel_multipliers
        self.context_dim = context_dim
        self.cross_attn_heads = cross_attn_heads

        # Project transformer hidden_dim -> UNet channel dim for cross-attention
        if context_proj_dim is not None:
            self.context_proj = nn.Linear(context_dim, context_proj_dim)
            self.effective_context_dim = context_proj_dim
        else:
            self.context_proj = nn.Identity()
            self.effective_context_dim = context_dim

        self.block_embed = BlockEmbedding(num_blocks, channels)
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model)
        )

        in_ch = channels
        self.encoder = nn.ModuleList()
        for mult in channel_multipliers:
            out_ch = channels * mult
            self.encoder.append(
                CrossAttnDownBlock3D(in_ch, out_ch, d_model, self.effective_context_dim, nheads=cross_attn_heads)
            )
            in_ch = out_ch
        self.middle = CrossAttnMiddleBlock3D(in_ch, d_model, self.effective_context_dim, nheads=cross_attn_heads)
        self.decoder = nn.ModuleList()
        for mult in reversed(channel_multipliers):
            out_ch = channels * mult
            self.decoder.append(
                CrossAttnUpBlock3D(in_ch, out_ch, d_model, self.effective_context_dim, nheads=cross_attn_heads)
            )
            in_ch = out_ch
        self.out_norm = nn.GroupNorm(min(8, channels), channels)
        self.out_proj = nn.Conv3d(channels, num_blocks, 1)

    def forward(
        self,
        noisy_blocks: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with cross-attention over transformer features.

        Args:
            noisy_blocks: [B, GX, GY, GZ] block IDs (noisy)
            timesteps: [B] diffusion timesteps
            context: [B, S, D] transformer last hidden state
            context_mask: [B, S] attention mask (1=valid, 0=padded)

        Returns:
            logits: [B, num_blocks, GX, GY, GZ]
        """
        x = self.block_embed(torch.clamp(noisy_blocks, 0, self.num_blocks - 1)).permute(0, 4, 1, 2, 3)
        t_emb = self.time_embed(sinusoidal_embedding(timesteps, self.d_model))

        # Project context if needed
        context_proj = self.context_proj(context)

        skips = []
        for down in self.encoder:
            x, skip = down(x, t_emb, context_proj, context_mask)
            skips.append(skip)
        x = self.middle(x, t_emb, context_proj, context_mask)
        for up in self.decoder:
            x = up(x, skips.pop(), t_emb, context_proj, context_mask)
        return self.out_proj(self.out_norm(x))

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            context: [B, S, D] transformer last hidden state
            context_mask: [B, S] attention mask
            num_steps: number of denoising steps (default: self.num_timesteps)
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)

        Returns:
            [B, GX, GY, GZ] block ID grid
        """
        self.eval()
        device = next(self.parameters()).device
        B = context.shape[0]
        steps = num_steps or self.num_timesteps
        GX, GY, GZ = self.grid_size
        x = torch.randint(0, self.num_blocks, (B, GX, GY, GZ), device=device)

        for t in range(steps - 1, -1, -1):
            logits = self.forward(
                x,
                torch.full((B,), t, device=device, dtype=torch.long),
                context,
                context_mask,
            )
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.shape[1])
                min_k = torch.topk(logits, k, dim=1)[0][:, -1, :, :, :].unsqueeze(1)
                logits[logits < min_k] = float("-inf")
            probs = torch.softmax(logits, dim=1)
            flat_probs = probs.permute(0, 2, 3, 4, 1).reshape(-1, self.num_blocks)
            if t > 0:
                pred = torch.multinomial(flat_probs, 1).reshape(B, GX, GY, GZ)
                mask = torch.rand_like(pred.float()) < ((t - 1) / steps)
                x = torch.where(mask, torch.randint(0, self.num_blocks, pred.shape, device=device), pred)
            else:
                x = torch.multinomial(flat_probs, 1).reshape(B, GX, GY, GZ)
            x = torch.clamp(x, 0, self.num_blocks - 1)
        return x


def train_diffusion_step(model, batch, optimizer, device, scaler=None):
    model.train()
    clean = torch.clamp(batch["voxel_ids"].to(device, non_blocking=True).long(), 0, model.num_blocks - 1)
    text_ids = batch["prompt_ids"].to(device, non_blocking=True)
    B = clean.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device)
    noise_mask = torch.rand_like(clean.float()) < (t.float() / model.num_timesteps).view(B, 1, 1, 1)
    noisy = torch.where(noise_mask, torch.randint(0, model.num_blocks, clean.shape, device=device), clean)
    use_amp = scaler is not None
    with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
        logits = model(noisy, t, text_ids)
        ce = torch.nn.functional.cross_entropy(logits, clean, reduction="none").reshape(-1)
        sample_weight = batch.get("sample_weight", torch.ones(B, device=device, dtype=torch.float32))
        if isinstance(sample_weight, torch.Tensor) and sample_weight.device != device:
            sample_weight = sample_weight.to(device)
        sample_weight_expanded = sample_weight.view(B, 1, 1, 1).expand_as(clean).reshape(-1)
        w = torch.where(clean.reshape(-1) == 0, 0.0625, 1.0) * sample_weight_expanded
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss.detach())


def train_transformer_diffusion_step(
    model: TransformerDiffusionModel,
    batch: dict,
    transformer_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> float:
    """Training step for TransformerDiffusionModel.

    The transformer_encoder is frozen - only the UNet parameters are trained.
    """
    model.train()
    clean = torch.clamp(batch["voxel_ids"].to(device, non_blocking=True).long(), 0, model.num_blocks - 1)
    B = clean.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device)
    noise_mask = torch.rand_like(clean.float()) < (t.float() / model.num_timesteps).view(B, 1, 1, 1)
    noisy = torch.where(noise_mask, torch.randint(0, model.num_blocks, clean.shape, device=device), clean)

    # Encode prompts with frozen transformer
    prompts = batch.get("prompt_text", None)
    if prompts is None:
        raise ValueError("batch must contain 'prompt_text' for transformer encoder")

    with torch.no_grad():
        encoded = transformer_encoder(prompts)
        context = encoded["last_hidden_state"].to(device)
        context_mask = encoded["attention_mask"].to(device)

    use_amp = scaler is not None
    with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
        logits = model(noisy, t, context, context_mask)
        ce = torch.nn.functional.cross_entropy(logits, clean, reduction="none").reshape(-1)
        sample_weight = batch.get("sample_weight", torch.ones(B, device=device, dtype=torch.float32))
        if isinstance(sample_weight, torch.Tensor) and sample_weight.device != device:
            sample_weight = sample_weight.to(device)
        sample_weight_expanded = sample_weight.view(B, 1, 1, 1).expand_as(clean).reshape(-1)
        w = torch.where(clean.reshape(-1) == 0, 0.0625, 1.0) * sample_weight_expanded
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss.detach())


def train_transformer_diffusion_step_cached(
    model: TransformerDiffusionModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> float:
    """Training step for TransformerDiffusionModel using pre-computed hidden states.
    
    No transformer_encoder needed - context comes from the batch directly.
    The cached states are float32 on CPU, so we cast them to match the model's dtype.
    """
    model.train()
    clean = torch.clamp(batch["voxel_ids"].to(device, non_blocking=True).long(), 0, model.num_blocks - 1)
    B = clean.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device)
    noise_mask = torch.rand_like(clean.float()) < (t.float() / model.num_timesteps).view(B, 1, 1, 1)
    noisy = torch.where(noise_mask, torch.randint(0, model.num_blocks, clean.shape, device=device), clean)

    # Use pre-computed hidden states from batch
    # Cast to model's dtype to avoid Half/Float mismatch
    model_dtype = next(model.parameters()).dtype
    context = batch["hidden_states"].to(device=device, dtype=model_dtype, non_blocking=True)
    context_mask = batch["attention_masks"].to(device=device, non_blocking=True)

    use_amp = scaler is not None
    with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
        logits = model(noisy, t, context, context_mask)
        ce = torch.nn.functional.cross_entropy(logits, clean, reduction="none").reshape(-1)
        sample_weight = batch.get("sample_weight", torch.ones(B, device=device, dtype=torch.float32))
        if isinstance(sample_weight, torch.Tensor) and sample_weight.device != device:
            sample_weight = sample_weight.to(device)
        sample_weight_expanded = sample_weight.view(B, 1, 1, 1).expand_as(clean).reshape(-1)
        w = torch.where(clean.reshape(-1) == 0, 0.0625, 1.0) * sample_weight_expanded
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss.detach())
