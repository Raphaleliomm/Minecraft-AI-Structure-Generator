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

from app.grad_accum import apply_optimizer_step, per_voxel_weight


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def clamp_num_steps(num_steps: int | None, num_timesteps: int) -> int:
    """Clamp sampling steps to the range the model was trained with.

    The timestep is fed directly into the sinusoidal embedding, so values above
    ``num_timesteps`` are outside the training distribution and produce garbage.
    Fewer steps are allowed (coarser schedule).
    """
    if not num_steps:
        return int(num_timesteps)
    return max(1, min(int(num_steps), int(num_timesteps)))


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
            # A large negative value instead of -inf: a fully masked row would
            # otherwise produce NaN in the softmax (and NaN poisons the whole run).
            attn_mask = mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(~attn_mask, -1e4)

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
                 d_model: int = 128, d_text: int = 64, channels: int = 64,
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

    def features(self, noisy_blocks, timesteps, text_ids) -> torch.Tensor:
        """UNet-Merkmale vor dem Ausgabekopf: ``[B, channels, GX, GY, GZ]``.

        Der Ausgabekopf erzeugt daraus die Logits ``[B, num_blocks, X*Y*Z]`` -
        mit grossem Blockvokabular und Grid der groesste Tensor im Training
        (32^3 * 17.690 Bloecke * Batch 16 * 4 Byte = 34,5 GiB, mehr als der
        gesamte VRAM einer 6-GB-Karte). Deshalb trennt dieses Modul die
        Merkmale vom Kopf: ``iter_logit_chunks()`` kann den Kopf - und damit den
        Speicherbedarf - in Voxel-Haeppchen rechnen, ohne die Numerik zu
        aendern (1x1x1-Konvolution ist punktweise).
        """
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
        return self.out_norm(x)

    def forward(self, noisy_blocks, timesteps, text_ids):
        return self.out_proj(self.features(noisy_blocks, timesteps, text_ids))

    @torch.no_grad()
    def sample(self, text_ids, num_steps=None, temperature=1.0, top_k=0,
               chunk_voxels: int | None = None):
        """Struktur erzeugen; ``chunk_voxels`` begrenzt die Logits im Speicher.

        Ohne ``chunk_voxels`` wird wie bisher der ganze Logits-Tensor
        ``[B, num_blocks, X*Y*Z]`` berechnet - bei Grid 32^3 und 17.690
        Blockzustaenden sind das 2,16 GiB pro Kopie. ``chunk_voxels`` (siehe
        ``app/training_memory.plan_generation``) rechnet Ausgabekopf und
        Sampling in Voxel-Haeppchen; Logits und Wahrscheinlichkeiten bleiben
        dabei identisch, nur die konkreten Zufallszahlen liegen anders.
        """
        self.eval()
        device = next(self.parameters()).device
        B = text_ids.shape[0]
        steps = clamp_num_steps(num_steps, self.num_timesteps)
        GX, GY, GZ = self.grid_size
        x = torch.zeros((B, GX, GY, GZ), dtype=torch.long, device=device)
        for t in range(steps - 1, -1, -1):
            features = self.features(
                x, torch.full((B,), t, device=device, dtype=torch.long), text_ids)
            pred = sample_denoising_step(
                self, features, temperature=temperature, top_k=top_k,
                chunk_voxels=chunk_voxels).reshape(B, GX, GY, GZ)
            if t > 0:
                mask = torch.rand_like(pred.float()) < ((t - 1) / steps)
                x = torch.where(mask, torch.zeros(pred.shape, dtype=torch.long, device=device), pred)
            else:
                x = pred
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

    def features(
        self,
        noisy_blocks: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """UNet-Merkmale vor dem Ausgabekopf: ``[B, channels, GX, GY, GZ]``.

        Der Ausgabekopf erzeugt daraus die Logits ``[B, num_blocks, GX, GY, GZ]``.
        Genau dieser Tensor war der Grund fuer "Cuda out of memory. Tried to
        allocate 34.55 GiB" (Grid 32^3, 17.690 Blockzustaende, Batch 16);
        ``iter_logit_chunks()`` rechnet den Kopf deshalb in Voxel-Haeppchen.

        Args:
            noisy_blocks: [B, GX, GY, GZ] block IDs (noisy)
            timesteps: [B] diffusion timesteps
            context: [B, S, D] transformer last hidden state
            context_mask: [B, S] attention mask (1=valid, 0=padded)
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
        return self.out_norm(x)

    def forward(
        self,
        noisy_blocks: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits ``[B, num_blocks, GX, GY, GZ]`` (ungechunkt).

        Fuer das Training immer ``features()`` + ``iter_logit_chunks()``
        benutzen - der ungechunkte Tensor sprengt bei Grid >= 32 und grossem
        Blockvokabular jeden VRAM.
        """
        return self.out_proj(self.features(noisy_blocks, timesteps, context, context_mask))

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
        return self.out_norm(x)

    def forward(
        self,
        noisy_blocks: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits ``[B, num_blocks, GX, GY, GZ]`` (ungechunkt)."""
        return self.out_proj(self.features(noisy_blocks, timesteps, context, context_mask))

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        chunk_voxels: int | None = None,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            context: [B, S, D] transformer last hidden state
            context_mask: [B, S] attention mask
            num_steps: number of denoising steps (clamped to self.num_timesteps)
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            chunk_voxels: voxel chunk for the output head. ``None`` builds the
                full logits tensor ``[B, num_blocks, X*Y*Z]`` (2.16 GiB per copy
                at grid 32^3 with 17,690 block states, which is what made
                "Generate" die with "Cuda out of memory" after training).
                ``app.training_memory.plan_generation()`` provides a value that
                fits the free VRAM.

        Returns:
            [B, GX, GY, GZ] block ID grid
        """
        self.eval()
        device = next(self.parameters()).device
        B = context.shape[0]
        steps = clamp_num_steps(num_steps, self.num_timesteps)
        GX, GY, GZ = self.grid_size
        x = torch.zeros((B, GX, GY, GZ), dtype=torch.long, device=device)

        for t in range(steps - 1, -1, -1):
            features = self.features(
                x,
                torch.full((B,), t, device=device, dtype=torch.long),
                context,
                context_mask,
            )
            pred = sample_denoising_step(
                self, features, temperature=temperature, top_k=top_k,
                chunk_voxels=chunk_voxels).reshape(B, GX, GY, GZ)
            if t > 0:
                # Match the training noise process (rand < t / num_timesteps).
                mask = torch.rand_like(pred.float()) < ((t - 1) / self.num_timesteps)
                x = torch.where(mask, torch.zeros(pred.shape, dtype=torch.long, device=device), pred)
            else:
                x = pred
            x = torch.clamp(x, 0, self.num_blocks - 1)
        return x


def create_noisy_blocks(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    num_timesteps: int,
    num_blocks: int,
    noise_block_prob: float = 0.20,
) -> torch.Tensor:
    """Create noisy blocks for diffusion training.

    Standard noise process: at noise positions (determined by timestep),
    blocks are set to air (0).

    Block injection (for noise_block_prob fraction of samples): at noise
    positions, random wrong block IDs are used instead of air. This teaches
    the model to remove incorrectly placed blocks, which occurs during
    sampling with temperature or bad predictions.

    Args:
        clean: [B, GX, GY, GZ] clean block IDs
        timesteps: [B] diffusion timesteps
        num_timesteps: total number of timesteps
        num_blocks: number of block types in vocabulary
        noise_block_prob: probability of injecting random blocks instead of air

    Returns:
        noisy: [B, GX, GY, GZ] noisy block IDs
    """
    B = clean.shape[0]
    device = clean.device

    # Standard noise mask: positions where rand < t/num_timesteps are noised
    noise_mask = torch.rand_like(clean.float()) < (timesteps.float() / num_timesteps).view(B, 1, 1, 1)

    # Determine which samples get random block injection (20% by default).
    # These samples are "clones" of the intermediate step with wrong blocks
    # added, so the model learns to remove incorrectly placed blocks.
    inject_blocks = torch.rand(B, device=device) < noise_block_prob

    # Generate random non-air block IDs for injection (1 to num_blocks-1)
    random_blocks = torch.randint(1, num_blocks, clean.shape, device=device, dtype=clean.dtype)

    # For injected samples: use random blocks at noise positions
    # For non-injected samples: use air (0) at noise positions (standard)
    noise_values = torch.where(
        inject_blocks.view(B, 1, 1, 1),
        random_blocks,
        torch.zeros_like(clean),
    )

    noisy = torch.where(noise_mask, noise_values, clean)
    return noisy


def logit_chunk_plan(features: torch.Tensor,
                     chunk_voxels: int | None = None) -> Tuple[int, int]:
    """(Voxel insgesamt, Voxel pro Chunk) fuer eine Merkmalskarte."""
    voxels = 1
    for size in features.shape[2:]:
        voxels *= int(size)
    if not chunk_voxels:
        return voxels, voxels
    return voxels, max(1, min(voxels, int(chunk_voxels)))


def iter_logit_chunks(model: nn.Module, features: torch.Tensor,
                      chunk_voxels: int | None = None):
    """Logits stueckweise erzeugen: ``(start, end, logits_chunk)``.

    ``features`` ist der Ausgang von ``model.features()`` (Form
    ``[B, channels, GX, GY, GZ]``); der Ausgabekopf ist eine 1x1x1-Konvolution
    und damit punktweise, laesst sich also beliebig entlang der (flach
    gedachten) Voxel-Achse aufteilen. Bei ``chunk_voxels=None`` wird genau ein
    Chunk erzeugt - identisch zum ungechunkten ``forward()``.
    """
    batch = int(features.shape[0])
    channels = int(features.shape[1])
    voxels, step = logit_chunk_plan(features, chunk_voxels)
    flat = features.reshape(batch, channels, voxels)
    for start in range(0, voxels, step):
        end = min(start + step, voxels)
        window = flat[:, :, start:end].reshape(batch, channels, 1, 1, end - start)
        yield start, end, model.out_proj(window)[:, :, 0, 0, :]


def sample_denoising_step(model: nn.Module, features: torch.Tensor, *,
                          temperature: float = 1.0, top_k: int = 0,
                          chunk_voxels: int | None = None) -> torch.Tensor:
    """Block-IDs eines Denoising-Schritts aus den UNet-Merkmalen ziehen.

    ``features`` kommt aus ``model.features()`` (Form ``[B, channels, GX, GY, GZ]``),
    zurueck kommen die Vorhersagen ``[B, GX*GY*GZ]``. Temperatur, Top-K und
    Softmax laufen je Voxel-Chunk; die Voxel werden weiterhin in der Reihenfolge
    0..N-1 abgearbeitet. Die Logits und damit die Wahrscheinlichkeiten sind
    exakt dieselben wie ungechunkt (mit ``top_k=1`` ist das Sampling
    deterministisch und das Ergebnis deshalb bitgleich), nur die konkreten
    Zufallszahlen liegen an einer anderen Stelle des Zufallsgenerators -
    Chunking ist also statistisch gleichwertig, aber nicht bitgleich.

    Warum ueberhaupt: ungechunkt kostet ein Schritt bei Grid 32^3 und 17.690
    Blockzustaenden ``17690 * 32768 * 4 Byte = 2,16 GiB`` - und wegen der
    Temperatur-, Top-K- und Softmax-Kopien liegt zeitweise das Vierfache im
    Speicher. Auf einer 6-GB-Karte endete das nach einem Trainingslauf in
    "Cuda out of memory" (dieselbe Ursache wie beim Training, nur ohne
    Rueckwaerts-Pass).
    """
    batch = int(features.shape[0])
    voxels = 1
    for size in features.shape[2:]:
        voxels *= int(size)
    predictions = torch.empty((batch, voxels), dtype=torch.long,
                              device=features.device)
    for start, end, chunk_logits in iter_logit_chunks(model, features, chunk_voxels):
        blocks = int(chunk_logits.shape[1])
        logits = chunk_logits
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        if int(top_k) > 0:
            k = min(int(top_k), blocks)
            min_k = torch.topk(logits, k, dim=1)[0][:, -1, :].unsqueeze(1)
            logits[logits < min_k] = float("-inf")
        probs = torch.softmax(logits, dim=1)                   # [B, Bloecke, Chunk]
        flat = probs.permute(0, 2, 1).reshape(-1, blocks)      # [B*Chunk, Bloecke]
        predictions[:, start:end] = torch.multinomial(flat, 1).reshape(batch, end - start)
    return predictions



def chunked_weighted_ce_loss(
    model: nn.Module,
    features: torch.Tensor,
    clean: torch.Tensor,
    batch: dict,
    *,
    device: torch.device,
    chunk_voxels: int | None = None,
    scaler=None,
    grad_scale: float = 1.0,
) -> float:
    """Gewichtete Kreuzentropie chunkweise vorwaerts UND rueckwaerts rechnen.

    ``grad_scale`` skaliert nur die *Gradienten* (der zurueckgegebene Loss bleibt
    der echte Mittelwert). Die Gradient-Akkumulation teilt damit den Anteil eines
    Mikro-Batches am Gesamt-Batch zu (siehe ``app/grad_accum.py``), ohne dass die
    angezeigte Verlustkurve verfaelscht wird.

    Rueckgabe ist der Gesamt-Loss ``Summe(ce * w) / Summe(w)`` - identisch zur
    ungechunkten Rechnung, genau wie die Gradienten (nur die Reihenfolge der
    Summation weicht ab).

    Zwei Stufen, damit die Rechenzeit nicht explodiert:
      1. Der Ausgabekopf laeuft in Voxel-Chunks auf ``features.detach()`` und
         sammelt nur den Gradienten bezueglich der Merkmale. Jeder Chunk wird
         sofort rueckwaerts gerechnet und danach freigegeben - deshalb liegt
         immer nur ein Chunk gleichzeitig im Speicher.
      2. EIN ``autograd.backward`` durch den UNet-Rumpf mit dem gesammelten
         Gradienten als Startwert.

    Wuerde man stattdessen pro Chunk ``loss.backward()`` aufrufen, muesste der
    (teure) Backward des Rumpfes fuer jeden Chunk erneut durchlaufen - bei 128
    Chunks ein Vielfaches der Rechenzeit (gemessen: 54 s statt ~3 s pro Batch).

    Der Aufrufer setzt ``model.train()``, ruft ``optimizer.zero_grad()`` vorher
    und danach ``clip_grad_norm_`` / ``optimizer.step()``. Mit AMP wird jeder
    Chunk-Loss skaliert und die Gradienten hinterher einmal entpackt
    (``scaler.unscale_``).

    Warum ueberhaupt: die Logits sind ``[B, num_blocks, X*Y*Z]``; bei Grid 32^3,
    17.690 Blockzustaenden und Batch 16 sind das 34,55 GiB - die Karte meldet
    dann "Tried to allocate 34.55 GiB" und das Training bricht ab. Mit Chunks
    liegt nur ein Haeppchen davon gleichzeitig im Speicher.
    """
    use_amp = scaler is not None
    batch_size = int(clean.shape[0])
    flat_clean = clean.reshape(batch_size, -1)

    # Gewichte siehe app/grad_accum.per_voxel_weight: dieselbe Definition nutzt
    # die Gradient-Akkumulation fuer die Anteile der Mikro-Batches.
    weight = per_voxel_weight(flat_clean, batch, batch_size)
    # Das Gewicht haengt nicht vom Modellausgang ab -> Nenner einmal bestimmen,
    # damit jeder Chunk korrekt skaliert werden kann.
    denominator = weight.sum().clamp_min(1.0)

    # Stufe 1: Kopf in Chunks, Gradient nur bezueglich der Merkmale.
    detached = features.detach().requires_grad_(True)
    total_loss = 0.0
    for start, end, logits in iter_logit_chunks(model, detached, chunk_voxels):
        with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
            ce = torch.nn.functional.cross_entropy(
                logits, flat_clean[:, start:end], reduction="none")
            chunk_loss = (ce * weight[:, start:end]).sum() / denominator
        total_loss += float(chunk_loss.detach())
        if grad_scale != 1.0:
            # Nur der Gradient wird skaliert (Anteil am Gesamt-Batch), nicht die
            # angezeigte Verlustkurve.
            chunk_loss = chunk_loss * float(grad_scale)
        if use_amp:
            scaler.scale(chunk_loss).backward()
        else:
            chunk_loss.backward()

    # Stufe 2: ein einziger Backward durch den Rumpf.
    seed = detached.grad
    if seed is None:
        raise RuntimeError(
            "Keine Chunk-Gradienten berechnet - features/out_proj ohne grad_fn?")
    if features.requires_grad:
        torch.autograd.backward(features, grad_tensors=seed)
    return total_loss


def train_diffusion_step(model, batch, optimizer, device, scaler=None, noise_block_prob=0.20,
                         chunk_voxels=None, accumulate=False, grad_scale=1.0):
    """Ein Trainingsschritt fuer das VoxelDiffusionModel.

    ``chunk_voxels`` begrenzt die Voxel, die gleichzeitig in Logits umgerechnet
    werden (siehe ``chunked_weighted_ce_loss``). ``None`` = alles auf einmal
    (altes Verhalten, nur fuer kleine Vokabulare/Grids geeignet).

    ``accumulate=True`` laesst den Optimizer-Schritt aus (Gradient-Akkumulation):
    der Aufrufer setzt ``zero_grad`` vorher und ruft nach dem letzten Mikro-Batch
    ``app.grad_accum.apply_optimizer_step()``. ``grad_scale`` ist der Anteil des
    Mikro-Batches am Gesamt-Batch.
    """
    model.train()
    clean = torch.clamp(batch["voxel_ids"].to(device, non_blocking=True).long(), 0, model.num_blocks - 1)
    text_ids = batch["prompt_ids"].to(device, non_blocking=True)
    B = clean.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device)
    noisy = create_noisy_blocks(clean, t, model.num_timesteps, model.num_blocks, noise_block_prob)
    use_amp = scaler is not None
    if not accumulate:
        optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
        features = model.features(noisy, t, text_ids)
    loss = chunked_weighted_ce_loss(model, features, clean, batch, device=device,
                                    chunk_voxels=chunk_voxels, scaler=scaler,
                                    grad_scale=grad_scale)
    if accumulate:
        return float(loss)
    apply_optimizer_step(model, optimizer, scaler)
    return float(loss)


def train_transformer_diffusion_step(
    model: TransformerDiffusionModel,
    batch: dict,
    transformer_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    noise_block_prob: float = 0.20,
    chunk_voxels: int | None = None,
    accumulate: bool = False,
    grad_scale: float = 1.0,
) -> float:
    """Training step for TransformerDiffusionModel.

    The transformer_encoder is frozen - only the UNet parameters are trained.
    ``chunk_voxels`` begrenzt die Logits pro Rechenschritt (siehe
    ``chunked_weighted_ce_loss``); ``None`` = ungechunkt.

    ``accumulate``/``grad_scale`` gehoeren zur Gradient-Akkumulation (siehe
    ``app/grad_accum.py``): mit ``accumulate=True`` wird nur rueckwaerts
    gerechnet, ``clip``/``step`` macht der Aufrufer am Ende des Fensters.
    """
    model.train()
    clean = torch.clamp(batch["voxel_ids"].to(device, non_blocking=True).long(), 0, model.num_blocks - 1)
    B = clean.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device)
    noisy = create_noisy_blocks(clean, t, model.num_timesteps, model.num_blocks, noise_block_prob)

    # Encode prompts with frozen transformer
    prompts = batch.get("prompt_text", None)
    if prompts is None:
        raise ValueError("batch must contain 'prompt_text' for transformer encoder")

    with torch.no_grad():
        encoded = transformer_encoder(prompts)
        context = encoded["last_hidden_state"].to(device)
        context_mask = encoded["attention_mask"].to(device)

    use_amp = scaler is not None
    if not accumulate:
        optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
        features = model.features(noisy, t, context, context_mask)
    loss = chunked_weighted_ce_loss(model, features, clean, batch, device=device,
                                    chunk_voxels=chunk_voxels, scaler=scaler,
                                    grad_scale=grad_scale)
    if accumulate:
        return float(loss.detach())
    apply_optimizer_step(model, optimizer, scaler)
    return float(loss.detach())


def train_transformer_diffusion_step_cached(
    model: TransformerDiffusionModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    noise_block_prob: float = 0.20,
    chunk_voxels: int | None = None,
    accumulate: bool = False,
    grad_scale: float = 1.0,
) -> float:
    """Training step for TransformerDiffusionModel using pre-computed hidden states.
    
    No transformer_encoder needed - context comes from the batch directly.
    The cached states are float32 on CPU, so we cast them to match the model's dtype.
    ``chunk_voxels`` begrenzt die Logits pro Rechenschritt (siehe
    ``chunked_weighted_ce_loss``); ``None`` = ungechunkt.

    ``accumulate``/``grad_scale`` gehoeren zur Gradient-Akkumulation (siehe
    ``app/grad_accum.py``): mit ``accumulate=True`` wird nur rueckwaerts
    gerechnet, ``clip``/``step`` macht der Aufrufer am Ende des Fensters.
    """
    model.train()
    clean = torch.clamp(batch["voxel_ids"].to(device, non_blocking=True).long(), 0, model.num_blocks - 1)
    B = clean.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device)
    noisy = create_noisy_blocks(clean, t, model.num_timesteps, model.num_blocks, noise_block_prob)

    # Use pre-computed hidden states from batch
    # Cast to model's dtype to avoid Half/Float mismatch
    model_dtype = next(model.parameters()).dtype
    context = batch["hidden_states"].to(device=device, dtype=model_dtype, non_blocking=True)
    context_mask = batch["attention_masks"].to(device=device, non_blocking=True)

    use_amp = scaler is not None
    if not accumulate:
        optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type if use_amp else "cpu", enabled=use_amp):
        features = model.features(noisy, t, context, context_mask)
    loss = chunked_weighted_ce_loss(model, features, clean, batch, device=device,
                                    chunk_voxels=chunk_voxels, scaler=scaler,
                                    grad_scale=grad_scale)
    if accumulate:
        return float(loss)
    apply_optimizer_step(model, optimizer, scaler)
    return float(loss)
