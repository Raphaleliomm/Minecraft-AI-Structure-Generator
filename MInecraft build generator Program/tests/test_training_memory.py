"""Tests fuer VRAM-Planung und die gechunkte Logits-Berechnung.

Hintergrund (Regression): Das Training der TF-Diffusion brach auf einer
GTX 1060 (6 GB) mit

    Cuda out of memory. Tried to allocate 34.55 GiB.

ab. Ursache war der Logits-Tensor ``[batch, num_blocks, X*Y*Z]``:
16 * 17.690 Blockzustaende * 32.768 Voxel * 4 Byte = 34,55 GiB. Diese Tests
sichern zu, dass
  * diese Zahl vom Planer erkannt und die Batch-Groesse reduziert wird,
  * die Chunk-Groesse in den freien Speicher passt,
  * die Self-Attention des VoxelTransformers (Voxel^2) als Grid-Limit erkannt
    wird,
  * Chunking Loss und Gradienten NICHT veraendert (Numerik-Paritaet).
"""
from __future__ import annotations

import torch

from app.diffusion_model import (
    TransformerDiffusionModel,
    VoxelDiffusionModel,
    iter_logit_chunks,
    train_diffusion_step,
    train_transformer_diffusion_step_cached,
)
from app.training_memory import (
    attention_bytes,
    describe_training_plan,
    format_gib,
    logits_bytes,
    plan_chunk_voxels,
    plan_diffusion_training,
    plan_transformer_training,
    vram_info,
)

GIB = 1024 ** 3
# Zahlen aus dem realen Trainingslauf (runs/tf_diffusion_cached_Phi-3.5-mini_*)
REAL_VOCAB = 17690
REAL_GRID = (32, 32, 32)
REAL_BATCH = 16


def test_logits_size_matches_the_reported_oom():
    """16 * 17.690 * 32^3 * 4 Byte = genau die 34,55 GiB aus der Fehlermeldung."""
    size = logits_bytes(REAL_BATCH, REAL_VOCAB, 32 ** 3)
    assert size == 16 * 17690 * 32768 * 4
    assert format_gib(size) == "34.55"


def test_transformer_attention_is_quadratic_in_voxels():
    """32^3: 6 Heads * 32768^2 * 4 Byte = 24 GiB pro Batch und Matrix."""
    per_matrix = attention_bytes(1, 6, 32 ** 3, 32 ** 3)
    assert format_gib(per_matrix) == "24.00"
    # Grid 16 hat 64x kleinere Attention-Flaechen
    assert attention_bytes(1, 6, 16 ** 3, 16 ** 3) * 64 == per_matrix


def test_chunk_plan_fits_the_budget():
    """Die gechunkte Logits-Berechnung muss in das Restbudget passen."""
    voxels = 32 ** 3
    chunk = plan_chunk_voxels(voxels, 16, REAL_VOCAB, 3 * GIB, overlap_bytes=0)
    assert 32 <= chunk < voxels
    assert chunk & (chunk - 1) == 0          # Zweierpotenz: gleich grosse Chunks
    per_voxel = 16 * REAL_VOCAB * 4 * 3.0    # Logits + Log-Softmax + Gradient
    assert chunk * per_voxel <= 3 * GIB


def test_chunk_plan_keeps_small_grids_unchunked():
    voxels = 16 ** 3
    assert plan_chunk_voxels(voxels, 4, 253, 2 * GIB) == voxels


def test_real_config_is_chunked_on_a_six_gigabyte_card():
    """Der reale Fall: 34,55 GiB Logits -> gechunkt statt am Stueck.

    Budget = 85 % der freien 6 GB (wie auf der GTX 1060 im Fehlerfall).
    """
    plan = plan_diffusion_training(
        REAL_GRID, REAL_BATCH, REAL_VOCAB, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=int(5.1 * GIB),
    )
    assert format_gib(plan.logits_bytes) == "34.55"
    assert plan.chunked
    assert plan.num_chunks > 1
    assert plan.fits
    assert plan.batch_size == REAL_BATCH          # Batch bleibt unveraendert
    # Kein einzelner Tensor (Logits, Log-Softmax, Gradient) reisst das Budget
    assert plan.chunk_voxels * REAL_BATCH * REAL_VOCAB * 4 * 3 <= plan.budget_bytes
    assert "VRAM-Plan" in plan.describe("de")


def test_tight_budget_reduces_the_batch_instead_of_crashing():
    """Wenn der VRAM knapp ist, wird der Batch verkleinert - nicht abgebrochen."""
    plan = plan_diffusion_training(
        REAL_GRID, 64, REAL_VOCAB, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=int(0.9 * GIB),
    )
    assert plan.fits
    assert plan.reduced
    assert plan.batch_size < 64
    assert "reduziert" in (plan.suggestion or "")


def test_tight_budget_keeps_the_effective_batch_via_accumulation():
    """Verkleinerter Batch + Akkumulation = derselbe Batch pro Optimizer-Schritt.

    Genau der Fall aus der Meldung "es reduziert meine Batch-Groesse": mit
    Gradient-Akkumulation laeuft das Training weiterhin mit dem gewuenschten
    Batch (hier 64), nur in mehreren Mikro-Batches.
    """
    budget = int(0.9 * GIB)
    plan = plan_diffusion_training(
        REAL_GRID, 64, REAL_VOCAB, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=budget,
    )
    assert plan.fits and plan.reduced
    assert plan.accumulated
    assert plan.effective_batch_size == 64           # Wunsch-Batch bleibt
    assert plan.micro_batch_size == plan.batch_size  # physisch kleiner
    assert plan.accum_steps * plan.batch_size >= 64
    assert plan.accum_steps == -(-64 // plan.batch_size)
    assert "Gradient Accumulation" in (plan.suggestion or "")
    assert "effektiv 64" in plan.describe("de")
    assert "effective 64" in plan.describe("en")

    # Ohne Akkumulation bleibt es beim alten Verhalten (reines Verkleinern).
    plain = plan_diffusion_training(
        REAL_GRID, 64, REAL_VOCAB, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=budget, allow_accumulation=False,
    )
    assert plain.batch_size == plan.batch_size
    assert not plain.accumulated
    assert plain.effective_batch_size == plain.batch_size


def test_transformer_plan_accumulates_without_changing_the_goal():
    """Auch der Transformer (Attention waechst mit Batch) nutzt die Akkumulation."""
    plan = plan_transformer_training(
        (16, 16, 16), 16, REAL_VOCAB, device=torch.device("cuda"),
        d_model=192, nhead=6, num_layers=5, dim_feedforward=768,
        budget_bytes=int(3.0 * GIB), window_size=0,
    )
    assert plan.fits and plan.reduced
    assert plan.batch_size == 1                  # physisch passt nur ein Beispiel
    assert plan.accum_steps == 16                # ...aber 16 pro Optimizer-Schritt
    assert plan.effective_batch_size == 16
    assert plan.accumulated


def test_endless_grid_cannot_fit_even_with_batch_one():
    """Grid 256^3 passt auf keine 6-GB-Karte - klare Meldung mit Vorschlag."""
    plan = plan_diffusion_training(
        (256, 256, 256), 1, REAL_VOCAB, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=int(5.1 * GIB),
    )
    assert plan.fits is False
    assert plan.limit in {"activations", "params"}
    assert "Grid" in (plan.suggestion or "")
    assert "VRAM zu klein" in describe_training_plan(plan, "de")


def test_transformer_plan_refuses_huge_attention_matrices():
    """Grid 32^3 sprengt jeden 6-GB-VRAM - der Plan bricht mit Hinweis ab."""
    plan = plan_transformer_training(
        REAL_GRID, 4, REAL_VOCAB, device=torch.device("cuda"),
        d_model=192, nhead=6, num_layers=5, dim_feedforward=768,
        budget_bytes=int(5.1 * GIB), window_size=0,
    )
    assert plan.fits is False
    assert plan.limit == "attention"
    assert plan.activation_bytes >= 144 * GIB  # 6 Matrizen a 24 GiB, plus Text
    assert "Grid 16x16x16" in (plan.suggestion or "")


def test_transformer_plan_allows_small_grid():
    plan = plan_transformer_training(
        (16, 16, 16), 2, 253, device=torch.device("cuda"),
        d_model=64, nhead=4, num_layers=3, dim_feedforward=128,
        budget_bytes=int(5.1 * GIB),
    )
    assert plan.fits is True
    assert plan.chunk_voxels == plan.voxels


def test_vram_info_cpu_has_a_budget():
    info = vram_info(torch.device("cpu"))
    assert info.device_type == "cpu"
    assert info.budget_bytes > 0


def test_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MINECRAFT_AI_VRAM_GUARD", "0")
    plan = plan_diffusion_training(REAL_GRID, REAL_BATCH, REAL_VOCAB,
                                   device=torch.device("cpu"))
    assert plan.fits is True
    assert "deaktiviert" in (plan.suggestion or "")


def test_chunk_size_can_be_forced_by_env(monkeypatch):
    """MINECRAFT_AI_LOGITS_CHUNK_VOXELS erzwingt eine feste Chunk-Groesse.

    Ohne die Variable waehlt der Plan die Groesse selbst (hier 512 Voxel);
    der Override wird auf 1..Voxel begrenzt und ersetzt nur die Chunk-Wahl -
    ein unbrauchbarer Wert wird ignoriert statt das Training zu verhindern.
    """
    monkeypatch.setenv("MINECRAFT_AI_LOGITS_CHUNK_VOXELS", "256")
    assert plan_chunk_voxels(32 ** 3, 16, REAL_VOCAB, 3 * GIB) == 256
    plan = plan_diffusion_training(
        REAL_GRID, REAL_BATCH, REAL_VOCAB, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=int(5.1 * GIB),
    )
    assert plan.chunk_voxels == 256
    assert plan.num_chunks == 32 ** 3 // 256
    assert plan.fits

    # kleiner als 16^3 wird auf die Voxelzahl des Grids begrenzt
    monkeypatch.setenv("MINECRAFT_AI_LOGITS_CHUNK_VOXELS", "999999")
    assert plan_chunk_voxels(16 ** 3, 4, 253, 2 * GIB) == 16 ** 3

    # Unsinn (0, negativ, Text) -> automatische Planung
    for bad in ("0", "-8", "abc", ""):
        monkeypatch.setenv("MINECRAFT_AI_LOGITS_CHUNK_VOXELS", bad)
        assert plan_chunk_voxels(32 ** 3, 16, REAL_VOCAB, 3 * GIB) == 512



# ─── Numerik: Chunking darf nichts aendern ───

NUM_BLOCKS = 12
GRID = (4, 4, 4)
B, S, D = 3, 5, 6


def _batch():
    generator = torch.Generator().manual_seed(7)
    return {
        "voxel_ids": torch.randint(0, NUM_BLOCKS, (B, *GRID), generator=generator),
        "hidden_states": torch.randn(B, S, D, generator=generator),
        "attention_masks": torch.ones(B, S),
        "sample_weight": torch.rand(B, generator=generator) + 0.5,
        "per_block_weight": torch.full((B,), 1.0),
        "per_air_weight": torch.full((B,), 3.0),
    }


def _transformer_diffusion() -> TransformerDiffusionModel:
    torch.manual_seed(1)
    return TransformerDiffusionModel(num_blocks=NUM_BLOCKS, grid_size=GRID, d_model=8,
                                     channels=4, channel_multipliers=(1, 2),
                                     num_timesteps=5, context_dim=D,
                                     cross_attn_heads=2, context_proj_dim=8)


def _voxel_diffusion() -> VoxelDiffusionModel:
    torch.manual_seed(1)
    return VoxelDiffusionModel(num_blocks=NUM_BLOCKS, text_vocab_size=10, grid_size=GRID,
                               d_model=8, d_text=4, channels=4,
                               channel_multipliers=(1, 2), num_timesteps=5)


def test_iter_logit_chunks_matches_unchunked_logits():
    model = _transformer_diffusion()
    features = model.features(torch.zeros(B, *GRID, dtype=torch.long),
                              torch.zeros(B, dtype=torch.long),
                              torch.zeros(B, S, D), torch.ones(B, S))
    full = model.out_proj(features)
    seen = torch.zeros(GRID[0] * GRID[1] * GRID[2], dtype=torch.long)
    for start, end, chunk in iter_logit_chunks(model, features, 5):
        assert chunk.shape == (B, NUM_BLOCKS, end - start)
        seen[start:end] += 1
        assert torch.allclose(chunk, full.reshape(B, NUM_BLOCKS, -1)[:, :, start:end])
    assert int(seen.sum()) == seen.numel()      # jede Voxel genau einmal


def _run_step(step, builder, chunk, use_prompt_ids=False):
    model = builder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _batch()
    if use_prompt_ids:
        batch["prompt_ids"] = torch.randint(0, 10, (B, 4))
    torch.manual_seed(11)
    loss = step(model, batch, optimizer, torch.device("cpu"), chunk_voxels=chunk)
    grads = {name: param.detach().clone() for name, param in model.named_parameters()}
    return loss, grads


def test_cached_step_loss_and_gradients_do_not_depend_on_chunking():
    loss_full, grads_full = _run_step(train_transformer_diffusion_step_cached,
                                      _transformer_diffusion, None)
    loss_chunked, grads_chunked = _run_step(train_transformer_diffusion_step_cached,
                                            _transformer_diffusion, 3)
    assert abs(loss_full - loss_chunked) < 1e-5
    for name, grad in grads_full.items():
        assert torch.allclose(grad, grads_chunked[name], atol=1e-6), name


def test_voxel_diffusion_step_loss_and_gradients_do_not_depend_on_chunking():
    loss_full, grads_full = _run_step(train_diffusion_step, _voxel_diffusion, None,
                                      use_prompt_ids=True)
    loss_chunked, grads_chunked = _run_step(train_diffusion_step, _voxel_diffusion, 3,
                                            use_prompt_ids=True)
    assert abs(loss_full - loss_chunked) < 1e-5
    for name, grad in grads_full.items():
        assert torch.allclose(grad, grads_chunked[name], atol=1e-6), name
