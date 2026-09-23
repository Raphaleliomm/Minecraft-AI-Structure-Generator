"""Tests fuer die Gradient-Akkumulation (``app/grad_accum.py``).

Regression: der VRAM-Guard hat die Batch-Groesse still verkleinert - das
Training lief dann mit weniger Beispielen pro Schritt als eingestellt (Default
ist z.B. Batch 16). Mit der Akkumulation bleibt die effektive Batch-Groesse
erhalten: der Batch wird in Mikro-Batches geteilt, die Gradienten werden mit
dem Anteil ``size / total`` gewichtet aufsummiert und der Optimizer schreitet
genau einmal pro Batch.

Diese Tests sichern zu, dass
  * die Aufteilung exakt und gleichmaessig ist (Summe = Batch-Groesse),
  * die Teilung des Batch-Dictionarys Tensoren UND Textlisten erwischt,
  * die aufsummierten Mikro-Batch-Gradienten dem Gradienten des grossen Batches
    entsprechen (gleiche Lernrate, kein Umrechnen noetig),
  * ein Schritt mit ``accumulate=True`` den Optimizer nicht anfasst.
"""
from __future__ import annotations

import pytest
import torch

from app.diffusion_model import (
    TransformerDiffusionModel,
    VoxelDiffusionModel,
    chunked_weighted_ce_loss,
    train_diffusion_step,
    train_transformer_diffusion_step_cached,
)
from app.grad_accum import (
    ENV,
    accumulation_enabled,
    apply_optimizer_step,
    batch_total_size,
    iter_micro_batches,
    micro_batch_sizes,
    per_sample_weight_mass,
    split_batch,
)

NUM_BLOCKS = 12
GRID = (4, 4, 4)
S, D = 5, 6
B = 4


def _batch(batch_size: int = B):
    generator = torch.Generator().manual_seed(7)
    return {
        "voxel_ids": torch.randint(0, NUM_BLOCKS, (batch_size, *GRID), generator=generator),
        "hidden_states": torch.randn(batch_size, S, D, generator=generator),
        "attention_masks": torch.ones(batch_size, S),
        "sample_weight": torch.rand(batch_size, generator=generator) + 0.5,
        "per_block_weight": torch.full((batch_size,), 1.0),
        "per_air_weight": torch.full((batch_size,), 3.0),
    }


def _transformer_diffusion() -> TransformerDiffusionModel:
    torch.manual_seed(1)
    model = TransformerDiffusionModel(num_blocks=NUM_BLOCKS, grid_size=GRID, d_model=8,
                                      channels=4, channel_multipliers=(1, 2),
                                      num_timesteps=5, context_dim=D,
                                      cross_attn_heads=2, context_proj_dim=8)
    model.eval()          # Dropout aus: die Features sollen vergleichbar sein
    return model


def _voxel_diffusion() -> VoxelDiffusionModel:
    torch.manual_seed(1)
    model = VoxelDiffusionModel(num_blocks=NUM_BLOCKS, text_vocab_size=10, grid_size=GRID,
                                d_model=8, d_text=4, channels=4,
                                channel_multipliers=(1, 2), num_timesteps=5)
    model.eval()
    return model


# ─── Aufteilung ───

def test_micro_batch_sizes_are_even_and_exact():
    """Summe = Batch-Groesse, nichts ueber der physischen Groesse, kein Rest-Batch."""
    assert micro_batch_sizes(10, 4) == (4, 3, 3)
    assert micro_batch_sizes(16, 4) == (4, 4, 4, 4)
    assert micro_batch_sizes(4, 8) == (4,)          # passt ungechunkt
    assert micro_batch_sizes(5, 5) == (5,)
    assert micro_batch_sizes(1, 1) == (1,)
    for total, max_size in ((7, 3), (64, 4), (9, 2), (3, 1), (64, 63)):
        sizes = micro_batch_sizes(total, max_size)
        assert sum(sizes) == total
        assert max(sizes) <= max_size
        assert max(sizes) - min(sizes) <= 1


def test_split_batch_keeps_tensors_texts_and_scalars():
    batch = {
        "voxel_ids": torch.arange(4 * 2).reshape(4, 2),
        "prompt_text": ["a", "b", "c", "d"],
        "epoch": 3,
    }
    parts = split_batch(batch, (3, 1))
    assert [part["voxel_ids"].tolist() for part in parts] == [
        [[0, 1], [2, 3], [4, 5]], [[6, 7]]]
    assert [part["prompt_text"] for part in parts] == [["a", "b", "c"], ["d"]]
    assert all(part["epoch"] == 3 for part in parts)


def test_iter_micro_batches_shares_sum_to_one_hundred_percent():
    batch = _batch()
    seen = []
    total_share = 0.0
    for micro, share in iter_micro_batches(batch, 3):
        seen.append(batch_total_size(micro))
        total_share += share
    assert seen == [2, 2]
    assert abs(total_share - 1.0) < 1e-9

    # Passt der Batch, gibt es genau einen Durchlauf mit Anteil 1.0 (schneller Pfad).
    single = list(iter_micro_batches(batch, B))
    assert len(single) == 1
    assert single[0][0] is batch
    assert single[0][1] == 1.0


def test_iter_micro_batches_can_share_by_weight_mass():
    """Mit ``sample_mass`` zaehlt die Gewichtsmasse, nicht die Beispielzahl."""
    batch = _batch()
    masses = torch.tensor([3.0, 1.0, 1.0, 1.0])
    shares = [share for _, share in iter_micro_batches(batch, 2, sample_mass=masses)]
    assert shares == [pytest.approx(4 / 6), pytest.approx(2 / 6)]

    # Unbrauchbare Masse (falsche Laenge) faellt auf die Beispielzahl zurueck.
    shares = [share for _, share in iter_micro_batches(batch, 2,
                                                       sample_mass=torch.tensor([1.0]))]
    assert shares == [pytest.approx(0.5), pytest.approx(0.5)]


def test_per_sample_weight_mass_matches_the_loss_weighting():
    """Die Masse ist genau der Nenner der gewichteten Kreuzentropie."""
    batch = _batch()
    ids = batch["voxel_ids"]
    air = (ids == 0).sum(dim=(1, 2, 3)).to(torch.float32)
    blocks = float(ids[0].numel()) - air

    masses = per_sample_weight_mass(ids, batch)
    expected = batch["sample_weight"] * (blocks * 1.0 + air * 3.0)
    assert masses.shape == (B,)
    assert torch.allclose(masses, expected)


def test_accumulation_is_on_by_default_and_switchable(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert accumulation_enabled()
    for value in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv(ENV, value)
        assert not accumulation_enabled(), value
    monkeypatch.setenv(ENV, "1")
    assert accumulation_enabled()


# ─── Gradienten: Akkumulation == ein grosser Batch ───

def _full_batch_grads(model, batch):
    """Gradienten eines Durchlaufs ueber den ganzen Batch (Referenz)."""
    clean = torch.clamp(batch["voxel_ids"], 0, NUM_BLOCKS - 1)
    timesteps = torch.zeros(B, dtype=torch.long)
    masks = torch.ones(B, S)
    model.zero_grad(set_to_none=True)
    loss = chunked_weighted_ce_loss(
        model, model.features(clean, timesteps, batch["hidden_states"], masks),
        clean, batch, device=torch.device("cpu"), chunk_voxels=5)
    grads = {name: param.grad.detach().clone()
             for name, param in model.named_parameters() if param.grad is not None}
    return loss, grads


def _accumulated_grads(model, batch, max_size: int, use_mass: bool):
    """Dasselbe ueber Mikro-Batches - jeder mit seinem Anteil am Batch.

    Jeder Mikro-Batch bekommt einen eigenen Vorwaertslauf (wie im echten
    Trainingslauf); ein gemeinsamer Graph liesse sich nicht zweimal rueckwaerts
    rechnen.
    """
    extended = {**batch, "timesteps": torch.zeros(B, dtype=torch.long),
                "masks": torch.ones(B, S)}
    masses = per_sample_weight_mass(batch["voxel_ids"], batch) if use_mass else None
    model.zero_grad(set_to_none=True)
    loss = 0.0
    for part, share in iter_micro_batches(extended, max_size, sample_mass=masses):
        clean = torch.clamp(part["voxel_ids"], 0, NUM_BLOCKS - 1)
        features = model.features(clean, part["timesteps"], part["hidden_states"],
                                  part["masks"])
        loss += share * chunked_weighted_ce_loss(
            model, features, clean, part, device=torch.device("cpu"),
            chunk_voxels=5, grad_scale=share)
    grads = {name: param.grad.detach().clone()
             for name, param in model.named_parameters() if param.grad is not None}
    return loss, grads


def _gradient_deviation(reference, other) -> float:
    """Groesste Abweichung bezogen auf den groessten Gradientenwert des Modells.

    Einzelne Parameter koennen fast vollstaendig aus der Differenz grosser
    Beitraege bestehen (z.B. ein ``skip.bias``); dort ist die relative
    Abweichung von float32-Rundungen gepraegt, obwohl der Schritt praktisch
    identisch ist. Der Bezug auf das Modellmaximum misst deshalb den Einfluss
    auf den Schritt.
    """
    scale = 0.0
    for grad in reference.values():
        scale = max(scale, float(grad.abs().max()))
    scale += 1e-12
    return max(float((other[name] - grad).abs().max()) / scale
               for name, grad in reference.items())


def test_accumulated_gradients_match_one_big_batch():
    """Zwei Mikro-Batches ergeben den Gradienten von Batch 4 (= Referenzlauf).

    Ohne diese Eigenschaft muesste die Lernrate beim Verkleinern angepasst
    werden - genau das ist der Grund fuer die Akkumulation.
    """
    model = _transformer_diffusion()
    batch = _batch()
    full_loss, full_grads = _full_batch_grads(model, batch)
    assert full_grads

    accumulated_loss, accumulated_grads = _accumulated_grads(
        model, batch, max_size=B // 2, use_mass=True)
    assert micro_batch_sizes(B, B // 2) == (2, 2)
    # Nur die Reihenfolge der Summation weicht ab (float32) - sonst exakt.
    # 1e-3 = 0.1 % des groessten Gradientenwerts, gemessen sind es 3e-4.
    assert abs(accumulated_loss - full_loss) / full_loss < 1e-4
    assert _gradient_deviation(full_grads, accumulated_grads) < 1e-3


def test_weight_masses_are_needed_for_exact_gradients():
    """Ohne Gewichtsmasse waeren die Gradienten deutlich andere.

    Der Loss wiegt Luft mit 3 (bzw. 75 im echten Training) und jedes Beispiel
    zusaetzlich mit ``sample_weight``. Teilt man den Batch nach Beispielzahl,
    bekommt ein Haeppchen aus luftigen Strukturen denselben Einfluss wie eines
    aus kompakten Bloecken - die summierten Gradienten sind dann nicht mehr die
    des grossen Batches. Dieser Test haelt fest, warum die Masse gebraucht wird.
    """
    model = _transformer_diffusion()
    batch = _batch()
    full_loss, full_grads = _full_batch_grads(model, batch)

    count_loss, count_grads = _accumulated_grads(model, batch, B // 2, use_mass=False)
    assert _gradient_deviation(full_grads, count_grads) > 0.01

    # Mit der Gewichtsmasse stimmt es dagegen exakt (bis auf float32-Rundung).
    mass_loss, mass_grads = _accumulated_grads(model, batch, B // 2, use_mass=True)
    assert abs(mass_loss - full_loss) / full_loss < 1e-4
    assert _gradient_deviation(full_grads, mass_grads) < 1e-3


# ─── Optimizer: genau ein Schritt pro Batch ───

def test_accumulated_step_defers_the_optimizer_step():
    """``accumulate=True`` sammelt nur - die Gewichte aendern sich erst nach dem Fenster."""
    model = _transformer_diffusion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    batch = _batch()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    shares = []
    for micro, share in iter_micro_batches(batch, B // 2):
        shares.append(share)
        train_transformer_diffusion_step_cached(
            model, micro, optimizer, torch.device("cpu"),
            chunk_voxels=5, accumulate=True, grad_scale=share)

    assert len(shares) == 2 and shares[0] < 1.0            # wirklich akkumuliert
    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), before[name]), name
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0
               for param in model.parameters())

    apply_optimizer_step(model, optimizer)
    changed = [name for name, param in model.named_parameters()
               if not torch.equal(param.detach(), before[name])]
    assert changed


def test_voxel_diffusion_step_accumulates_too():
    """Gleiches Verhalten fuer das VoxelDiffusionModel (eigener Trainingsschritt)."""
    model = _voxel_diffusion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    batch = _batch()
    batch["prompt_ids"] = torch.randint(0, 10, (B, 4))
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    for micro, share in iter_micro_batches(batch, B // 2):
        train_diffusion_step(model, micro, optimizer, torch.device("cpu"),
                             chunk_voxels=5, accumulate=True, grad_scale=share)

    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), before[name]), name
    apply_optimizer_step(model, optimizer)
    assert any(not torch.equal(param.detach(), before[name])
               for name, param in model.named_parameters())


def test_single_pass_keeps_the_old_behaviour():
    """Ohne Akkumulation (ein Mikro-Batch) laeuft der Schritt wie vorher durch."""
    model = _transformer_diffusion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    batch = _batch()
    loss = train_transformer_diffusion_step_cached(
        model, batch, optimizer, torch.device("cpu"), chunk_voxels=5)
    assert loss > 0
    assert any(not torch.equal(param.detach(), before[name])
               for name, param in model.named_parameters())
