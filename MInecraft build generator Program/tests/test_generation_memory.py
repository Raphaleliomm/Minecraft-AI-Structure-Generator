"""Tests fuer die Speicherplanung *und* die gechunkte Logits-Berechnung der
Generierung.

Hintergrund (Regression): Nach einem Trainingslauf endete der Klick auf
"Generieren" im Tab *Generate* mit

    Cuda out of memory.

Das Training war bereits gechunkt, ``sample()`` rechnete die Logits aber wieder
am Stueck: Grid 32^3 mit 17.690 Blockzustaenden sind ``17690 * 32768 * 4 Byte =
2,16 GiB`` pro Tensor und mit Temperatur-, Top-K-, Softmax- und Permute-Kopie
kurzzeitig ~8,6 GiB. Dazu kam, dass der Caching-Allocator den VRAM-Peak des
Trainings reserviert haelt (``mem_get_info`` meldet ihn als belegt).

Diese Tests sichern zu, dass
  * der Planer die Chunk-Groesse aus dem freien Budget ableitet,
  * Chunking die Logits nicht veraendert (deterministisches Sampling mit
    ``top_k=1`` bzw. Argmax ist bitgleich),
  * die Generierung den Chunk-Pfad wirklich benutzt (``sample()`` und
    ``VoxelTransformer.generate()``).
"""
from __future__ import annotations

import torch

from app.diffusion_model import (
    TransformerDiffusionModel,
    VoxelDiffusionModel,
    iter_logit_chunks,
    sample_denoising_step,
)
from app.training_memory import (
    describe_generation_plan,
    format_gib,
    logits_bytes,
    plan_generation,
    transformer_estimates,
    transformer_generation_bytes,
    unet_activation_bytes,
    unet_inference_bytes,
)
from model import SharedWeightVoxelTransformer

GIB = 1024 ** 3
# Zahlen aus dem realen Trainingslauf (runs/tf_diffusion_cached_Phi-3.5-mini_*)
REAL_VOCAB = 17690
REAL_GRID = (32, 32, 32)
# Freier Speicher einer GTX 1060, nachdem das Training Speicher hinterlassen hat
REAL_BUDGET = int(0.9 * GIB)


# ─── Planung ───

def test_generation_logits_match_the_reported_oom():
    """Grid 32^3 mit 17.690 Blockzustaenden: 2,16 GiB pro Logits-Kopie."""
    size = logits_bytes(1, REAL_VOCAB, 32 ** 3)
    assert size == 17690 * 32768 * 4
    assert format_gib(size) == "2.16"


def test_generation_plan_chunks_the_real_case():
    """Der reale Fall: der ungechunkte Tensor passt nicht, gechunkt schon."""
    plan = plan_generation(REAL_GRID, REAL_VOCAB, device=None, batch_size=1,
                           activation_bytes=0, budget_bytes=REAL_BUDGET)
    assert plan.fits
    assert plan.chunked
    assert plan.chunk_voxels < plan.voxels
    assert plan.chunk_voxels & (plan.chunk_voxels - 1) == 0      # Zweierpotenz
    assert plan.logits_bytes == logits_bytes(1, REAL_VOCAB, 32 ** 3)
    # Jeder Chunk (Logits + Log-Softmax/Softmax-Kopien) passt in das Budget
    assert plan.estimated_bytes <= plan.budget_bytes
    assert "Chunks a" in plan.describe("de")


def test_generation_plan_fits_small_grids_unchunked():
    plan = plan_generation((16, 16, 16), 253, device=None, budget_bytes=2 * GIB)
    assert plan.fits
    assert not plan.chunked
    assert plan.chunk_voxels == 16 ** 3
    assert plan.num_chunks == 1


def test_generation_plan_reports_hopeless_configurations():
    """Ohne freien Speicher gibt es eine Meldung mit Vorschlag statt OOM."""
    plan = plan_generation(REAL_GRID, REAL_VOCAB, device=None, budget_bytes=32,
                           activation_bytes=16 * 1024 ** 2)
    assert not plan.fits
    assert plan.suggestion
    text = describe_generation_plan(plan, "de")
    assert "VRAM zu klein" in text


def test_generation_plan_respects_the_guard_switch(monkeypatch):
    monkeypatch.setenv("MINECRAFT_AI_VRAM_GUARD", "0")
    plan = plan_generation(REAL_GRID, REAL_VOCAB, device=None,
                           budget_bytes=REAL_BUDGET)
    assert plan.fits                       # Guard aus -> wie vorher ungechunkt
    assert not plan.chunked
    assert "Guard deaktiviert" in plan.describe("de")


def test_generation_plan_respects_the_chunk_override(monkeypatch):
    monkeypatch.setenv("MINECRAFT_AI_LOGITS_CHUNK_VOXELS", "256")
    plan = plan_generation(REAL_GRID, REAL_VOCAB, device=None,
                           budget_bytes=REAL_BUDGET)
    assert plan.chunk_voxels == 256
    assert plan.num_chunks == 32 ** 3 // 256


def test_inference_activations_are_smaller_than_training():
    """Ohne Autograd braucht der UNet-Vorwaerts-Lauf deutlich weniger."""
    training = unet_activation_bytes(1, 16, (1, 2, 2), REAL_GRID, 2, 64)
    inference = unet_inference_bytes(1, 16, (1, 2, 2), REAL_GRID, 2, 64)
    assert 0 < inference < training


def test_transformer_generation_attention_is_the_limit():
    """Die Self-Attention (Voxel^2) laesst sich nicht chunken.

    32^3: 6 Heads * 32768^2 * 4 Byte = 24 GiB pro Layer - auch ohne Backward
    passt das nicht in 5,1 GiB, also meldet der Plan das statt abzustuerzen.
    """
    attn, rest = transformer_generation_bytes(REAL_GRID, 1, REAL_VOCAB, 192, 6, 5, 768, window_size=0)
    assert attn >= 120 * GIB  # 5 Layer * 24 GiB, plus Text-Attention
    train_attn, _, _ = transformer_estimates(REAL_GRID, 1, REAL_VOCAB, 192, 6, 5, 768, window_size=0)
    assert attn < train_attn                            # Training haelt Kopien
    plan = plan_generation(REAL_GRID, REAL_VOCAB, device=None,
                           activation_bytes=attn + rest,
                           budget_bytes=int(5.1 * GIB))
    assert not plan.fits
    assert plan.suggestion

    # 16^3 ist 64x kleiner und passt mit kleinem Vokabular
    attn16, rest16 = transformer_generation_bytes((16, 16, 16), 1, 253, 128, 4, 4, 512, window_size=0)
    assert plan_generation((16, 16, 16), 253, device=None,
                           activation_bytes=attn16 + rest16,
                           budget_bytes=int(5.1 * GIB)).fits


# ─── Numerik: Chunking darf das Sampling nicht veraendern ───

NUM_BLOCKS = 23
GRID = (5, 5, 5)
BATCH, PROMPT_LEN, CONTEXT_DIM = 2, 4, 8


def _diffusion_model() -> VoxelDiffusionModel:
    torch.manual_seed(1)
    return VoxelDiffusionModel(num_blocks=NUM_BLOCKS, text_vocab_size=11,
                               grid_size=GRID, d_model=8, d_text=6, channels=8,
                               channel_multipliers=(1, 2), num_timesteps=4)


def _tf_diffusion_model() -> TransformerDiffusionModel:
    torch.manual_seed(2)
    return TransformerDiffusionModel(
        num_blocks=NUM_BLOCKS, grid_size=GRID, d_model=8, channels=8,
        channel_multipliers=(1, 2), num_timesteps=4, context_dim=CONTEXT_DIM,
        cross_attn_heads=2, context_proj_dim=16)


def _transformer() -> SharedWeightVoxelTransformer:
    torch.manual_seed(3)
    return SharedWeightVoxelTransformer(
        text_vocab_size=11, block_vocab_size=NUM_BLOCKS, grid_size=GRID,
        d_model=8, nhead=2, num_layers=2, dim_feedforward=16, dropout=0.0)


def test_chunked_logits_are_bit_identical():
    """Der 1x1x1-Ausgabekopf ist punktweise -> Chunks aendern nichts."""
    model = _tf_diffusion_model().eval()
    context = torch.randn(BATCH, PROMPT_LEN, CONTEXT_DIM)
    features = model.features(
        torch.zeros(BATCH, *GRID, dtype=torch.long),
        torch.zeros(BATCH, dtype=torch.long), context)
    with torch.no_grad():
        reference = model.out_proj(features).flatten(2)
        parts = [chunk.reshape(chunk.shape[0], chunk.shape[1], -1)
                 for _, _, chunk in iter_logit_chunks(model, features, 7)]
    assert torch.equal(reference, torch.cat(parts, dim=2))


def test_diffusion_sampling_is_chunk_size_independent():
    """top_k=1 macht das Sampling deterministisch: nur das Argmax hat Masse 1.

    Frei gewaehlte Zufallszahlen liegen nach dem Chunking an einer anderen
    Stelle des Zufallsgenerators (statistisch gleichwertig, nicht bitgleich) -
    die Logits/Wahrscheinlichkeiten sind aber identisch.
    """
    model = _diffusion_model()
    text_ids = torch.randint(0, 11, (BATCH, PROMPT_LEN))
    torch.manual_seed(5)
    full = model.sample(text_ids, num_steps=4, temperature=0.7, top_k=1)
    torch.manual_seed(5)
    chunked = model.sample(text_ids, num_steps=4, temperature=0.7, top_k=1,
                           chunk_voxels=6)
    assert torch.equal(full, chunked)


def test_tf_diffusion_sampling_is_chunk_size_independent():
    model = _tf_diffusion_model()
    context = torch.randn(BATCH, PROMPT_LEN, CONTEXT_DIM)
    mask = torch.ones(BATCH, PROMPT_LEN, dtype=torch.long)
    torch.manual_seed(7)
    full = model.sample(context, mask, num_steps=4, temperature=0.8, top_k=1)
    torch.manual_seed(7)
    chunked = model.sample(context, mask, num_steps=4, temperature=0.8, top_k=1,
                           chunk_voxels=6)
    assert torch.equal(full, chunked)


def test_sample_defaults_are_unchanged():
    """Ohne chunk_voxels bleibt alles wie vorher (Form, Typ, Wertebereich)."""
    model = _diffusion_model()
    text_ids = torch.randint(0, 11, (BATCH, PROMPT_LEN))
    grid = model.sample(text_ids, num_steps=2)
    assert tuple(grid.shape) == (BATCH, *GRID)
    assert grid.dtype == torch.long
    assert 0 <= int(grid.min()) and int(grid.max()) < NUM_BLOCKS

    tf_model = _tf_diffusion_model()
    context = torch.randn(BATCH, PROMPT_LEN, CONTEXT_DIM)
    out = tf_model.sample(context, None, num_steps=2)
    assert tuple(out.shape) == (BATCH, *GRID)


def test_top_k_restricts_the_sampled_blocks():
    """Mit top_k=3 darf nur einer der drei wahrscheinlichsten Bloecke kommen."""
    model = _tf_diffusion_model().eval()
    context = torch.randn(BATCH, PROMPT_LEN, CONTEXT_DIM)
    features = model.features(
        torch.zeros(BATCH, *GRID, dtype=torch.long),
        torch.zeros(BATCH, dtype=torch.long), context)
    with torch.no_grad():
        logits = model.out_proj(features).flatten(2)               # [B, Bloecke, Voxel]
        sampled = sample_denoising_step(model, features, temperature=1.0, top_k=3,
                                        chunk_voxels=6).reshape(BATCH, -1)
    top = logits.topk(3, dim=1).indices                            # [B, 3, Voxel]
    assert bool((top == sampled.unsqueeze(1)).any(dim=1).all())


def test_transformer_generate_is_chunk_size_independent_with_argmax():
    model = _transformer()
    prompt_ids = torch.randint(0, 11, (BATCH, 3))
    full = model.generate(prompt_ids, temperature=0.0, top_k=8)
    chunked = model.generate(prompt_ids, temperature=0.0, top_k=8, chunk_voxels=7)
    assert torch.equal(full, chunked)
    assert tuple(chunked.shape) == (BATCH, *GRID)
    assert int(chunked.max()) < NUM_BLOCKS


def test_transformer_output_chunks_reproduce_forward():
    model = _transformer().eval()
    prompt_ids = torch.randint(0, 11, (BATCH, 3))
    with torch.no_grad():
        reference = model.forward(prompt_ids).flatten(1)
        collected = torch.cat([chunk.flatten(1) for _, _, chunk in
                               model.iter_output_chunks(prompt_ids, 7)], dim=1)
    assert collected.shape == reference.shape
    # Punktweiser Kopf: nur Gleitkomma-Rundung, kein systematischer Unterschied
    assert torch.allclose(reference, collected, atol=1e-5)


# ─── Verdrahtung im Generate-Tab ───

class _FakeVar:
    def __init__(self):
        self.text = ""

    def set(self, value):
        self.text = value


class _FakeEncoder:
    """Ersatz fuer FrozenTransformerEncoder: liefert einen festen Kontext."""

    display_name = "FakeEncoder"

    def __init__(self, tokens: int = 3, dim: int = CONTEXT_DIM):
        self.tokens = tokens
        self.dim = dim

    def __call__(self, _prompts):
        return {"last_hidden_state": torch.zeros(1, self.tokens, self.dim),
                "attention_mask": torch.ones(1, self.tokens, dtype=torch.long)}


def _fake_app(**state):
    """Minimal-Attrappe der App (wie tests/test_preview_render.py)."""
    from types import SimpleNamespace

    from app.main_app import MinecraftStructureApp

    done, failed = [], []
    app = SimpleNamespace(
        config=SimpleNamespace(language="de", gpu_enabled=False),
        status_var=_FakeVar(), info_var=_FakeVar(),
        after=lambda _delay, callback, *args: callback(*args),
        _generation_done=lambda info: done.append(info),
        _generation_failed=lambda error: failed.append(error),
        done=done, failed=failed,
    )
    # Methoden der echten Klasse an die Attrappe binden
    app._tr = lambda key, **kwargs: MinecraftStructureApp._tr(app, key, **kwargs)
    app._plan_generation = (
        lambda device: MinecraftStructureApp._plan_generation(app, device))
    app.__dict__.update(state)
    return app


def test_generate_worker_passes_the_planned_chunk_to_sample(monkeypatch):
    """Der geplante Chunk muss bei sample() ankommen (TF-Diffusion)."""
    from types import SimpleNamespace

    from app.main_app import MinecraftStructureApp

    monkeypatch.setenv("MINECRAFT_AI_LOGITS_CHUNK_VOXELS", "32")
    model = _tf_diffusion_model()
    seen = {}

    def fake_sample(context, context_mask=None, num_steps=None, temperature=1.0,
                    top_k=0, chunk_voxels=None):
        seen["chunk_voxels"] = chunk_voxels
        seen["steps"] = num_steps
        grid = torch.zeros(1, *GRID, dtype=torch.long)
        grid[0, 1:3, 1:3, 1:3] = 5
        return grid

    model.sample = fake_sample

    app = _fake_app(
        model_type="transformer_diffusion", tf_diffusion_model=model,
        tf_encoder=_FakeEncoder(),
        voxel_tokenizer=SimpleNamespace(id_to_block=["minecraft:air", "minecraft:stone"]),
        generated_grid=None, generated_np=None, id_to_block=None,
    )
    MinecraftStructureApp._generate_worker(
        app, "kleine huette", {"temperature": 0.7, "top_k": 5, "diffusion_steps": 99})

    assert not app.failed
    assert seen["chunk_voxels"] == 32                 # Chunk aus dem Planer
    assert seen["steps"] == 4                         # auf num_timesteps begrenzt
    assert app.done and "TF-Diffusion" in app.done[0]
    assert "Chunks a 32 Voxel" in app.status_var.text
    assert tuple(app.generated_grid.shape) == (2, 2, 2)


def test_generate_worker_passes_the_planned_chunk_to_generate(monkeypatch):
    """Dasselbe fuer den Transformer-Pfad (generate statt sample)."""
    from types import SimpleNamespace

    from app.main_app import MinecraftStructureApp

    monkeypatch.setenv("MINECRAFT_AI_LOGITS_CHUNK_VOXELS", "16")
    model = _transformer()
    seen = {}

    def fake_generate(prompt_ids, temperature=0.9, top_k=40, chunk_voxels=None):
        seen["chunk_voxels"] = chunk_voxels
        grid = torch.zeros(1, *GRID, dtype=torch.long)
        grid[0, 0:2, 0:2, 0:2] = 4
        return grid

    model.generate = fake_generate
    app = _fake_app(
        model_type="transformer", transformer_model=model,
        prompt_tokenizer=SimpleNamespace(encode=lambda _text: torch.tensor([1, 2, 3])),
        voxel_tokenizer=SimpleNamespace(id_to_block=["minecraft:air", "minecraft:stone"]),
        generated_grid=None, generated_np=None, id_to_block=None,
    )
    MinecraftStructureApp._generate_worker(
        app, "kleine huette", {"temperature": 0.9, "top_k": 40, "diffusion_steps": 50})

    assert not app.failed
    assert seen["chunk_voxels"] == 16
    assert app.done and "Transformer" in app.done[0]


def test_generate_worker_reports_hopeless_vram_instead_of_crashing(monkeypatch):
    """Passt es nicht, gibt es eine Meldung - und sample() wird nicht gerufen."""
    from types import SimpleNamespace

    import app.main_app as main_app
    from app.training_memory import GenerationPlan

    called = []
    model = _tf_diffusion_model()
    model.sample = lambda *args, **kwargs: called.append(kwargs)

    hopeless = GenerationPlan(
        chunk_voxels=32, voxels=GRID[0] ** 3, logits_bytes=2 * GIB,
        activation_bytes=GIB, estimated_bytes=3 * GIB, budget_bytes=GIB,
        available_bytes=GIB, fits=False)
    monkeypatch.setattr(main_app, "plan_generation", lambda *a, **kw: hopeless)

    app = _fake_app(
        model_type="transformer_diffusion", tf_diffusion_model=model,
        tf_encoder=_FakeEncoder(),
        voxel_tokenizer=SimpleNamespace(id_to_block=["minecraft:air"]),
        generated_grid=None, generated_np=None, id_to_block=None,
    )
    main_app.MinecraftStructureApp._generate_worker(
        app, "kleine huette", {"temperature": 0.7, "top_k": 5, "diffusion_steps": 10})

    assert not called
    assert app.failed and "VRAM zu klein fuer die Generierung" in app.failed[0]
    assert "VRAM zu klein" in app.status_var.text
