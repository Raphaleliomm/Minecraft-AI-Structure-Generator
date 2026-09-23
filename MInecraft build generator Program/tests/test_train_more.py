"""Tests fuer "Weiter trainieren" (Model-Manager -> Train More).

Der Dialog selbst braucht Tk; diese Tests decken die Logik dahinter ab, mit
einer App-Attrappe wie in test_encoder_fallback.py:

  * der Checkpoint bestimmt Grid, Architektur, Vokabular und Encoder
  * der Trainer wird mit ``continue_from`` gestartet (frueher wurde stattdessen
    ein neues Modell trainiert bzw. bei TF-Diffusion gar nichts)
  * inkompatible Caches/Encoder brechen mit einer klaren Meldung ab
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import app.main_app as main_app
import app.model_manager as model_manager
from app.constants import grid_size_label
from app.diffusion_model import TransformerDiffusionModel
from app.main_app import MinecraftStructureApp
from app.model_manager import ModelEntry, ModelRegistry
from app.training_memory import GIB, plan_diffusion_training

GRID = (16, 16, 16)
CONTEXT_DIM = 6
ENCODER = "Test-Encoder"
NAME = "tf_diffusion_test"


class _FakeEncoder:
    """Attrappe von FrozenTransformerEncoder (nur was geprueft wird)."""

    def __init__(self, name: str, hidden_dim: int = CONTEXT_DIM):
        self.display_name = name
        self.hidden_dim = hidden_dim
        self.max_length = 32
        self.device = torch.device("cpu")


def _write_checkpoint(runs_dir, name: str = NAME, *, context_dim: int = CONTEXT_DIM,
                      encoder: str = ENCODER, epoch: int = 7, grid=GRID,
                      write_vocab: bool = True):
    """Kleinen TF-Diffusion-Checkpoint schreiben (wie das Training ihn ablegt)."""
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = TransformerDiffusionModel(
        num_blocks=5, grid_size=grid, d_model=16, channels=8,
        channel_multipliers=(1, 2), num_timesteps=4,
        context_dim=context_dim, cross_attn_heads=2, context_proj_dim=32)
    torch.save({
        "model_state": model.state_dict(), "grid_size": grid, "block_vocab_size": 5,
        "num_blocks": 5, "d_model": 16, "channels": 8, "channel_multipliers": [1, 2],
        "num_timesteps": 4, "context_dim": context_dim, "cross_attn_heads": 2,
        "context_proj_dim": 32,
        "encoder_config": {"display_name": encoder, "hf_id": "test/encoder",
                           "hidden_dim": context_dim},
        "augmentation_diversity": 1, "allow_vertical_movement": False,
        "epoch": epoch, "loss": 0.5,
    }, run_dir / "model.pt")
    if write_vocab:
        (run_dir / "prompt_vocab.json").write_text(json.dumps({
            "token_to_id": {"<pad>": 0, "<unk>": 1, "hut": 2}, "max_length": 8,
        }), encoding="utf-8")
        (run_dir / "block_vocab.json").write_text(json.dumps({
            "block_to_id": {"minecraft:air": 0, "minecraft:stone": 1},
            "block_entity_types": [],
        }), encoding="utf-8")
    return run_dir


def _registry(monkeypatch, runs_dir) -> ModelRegistry:
    monkeypatch.setattr(model_manager, "MODELS_DIR", runs_dir)
    registry = ModelRegistry()
    registry.discover()
    return registry


def _stub_app(registry, encoder: _FakeEncoder | None = None, **state):
    """App-Attrappe mit sofort ausgefuehrtem ``after`` (wie im Main-Thread)."""
    app = SimpleNamespace(
        model_registry=registry,
        tf_encoder=encoder,
        config=SimpleNamespace(language="de", gpu_enabled=False),
        status_var=SimpleNamespace(set=lambda _value: None),
        after=lambda _delay, callback, *args: callback(*args),
        _tr=lambda key, **kwargs: key,
        _discover_models=lambda: None,
    )
    app.encoder_requests = []

    def ensure_encoder(name: str) -> _FakeEncoder:
        app.encoder_requests.append(name)
        return app.fake_encoder or _FakeEncoder(name)

    app.fake_encoder = None
    app._ensure_encoder_for_continue = ensure_encoder
    app.__dict__.update(state)
    return app


# ─── Grid-Groesse aus dem Checkpoint ───

def test_grid_size_label_is_the_reverse_lookup():
    assert grid_size_label(GRID) == "16×16×16"
    assert grid_size_label((32, 32, 32)).startswith("32×32×32")
    assert grid_size_label((8, 8, 8)) is None      # nicht im Dropdown


def test_checkpoint_metadata_exposes_context_dim(tmp_path):
    run_dir = _write_checkpoint(tmp_path)
    entry = ModelEntry(name=NAME, model_type="transformer_diffusion", path=run_dir,
                       checkpoint_path=run_dir / "model.pt")
    entry.load_metadata()
    assert entry.context_dim == CONTEXT_DIM
    assert entry.cross_attn_heads == 2
    assert entry.encoder_name == ENCODER
    assert entry.epochs_trained == 7


# ─── Zustand fuer "Weiter trainieren" einsammeln ───

def test_prepare_continue_reads_the_checkpoint(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path, epoch=7)
    app = _stub_app(_registry(monkeypatch, tmp_path))

    state = MinecraftStructureApp._prepare_tf_diffusion_continue(
        app, NAME, torch.device("cpu"), False, None)

    assert state["grid_size"] == GRID
    assert (state["channels"], state["channel_multipliers"]) == (8, (1, 2))
    assert state["d_model"] == 16
    assert state["num_timesteps"] == 4
    assert state["context_dim"] == CONTEXT_DIM
    assert state["context_proj_dim"] == 32
    assert state["num_blocks"] == 5
    assert state["start_epoch"] == 7
    assert state["encoder_config"]["display_name"] == ENCODER
    # Der Encoder des Checkpoints wird geladen (hier: die Attrappe), nicht der
    # gerade im Training-Tab ausgewaehlte.
    assert app.encoder_requests == [ENCODER]
    # Vokabular kommt aus dem Checkpoint-Ordner
    assert state["voxel_tokenizer"].block_to_id["minecraft:stone"] == 1
    assert "hut" in state["prompt_tokenizer"].token_to_id


def test_prepare_continue_rejects_missing_vocab(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path, write_vocab=False)
    app = _stub_app(_registry(monkeypatch, tmp_path))
    with pytest.raises(RuntimeError, match="tm_missing_vocab"):
        MinecraftStructureApp._prepare_tf_diffusion_continue(
            app, NAME, torch.device("cpu"), False, None)


def test_prepare_continue_rejects_foreign_encoder(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path)
    app = _stub_app(_registry(monkeypatch, tmp_path),
                    fake_encoder=_FakeEncoder(ENCODER, hidden_dim=99))
    with pytest.raises(RuntimeError, match="tm_encoder_dim_mismatch"):
        MinecraftStructureApp._prepare_tf_diffusion_continue(
            app, NAME, torch.device("cpu"), False, None)


def test_prepare_continue_rejects_incompatible_cache(tmp_path, monkeypatch):
    from app import hidden_state_cache

    _write_checkpoint(tmp_path)
    app = _stub_app(_registry(monkeypatch, tmp_path))

    # Cache eines anderen Encoders (falsche Dimension)
    monkeypatch.setattr(hidden_state_cache, "load_hidden_states_raw",
                        lambda cache_dir, mmap=None: {"metadata": {
                            "hidden_dim": 99, "encoder_name": ENCODER}})
    with pytest.raises(RuntimeError, match="tm_cache_dim_mismatch"):
        MinecraftStructureApp._prepare_tf_diffusion_continue(
            app, NAME, torch.device("cpu"), True, ENCODER)

    # Passende Dimension, aber ein anderer Encoder -> andere Textmerkmale
    monkeypatch.setattr(hidden_state_cache, "load_hidden_states_raw",
                        lambda cache_dir, mmap=None: {"metadata": {
                            "hidden_dim": CONTEXT_DIM, "encoder_name": "Anderer"}})
    with pytest.raises(RuntimeError, match="tm_cache_encoder_mismatch"):
        MinecraftStructureApp._prepare_tf_diffusion_continue(
            app, NAME, torch.device("cpu"), True, ENCODER)
    # Mit Cache wird kein Encoder geladen
    assert app.encoder_requests == []


# ─── Modell bauen: neu oder fortsetzen ───

def test_build_tf_diffusion_model_continues_checkpoint(tmp_path, monkeypatch):
    run_dir = _write_checkpoint(tmp_path)
    app = _stub_app(_registry(monkeypatch, tmp_path))
    state = MinecraftStructureApp._prepare_tf_diffusion_continue(
        app, NAME, torch.device("cpu"), False, None)
    config = {"channels": state["channels"],
              "channel_multipliers": state["channel_multipliers"],
              "d_model": state["d_model"], "cross_attn_heads": state["cross_attn_heads"]}

    model = MinecraftStructureApp._build_tf_diffusion_model(
        app, config, state["grid_size"], state["num_timesteps"], state["context_dim"],
        len(state["voxel_tokenizer"].id_to_block), state, torch.device("cpu"))

    saved = torch.load(run_dir / "model.pt", map_location="cpu")["model_state"]
    assert model.num_blocks == 5
    assert model.context_dim == CONTEXT_DIM
    assert torch.allclose(model.state_dict()["time_embed.0.weight"],
                          saved["time_embed.0.weight"])
    assert "state_dict" not in state      # grosse Tensor-Kopie ist freigegeben


def test_build_tf_diffusion_model_starts_fresh_without_state():
    app = _stub_app(ModelRegistry())
    config = {"channels": 8, "channel_multipliers": (1, 2),
              "d_model": 16, "cross_attn_heads": 2}
    model = MinecraftStructureApp._build_tf_diffusion_model(
        app, config, GRID, 4, CONTEXT_DIM, 9, None, torch.device("cpu"))
    assert model.num_blocks == 9                 # Vokabular des Datensatzes
    assert model.effective_context_dim == 32     # d_model * 2 wie beim Training


# ─── Checkpoint laden bzw. schon geladenes Modell wiederverwenden ───

def test_load_continue_checkpoint_reuses_the_loaded_model(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path, "loaded_model")
    _write_checkpoint(tmp_path, "other_model")
    registry = _registry(monkeypatch, tmp_path)
    loaded = object()
    calls = []
    app = SimpleNamespace(
        model_registry=registry,
        _discover_models=lambda: None,
        transformer_model=loaded,
        current_transformer_name="loaded_model",
        _load_transformer_entry=lambda name, device: calls.append(name) or True,
        _load_diffusion_entry=lambda name, device: calls.append(name) or True,
        _load_tf_diffusion_entry=lambda name, device: calls.append(name) or True,
    )
    load = MinecraftStructureApp._load_continue_checkpoint
    assert load(app, "transformer", "loaded_model", "cpu") is True
    assert calls == []                     # schon geladen -> kein zweiter Ladevorgang
    assert load(app, "transformer", "other_model", "cpu") is True
    assert calls == ["other_model"]        # anderes Modell -> laden
    assert load(app, "transformer", "gibt_es_nicht", "cpu") is False


# ─── Start: richtiger Worker mit den Dialogwerten ───

class _FakeThread:
    """Sammelt die gestarteten Worker, statt sie auszufuehren."""

    started: list = []

    def __init__(self, target=None, args=(), daemon=None, **kwargs):
        self.target = target
        self.args = args

    def start(self) -> None:
        _FakeThread.started.append((self.target, self.args))


def _setter():
    return SimpleNamespace(set=lambda _value: None)


def _configurable():
    return SimpleNamespace(configure=lambda **kwargs: None)


def _start_training_app() -> SimpleNamespace:
    app = SimpleNamespace()
    app.status_messages = []
    app.training_running = False
    app._selected_arch = {"d_model": 192, "nhead": 6, "num_layers": 5,
                          "dim_feedforward": 768}
    app.status_var = SimpleNamespace(set=lambda value: app.status_messages.append(value))
    # Wie im Main-Thread: after() fuehrt den Rueckruf sofort aus.
    app.after = lambda _delay, callback, *args: callback(*args)
    app.train_epoch_bar = _setter()
    app.train_batch_bar = _setter()
    app.train_btn_transformer = _configurable()
    app.train_btn_diffusion = _configurable()
    app.train_btn_tf_diffusion = _configurable()
    app.train_stop_btn = _configurable()
    app.augmentation_diversity_var = SimpleNamespace(get=lambda: 1)
    app.setting_aug_vertical = SimpleNamespace(get=lambda: False)
    app._collect_training_params = lambda: {"epochs": 1, "batch_size": 1,
                                            "lr": 1e-3, "air_weight": 75.0}
    app._tr = lambda key, **kwargs: key
    app._training_worker = "worker-sentinel"
    app._read_int_entry = MinecraftStructureApp._read_int_entry.__get__(app)
    return app


def test_start_training_rejects_unknown_types(monkeypatch):
    """Frueher lief 'Train More' bei TF-Diffusion still ins Leere."""
    monkeypatch.setattr(main_app, "threading", SimpleNamespace(Thread=_FakeThread))
    _FakeThread.started = []
    app = _start_training_app()

    MinecraftStructureApp._start_training(app, "transformer_diffusion")

    assert app.status_messages[-1] == "status_unknown_train_type"
    assert _FakeThread.started == []          # kein Worker gestartet


def test_start_training_forwards_continue_and_overrides(monkeypatch):
    monkeypatch.setattr(main_app, "threading", SimpleNamespace(Thread=_FakeThread))
    _FakeThread.started = []
    app = _start_training_app()

    MinecraftStructureApp._start_training(
        app, "transformer", grid_size=(32, 32, 32),
        overrides={"epochs": 3, "batch_size": 2, "lr": 5e-4, "air_weight": 80.0,
                   "augmentation_diversity": 2, "allow_vertical_movement": True},
        continue_from="my_model")

    target, args = _FakeThread.started[-1]
    assert target == "worker-sentinel"
    model_type, grid_size, aug, vertical, params = args
    # Das Grid kommt aus dem Checkpoint, nicht aus dem Training-Tab
    assert (model_type, grid_size, aug, vertical) == ("transformer", (32, 32, 32), 2, True)
    assert params["continue_from"] == "my_model"
    assert (params["epochs"], params["batch_size"], params["lr"]) == (3, 2, 5e-4)
    assert params["air_weight"] == 80.0


def test_apply_training_plan_hands_back_the_accumulated_batch():
    """Passt der Wunsch-Batch nicht, laufen trotzdem alle Beispiele pro Schritt.

    Der Plan liefert dann (effektiver Batch, Chunk, Mikro-Batch) und die
    Statuszeile nennt die Aufteilung - frueher wurde der Batch einfach kleiner
    und die Trainingsdynamik aenderte sich.
    """
    app = _start_training_app()
    app.config = SimpleNamespace(language="de")
    plan = plan_diffusion_training(
        (32, 32, 32), 64, 17690, device=torch.device("cuda"),
        channels=16, channel_multipliers=(1, 2, 2), d_model=32,
        cross_attn_heads=2, context_dim=3072, context_len=64,
        budget_bytes=int(0.9 * GIB),
    )
    assert plan.accumulated

    batch, chunk, micro = MinecraftStructureApp._apply_training_plan(app, plan, 64)

    assert batch == 64                      # der Wunsch-Batch bleibt der Batch
    assert micro == plan.batch_size < 64    # gerechnet wird in kleineren Haeppchen
    assert chunk == plan.chunk_voxels
    assert "Gradient Accumulation" in app.status_messages[-1]
    assert app.status_messages[-1].endswith("grad_accum_status")


def test_apply_training_plan_without_accumulation_does_not_mention_it():
    """Ohne Reduzierung bleibt die Statuszeile wie bisher."""
    app = _start_training_app()
    app.config = SimpleNamespace(language="de")
    plan = plan_diffusion_training(
        (16, 16, 16), 4, 253, device=torch.device("cpu"),
    )
    batch, chunk, micro = MinecraftStructureApp._apply_training_plan(app, plan, 4)
    assert (batch, micro) == (4, 4)
    assert not app.status_messages[-1].endswith("grad_accum_status")


# ─── Dialogwerte im Training-Tab spiegeln (Anzeige) ───

def _sync_app() -> SimpleNamespace:
    """App-Attrappe fuer _sync_training_widgets (alle Widgets als Attrappen)."""
    app = SimpleNamespace()
    app.panels = []
    app.train_model_type_selector = SimpleNamespace(set=lambda v: app.panels.append(v))
    app._on_train_type_change = lambda value: app.panels.append(value)
    app.grid = {"label": "16×16×16"}
    app.grid_size_var = SimpleNamespace(
        get=lambda: app.grid["label"],
        set=lambda v: app.grid.__setitem__("label", v))
    app.entries = {}

    def entry(name: str):
        widget = SimpleNamespace(values=[], delete=lambda a, b: widget.values.clear(),
                                 insert=lambda index, value: widget.values.append(value))
        app.entries[name] = widget
        return widget

    app.train_epochs_entry = entry("epochs")
    app.train_batch_entry = entry("batch")
    app.train_lr_entry = entry("lr")
    app.tf_diff_steps_entry = entry("tf_steps")
    app.diff_train_steps_entry = entry("diff_steps")
    app.augmentation_diversity_var = SimpleNamespace(set=lambda v: None)
    app._update_aug_label = lambda value: None
    app.air_weight_var = SimpleNamespace(set=lambda v: None)
    app._update_air_weight_label = lambda value: None
    app.vertical = {"selected": False}
    app.setting_aug_vertical = SimpleNamespace(
        select=lambda: app.vertical.__setitem__("selected", True),
        deselect=lambda: app.vertical.__setitem__("selected", False))
    app.tab = {"view": None}
    app.tab_view = SimpleNamespace(set=lambda v: app.tab.__setitem__("view", v))
    app.encoder = {"name": None}
    app.tf_encoder_combo = SimpleNamespace(set=lambda v: app.encoder.__setitem__("name", v))
    app.cache_used = {"value": None}
    app.tf_use_cached_hs_var = SimpleNamespace(set=lambda v: app.cache_used.__setitem__("value", v))
    app.cache = {"name": None}
    app.tf_cache_selector = SimpleNamespace(set=lambda v: app.cache.__setitem__("name", v))
    app._tr = lambda key, **kwargs: key
    return app


def _entry(model_type: str, **metadata) -> ModelEntry:
    from pathlib import Path
    entry = ModelEntry(name="m", model_type=model_type,
                       path=Path("runs") / "m", checkpoint_path=Path("runs") / "m" / "model.pt")
    entry.grid_size = (32, 32, 32)
    for key, value in metadata.items():
        setattr(entry, key, value)
    return entry


def test_sync_training_widgets_mirrors_grid_and_params():
    app = _sync_app()
    MinecraftStructureApp._sync_training_widgets(app, _entry("transformer"), {
        "epochs": 3, "batch_size": 2, "lr": 0.0005, "air_weight": 80.0,
        "augmentation_diversity": 2, "allow_vertical_movement": True})

    assert app.grid["label"].startswith("32×32×32")     # Grid aus dem Checkpoint
    assert app.entries["epochs"].values == ["3"]
    assert app.entries["batch"].values == ["2"]
    assert app.entries["lr"].values == ["0.0005"]
    assert app.vertical["selected"] is True
    assert app.panels == ["Transformer", "Transformer"]
    assert app.tab["view"] == "tab_training"


def test_sync_training_widgets_mirrors_encoder_and_cache():
    app = _sync_app()
    MinecraftStructureApp._sync_training_widgets(
        app, _entry("transformer_diffusion", encoder_name=ENCODER, num_timesteps=4,
                    context_dim=CONTEXT_DIM),
        {"epochs": 1, "batch_size": 1, "lr": 1e-3, "air_weight": 75.0,
         "augmentation_diversity": 0, "allow_vertical_movement": False,
         "use_cached": True, "cache_name": ENCODER})

    assert app.panels == ["TF-Diffusion", "TF-Diffusion"]
    assert app.encoder["name"] == ENCODER
    assert app.cache_used["value"] is True
    assert app.cache["name"] == ENCODER
    assert app.entries["tf_steps"].values == ["4"]     # Schritte des Checkpoints
    assert app.vertical["selected"] is False
