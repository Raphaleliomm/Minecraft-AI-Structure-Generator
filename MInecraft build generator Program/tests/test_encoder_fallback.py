"""Tests fuer den Text-Encoder: VRAM oder (nach Rueckfrage) System-RAM.

Hintergrund: Auf einer 6-GB-Karte laesst sich ``Phi-3.5-mini`` (fp16 ~7,6 GB)
nicht in den VRAM laden - die TF-Diffusion-Generierung meldete danach nur
"kein Modell". Die App fragt jetzt einmal je Encoder, ob er stattdessen in den
System-RAM geladen wird und rechnet dann auf der CPU.

Die Tests ersetzen ``FrozenTransformerEncoder`` durch eine Attrappe - es wird
also kein echtes Modell (und kein VRAM) geladen.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

import app.main_app as main_app
from app.main_app import MinecraftStructureApp
from app.training_memory import GIB, system_ram_bytes
from app.transformer_encoder import (
    ENCODER_PARAMS_B,
    MODEL_NAMES,
    encoder_working_set_bytes,
)

MODEL = "Phi-3.5-mini"
TOTAL_VRAM = 6 * GIB


class _FakeEncoder:
    """Attrappe von ``FrozenTransformerEncoder`` (merkt sich device/dtype)."""

    create_calls: list = []          # erfolgreich gebaute Encoder
    attempts: list = []              # jeder Aufruf, auch fehlgeschlagene
    fail_on_cuda = False

    def __init__(self, model_name: str, device="cpu", dtype=torch.float32,
                 compile_model: bool = False):
        _FakeEncoder.attempts.append((model_name, torch.device(device).type))
        if self.fail_on_cuda and torch.device(device).type == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 7.60 GiB.")
        self.display_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.hidden_dim = 8
        self.max_length = 512
        _FakeEncoder.create_calls.append((model_name, self.device.type, dtype))


def _patch_cuda(monkeypatch, free_bytes: int, available: bool = True) -> None:
    monkeypatch.setattr(main_app.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(main_app.torch.cuda, "mem_get_info",
                        lambda index=None: (free_bytes, TOTAL_VRAM))


def _app(monkeypatch, dialog_answer=None, encoder_cls=_FakeEncoder, **state):
    """App-Attrappe mit sofort ausgefuehrtem ``after`` (wie im Main-Thread)."""
    monkeypatch.setattr(main_app, "FrozenTransformerEncoder", encoder_cls)
    _FakeEncoder.create_calls = []          # Attrappen notieren dort ihre Aufrufe
    _FakeEncoder.attempts = []
    questions = []

    app = SimpleNamespace(
        config=SimpleNamespace(language="de", gpu_enabled=True),
        status_var=SimpleNamespace(set=lambda _value: None),
        after=lambda _delay, callback, *args: callback(*args),
        _encoder_cpu_choice={},
        _encoder_load_lock=threading.Lock(),
        _tr=lambda key, **kwargs: MinecraftStructureApp._tr(
            SimpleNamespace(config=SimpleNamespace(language="de")), key, **kwargs),
    )
    if dialog_answer is None:
        app._ask_yes_no_blocking = lambda title, question: pytest.fail(
            "Es haette kein Dialog erscheinen duerfen")
    else:
        def ask(title, question):
            questions.append((title, question))
            return dialog_answer
        app._ask_yes_no_blocking = ask
    app.questions = questions
    app.__dict__.update(state)
    for name in ("_encoder_vram_fits", "_request_encoder_cpu", "_encoder_cpu_dtype",
                 "_build_tf_encoder", "_load_tf_encoder_on_cpu", "_create_tf_encoder",
                 "_create_tf_encoder_locked"):
        setattr(app, name, getattr(MinecraftStructureApp, name).__get__(app))
    return app


# ─── Groessenschaetzung ───

def test_encoder_sizes_are_plausible():
    """Phi-3.5-mini: fp16 ~7,6 GB, fp32 ~15 GB (die Zahlen aus der Messung)."""
    assert encoder_working_set_bytes(MODEL) == int(3.8e9 * 2)
    assert encoder_working_set_bytes(MODEL, item_bytes=4) == int(3.8e9 * 4)
    assert encoder_working_set_bytes("Flan-T5-small") == int(0.035e9 * 2)
    assert encoder_working_set_bytes("Unbekannt") == 0


def test_every_supported_encoder_has_an_estimate():
    """Kein Encoder der GUI darf ohne Schaetzung dastehen."""
    missing = [name for name in MODEL_NAMES if name not in ENCODER_PARAMS_B]
    assert missing == []


def test_system_ram_bytes_is_plausible():
    ram = system_ram_bytes()
    assert ram is None or ram > 0



# ─── Entscheidung VRAM / RAM ───

def test_encoder_stays_on_gpu_when_it_fits(monkeypatch):
    _patch_cuda(monkeypatch, free_bytes=8 * GIB)
    app = _app(monkeypatch)                     # kein Dialog erlaubt
    encoder = app._create_tf_encoder(MODEL)
    assert encoder.device.type == "cuda"
    assert encoder.dtype == torch.float16


def test_encoder_asks_and_moves_to_ram(monkeypatch):
    """Zu wenig VRAM: Rueckfrage -> CPU in fp16 (fp32 waere ~15 GB)."""
    _patch_cuda(monkeypatch, free_bytes=1 * GIB)
    app = _app(monkeypatch, dialog_answer=True)
    encoder = app._create_tf_encoder(MODEL)
    assert encoder.device.type == "cpu"
    assert encoder.dtype == torch.float16
    assert app.questions, "es muss gefragt werden"
    _title, question = app.questions[0]
    # 3,8 Mrd. Parameter in fp16 = 7,08 GiB (gerundet 7,1)
    assert MODEL in question and "7.1" in question and "1.00" in question


def test_encoder_tries_the_gpu_when_the_user_says_no(monkeypatch):
    _patch_cuda(monkeypatch, free_bytes=1 * GIB)
    app = _app(monkeypatch, dialog_answer=False)
    encoder = app._create_tf_encoder(MODEL)
    assert encoder.device.type == "cuda"        # der Nutzer wollte es so
    assert app._encoder_cpu_choice[MODEL] is False


def test_encoder_oom_asks_afterwards(monkeypatch):
    """Schaetzung zu optimistisch -> nach dem OOM wird gefragt und gewechselt."""
    class _FailingOnCuda(_FakeEncoder):
        fail_on_cuda = True

    _patch_cuda(monkeypatch, free_bytes=8 * GIB)     # Schaetzung sagt "passt"
    app = _app(monkeypatch, dialog_answer=True, encoder_cls=_FailingOnCuda)
    encoder = app._create_tf_encoder(MODEL)
    assert encoder.device.type == "cpu"
    assert app.questions
    # erst GPU (schlaegt fehl), dann CPU - und nur der CPU-Bau zaehlt als Erfolg
    assert _FakeEncoder.attempts == [(MODEL, "cuda"), (MODEL, "cpu")]
    assert _FakeEncoder.create_calls == [(MODEL, "cpu", torch.float16)]


def test_encoder_oom_without_consent_raises_clear_error(monkeypatch):
    class _FailingOnCuda(_FakeEncoder):
        fail_on_cuda = True

    _patch_cuda(monkeypatch, free_bytes=8 * GIB)
    app = _app(monkeypatch, dialog_answer=False, encoder_cls=_FailingOnCuda)
    with pytest.raises(RuntimeError) as excinfo:
        app._create_tf_encoder(MODEL)
    assert "passt nicht in den VRAM" in str(excinfo.value)


def test_encoder_question_is_asked_only_once(monkeypatch):
    _patch_cuda(monkeypatch, free_bytes=1 * GIB)
    app = _app(monkeypatch, dialog_answer=True)
    first = app._create_tf_encoder(MODEL)
    second = app._create_tf_encoder(MODEL)
    assert (first.device.type, second.device.type) == ("cpu", "cpu")
    assert len(app.questions) == 1


def test_encoder_choice_is_forgotten_when_the_ram_load_fails(monkeypatch):
    """Scheitert der RAM-Ladepfad, wird beim naechsten Mal wieder gefragt."""
    class _FailAlways(_FakeEncoder):
        fail_on_cuda = True

        def __init__(self, *args, **kwargs):
            raise MemoryError("DefaultCPUAllocator: can't allocate memory")

    _patch_cuda(monkeypatch, free_bytes=1 * GIB)
    app = _app(monkeypatch, dialog_answer=True, encoder_cls=_FailAlways)
    with pytest.raises(MemoryError):
        app._create_tf_encoder(MODEL)
    assert MODEL not in app._encoder_cpu_choice


def test_encoder_cpu_dtype_uses_fp32_for_small_encoders(monkeypatch):
    monkeypatch.setattr(main_app, "system_ram_bytes", lambda: 16 * GIB)
    app = _app(monkeypatch)
    assert app._encoder_cpu_dtype("Flan-T5-small") == torch.float32
    assert app._encoder_cpu_dtype(MODEL) == torch.float16      # ~15 GB in fp32
    # Ohne RAM-Info bleibt es bei fp32 fuer kleine Encoder
    monkeypatch.setattr(main_app, "system_ram_bytes", lambda: None)
    assert app._encoder_cpu_dtype("Flan-T5-base") == torch.float32


def test_encoder_is_loaded_on_cpu_without_gpu(monkeypatch):
    _patch_cuda(monkeypatch, free_bytes=1 * GIB, available=False)
    app = _app(monkeypatch)                     # kein Dialog erlaubt
    encoder = app._create_tf_encoder("Flan-T5-small")
    assert encoder.device.type == "cpu"
    assert encoder.dtype == torch.float32


# ─── Dialog-Marshalling ───

def test_ask_yes_no_marshals_to_the_main_thread(monkeypatch):
    """Der Dialog darf nicht im Worker-Thread entstehen: ``after`` nutzen."""
    import tkinter.messagebox as messagebox

    monkeypatch.setattr(messagebox, "askyesno", lambda title, question: True)
    scheduled = []

    app = SimpleNamespace(
        after=lambda delay, callback, *args: scheduled.append(callback) or callback())
    assert MinecraftStructureApp._ask_yes_no_blocking(app, "Titel", "Frage") is True
    assert scheduled, "'after' wurde nicht benutzt"


def test_ask_yes_no_returns_the_answer(monkeypatch):
    import tkinter.messagebox as messagebox

    for answer in (True, False):
        monkeypatch.setattr(messagebox, "askyesno", lambda t, q, a=answer: a)
        app = SimpleNamespace(after=lambda delay, callback, *args: threading.Thread(
            target=callback, daemon=True).start())
        assert MinecraftStructureApp._ask_yes_no_blocking(app, "T", "F") is answer
