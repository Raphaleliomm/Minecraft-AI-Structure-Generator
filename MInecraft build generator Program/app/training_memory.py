"""VRAM-Abschaetzung und Chunk-Planung fuer das Training.

Warum das noetig ist: Das Diffusionsmodell berechnet die Logits als EINEN
Tensor ``[batch, num_blocks, X*Y*Z]``. Mit 17.690 Blockzustaenden (Rotationen
zaehlen mit), 32^3 Voxeln und Batch 16 sind das

    16 * 17.690 * 32.768 * 4 Byte = 34,55 GiB

- die Grafikkarte (GTX 1060, 6 GB) bricht dann nach dem minutenlangen Laden der
Trainingsdaten mit "Cuda out of memory. Tried to allocate 34.55 GiB" ab.
Seit ``app/diffusion_model.py`` die Logits in Voxel-Chunks berechnet, muss nur
noch *ein* Chunk gleichzeitig im Speicher liegen; dieses Modul rechnet aus, wie
gross die Chunks und der Batch sein duerfen.

Die Rechnung ist eine bewusste Abschaetzung (keine exakte Buchhaltung): Logits
(inkl. Log-Softmax- und Gradienten-Kopie), Aktivierungen des UNet,
Cross-Attention-Matrizen ``[B, heads, Voxel, Text]`` und Modellzustaende
(Gewichte + Gradienten + Adam) werden addiert. Liegt das Ergebnis ueber dem
freien Speicher, wird die Batch-Groesse reduziert; passt selbst Batch 1 nicht,
meldet der Plan ``fits=False`` samt groesstem Verursacher.

Eine reduzierte Batch-Groesse aendert aber das Training. Deshalb liefert der
Plan zusaetzlich ``accum_steps``: so viele Mikro-Batches halten die *effektive*
Batch-Groesse (``effective_batch_size``), sodass ein Optimizer-Schritt weiterhin
``batch_size`` Beispiele sieht (Gradient-Akkumulation, siehe
``app/grad_accum.py``). Mit ``allow_accumulation=False`` bleibt es beim reinen
Verkleinern.

Abschaltbar ueber ``MINECRAFT_AI_VRAM_GUARD=0``, feste Chunk-Groesse ueber
``MINECRAFT_AI_LOGITS_CHUNK_VOXELS``, Akkumulation ueber
``MINECRAFT_AI_GRAD_ACCUM=0``.

Dasselbe Problem traf danach die *Generierung*: ``sample()`` rechnete die
Logits wieder am Stueck. Bei Grid 32^3 und 17.690 Blockzustaenden sind das
``17690 * 32768 * 4 Byte = 2,16 GiB`` - und die Sampling-Schleife haelt
davon wegen Temperatur, Top-K, Softmax und Permute zeitweise vier Kopien
(rund 8,6 GiB). Auf der GTX 1060 endete der Klick auf "Generieren" daher mit
``Cuda out of memory``, sobald das Training zuvor Speicher hinterlassen hatte.
``plan_generation()`` liefert dafuer die Chunk-Groesse, die in den freien
Speicher passt (siehe auch ``app/diffusion_model.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import torch

from app.constants import ALLOWED_GRID_SIZES

GIB = 1024 ** 3
MIB = 1024 ** 2

# Logits + Log-Softmax-Kopie in F.cross_entropy + Gradient des Backward-Passes.
CHUNK_OVERHEAD = 3.0
# Obergrenze fuer die *Effizienz*: sehr grosse Chunks sind nicht nur teurer im
# Speicher, sondern auch deutlich langsamer (gemessen auf einer GTX 1060,
# Grid 32^3/Batch 16/17.690 Bloecke: 512 Voxel -> 5,7 s pro Batch, 1024 Voxel
# -> 16,7 s, 2048 Voxel -> 44,6 s). Sehr kleine Chunks verlieren dagegen an
# Kernel-/Python-Overhead. 768 MiB Logits pro Chunk treffen den Sweet Spot.
TARGET_CHUNK_BYTES = 768 * MIB
# Es wird nur ein Teil des freien Speichers verplant: CUDA-Fragmentierung,
# cuDNN-Workspaces und (bei "live"-Training) der Text-Encoder brauchen ebenfalls
# Platz. Bewusst vorsichtig, aber nicht so klein, dass sinnvolle
# Konfigurationen verhindert werden.
DEFAULT_SAFETY_FRACTION = 0.85
# Budget fuer reines CPU-Training: auch im RAM ist ein 34-GiB-Tensor toedlich.
CPU_BUDGET_BYTES = 2 * GIB
# Darunter dominiert der Python-/Kernel-Overhead pro Chunk den Nutzen.
MIN_CHUNK_VOXELS = 32
# Gewichte + Gradienten + Adam-Zustaende (2 Momente) je Parameter.
PARAM_STATE_FACTOR = 3
# Self-Attention des VoxelTransformers: pro Layer bleibt die Softmax-Ausgabe
# fuer den Backward-Pass erhalten, dazu eine transiente Kopie.
ATTENTION_TRANSIENT_COPIES = 1.0
# Bei der Generierung gibt es keinen Rueckwaerts-Pass (keine Gradienten, keine
# Log-Softmax-Kopie fuer autograd, keine Adam-Zustaende) - die Aktivierungen
# sind deshalb deutlich kleiner als die von ``unet_activation_bytes``
# geschaetzten Trainingswerte.
INFERENCE_ACTIVATION_FACTOR = 0.35
# Textlaenge, mit der die Cross-Attention-Matrizen beim Planen der Generierung
# veranschlagt werden. 128 Token ist bewusst grosszuegig: die Prompts im
# Hidden-State-Cache liegen bei ~40-64 Token (95%-Quantil 64).
GENERATION_CONTEXT_TOKENS = 128

GUARD_ENV = "MINECRAFT_AI_VRAM_GUARD"
CHUNK_ENV = "MINECRAFT_AI_LOGITS_CHUNK_VOXELS"


def format_gib(num_bytes: float) -> str:
    """Menschenlesbare GiB-Angabe (2 Nachkommastellen, wie torch sie meldet)."""
    return f"{float(num_bytes) / GIB:.2f}"


def guard_enabled() -> bool:
    value = os.environ.get(GUARD_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _env_chunk_override() -> Optional[int]:
    value = os.environ.get(CHUNK_ENV, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def voxel_count(grid_size: Tuple[int, int, int]) -> int:
    gx, gy, gz = (int(v) for v in grid_size)
    return max(1, gx * gy * gz)


@dataclass(frozen=True)
class VramInfo:
    """Freier/gesamter Speicher des Trainingsgeraets."""

    total_bytes: int
    free_bytes: int
    budget_bytes: int
    device_type: str = "cpu"

    @property
    def total_gib(self) -> str:
        return format_gib(self.total_bytes)

    @property
    def free_gib(self) -> str:
        return format_gib(self.free_bytes)

    @property
    def budget_gib(self) -> str:
        return format_gib(self.budget_bytes)


@dataclass(frozen=True)
class TrainingPlan:
    """Ergebnis der Speicherplanung fuer einen Trainingslauf."""

    batch_size: int
    requested_batch_size: int
    chunk_voxels: int
    voxels: int
    logits_bytes: int
    activation_bytes: int
    param_bytes: int
    estimated_bytes: int
    budget_bytes: int
    available_bytes: int
    fits: bool
    limit: str = "ok"
    suggestion: Optional[str] = None
    details: Dict[str, float] = field(default_factory=dict)
    # Anzahl Mikro-Batches pro Optimizer-Schritt (> 1 = Gradient-Akkumulation).
    accum_steps: int = 1

    @property
    def reduced(self) -> bool:
        return self.batch_size < self.requested_batch_size

    @property
    def micro_batch_size(self) -> int:
        """Physischer Batch pro Vorwaerts-/Rueckwaerts-Lauf (das ist ``batch_size``)."""
        return max(1, int(self.batch_size))

    @property
    def effective_batch_size(self) -> int:
        """Batch-Groesse, die ein Optimizer-Schritt sieht (mit Akkumulation wie gewuenscht)."""
        return self.micro_batch_size * max(1, int(self.accum_steps))

    @property
    def accumulated(self) -> bool:
        """Werden die Gradienten ueber mehrere Mikro-Batches gesammelt?"""
        return max(1, int(self.accum_steps)) > 1

    @property
    def chunked(self) -> bool:
        return 0 < self.chunk_voxels < self.voxels

    @property
    def num_chunks(self) -> int:
        if not self.chunked:
            return 1
        return (self.voxels + self.chunk_voxels - 1) // self.chunk_voxels

    def _batch_text(self, language: str = "de") -> str:
        """Batch-Angabe fuer die Statuszeile - mit effektiver Groesse bei Akkumulation."""
        if not self.accumulated:
            return f"Batch {self.batch_size}" if language == "de" else f"batch {self.batch_size}"
        if language == "de":
            return f"Batch {self.batch_size} (effektiv {self.effective_batch_size})"
        return f"batch {self.batch_size} (effective {self.effective_batch_size})"

    def describe(self, language: str = "de") -> str:
        """Kurze Beschreibung; bewusst unabhaengig von app/translations.py, weil
        das Modul auch vom CLI-Training (train.py) benutzt wird."""
        if language == "de":
            text = (f"VRAM-Plan: Logits {format_gib(self.logits_bytes)} GiB "
                    f"({self.num_chunks} Chunks a {self.chunk_voxels} Voxel), "
                    f"{self._batch_text('de')}, geschaetzt "
                    f"{format_gib(self.estimated_bytes)} von "
                    f"{format_gib(self.budget_bytes)} GiB nutzbar")
        else:
            text = (f"VRAM plan: logits {format_gib(self.logits_bytes)} GiB "
                    f"({self.num_chunks} chunks of {self.chunk_voxels} voxels), "
                    f"{self._batch_text('en')}, estimated "
                    f"{format_gib(self.estimated_bytes)} of "
                    f"{format_gib(self.budget_bytes)} GiB usable")
        if self.suggestion:
            text += f" - {self.suggestion}"
        return text


# ─── Speicher des Geraets ───

def vram_info(device: Optional[torch.device] = None,
              safety_fraction: float = DEFAULT_SAFETY_FRACTION,
              reserve_bytes: int = 0) -> VramInfo:
    """Freien Speicher des Trainingsgeraets bestimmen (CPU: festes Budget)."""
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except (RuntimeError, TypeError):  # sehr alte torch-Version ohne mem_get_info
            props = torch.cuda.get_device_properties(index)
            total_bytes = int(props.total_memory)
            free_bytes = int(total_bytes - torch.cuda.memory_allocated(index))
        budget = max(0, int(int(free_bytes) * float(safety_fraction)) - int(reserve_bytes))
        return VramInfo(total_bytes=int(total_bytes), free_bytes=int(free_bytes),
                        budget_bytes=budget, device_type="cuda")
    return VramInfo(total_bytes=CPU_BUDGET_BYTES, free_bytes=CPU_BUDGET_BYTES,
                    budget_bytes=CPU_BUDGET_BYTES, device_type="cpu")


def _budget_info(device: Optional[torch.device],
                 safety_fraction: float = DEFAULT_SAFETY_FRACTION,
                 reserve_bytes: int = 0,
                 budget_bytes: Optional[int] = None) -> VramInfo:
    """VramInfo des Geraets - oder ein festes Budget (Tests/Reproduzierbarkeit)."""
    info = vram_info(device, safety_fraction=safety_fraction,
                     reserve_bytes=reserve_bytes)
    if budget_bytes is None:
        return info
    fixed = max(0, int(budget_bytes))
    return replace(info, budget_bytes=fixed, free_bytes=fixed)


def system_ram_bytes() -> Optional[int]:
    """Freier physischer Arbeitsspeicher in Byte (None, wenn unbekannt).

    Wird nur fuer den Hinweis gebraucht, ob ein Text-Encoder, der nicht in den
    VRAM passt, ueberhaupt in den RAM geladen werden kann. Ohne zusaetzliche
    Abhaengigkeit (psutil) ueber die Windows-API; auf anderen Systemen None.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:  # noqa: BLE001 - reine Zusatzinfo, darf nie stoeren
        return None


# ─── Groessen einzelner Tensoren ───

def logits_elements(batch_size: int, num_blocks: int, voxels: int) -> int:
    """Elemente des Logits-Tensors ``[batch, num_blocks, Voxel]``."""
    return max(1, int(batch_size)) * max(1, int(num_blocks)) * max(1, int(voxels))


def logits_bytes(batch_size: int, num_blocks: int, voxels: int,
                 item_bytes: int = 4) -> int:
    """Bytes des Logits-Tensors (der OOM-Verursacher im Diffusions-Training)."""
    return logits_elements(batch_size, num_blocks, voxels) * int(item_bytes)


def attention_bytes(batch_size: int, nhead: int, voxels: int,
                    context_len: int, item_bytes: int = 4) -> int:
    """Bytes einer Attention-Matrix ``[B, heads, Voxel, Text]``."""
    return (max(1, int(batch_size)) * max(1, int(nhead)) * max(1, int(voxels))
            * max(1, int(context_len)) * int(item_bytes))



# ─── Modellgroessen (Schaetzung) ───

def unet_parameter_bytes(channels: int, multipliers: Tuple[int, ...],
                         d_model: int, num_blocks: int,
                         context_dim: int) -> int:
    """Parameter des UNet (Gewichte) - grob, aber in der richtigen Groessenordnung."""
    params = 0
    in_ch = int(channels)
    for mult in multipliers:
        out_ch = int(channels) * int(mult)
        params += 27 * in_ch * out_ch          # conv1 (3x3x3)
        params += 27 * out_ch * out_ch         # conv2
        params += 4 * out_ch * out_ch          # cross-attention q/k/v/out
        params += 2 * d_model * out_ch         # time-mlp
        in_ch = out_ch
    params += 2 * (27 * in_ch * in_ch + 4 * in_ch * in_ch)      # Middle
    for mult in reversed(list(multipliers)):
        out_ch = int(channels) * int(mult)
        params += 64 * in_ch * out_ch          # conv_transpose (4x4x4)
        params += 27 * (2 * out_ch) * out_ch + 27 * out_ch * out_ch
        params += 4 * out_ch * out_ch
        params += 2 * d_model * out_ch
        in_ch = out_ch
    params += num_blocks * int(channels) + num_blocks           # out_proj 1x1x1
    params += 2 * d_model * d_model * 2 + d_model * d_model * 4  # time embedding
    params += 2 * d_model * int(context_dim)                    # context projection
    return max(1, params) * 4


def _cross_attn_element_count(nhead: int, voxels: int, context_len: int) -> int:
    """Elemente, die eine Cross-Attention gleichzeitig haelt.

    Gemessen an ``CrossAttention.forward``: Score-Matrix (``q @ k``), die Kopie
    aus ``masked_fill``, die Softmax-Ausgabe und die Dropout-Kopie.
    """
    return 4 * max(1, int(nhead)) * max(1, int(voxels)) * max(1, int(context_len))


def _resblock_element_count(in_ch: int, out_ch: int, voxels: int) -> int:
    """Elemente eines ResBlocks (GroupNorm/SiLU/Konvolutionen + Skip)."""
    return (4 * (int(in_ch) + int(out_ch)) + 2 * int(out_ch)) * max(1, int(voxels))


def unet_activation_bytes(batch_size: int, channels: int,
                          multipliers: Tuple[int, ...],
                          grid_size: Tuple[int, int, int], nhead: int,
                          context_len: int, item_bytes: int = 4) -> int:
    """Aktivierungen eines UNet-Trainingsschritts (Schaetzung).

    Blockweise Summe ueber Encoder, Middle und Decoder - kalibriert an
    Messungen auf einer GTX 1060 (Grid 32^3, Batch 16, channels 16,
    Multiplikatoren (1,2,2), 2 Heads, 64 Text-Tokens): gemessen ~3,0 GiB,
    Schaetzung ~3,4 GiB (bewusst etwas zu hoch, damit der Plan nicht zu
    optimistisch ist - zu niedrig geschaetzt kostet ein CUDA-OOM).
    """
    batch = max(1, int(batch_size))
    full_voxels = voxel_count(grid_size)
    voxels = full_voxels
    total = 0.0
    in_ch = int(channels)

    for mult in multipliers:                     # Encoder: DownBlock3D
        out_ch = int(channels) * int(mult)
        total += _resblock_element_count(in_ch, out_ch, voxels)
        total += _cross_attn_element_count(nhead, voxels, context_len)
        total += 2 * out_ch * voxels             # Skip + Downsampling-Ausgang
        voxels = max(1, voxels // 8)
        in_ch = out_ch

    for _ in range(2):                           # MiddleBlock3D (2 ResBlocks)
        total += _resblock_element_count(in_ch, in_ch, voxels)
        total += _cross_attn_element_count(nhead, voxels, context_len)

    for mult in reversed(list(multipliers)):     # Decoder: UpBlock3D
        out_ch = int(channels) * int(mult)
        voxels = min(full_voxels, voxels * 8)
        total += _resblock_element_count(out_ch, out_ch, voxels)
        total += _cross_attn_element_count(nhead, voxels, context_len)
        total += 2 * out_ch * voxels             # Upsampling + Konkatenation

    total += int(channels) * full_voxels         # Ausgang vor out_proj
    return int(total * batch * int(item_bytes))


# ─── Chunk- und Batch-Planung ───

def plan_chunk_voxels(voxels: int, batch_size: int, num_blocks: int,
                      budget_bytes: int, item_bytes: int = 4,
                      overlap_bytes: int = 0) -> int:
    """Groesste sinnvolle Chunk-Groesse fuer die Logits-Berechnung.

    Ungechunkt kostet der Logits-Tensor ``batch * num_blocks * voxels``
    Elemente. Im Chunk-Betrieb kostet *jede* Voxelposition ``batch *
    num_blocks`` Elemente; mit den Temporaries der Kreuzentropie (Faktor
    ``CHUNK_OVERHEAD``) folgt das Speicher-Limit direkt aus dem freien Budget.
    Zusaetzlich wird auf ``TARGET_CHUNK_BYTES`` gedeckelt: zu grosse Chunks sind
    messbar langsamer (Speicherbandbreite), zu kleine kosten Overhead.
    """
    voxels = max(1, int(voxels))
    override = _env_chunk_override()
    if override is not None:
        return max(1, min(voxels, override))

    per_voxel = (max(1, int(batch_size)) * max(1, int(num_blocks))
                 * int(item_bytes) * CHUNK_OVERHEAD)
    if per_voxel <= 0:
        return voxels
    limits = [voxels, max(1, int(TARGET_CHUNK_BYTES / max(1, int(batch_size))
                                 / max(1, int(num_blocks)) / int(item_bytes)))]
    remaining = int(budget_bytes) - int(overlap_bytes)
    if remaining > 0:
        limits.append(int(remaining / per_voxel))
    chunk = max(MIN_CHUNK_VOXELS, min(limits))
    if chunk >= voxels:
        return voxels
    # Auf Zweierpotenz abrunden: gleich grosse Chunks, nur der letzte bleibt klein.
    power = 1
    while power * 2 <= chunk:
        power *= 2
    return max(MIN_CHUNK_VOXELS, min(voxels, power))


_BATCH_STEPS = (1, 2, 4, 8, 16, 32, 64, 128)


def _next_lower_batch(batch: int) -> Optional[int]:
    """Naechstkleinere uebliche Batch-Groesse (fuer die Reduzierung)."""
    smaller = [value for value in _BATCH_STEPS if value < batch]
    return smaller[-1] if smaller else None


def _accumulation_steps(requested: int, micro: int) -> int:
    """Wie viele Mikro-Batches braucht der gewuenschte Batch?"""
    return -(-max(1, int(requested)) // max(1, int(micro)))


def _reduction_plan(plan: TrainingPlan, requested: int, allow_accumulation: bool,
                    reason: str) -> TrainingPlan:
    """Reduzierten Batch beschreiben - mit oder ohne Gradient-Akkumulation.

    Mit Akkumulation bleibt die *effektive* Batch-Groesse erhalten: ein
    Optimizer-Schritt sieht weiterhin ``requested`` Beispiele, nur eben in
    ``accum_steps`` Haeppchen (siehe ``app/grad_accum.py``). Ohne sie wird wie
    vorher nur verkleinert - das aendert die Trainingsdynamik.
    """
    steps = _accumulation_steps(requested, plan.batch_size) if allow_accumulation else 1
    plan = replace(plan, accum_steps=steps)
    if plan.accumulated:
        return replace(plan, suggestion=(
            f"Batch-Groesse von {requested} auf {plan.batch_size} reduziert, aber als "
            f"{steps} Mikro-Batches akkumuliert (effektiv {plan.effective_batch_size}, "
            f"Gradient Accumulation) - {reason}"))
    return replace(plan, suggestion=(
        f"Batch-Groesse von {requested} auf {plan.batch_size} reduziert - {reason}"))


def plan_diffusion_training(
    grid_size: Tuple[int, int, int],
    batch_size: int,
    num_blocks: int,
    *,
    device: Optional[torch.device] = None,
    channels: int = 16,
    channel_multipliers: Tuple[int, ...] = (1, 2, 2),
    d_model: int = 32,
    cross_attn_heads: int = 2,
    context_dim: int = 3072,
    context_len: int = 64,
    item_bytes: int = 4,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    reserve_bytes: int = 0,
    forced_chunk_voxels: Optional[int] = None,
    budget_bytes: Optional[int] = None,
    allow_accumulation: bool = True,
) -> TrainingPlan:
    """Plan fuer Diffusions-Training (VoxelDiffusionModel/TransformerDiffusionModel).

    Die Logits werden in Chunks berechnet, deshalb ist der Engpass nur noch die
    Aktivierungsmenge des UNet plus *ein* Chunk. Passt die gewuenschte
    Batch-Groesse nicht, wird die groesste passende gewaehlt; passt selbst
    Batch 1 nicht, ist ``fits=False`` und ``limit`` nennt den Verursacher.

    Mit ``allow_accumulation`` (Standard) bleibt die effektive Batch-Groesse
    erhalten: ``accum_steps`` sagt, in wie viele Mikro-Batches der Wunsch-Batch
    dafuer geteilt wird (siehe ``app/grad_accum.py``).

    ``budget_bytes`` ueberschreibt den erkannten freien Speicher (Tests).
    """
    voxels = voxel_count(grid_size)
    requested = max(1, min(128, int(batch_size)))
    info = _budget_info(device, safety_fraction, reserve_bytes, budget_bytes)

    def _plan_for(batch: int) -> TrainingPlan:
        activation = unet_activation_bytes(batch, channels, channel_multipliers,
                                           grid_size, cross_attn_heads, context_len,
                                           item_bytes)
        params = unet_parameter_bytes(channels, channel_multipliers, d_model,
                                      num_blocks, context_dim) * PARAM_STATE_FACTOR
        base = activation + params
        if forced_chunk_voxels:
            chunk = max(1, min(voxels, int(forced_chunk_voxels)))
        else:
            chunk = plan_chunk_voxels(voxels, batch, num_blocks, info.budget_bytes,
                                      item_bytes, overlap_bytes=base)
        # Achtung: der Chunk kostet batch * num_blocks je Voxelposition.
        chunk_bytes = int(logits_bytes(batch, num_blocks, chunk, item_bytes)
                          * CHUNK_OVERHEAD)
        estimated = base + chunk_bytes
        return TrainingPlan(
            batch_size=batch, requested_batch_size=requested,
            chunk_voxels=int(chunk), voxels=voxels,
            logits_bytes=logits_bytes(batch, num_blocks, voxels, item_bytes),
            activation_bytes=activation, param_bytes=params,
            estimated_bytes=estimated, budget_bytes=info.budget_bytes,
            available_bytes=info.free_bytes,
            fits=estimated <= info.budget_bytes,
            limit="ok",
            details={"chunk_bytes": float(chunk_bytes),
                     "chunk_logits_bytes": float(logits_bytes(batch, num_blocks,
                                                              chunk, item_bytes))},
        )

    if not guard_enabled():
        plan = _plan_for(requested)
        return replace(plan, fits=True,
                       suggestion="VRAM-Guard deaktiviert (MINECRAFT_AI_VRAM_GUARD=0)")

    candidate = requested
    plan = _plan_for(candidate)
    while not plan.fits:
        lower = _next_lower_batch(candidate)
        if lower is None:
            break
        candidate = lower
        plan = _plan_for(candidate)

    if plan.fits:
        if plan.reduced:
            return _reduction_plan(
                plan, requested, allow_accumulation,
                reason=(f"Logits {format_gib(plan.logits_bytes)} GiB, "
                        f"{plan.num_chunks} Chunks"))
        return plan

    hint = largest_fitting_grid(info.budget_bytes, num_blocks, channels,
                                channel_multipliers, d_model, cross_attn_heads,
                                context_len, item_bytes)
    limit = "activations"
    if plan.param_bytes > plan.activation_bytes:
        limit = "params"
    return replace(plan, limit=limit, suggestion=(
        f"Batch 1 passt nicht: Aktivierungen "
        f"{format_gib(plan.activation_bytes)} GiB"
        + (f", groesstes passendes Grid {hint}" if hint else "")))


def largest_fitting_grid(budget_bytes: int, num_blocks: int, channels: int,
                         channel_multipliers: Tuple[int, ...], d_model: int,
                         cross_attn_heads: int, context_len: int,
                         item_bytes: int = 4) -> Optional[str]:
    """Groesstes Grid aus ALLOWED_GRID_SIZES, das mit Batch 1 noch passt."""
    best_label = None
    params = unet_parameter_bytes(channels, channel_multipliers, d_model,
                                  num_blocks, 3072) * PARAM_STATE_FACTOR
    for grid in sorted(ALLOWED_GRID_SIZES.values(), key=voxel_count):
        voxels = voxel_count(grid)
        activation = unet_activation_bytes(1, channels, channel_multipliers, grid,
                                           cross_attn_heads, context_len, item_bytes)
        remaining = int(budget_bytes) - activation - params
        chunk = plan_chunk_voxels(voxels, 1, num_blocks, remaining, item_bytes)
        chunk_bytes = int(logits_bytes(1, num_blocks, chunk, item_bytes) * CHUNK_OVERHEAD)
        if activation + params + chunk_bytes <= budget_bytes:
            best_label = f"{grid[0]}x{grid[1]}x{grid[2]}"
        else:
            break
    return best_label


# ─── Shared-Weight-Transformer (model.py) ───

def _transformer_layer_bytes(d_model: int, num_layers: int,
                             dim_feedforward: int) -> int:
    """Parameter des VoxelTransformer (Gewichte), grob."""
    per_layer = (4 * d_model * d_model        # self-attention
                 + 3 * d_model * d_model      # cross-attention
                 + 2 * d_model * dim_feedforward
                 + 6 * d_model)
    return max(1, per_layer * max(1, num_layers)) * 4


def transformer_estimates(grid_size: Tuple[int, int, int], batch: int,
                          block_vocab_size: int, d_model: int, nhead: int,
                          num_layers: int, dim_feedforward: int,
                          item_bytes: int = 4) -> Tuple[int, int, int]:
    """(Attention-Bytes, Logits-Bytes, Rest) fuer den VoxelTransformer.

    Die Self-Attention laeuft ueber ALLE Voxel und ist nicht chunkbar, deshalb
    ist sie der harte Engpass: bei Grid 32^3 sind das 6 Heads * 32768^2 * 4 Byte
    = 24 GiB pro Matrix und Layer. Gerechnet wird eine Matrix je Layer plus eine
    transiente Kopie (``ATTENTION_TRANSIENT_COPIES``) - Grid 16^3 kostet 64x
    weniger.
    """
    voxels = voxel_count(grid_size)
    matrix = attention_bytes(batch, nhead, voxels, voxels, item_bytes)
    attn = int(matrix * (max(1, int(num_layers)) * 1.0 + ATTENTION_TRANSIENT_COPIES))
    logits = logits_bytes(batch, block_vocab_size, voxels, item_bytes)
    embeddings = ((block_vocab_size + 1) * d_model + voxels * d_model
                  + 2 * d_model)
    states = (_transformer_layer_bytes(d_model, num_layers, dim_feedforward)
              + embeddings * 4) * PARAM_STATE_FACTOR
    sequence = batch * voxels * d_model * item_bytes
    return attn, logits, states + sequence


def plan_transformer_training(
    grid_size: Tuple[int, int, int],
    batch_size: int,
    block_vocab_size: int,
    *,
    device: Optional[torch.device] = None,
    d_model: int = 192,
    nhead: int = 6,
    num_layers: int = 5,
    dim_feedforward: int = 768,
    item_bytes: int = 4,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    reserve_bytes: int = 0,
    budget_bytes: Optional[int] = None,
    allow_accumulation: bool = True,
) -> TrainingPlan:
    """Plan fuer den Shared-Weight-Transformer (Grid-Limit + Batch-Reduktion).

    Hier ist die Self-Attention ``[B, heads, Voxel, Voxel]`` der Engpass - sie
    laesst sich nicht chunken, waechst aber (wie der Rest) mit der Batch-Groesse.
    Der Plan reduziert deshalb die *physische* Batch-Groesse; mit
    ``allow_accumulation`` (Standard) haelt ``accum_steps`` die effektive
    Batch-Groesse dabei (siehe ``app/grad_accum.py``).

    ``budget_bytes`` ueberschreibt den erkannten freien Speicher (Tests).
    """
    voxels = voxel_count(grid_size)
    requested = max(1, min(128, int(batch_size)))
    info = _budget_info(device, safety_fraction, reserve_bytes, budget_bytes)

    def _plan_for(batch: int) -> TrainingPlan:
        attn, logits, rest = transformer_estimates(grid_size, batch,
                                                   block_vocab_size, d_model,
                                                   nhead, num_layers,
                                                   dim_feedforward, item_bytes)
        return TrainingPlan(
            batch_size=batch, requested_batch_size=requested,
            chunk_voxels=voxels, voxels=voxels, logits_bytes=logits,
            activation_bytes=attn, param_bytes=rest,
            estimated_bytes=attn + logits + rest,
            budget_bytes=info.budget_bytes, available_bytes=info.free_bytes,
            fits=attn + logits + rest <= info.budget_bytes, limit="ok",
        )

    if not guard_enabled():
        return replace(_plan_for(requested), fits=True,
                       suggestion="VRAM-Guard deaktiviert (MINECRAFT_AI_VRAM_GUARD=0)")

    candidate = requested
    plan = _plan_for(candidate)
    while not plan.fits:
        lower = _next_lower_batch(candidate)
        if lower is None:
            break
        candidate = lower
        plan = _plan_for(candidate)

    if plan.fits:
        if plan.reduced:
            return _reduction_plan(
                plan, requested, allow_accumulation,
                reason=f"Attention {format_gib(plan.activation_bytes)} GiB")
        return plan

    hint = None
    for grid in sorted(ALLOWED_GRID_SIZES.values(), key=voxel_count):
        if voxel_count(grid) >= voxels:
            continue
        attn, logits, rest = transformer_estimates(grid, 1, block_vocab_size,
                                                   d_model, nhead, num_layers,
                                                   dim_feedforward, item_bytes)
        if attn + logits + rest <= info.budget_bytes:
            hint = f"{grid[0]}x{grid[1]}x{grid[2]}"
    return replace(plan, limit="attention", suggestion=(
        f"Attention-Matrix {format_gib(plan.activation_bytes)} GiB "
        f"(waechst mit Voxel^2, nicht chunkbar)"
        + (f" - mit Batch 1 passend waere Grid {hint}" if hint else "")))


def describe_training_plan(plan: TrainingPlan, language: str = "de") -> str:
    """Text fuer Protokoll/Status (ohne Tk, daher auch im Thread nutzbar)."""
    if not plan.fits:
        if language == "de":
            text = (f"VRAM zu klein: Logits {format_gib(plan.logits_bytes)} GiB, "
                    f"geschaetzter Bedarf {format_gib(plan.estimated_bytes)} GiB, "
                    f"nutzbar {format_gib(plan.budget_bytes)} GiB ({plan.limit})")
        else:
            text = (f"Not enough VRAM: logits {format_gib(plan.logits_bytes)} GiB, "
                    f"estimated need {format_gib(plan.estimated_bytes)} GiB, "
                    f"usable {format_gib(plan.budget_bytes)} GiB ({plan.limit})")
        if plan.suggestion:
            text += f" - {plan.suggestion}"
        return text
    return plan.describe(language)



# ─── Generierung (nur Vorwaerts-Rechnung) ───

@dataclass(frozen=True)
class GenerationPlan:
    """Ergebnis der Speicherplanung fuer einen Generierungslauf.

    ``logits_bytes`` ist - wie bei :class:`TrainingPlan` - der *ungechunkte*
    Tensor, damit die Meldung direkt mit der Fehlermeldung von CUDA
    vergleichbar bleibt; ``estimated_bytes`` rechnet dagegen mit einem Chunk.
    """

    chunk_voxels: int
    voxels: int
    logits_bytes: int
    activation_bytes: int
    estimated_bytes: int
    budget_bytes: int
    available_bytes: int
    fits: bool
    suggestion: Optional[str] = None

    @property
    def chunked(self) -> bool:
        return 0 < self.chunk_voxels < self.voxels

    @property
    def num_chunks(self) -> int:
        if not self.chunked:
            return 1
        return (self.voxels + self.chunk_voxels - 1) // self.chunk_voxels

    def describe(self, language: str = "de") -> str:
        if language == "de":
            text = (f"VRAM-Plan (Generierung): Logits {format_gib(self.logits_bytes)} GiB "
                    f"({self.num_chunks} Chunks a {self.chunk_voxels} Voxel), "
                    f"geschaetzt {format_gib(self.estimated_bytes)} von "
                    f"{format_gib(self.budget_bytes)} GiB nutzbar")
        else:
            text = (f"VRAM plan (generation): logits {format_gib(self.logits_bytes)} GiB "
                    f"({self.num_chunks} chunks of {self.chunk_voxels} voxels), "
                    f"estimated {format_gib(self.estimated_bytes)} of "
                    f"{format_gib(self.budget_bytes)} GiB usable")
        if self.suggestion:
            text += f" - {self.suggestion}"
        return text


def unet_inference_bytes(batch_size: int, channels: int,
                         multipliers: Tuple[int, ...],
                         grid_size: Tuple[int, int, int], nhead: int = 1,
                         context_len: int = 1, item_bytes: int = 4) -> int:
    """Aktivierungen eines UNet-Vorwaerts-Laufs (Schaetzung, ohne Backward).

    Baut auf :func:`unet_activation_bytes` auf und skaliert mit
    ``INFERENCE_ACTIVATION_FACTOR``: ohne Autograd existieren weder die
    gespeicherten Zwischenergebnisse noch die Gradienten-Kopien.
    """
    return int(unet_activation_bytes(batch_size, channels, multipliers, grid_size,
                                     nhead, context_len, item_bytes)
               * INFERENCE_ACTIVATION_FACTOR)


def transformer_generation_bytes(grid_size: Tuple[int, int, int], batch_size: int,
                                 block_vocab_size: int, d_model: int, nhead: int,
                                 num_layers: int, dim_feedforward: int,
                                 item_bytes: int = 4) -> Tuple[int, int]:
    """(Attention-Bytes, Rest) fuer einen Vorwaerts-Lauf des VoxelTransformers.

    Wie :func:`transformer_estimates`, aber ohne Trainings-Zuschlaege: nur *eine*
    Score-Matrix je Layer (kein Backward, keine transiente Kopie) und nur die
    Gewichte (keine Gradienten, keine Adam-Zustaende).
    """
    voxels = voxel_count(grid_size)
    attn = attention_bytes(batch_size, nhead, voxels, voxels,
                           item_bytes) * max(1, int(num_layers))
    embeddings = ((block_vocab_size + 1) * d_model + voxels * d_model + 2 * d_model) * 4
    states = _transformer_layer_bytes(d_model, num_layers, dim_feedforward) + embeddings
    sequence = max(1, int(batch_size)) * voxels * d_model * item_bytes
    return int(attn), int(states + sequence)


def plan_generation(
    grid_size: Tuple[int, int, int],
    num_blocks: int,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    activation_bytes: int = 0,
    item_bytes: int = 4,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    budget_bytes: Optional[int] = None,
) -> GenerationPlan:
    """Chunk-Groesse fuer die Logits *beim Generieren* bestimmen.

    ``activation_bytes`` sind die nicht chunkbaren Kosten (UNet-Aktivierungen
    bzw. Attention + Gewichte des Transformers); der Rest des Budgets steht den
    Logits zur Verfuegung. Ist selbst der kleinste Chunk zu gross, meldet der
    Plan ``fits=False`` samt Vorschlag statt eines CUDA-Absturzes.

    Wichtig: vor dem Aufruf ``torch.cuda.empty_cache()`` ausfuehren. Die
    Planung misst den *treiber*-freien Speicher (``mem_get_info``), und der
    PyTorch-Caching-Allocator behaelt die Bloecke eines Trainingslaufs sonst
    als "belegt" - der Plan waere dann unnoetig pessimistisch.
    """
    voxels = voxel_count(grid_size)
    requested = max(1, int(batch_size))
    info = _budget_info(device, safety_fraction, 0, budget_bytes)
    full_logits = logits_bytes(requested, num_blocks, voxels, item_bytes)

    def _plan_for(chunk: int) -> GenerationPlan:
        chunk_bytes = int(logits_bytes(requested, num_blocks, chunk, item_bytes)
                          * CHUNK_OVERHEAD)
        total = int(activation_bytes) + chunk_bytes
        return GenerationPlan(
            chunk_voxels=chunk, voxels=voxels, logits_bytes=full_logits,
            activation_bytes=int(activation_bytes), estimated_bytes=total,
            budget_bytes=info.budget_bytes, available_bytes=info.free_bytes,
            fits=total <= info.budget_bytes,
        )

    if not guard_enabled():
        return replace(_plan_for(voxels), fits=True,
                       suggestion="VRAM-Guard deaktiviert (MINECRAFT_AI_VRAM_GUARD=0)")

    remaining = info.budget_bytes - int(activation_bytes)
    chunk = plan_chunk_voxels(voxels, requested, num_blocks, max(0, remaining),
                              item_bytes)
    plan = _plan_for(chunk)
    if plan.fits:
        return plan

    # Selbst der kleinste Chunk passt nicht mehr -> kleineres Grid vorschlagen.
    hint = None
    for grid in sorted(ALLOWED_GRID_SIZES.values(), key=voxel_count):
        smaller = voxel_count(grid)
        if smaller >= voxels:
            continue
        smaller_chunk = plan_chunk_voxels(smaller, requested, num_blocks,
                                          max(0, remaining), item_bytes)
        cost = int(logits_bytes(requested, num_blocks, smaller_chunk, item_bytes)
                   * CHUNK_OVERHEAD)
        if int(activation_bytes) + cost <= info.budget_bytes:
            hint = f"{grid[0]}x{grid[1]}x{grid[2]}"
    return replace(plan, suggestion=(
        f"Aktivierungen {format_gib(activation_bytes)} GiB, "
        f"kleinster Chunk {format_gib(plan.estimated_bytes - int(activation_bytes))} GiB"
        + (f" - mit kleinerem Grid ({hint}) wuerde es passen" if hint else "")))


def describe_generation_plan(plan: GenerationPlan, language: str = "de") -> str:
    """Text fuer Statuszeile/Protokoll (ohne Tk, daher auch im Thread nutzbar)."""
    if not plan.fits:
        if language == "de":
            text = (f"Generierung: VRAM zu klein - Logits {format_gib(plan.logits_bytes)} GiB, "
                    f"geschaetzter Bedarf {format_gib(plan.estimated_bytes)} GiB, "
                    f"nutzbar {format_gib(plan.budget_bytes)} GiB")
        else:
            text = (f"Generation: not enough VRAM - logits {format_gib(plan.logits_bytes)} GiB, "
                    f"estimated need {format_gib(plan.estimated_bytes)} GiB, "
                    f"usable {format_gib(plan.budget_bytes)} GiB")
        if plan.suggestion:
            text += f" - {plan.suggestion}"
        return text
    return plan.describe(language)
