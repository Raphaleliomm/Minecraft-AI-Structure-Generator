"""Gradient-Akkumulation: grosser Batch trotz kleinem VRAM.

Der VRAM-Guard (``app/training_memory.py``) verkleinert die Batch-Groesse, wenn
Aktivierungen, Logits oder Optimizer-Zustaende nicht in den freien Speicher
passen. Diese Verkleinerung aendert aber das Training: sechzehn Beispiele in
einem Schritt verhalten sich anders als vier Schritte mit je vier Beispielen.

Dieses Modul behaelt die *effektive* Batch-Groesse bei:

  * Der Batch wird in Mikro-Batches geteilt, die einzeln vorwaerts und
    rueckwaerts gerechnet werden - der Speicherbedarf entspricht dem
    Mikro-Batch, nicht dem ganzen Batch.
  * Der Verlust jedes Mikro-Batches geht mit seinem Anteil am Gesamt-Batch
    gewichtet in die Gradienten ein, damit die Summe der Gradienten dem
    Mittelwert ueber den ganzen Batch entspricht (nicht der Summe - das waere
    ein Vielfaches der Lernrate).
  * ``optimizer.step()`` passiert genau einmal pro Batch und danach wird
    geclippt (wie beim grossen Batch, nicht pro Mikro-Batch).

Der Anteil wird an der *Gewichtsmasse* gemessen (``sample_mass``): der Loss ist
ein gewichteter Mittelwert ``sum(ce * w) / sum(w)``, und die Gewichte
unterscheiden sich stark (Luft-Gewicht 75 gegen Block 1, dazu das
Sample-Gewicht). Wuerde man nach Beispielzahl teilen, bekaeme ein Haeppchen aus
luftigen Strukturen denselben Einfluss wie eines aus kompakten Bloecken - die
Gradienten waeren dann nicht mehr die des grossen Batches (gemessen: bis zu
180 % Abweichung). Mit der Gewichtsmasse gilt dagegen exakt

    Gradient = Summe_i (m_i / M) * Gradient_i(Teil-Loss) = Summe_i Gradient(S_i) / M

also bis auf die Reihenfolge der Summation genau der Optimizer-Schritt des
grossen Batches - ohne die Lernrate anzupassen. Die Masse liefert
``per_sample_weight_mass()`` (dieselbe Formel wie die gewichtete
Kreuzentropie); ohne sie teilt ``iter_micro_batches`` nach Beispielzahl.

Abschaltbar ueber ``MINECRAFT_AI_GRAD_ACCUM=0`` (dann wird die Batch-Groesse
wie vorher einfach verkleinert) bzw. ueber den Schalter im Training-Tab.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

ENV = "MINECRAFT_AI_GRAD_ACCUM"


def accumulation_enabled() -> bool:
    """Standard fuer die Akkumulation (an, solange die Umgebungsvariable nicht aus ist)."""
    value = os.environ.get(ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def micro_batch_sizes(total: int, max_size: int) -> Tuple[int, ...]:
    """``total`` Beispiele moeglichst gleichmaessig in Haeppchen <= ``max_size``.

    Die Summe ist exakt ``total``, damit ein Optimizer-Schritt genau den
    gewuenschten Batch sieht. Beispiel ``total=10, max_size=4`` -> ``(4, 3, 3)``
    und nicht ``(4, 4, 2)``: zwei gleich grosse Haeppchen sind stabiler als ein
    doppelt so schwerer Rest.

    Passt der ganze Batch (``max_size >= total``), ist das Ergebnis ``(total,)``
    - der Aufrufer kann dann den schnellen Pfad ohne Kopien nehmen.
    """
    total = max(1, int(total))
    max_size = max(1, int(max_size))
    if max_size >= total:
        return (total,)
    steps = -(-total // max_size)               # ceil(total / max_size)
    sizes: List[int] = []
    remaining = total
    for index in range(steps):
        left = steps - index
        size = min(max_size, -(-remaining // left))
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def _batch_length(value: Any) -> Optional[int]:
    """Laenge der Batch-Dimension eines Batch-Werts (``None`` = kein Batch-Feld)."""
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.dim() > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


# ─── Gewichte der gewichteten Kreuzentropie ───
# Der Loss lautet sum(ce * w) / sum(w) mit einem Gewicht je Voxel. Diese Helfer
# sind die einzige Definition dieser Gewichte (app/diffusion_model.py nutzt sie
# ebenfalls) - sonst wuerde die Akkumulation mit anderen Gewichten rechnen als
# das Training.

def weight_vector(batch: Dict[str, Any], key: str, batch_size: int,
                  device: torch.device) -> torch.Tensor:
    """Gewichtsvektor aus dem Batch holen (fehlend -> Einsen, wie bisher)."""
    weights = batch.get(key)
    if not isinstance(weights, torch.Tensor):
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    return weights.to(device=device, dtype=torch.float32).reshape(-1)


def per_voxel_weight(flat_ids: torch.Tensor, batch: Dict[str, Any],
                     batch_size: int) -> torch.Tensor:
    """Gewichtsmatrix ``[B, Voxel]`` der gewichteten Kreuzentropie.

    Luft (ID 0) bekommt das Luft-Gewicht, alles andere das Block-Gewicht und
    jeder Block zusaetzlich das Sample-Gewicht. ``flat_ids`` sind die auf
    ``[B, Voxel]`` reduzierten Block-IDs - das Klemmen der IDs spielt fuer die
    Luft-Maske keine Rolle, weil dabei keine neue Null entstehen kann.
    """
    sample_weight = weight_vector(batch, "sample_weight", batch_size, flat_ids.device)
    per_block = weight_vector(batch, "per_block_weight", batch_size, flat_ids.device)
    per_air = weight_vector(batch, "per_air_weight", batch_size, flat_ids.device)
    return torch.where(flat_ids == 0, per_air.view(batch_size, 1),
                       per_block.view(batch_size, 1)) * sample_weight.view(batch_size, 1)


def per_sample_weight_mass(voxel_ids: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
    """Summe der Loss-Gewichte je Beispiel (``[B]``).

    Genau der Wert, durch den die gewichtete Kreuzentropie teilt - und deshalb
    der richtige Anteil fuer die Gradient-Akkumulation. Wird auf den
    (guenstigen) CPU-Tensoren des Batches gerechnet: das ist billig gegenueber
    dem Training und muss nicht in den VRAM.

    Der ``clean``-Clamp der Trainingsschritte ist hier nicht noetig: er kann nur
    IDs oberhalb des Vokabulars veraendern und damit nie eine Luft-ID (0)
    erzeugen.
    """
    batch_size = int(voxel_ids.shape[0])
    flat = voxel_ids.reshape(batch_size, -1)
    return per_voxel_weight(flat, batch, batch_size).sum(dim=1)


def batch_total_size(batch: Dict[str, Any]) -> int:
    """Batch-Groesse eines Batch-Dictionaries.

    Batch-Dictionaries (Dataset-Default-Collate wie ``collate_context_batch``)
    enthalten immer mindestens ein Tensor-Feld, dessen erste Dimension die
    Batch-Groesse ist - die Suche nach dem ersten solchen Feld ist deshalb
    ausreichend und unabhaengig vom konkreten Layout.
    """
    for value in batch.values():
        length = _batch_length(value)
        if length is not None:
            return length
    return 1


def split_batch(batch: Dict[str, Any], sizes: Sequence[int]) -> List[Dict[str, Any]]:
    """Batch-Dictionary entlang der Batch-Dimension in Mikro-Batches teilen.

    Geteilt werden Tensoren (Dimension 0) und Listen/Sequenzen (z.B.
    ``prompt_text``); alle anderen Felder (Skalare, ``None``) werden unveraendert
    in jeden Mikro-Batch uebernommen. Die Summe der ``sizes`` muss der
    Batch-Groesse entsprechen.
    """
    total = batch_total_size(batch)
    parts: List[Dict[str, Any]] = [{} for _ in sizes]
    offset = 0
    for index, size in enumerate(sizes):
        end = offset + size
        for key, value in batch.items():
            length = _batch_length(value)
            if length is None or length != total:
                parts[index][key] = value
            else:
                parts[index][key] = value[offset:end]
        offset = end
    return parts


def iter_micro_batches(batch: Dict[str, Any], max_size: int,
                       sample_mass: Optional[torch.Tensor] = None,
                       ) -> Iterator[Tuple[Dict[str, Any], float]]:
    """Mikro-Batches samt Anteil am Gesamt-Batch durchlaufen.

    Der Anteil multipliziert den Verlust jedes Mikro-Batches, damit die Summe der
    Gradienten dem Mittelwert ueber den ganzen Batch entspricht (nicht dem
    ``accum_steps``-fachen). Der zurueckgegebene Loss bleibt der echte
    Mittelwert des Mikro-Batches - die Anzeige wird also nicht verfaelscht.

    ``sample_mass`` ist das Gewicht je Beispiel (siehe
    ``per_sample_weight_mass``). Ohne sie wird nach Beispielzahl geteilt, mit ihr
    nach Gewichtsmasse - nur damit ist die Summe der Gradienten exakt der
    Gradient des grossen Batches, auch bei sehr unterschiedlich gewichteten
    Beispielen (Luft-Gewicht!). Die Summe der Anteile ist immer 1.0.

    Passt der ganze Batch in den VRAM (``max_size >= total``), wird er
    unveraendert mit Anteil 1.0 zurueckgegeben - der schnelle Pfad ohne Kopien,
    den die Trainingsschleifen auch ohne Akkumulation nutzen.
    """
    total = batch_total_size(batch)
    sizes = micro_batch_sizes(total, max_size)
    if len(sizes) == 1:
        yield batch, 1.0
        return
    shares = _shares(sizes, sample_mass)
    for micro, share in zip(split_batch(batch, sizes), shares):
        yield micro, share


def _shares(sizes: Sequence[int], sample_mass: Optional[torch.Tensor]) -> List[float]:
    """Anteile der Mikro-Batches am Gesamt-Batch (Summe 1.0)."""
    total = float(sum(sizes))
    if sample_mass is None:
        return [size / total for size in sizes]
    masses = torch.as_tensor(sample_mass).reshape(-1).to(torch.float64)
    if int(masses.numel()) != len(sizes) and int(masses.numel()) != int(total):
        return [size / total for size in sizes]
    if int(masses.numel()) == int(total):
        # Masse je Beispiel: auf die Mikro-Batches aufsummieren.
        masses = torch.stack([masses[start:start + size].sum()
                              for start, size in _windows(sizes)])
    mass_total = float(masses.sum())
    if mass_total <= 0.0:
        return [size / total for size in sizes]
    return [float(mass) / mass_total for mass in masses]


def _windows(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    """(Start, Groesse) je Mikro-Batch."""
    windows: List[Tuple[int, int]] = []
    offset = 0
    for size in sizes:
        windows.append((offset, size))
        offset += size
    return windows


def apply_optimizer_step(model, optimizer, scaler=None, max_norm: float = 1.0) -> None:
    """Gradienten clippen und EINEN Optimizer-Schritt ausfuehren (AMP-fest).

    Wird am Ende eines Akkumulations-Fensters aufgerufen - genau einmal pro
    Batch, nicht pro Mikro-Batch. Mit AMP werden die skalierten Gradienten
    vorher einmal entpackt (``scaler.unscale_``), sonst haengt der Clip-Faktor
    vom Loss-Scaling ab.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
