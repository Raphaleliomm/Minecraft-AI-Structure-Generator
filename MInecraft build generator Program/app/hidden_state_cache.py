"""Pre-computed hidden states cache for TF-Diffusion training.
Caches encoder outputs once so training doesn't need to re-encode prompts every epoch."""
from __future__ import annotations

import json
import hashlib
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import numpy as np

from app.paths import CACHE_DIR
from app.transformer_encoder import FrozenTransformerEncoder
from dataset import split_prompt_text

logger = logging.getLogger(__name__)

# Der Cache ist mehrere GB gross (z.B. 1.7 GB fuer 6367 Prompts mit hidden_dim
# 3072). Ein normales torch.load kopiert ihn komplett in den RAM; mit mmap
# bleibt er auf der Platte und der Speicherbedarf sinkt auf die tatsaechlich
# gelesenen Zeilen. Abschaltbar ueber MINECRAFT_AI_CACHE_MMAP=0.
CACHE_MMAP_ENV = "MINECRAFT_AI_CACHE_MMAP"


def _mmap_enabled(mmap: Optional[bool] = None) -> bool:
    if mmap is not None:
        return bool(mmap)
    value = os.environ.get(CACHE_MMAP_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _load_cache_file(data_path: Path, mmap: bool = True) -> Dict:
    """Cache-Datei laden - bevorzugt als Speicher-Mapping.

    Reihenfolge:
      1. ``mmap=True`` + ``weights_only=True`` (kein RAM-Kopie, schnellster Weg)
      2. ``map_location="cpu"`` + ``weights_only=True`` (z.B. auf Dateisystemen
         ohne mmap-Unterstuetzung)
      3. klassisches ``torch.load`` (alte Caches mit eigenen Objekten)

    Die mmap-Variante ist nur lesbar genutzt: das Dataset schneidet Zeilen
    heraus und der Collate stapelt sie in neue Tensoren, es wird nie in den
    Cache zurueckgeschrieben.
    """
    if _mmap_enabled(mmap):
        try:
            return torch.load(data_path, mmap=True, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - jeder Fehler -> Fallback
            logger.info("mmap-Laden nicht moeglich (%s) - lade normal", exc)
    try:
        return torch.load(data_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - z.B. alter Pickle-Inhalt
        logger.info("weights_only-Laden nicht moeglich (%s) - lade klassisch", exc)
        return torch.load(data_path, map_location="cpu")


def cache_key(encoder_name: str) -> str:
    """Generate a filesystem-safe cache key from encoder name."""
    return encoder_name.replace("/", "_").replace(" ", "_")


def _file_hash(path: Path) -> str:
    """SHA256 hash of a file's content."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _file_signature(path: Path) -> Tuple[float, int]:
    """Cheap change detection: (mtime, size).

    Hashing every structure file on every cache check meant reading gigabytes
    from disk whenever the user changed the cache selection.
    """
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)


def _split_prompts(text: str) -> List[str]:
    """Split a prompt text on '---' separators, filter out metadata lines.

    Delegates to dataset.split_prompt_text so cache and dataset always agree.
    """
    return split_prompt_text(text)


def _predict_token_lengths(
    encoder: FrozenTransformerEncoder,
    prompts: List[str],
) -> Optional[List[int]]:
    """Token count per prompt, measured with the encoder's own tokenizer.

    Knowing the lengths up front allows the cache to be written into one
    pre-sized buffer instead of holding one tensor per prompt plus a padded copy
    plus the ``torch.cat`` result (three copies of the whole data set).

    Returns ``None`` when the encoder has no usable tokenizer (custom/stub
    encoders) - the caller then falls back to collecting the ragged per-prompt
    tensors and concatenating them.
    """
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        return None
    max_length = int(getattr(encoder, "max_length", 512) or 512)
    lengths: List[int] = []
    try:
        for start in range(0, len(prompts), 256):
            chunk = prompts[start:start + 256]
            encoded = tokenizer(chunk, padding=False, truncation=True, max_length=max_length)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
    except Exception:  # noqa: BLE001 - any tokenizer quirk uses the safe fallback
        logger.warning("Token-Laengen konnten nicht vorab bestimmt werden", exc_info=True)
        return None
    if len(lengths) != len(prompts):
        return None
    return [int(length) for length in lengths]


def compute_hidden_states(
    encoder: FrozenTransformerEncoder,
    encoder_name: str,
    schem_files: List[Path],
    txt_files: List[Path],
    status_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """Compute hidden states for all training data and cache them.
    
    Handles multi-prompt .txt files (prompts separated by '---').
    Each prompt becomes its own cache entry, and the file list tracks
    which schem/prompt combination produced each row.

    The cache is written *ragged* (no padding): all rows are concatenated along
    the token axis and their lengths are stored next to them. Padding every row
    to the longest prompt of the data set is what made this step blow up - with
    6367 prompts (median 14 tokens, longest 512) and hidden_dim=3072 the padded
    array needs ~20 GB in fp16, the ragged one only ~1.7 GB. The previous
    implementation also held three copies of it at once (per-prompt list +
    padded copies + torch.cat result), which ended in
    "DefaultCPUAllocator: not enough memory". Padding now happens per batch
    (see dataset.collate_context_batch).
    
    Args:
        encoder: Loaded frozen transformer encoder
        encoder_name: Display name of the encoder (for cache key)
        schem_files: List of .schem file paths
        txt_files: List of corresponding .txt prompt file paths
        status_callback: Optional callback for status updates
        
    Returns:
        Path to the cache directory
    """
    key = cache_key(encoder_name)
    out_dir = CACHE_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: every prompt becomes exactly one cache row ──
    # Row info: (schem, txt, schem signature, txt signature, prompt index, prompt)
    entries: List[Tuple[Path, Path, Tuple[float, int], Tuple[float, int], int, str]] = []
    for schem_path, txt_path in zip(schem_files, txt_files):
        prompts = _split_prompts(txt_path.read_text(encoding="utf-8"))
        schem_sig = _file_signature(schem_path)
        txt_sig = _file_signature(txt_path)
        for pi, prompt in enumerate(prompts):
            entries.append((schem_path, txt_path, schem_sig, txt_sig, pi, prompt))

    if not entries:
        raise ValueError("Keine Prompts zum Encodieren gefunden")

    total_prompts = len(entries)
    predicted_lengths = _predict_token_lengths(encoder, [entry[5] for entry in entries])
    capacity = int(sum(predicted_lengths)) if predicted_lengths is not None else 0

    # ── Pass 2: encode and write straight into a pre-sized buffer ──
    # Rows are stored back to back WITHOUT padding; the row offsets are the
    # cumulative sums of 'lengths'. Padding to the batch maximum happens later
    # in dataset.collate_context_batch(), where only a handful of rows at a time
    # are involved.
    hidden_buffer: Optional[torch.Tensor] = None
    mask_buffer: Optional[torch.Tensor] = None
    hidden_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    lengths: List[int] = []
    file_list_items: List[Dict] = []
    cursor = 0

    if status_callback:
        status_callback(f"Encodiere {total_prompts} Prompts ({capacity} Tokens)")

    for processed, (schem_path, txt_path, schem_sig, txt_sig, pi, prompt) in enumerate(entries, start=1):
        # Throttled status updates: one Tk event per prompt flooded the UI thread
        # for data sets with thousands of prompts.
        if status_callback and (processed == 1 or processed % 25 == 0):
            status_callback(f"Encodiere {processed}/{total_prompts}: {schem_path.name} prompt {pi + 1}")

        with torch.no_grad():
            encoded = encoder([prompt])
            hs = encoded["last_hidden_state"].detach().cpu()  # [1, seq_len, hidden_dim]
            mask = encoded["attention_mask"].detach().cpu()   # [1, seq_len]
        if hs.dim() == 3:
            hs = hs[0]      # [seq_len, hidden_dim]
        if mask.dim() == 2:
            mask = mask[0]  # [seq_len]

        row_len = int(hs.shape[0])
        if predicted_lengths is not None:
            if hidden_buffer is None:
                # ONE flat buffer for the whole data set: [total_tokens, hidden_dim]
                hidden_buffer = torch.empty((max(capacity, 1), int(hs.shape[1])), dtype=hs.dtype)
                mask_buffer = torch.zeros((max(capacity, 1),), dtype=mask.dtype)
            if cursor + row_len > hidden_buffer.shape[0]:
                # Only happens if the tokenizer disagrees with encoder() - grow once.
                new_capacity = max(cursor + row_len, int(hidden_buffer.shape[0] * 1.25) + 512)
                grown_hidden = torch.empty((new_capacity, hidden_buffer.shape[1]),
                                           dtype=hidden_buffer.dtype)
                grown_hidden[:cursor] = hidden_buffer[:cursor]
                grown_mask = torch.zeros((new_capacity,), dtype=mask_buffer.dtype)
                grown_mask[:cursor] = mask_buffer[:cursor]
                hidden_buffer, mask_buffer = grown_hidden, grown_mask
            hidden_buffer[cursor:cursor + row_len] = hs
            mask_buffer[cursor:cursor + row_len] = mask
        else:
            # Fallback without a usable tokenizer: keep the unpadded rows and
            # concatenate them at the end (2x the final size, not 3x + padding).
            hidden_chunks.append(hs)
            mask_chunks.append(mask)

        cursor += row_len
        lengths.append(row_len)
        schem_mtime, schem_size = schem_sig
        txt_mtime, txt_size = txt_sig
        file_list_items.append({
            "schem": schem_path.name,
            "txt": txt_path.name,
            "mtime": schem_mtime,
            "size": schem_size,
            "txt_mtime": txt_mtime,
            "txt_size": txt_size,
            "prompt_index": pi,
            "prompt_preview": prompt[:80],
        })

    if hidden_buffer is not None:
        if cursor == hidden_buffer.shape[0]:
            hidden_states = hidden_buffer
            attention_masks = mask_buffer
        else:
            # Trim the (rare) over-allocation - do not keep or save slack.
            hidden_states = hidden_buffer[:cursor].clone()
            attention_masks = mask_buffer[:cursor].clone()
    else:
        hidden_states = torch.cat(hidden_chunks, dim=0)
        attention_masks = torch.cat(mask_chunks, dim=0)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    total_tokens = int(hidden_states.shape[0])
    max_seq_len = max(lengths)

    # Save: concatenated (unpadded) rows plus their lengths. Row offsets are
    # cumsum(lengths); see MultiSourceSchematicDataset._set_hidden_states().
    torch.save({
        "hidden_states": hidden_states,
        "attention_masks": attention_masks,
        "lengths": lengths_tensor,
        "layout": "ragged",
    }, out_dir / "data.pt")

    size_mb = (hidden_states.numel() * hidden_states.element_size()
               + attention_masks.numel() * attention_masks.element_size()) / 1_000_000

    metadata = {
        "encoder_name": encoder_name,
        "encoder_hf_id": getattr(encoder, "hf_id", ""),
        "hidden_dim": int(hidden_states.shape[1]),
        "layout": "ragged",
        "dtype": str(hidden_states.dtype).replace("torch.", ""),
        "num_samples": total_prompts,   # Total prompt entries
        "num_files": len(schem_files),  # Original schematic files
        "total_tokens": total_tokens,
        "max_seq_len": max_seq_len,
        "size_mb": round(size_mb, 1),
        "created": datetime.now().isoformat(),
        "files": file_list_items,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if status_callback:
        status_callback(f"{total_prompts} Hidden States ({len(schem_files)} Dateien, "
                        f"{size_mb:.0f} MB, max {max_seq_len} Tokens)")

    return out_dir


def load_hidden_states(encoder_name: str,
                       mmap: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached hidden states and attention masks.

    The cache is stored ragged: 'hidden_states' is the flat
    [total_tokens, hidden_dim] tensor, 'attention_masks' its flat
    [total_tokens] counterpart. Use load_hidden_states_raw() when the per-row
    lengths (row offsets) are needed as well.

    ``mmap`` (Standard: an) laesst den Cache auf der Platte und vermeidet die
    GB-grosse RAM-Kopie beim Trainingsstart.
    
    Returns:
        (hidden_states, attention_masks) tensors
    """
    key = cache_key(encoder_name)
    data_path = CACHE_DIR / key / "data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Kein Cache für Encoder '{encoder_name}' gefunden. "
                                f"Bitte zuerst Hidden States vorberechnen.")
    data = _load_cache_file(data_path, mmap=mmap)
    return data["hidden_states"], data["attention_masks"]


def load_hidden_states_raw(cache_dir: Path, mmap: Optional[bool] = None) -> Dict:
    """Load cached hidden states from a specific cache directory path.

    ``mmap`` (Standard: an) mappt die Datei statt sie in den RAM zu kopieren -
    der 1.7-GB-Cache eines 6367-Prompt-Datensatzes belegte vorher sofort den
    doppelten Speicher und brauchte beim Start mehrere Sekunden.

    Returns dict with:
        'hidden_states'   - flat [total_tokens, hidden_dim], rows concatenated
        'attention_masks' - flat [total_tokens]
        'lengths'         - [num_prompts] token count per row
        'layout'          - "ragged" (new) or "padded" (converted legacy cache)
        'metadata'        - parsed metadata.json

    Caches written before the ragged layout exist are rectangular
    [N, seq_len, hidden_dim] tensors with every row padded to the longest
    prompt. They are converted on load so that callers only ever see the flat
    form (row offsets = cumsum of 'lengths').
    """
    data_path = cache_dir / "data.pt"
    meta_path = cache_dir / "metadata.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden in {cache_dir}")
    data = _load_cache_file(data_path, mmap=mmap)
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Metadaten des Hidden-State-Caches unlesbar (%s): %s", meta_path, exc)

    hidden_states = data["hidden_states"]
    attention_masks = data["attention_masks"]

    if hidden_states.dim() == 3:
        # Legacy padded cache -> drop the padding (all rows are all-ones masks).
        lengths = attention_masks.reshape(attention_masks.shape[0], -1).sum(dim=1).to(torch.long)
        hidden_states = torch.cat(
            [hidden_states[i, :int(lengths[i])] for i in range(hidden_states.shape[0])], dim=0)
        attention_masks = torch.ones(int(lengths.sum()), dtype=torch.long)
        layout = "padded"
    else:
        lengths = data.get("lengths")
        if lengths is None:
            raise ValueError(
                f"Hidden-State-Cache ohne Zeilenlaengen (unbekanntes Format): {data_path}")
        lengths = lengths.to(torch.long).reshape(-1)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        attention_masks = attention_masks.reshape(-1)
        layout = "ragged"

    return {
        "hidden_states": hidden_states,
        "attention_masks": attention_masks,
        "lengths": lengths,
        "layout": layout,
        "metadata": metadata,
    }


def _entry_is_current(entry: Dict, schem_path: Path, txt_path: Path) -> bool:
    """Check a single cache entry against the current files.

    New caches store (mtime, size) signatures. Older caches only have SHA256
    hashes - those are still compared (slow, but keeps old caches usable).
    """
    try:
        mtime, size = _file_signature(schem_path)
        txt_mtime, txt_size = _file_signature(txt_path)
    except OSError:
        return False

    if "mtime" in entry and "size" in entry:
        if float(entry["mtime"]) != mtime or int(entry["size"]) != size:
            return False
        if "txt_mtime" in entry and "txt_size" in entry:
            if float(entry["txt_mtime"]) != txt_mtime or int(entry["txt_size"]) != txt_size:
                return False
        return True

    if "hash" in entry:
        try:
            return bool(entry.get("hash") == _file_hash(schem_path)
                        and entry.get("txt_hash") == _file_hash(txt_path))
        except OSError:
            return False

    return False


def validate_cache(encoder_name: str, schem_files: List[Path], txt_files: List[Path]) -> Dict:
    """Check if the cache is still valid (same files, same content signature).
    
    Handles multi-prompt txt files (prompts separated by '---').
    Checks that all schem files still match their cached signature and that
    the number of prompts matches.
    
    Returns:
        {"valid": bool, "cached": int, "total": int, "missing": int,
         "changed": int, "new_files": list, "message": str}
    """
    key = cache_key(encoder_name)
    cache_dir = CACHE_DIR / key
    
    # Count total prompts from all txt files (like dataset discovery does)
    total_prompts = 0
    for schem_path, txt_path in zip(schem_files, txt_files):
        prompts = _split_prompts(txt_path.read_text(encoding="utf-8"))
        total_prompts += len(prompts)
    
    result = {
        "valid": False,
        "cached": 0,
        "total": total_prompts,
        "num_files": len(schem_files),
        "missing": 0,
        "changed": 0,
        "new_files": [],
        "message": "",
    }

    if not cache_dir.exists():
        result["message"] = "Kein Cache vorhanden"
        result["missing"] = total_prompts
        return result

    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        result["message"] = "Cache beschädigt (keine metadata.json)"
        result["missing"] = total_prompts
        return result

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cache-Metadaten unlesbar (%s): %s", meta_path, exc)
        result["message"] = "Cache beschädigt (metadata.json ungültig)"
        result["missing"] = total_prompts
        return result

    cached_entries = metadata.get("files", [])
    result["cached"] = len(cached_entries)

    # Build lookup: for each schem file, collect all cached prompt entries
    cache_by_schem: Dict[str, List[dict]] = {}
    for entry in cached_entries:
        schem_name = entry.get("schem", "")
        if schem_name not in cache_by_schem:
            cache_by_schem[schem_name] = []
        cache_by_schem[schem_name].append(entry)

    # Check current files
    new_files = []
    changed_files = []
    current_names = set()
    expected_prompt_count = 0

    for schem_path, txt_path in zip(schem_files, txt_files):
        current_names.add(schem_path.name)
        try:
            prompts = _split_prompts(txt_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("%s konnte nicht gelesen werden (%s)", txt_path, exc)
            prompts = []
        expected_prompt_count += len(prompts)

        if schem_path.name not in cache_by_schem:
            new_files.append(schem_path.name)
        else:
            # mtime/size signature first (cheap), hashes only for legacy caches
            cached_entries_for_file = cache_by_schem[schem_path.name]
            if not any(
                _entry_is_current(entry, schem_path, txt_path)
                for entry in cached_entries_for_file
            ):
                changed_files.append(schem_path.name)

    # Check for deleted files
    deleted = [s for s in cache_by_schem if s not in current_names]

    # Check if prompt counts match
    prompt_count_mismatch = expected_prompt_count != len(cached_entries)
    # But also consider files that have changed - their prompt count might differ
    if prompt_count_mismatch and not new_files and not changed_files and not deleted:
        # Only flag as changed if all files are present but prompt count differs
        changed_files.append("(geänderte Prompt-Anzahl)")

    result["missing"] = len(new_files) + len(deleted)
    result["changed"] = len(changed_files)
    result["new_files"] = new_files
    result["valid"] = (result["missing"] == 0 and result["changed"] == 0)

    if result["valid"]:
        result["message"] = f"Cache aktuell ({result['cached']} Hidden States, {result['num_files']} Dateien)"
    else:
        parts = []
        if new_files:
            parts.append(f"{len(new_files)} neue Dateien")
        if deleted:
            parts.append(f"{len(deleted)} gelöschte Dateien")
        if changed_files:
            parts.append(f"{len(changed_files)} geänderte Dateien")
        result["message"] = f"Cache nicht aktuell: {', '.join(parts)}"
    
    return result


def delete_cache(encoder_name: str) -> bool:
    """Delete cached hidden states for an encoder."""
    key = cache_key(encoder_name)
    cache_dir = CACHE_DIR / key
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        return True
    return False


def list_caches() -> List[Dict]:
    """List all available caches with metadata."""
    if not CACHE_DIR.exists():
        return []
    caches = []
    for cache_dir in sorted(CACHE_DIR.iterdir()):
        if not cache_dir.is_dir():
            continue
        meta_path = cache_dir / "metadata.json"
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                caches.append({
                    "encoder_name": metadata.get("encoder_name", cache_dir.name),
                    "num_samples": metadata.get("num_samples", 0),
                    "created": metadata.get("created", "?"),
                    "hidden_dim": metadata.get("hidden_dim", 0),
                    "size_mb": sum(f.stat().st_size for f in cache_dir.glob("*") if f.is_file()) / 1_000_000,
                })
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Cache-Eintrag unlesbar (%s): %s", meta_path, exc)
    return caches