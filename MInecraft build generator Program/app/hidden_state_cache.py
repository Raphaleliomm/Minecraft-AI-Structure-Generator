"""Pre-computed hidden states cache for TF-Diffusion training.
Caches encoder outputs once so training doesn't need to re-encode prompts every epoch."""
from __future__ import annotations

import json
import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import numpy as np

from app.transformer_encoder import FrozenTransformerEncoder

CACHE_DIR = Path("runs/hidden_states_cache")


def cache_key(encoder_name: str) -> str:
    """Generate a filesystem-safe cache key from encoder name."""
    return encoder_name.replace("/", "_").replace(" ", "_")


def _file_hash(path: Path) -> str:
    """SHA256 hash of a file's content."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _split_prompts(text: str) -> List[str]:
    """Split a prompt text on '---' separators, filter out metadata lines."""
    META_PREFIXES = ("source:", "aug:", "author:", "name:", "url:", "id:")
    parts = [p.strip() for p in text.split("---")]
    prompts = []
    for part in parts:
        if not part:
            continue
        # Skip metadata-only lines
        if any(part.lower().startswith(prefix) for prefix in META_PREFIXES):
            continue
        prompts.append(part)
    return prompts


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

    all_hidden_states = []
    all_attention_masks = []
    file_list_items = []  # Each entry corresponds to one prompt

    device = encoder.device

    # First pass: collect all prompts and encode them
    total_prompts = 0
    for schem_path, txt_path in zip(schem_files, txt_files):
        prompts = _split_prompts(txt_path.read_text(encoding="utf-8"))
        total_prompts += len(prompts)

    processed = 0
    for i, (schem_path, txt_path) in enumerate(zip(schem_files, txt_files)):
        # Read and split prompts
        prompts = _split_prompts(txt_path.read_text(encoding="utf-8"))
        schem_hash = _file_hash(schem_path)

        for pi, prompt in enumerate(prompts):
            processed += 1
            if status_callback:
                status_callback(f"Encodiere {processed}/{total_prompts}: {schem_path.name} prompt {pi+1}/{len(prompts)}")

            # Encode prompt
            with torch.no_grad():
                encoded = encoder([prompt])
                hs = encoded["last_hidden_state"].cpu()  # [1, seq_len, hidden_dim]
                mask = encoded["attention_mask"].cpu()   # [1, seq_len]

            all_hidden_states.append(hs)
            all_attention_masks.append(mask)
            file_list_items.append({
                "schem": schem_path.name,
                "txt": txt_path.name,
                "hash": schem_hash,
                "prompt_index": pi,
                "prompt_preview": prompt[:80],
            })

    if not all_hidden_states:
        raise ValueError("Keine Prompts zum Encodieren gefunden")

    # Pad all tensors to the same sequence length before stacking
    max_seq_len = max(hs.shape[1] for hs in all_hidden_states)
    hidden_dim = all_hidden_states[0].shape[2]
    
    padded_hs = []
    padded_masks = []
    for hs, mask in zip(all_hidden_states, all_attention_masks):
        seq_len = hs.shape[1]
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            hs = torch.nn.functional.pad(hs, (0, 0, 0, pad_len), value=0.0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        padded_hs.append(hs)
        padded_masks.append(mask)
    
    stacked_hs = torch.cat(padded_hs, dim=0)  # [N, max_seq_len, hidden_dim]
    stacked_mask = torch.cat(padded_masks, dim=0)  # [N, max_seq_len]

    # Save
    torch.save({"hidden_states": stacked_hs, "attention_masks": stacked_mask}, out_dir / "data.pt")

    metadata = {
        "encoder_name": encoder_name,
        "encoder_hf_id": encoder.hf_id,
        "hidden_dim": encoder.hidden_dim,
        "num_samples": len(all_hidden_states),  # Total prompt entries
        "num_files": len(schem_files),          # Original schematic files
        "created": datetime.now().isoformat(),
        "files": file_list_items,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if status_callback:
        status_callback(f"{len(all_hidden_states)} Hidden States ({total_prompts} Prompts, {len(schem_files)} Dateien)")

    return out_dir


def load_hidden_states(encoder_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached hidden states and attention masks.
    
    Returns:
        (hidden_states, attention_masks) tensors
    """
    key = cache_key(encoder_name)
    data_path = CACHE_DIR / key / "data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Kein Cache für Encoder '{encoder_name}' gefunden. "
                                f"Bitte zuerst Hidden States vorberechnen.")
    data = torch.load(data_path, map_location="cpu")
    return data["hidden_states"], data["attention_masks"]


def load_hidden_states_raw(cache_dir: Path) -> Dict:
    """Load cached hidden states from a specific cache directory path.
    
    Returns dict with 'hidden_states', 'attention_masks', and 'metadata'.
    """
    data_path = cache_dir / "data.pt"
    meta_path = cache_dir / "metadata.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden in {cache_dir}")
    data = torch.load(data_path, map_location="cpu")
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "hidden_states": data["hidden_states"],
        "attention_masks": data["attention_masks"],
        "metadata": metadata,
    }


def validate_cache(encoder_name: str, schem_files: List[Path], txt_files: List[Path]) -> Dict:
    """Check if the cache is still valid (same files, same hashes).
    
    Handles multi-prompt txt files (prompts separated by '---').
    Validates that all schem files still have the same hash and that
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
    except Exception:
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
        prompts = _split_prompts(txt_path.read_text(encoding="utf-8"))
        expected_prompt_count += len(prompts)

        if schem_path.name not in cache_by_schem:
            new_files.append(schem_path.name)
        else:
            current_hash = _file_hash(schem_path)
            # Check if hash matches ANY entry for this schem file
            cached_entries_for_file = cache_by_schem[schem_path.name]
            hash_matches = any(c.get("hash") == current_hash for c in cached_entries_for_file)
            if not hash_matches:
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
            except Exception:
                pass
    return caches