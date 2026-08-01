"""Kaggle Export — packt Trainingsskript, Daten und Notebook für Kaggle GPU Training."""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def create_kaggle_export(
    output_dir: str = "exports",
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1.5e-3,
    aug_diversity: int = 1,
    allow_vertical: bool = False,
    grid_size: Tuple[int, int, int] = (16, 16, 16),
    model_type: str = "transformer",
    data_dirs: Optional[List[str]] = None,
    architecture: Optional[dict] = None,
    tf_unet_config: Optional[dict] = None,
    air_weight: float = 75.0,
    encoder_name: Optional[str] = None,
    context_dim: Optional[int] = None,
) -> Path:
    """Create a complete Kaggle export package.

    Parameters
    ----------
    output_dir : str
        Directory where the export folder will be created.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Peak learning rate.
    aug_diversity : int
        Augmentation diversity (0-5).
    allow_vertical : bool
        Allow vertical movement during augmentation.
    grid_size : tuple[int, int, int]
        Target grid size (e.g. (16,16,16)).
    model_type : str
        Model type — "transformer", "diffusion", or "transformer_diffusion".
    data_dirs : list[str] | None
        List of training data directories to include.
    architecture : dict | None
        Architecture dict for transformer (d_model, nhead, num_layers, dim_feedforward).
    tf_unet_config : dict | None
        UNet config for transformer_diffusion (channels, channel_multipliers, d_model, cross_attn_heads).
    air_weight : float
        Air weight factor for loss weighting (default: 75.0).

    Returns
    -------
    Path
        Path to the created export directory.
    """
    if data_dirs is None:
        data_dirs = [
            "Trainingsdaten good thoroughly analyzed",
            "Trainingsdaten zu gross vorerst ausgelagert",
        ]

    gx, gy, gz = grid_size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(output_dir) / f"kaggle_export_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Copy source files ───
    _copy_source_files(export_dir)

    # ─── 2. Create training data ZIP ───
    _package_training_data(export_dir, data_dirs)

    # ─── 3. Create Kaggle notebook ───
    _create_notebook(
        export_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aug_diversity=aug_diversity,
        allow_vertical=allow_vertical,
        grid_size=(gx, gy, gz),
        model_type=model_type,
        architecture=architecture,
        tf_unet_config=tf_unet_config,
        air_weight=air_weight,
        encoder_name=encoder_name,
        context_dim=context_dim,
    )

    # ─── 4. Create README ───
    _create_readme(export_dir, epochs, batch_size, learning_rate, grid_size,
                   model_type, architecture, tf_unet_config, air_weight)

    return export_dir


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _copy_source_files(export_dir: Path) -> None:
    """Copy train.py, model.py, dataset.py, requirements.txt, and app/ into the export dir."""
    src_dir = Path(".")

    files_to_copy = [
        "train.py",
        "model.py",
        "dataset.py",
        "requirements.txt",
    ]

    for fname in files_to_copy:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(export_dir / fname))
        else:
            # Write a minimal stub so the notebook still works
            (export_dir / fname).write_text(
                f"# {fname} — automatically generated placeholder\n"
                f"# The original was not found at {src.resolve()}\n"
            )

    # Copy the app/ directory (needed for diffusion_model.py, transformer_encoder.py, etc.)
    app_src = src_dir / "app"
    app_dst = export_dir / "app"
    if app_src.exists() and app_src.is_dir():
        shutil.copytree(str(app_src), str(app_dst), dirs_exist_ok=True)


def _package_training_data(export_dir: Path, data_dirs: List[str]) -> None:
    """Create a ZIP file containing all training schematics & text files."""
    zip_path = export_dir / "training_data.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for dir_path_str in data_dirs:
            dir_path = Path(dir_path_str)
            if not dir_path.exists():
                print(f"  ⚠️ Data directory not found, skipping: {dir_path}")
                continue
            for fpath in sorted(dir_path.rglob("*")):
                if fpath.is_file() and fpath.suffix.lower() in {".schem", ".schematic", ".txt", ".json"}:
                    arcname = f"{dir_path.name}/{fpath.relative_to(dir_path)}"
                    zf.write(str(fpath), arcname)
    print(f"  📦 Training data ZIP created: {zip_path}")


def _create_notebook(
    export_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    aug_diversity: int,
    allow_vertical: bool,
    grid_size: Tuple[int, int, int],
    model_type: str,
    architecture: Optional[dict] = None,
    tf_unet_config: Optional[dict] = None,
    air_weight: float = 75.0,
    encoder_name: Optional[str] = None,
    context_dim: Optional[int] = None,
) -> None:
    """Create a Kaggle-compatible Jupyter notebook that runs training."""
    gx, gy, gz = grid_size
    allow_vertical_flag = "true" if allow_vertical else "false"

    # Determine architecture based on model type
    if model_type == "transformer":
        if architecture:
            d_model = architecture.get("d_model", 192)
            nhead = architecture.get("nhead", 6)
            num_layers = architecture.get("num_layers", 5)
            dim_ff = architecture.get("dim_feedforward", 768)
        else:
            d_model, nhead, num_layers, dim_ff = 192, 6, 5, 768
    elif model_type == "diffusion":
        d_model, nhead, num_layers, dim_ff = 128, 0, 0, 0  # Not used for diffusion
    elif model_type == "transformer_diffusion":
        if tf_unet_config:
            channels = tf_unet_config.get("channels", 32)
            ch_mult = tf_unet_config.get("channel_multipliers", (1, 2, 2))
            tf_d_model = tf_unet_config.get("d_model", 64)
            ca_heads = tf_unet_config.get("cross_attn_heads", 4)
        else:
            channels, ch_mult, tf_d_model, ca_heads = 32, (1, 2, 2), 64, 4
        # Use provided encoder info or default to Phi-3.5
        enc_name = encoder_name or "Phi-3.5-mini"
        ctx_dim = context_dim or 768
    else:
        d_model, nhead, num_layers, dim_ff = 192, 6, 5, 768

    # Build model creation code based on type
    if model_type == "transformer":
        model_code = _build_transformer_model_code(d_model, nhead, num_layers, dim_ff)
        train_code = _build_transformer_train_code(air_weight)
        save_code = _build_transformer_save_code(d_model, nhead, num_layers, dim_ff)
        model_desc = "Shared-Weight Voxel Transformer"
    elif model_type == "diffusion":
        model_code = _build_diffusion_model_code()
        train_code = _build_diffusion_train_code(air_weight)
        save_code = _build_diffusion_save_code()
        model_desc = "3D Voxel Diffusion Model"
    elif model_type == "transformer_diffusion":
        model_code = _build_tf_diffusion_model_code(channels, ch_mult, tf_d_model, ca_heads, enc_name, ctx_dim)
        train_code = _build_tf_diffusion_train_code(air_weight, enc_name, ctx_dim)
        save_code = _build_tf_diffusion_save_code(channels, ch_mult, tf_d_model, ca_heads)
        model_desc = "Transformer Diffusion Model"
    else:
        model_code = _build_transformer_model_code(192, 6, 5, 768)
        train_code = _build_transformer_train_code(air_weight)
        save_code = _build_transformer_save_code(192, 6, 5, 768)
        model_desc = "Shared-Weight Voxel Transformer"

    notebook = {
        "cells": [
            _md_cell(
                f"# Minecraft Structure Generator — Training on Kaggle\n\n"
                f"**Settings:** Grid {gx}×{gy}×{gz}  ·  "
                f"{epochs} epochs  ·  batch {batch_size}  ·  "
                f"lr {learning_rate:.2e}\n\n"
                f"This notebook trains the **{model_desc}** "
                f"using the packaged training data.\n\n"
                f"**Air Weight:** {air_weight}\n\n"
                "---"
            ),
            _code_cell(
                "# ⚡ Install dependencies\n"
                "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q\n"
                "!pip install nbtlib matplotlib numpy tqdm -q\n"
                "!pip install --upgrade sympy -q\n\n"
                "import sys, os, json, math, random, glob, zipfile, shutil, time, warnings\n"
                "warnings.filterwarnings('ignore')\n"
                "from pathlib import Path\n"
                "import torch\n"
                "import numpy as np\n\n"
                "print('✅ Dependencies installed')"
            ),
            _code_cell(
                "# 📂 Extract training data\n"
                "DATA_ZIP = '/kaggle/input/training_data.zip'\n"
                "# The data is bundled in the notebook directory;\n"
                "# we also copy it from the working dir if different.\n"
                "data_zip_path = 'training_data.zip'\n"
                "if os.path.exists(data_zip_path):\n"
                "    import zipfile\n"
                "    with zipfile.ZipFile(data_zip_path, 'r') as zf:\n"
                "        zf.extractall('data')\n"
                "    print('✅ Extracted training_data.zip to data/')\n"
                "else:\n"
                "    print('⚠️ training_data.zip not found; creating empty directories')\n"
                "    os.makedirs('data', exist_ok=True)\n\n"
                "# List what we got\n"
                "for p in sorted(Path('data').rglob('*.schem'))[:10]:\n"
                "    print(f'  {p}')\n"
                "total = len(list(Path('data').rglob('*.schem')))\n"
                "print(f'📊 Total .schem files: {total}')"
            ),
            _code_cell(
                "# 🧠 Import model & dataset\n"
                "sys.path.insert(0, '.')\n"
                "from dataset import MultiSourceSchematicDataset, PromptTokenizer, VoxelTokenizer\n"
                + (f"from model import SharedWeightVoxelTransformer\n" if model_type == "transformer" else "")
                + (f"from app.diffusion_model import VoxelDiffusionModel, train_diffusion_step\n" if model_type == "diffusion" else "")
                + (f"from app.diffusion_model import TransformerDiffusionModel, train_transformer_diffusion_step\n" if model_type == "transformer_diffusion" else "")
                + "from torch.utils.data import DataLoader\n\n"
                "print('✅ Modules imported')"
            ),
            _code_cell(
                "# ⚙️ Settings (auto-filled from GUI)\n"
                f"GRID_SIZE = ({gx}, {gy}, {gz})\n"
                f"EPOCHS = {epochs}\n"
                f"BATCH_SIZE = {batch_size}\n"
                f"LR = {learning_rate}\n"
                f"AUG_DIVERSITY = {aug_diversity}\n"
                f"ALLOW_VERTICAL = {allow_vertical_flag}\n"
                f"AIR_WEIGHT = {air_weight}\n"
                f"NOISE_BLOCK_PROB = 0.20  # 20% of steps get random wrong blocks injected (learn to remove bad blocks)\n\n"
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                "print(f'🚀 Device: {DEVICE}')\n"
                "if torch.cuda.is_available():\n"
                "    print(f'   GPU: {torch.cuda.get_device_name(0)}')\n"
                "    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
            ),
            _code_cell(
                "# 📦 Load dataset\n"
                "data_dirs = [str(p) for p in Path('data').iterdir() if p.is_dir()]\n"
                "print(f'Data directories: {data_dirs}')\n\n"
                "dataset = MultiSourceSchematicDataset(\n"
                "    data_dirs,\n"
                "    target_size=GRID_SIZE,\n"
                "    max_voxels=400_000,\n"
                "    augmentation_diversity=AUG_DIVERSITY,\n"
                "    allow_vertical_movement=ALLOW_VERTICAL,\n"
                "    air_weight_factor=AIR_WEIGHT,\n"
                ")\n\n"
                "print(f'📊 Dataset size: {len(dataset)}')\n"
                "print(f'🧱 Block vocab: {len(dataset.voxel_tokenizer.id_to_block)}')\n"
                "print(f'🔤 Prompt vocab: {len(dataset.prompt_tokenizer.token_to_id)}')\n\n"
                "loader = DataLoader(\n"
                "    dataset, batch_size=BATCH_SIZE, shuffle=True,\n"
                "    num_workers=2, pin_memory=(DEVICE.type == 'cuda'),\n"
                ")"
            ),
            _code_cell(model_code),
            _code_cell(train_code),
            _code_cell(
                "# ✅ Training complete\n"
                "elapsed = time.time() - start_time\n"
                "print(f'✅ Training completed in {elapsed/60:.1f} min')\n"
                "print(f'📊 Best loss: {best_loss:.4f}')\n"
                "print(f'📁 Model saved to model.pt')\n\n"
                "# Download link for the model\n"
                "print('\\n⬇️ To download the trained model, find model.pt in the output tab.')"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (GPU)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path = export_dir / "kaggle_notebook.ipynb"
    with open(str(notebook_path), "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"  📓 Notebook created: {notebook_path}")


def _build_transformer_model_code(d_model, nhead, num_layers, dim_ff):
    return (
        "# 🏗️ Create model\n"
        "model = SharedWeightVoxelTransformer(\n"
        "    text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),\n"
        "    block_vocab_size=len(dataset.voxel_tokenizer.id_to_block),\n"
        "    grid_size=GRID_SIZE,\n"
        f"    d_model={d_model},\n"
        f"    nhead={nhead},\n"
        f"    num_layers={num_layers},\n"
        f"    dim_feedforward={dim_ff},\n"
        "    dropout=0.1,\n"
        ").to(DEVICE)\n\n"
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'🧠 Trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)')\n\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)\n\n"
        "# Cosine LR scheduler with warmup\n"
        "total_steps = EPOCHS * len(loader)\n"
        "warmup_steps = 500\n\n"
        "def lr_lambda(step):\n"
        "    if step < warmup_steps:\n"
        "        return float(step) / max(1, warmup_steps)\n"
        "    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)\n"
        "    return 0.5 * (1.0 + math.cos(progress * math.pi))\n\n"
        "scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)"
    )


def _build_transformer_train_code(air_weight):
    return (
        "# 🎯 Training Loop\n"
        "best_loss = float('inf')\n"
        "loss_history = []\n"
        "start_time = time.time()\n\n"
        "for epoch in range(1, EPOCHS + 1):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    num_batches = 0\n\n"
        "    for batch in loader:\n"
        "        prompt_ids = batch['prompt_ids'].to(DEVICE, non_blocking=True)\n"
        "        target = batch['voxel_ids'].to(DEVICE, non_blocking=True).reshape(prompt_ids.shape[0], -1)\n"
        "        target = model.safe_clamp_target(target)\n\n"
        "        logits = model(prompt_ids)\n"
        "        target_flat = target.reshape(-1)\n"
        "        sample_weight = batch['sample_weight'].to(DEVICE).view(-1, 1)\n"
        "        sample_weight = sample_weight.expand_as(target).reshape(-1)\n"
        "        per_block_w = batch['per_block_weight'].to(DEVICE).view(-1, 1).expand_as(target).reshape(-1)\n"
        "        per_air_w = batch['per_air_weight'].to(DEVICE).view(-1, 1).expand_as(target).reshape(-1)\n"
        "        weight_per_token = torch.where(target_flat == 0, per_air_w, per_block_w) * sample_weight\n"
        "        logp = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)\n"
        "        nll = torch.nn.functional.nll_loss(logp, target_flat, reduction='none')\n"
        "        loss = (nll * weight_per_token).sum() / weight_per_token.sum().clamp_min(1.0)\n\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        loss.backward()\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        "        optimizer.step()\n"
        "        scheduler.step()\n\n"
        "        total_loss += float(loss.detach())\n"
        "        num_batches += 1\n\n"
        "    avg_loss = total_loss / max(num_batches, 1)\n"
        "    loss_history.append(avg_loss)\n"
        "    elapsed = time.time() - start_time\n\n"
        "    print(f'epoch={epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s')\n\n"
        "    if avg_loss < best_loss:\n"
        "        best_loss = avg_loss\n"
        "        torch.save({\n"
        "            'model_state': model.state_dict(),\n"
        "            'grid_size': GRID_SIZE,\n"
        "            'text_vocab_size': len(dataset.prompt_tokenizer.token_to_id),\n"
        "            'block_vocab_size': len(dataset.voxel_tokenizer.id_to_block),\n"
        f"            'd_model': {d_model},\n"
        f"            'nhead': {nhead},\n"
        f"            'layers': {num_layers},\n"
        f"            'dim_feedforward': {dim_ff},\n"
        "            'augmentation_diversity': AUG_DIVERSITY,\n"
        "            'allow_vertical_movement': ALLOW_VERTICAL,\n"
        "            'epoch': epoch,\n"
        "            'loss': avg_loss,\n"
        "        }, 'model.pt')\n"
        "        print(f'  → new best model saved (loss={avg_loss:.4f})')"
    ).replace("{d_model}", str(d_model)).replace("{nhead}", str(nhead)).replace("{num_layers}", str(num_layers)).replace("{dim_ff}", str(dim_ff))


def _build_transformer_save_code(d_model, nhead, num_layers, dim_ff):
    return ""  # Save is handled in train code


def _build_diffusion_model_code():
    return (
        "# 🏗️ Create model\n"
        "from app.diffusion_model import VoxelDiffusionModel, train_diffusion_step\n"
        "model = VoxelDiffusionModel(\n"
        "    num_blocks=len(dataset.voxel_tokenizer.id_to_block),\n"
        "    text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),\n"
        "    grid_size=GRID_SIZE,\n"
        "    d_model=128, d_text=64, channels=64,\n"
        "    channel_multipliers=(1, 2, 2), num_timesteps=50,\n"
        ").to(DEVICE)\n\n"
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'🧠 Trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)')\n\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)"
    )


def _build_diffusion_train_code(air_weight):
    return (
        "# 🎯 Training Loop\n"
        "best_loss = float('inf')\n"
        "loss_history = []\n"
        "start_time = time.time()\n\n"
        "for epoch in range(1, EPOCHS + 1):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    num_batches = 0\n\n"
        "    for batch in loader:\n"
        "        loss = train_diffusion_step(model, batch, optimizer, DEVICE, noise_block_prob=NOISE_BLOCK_PROB)\n"
        "        total_loss += loss\n"
        "        num_batches += 1\n\n"
        "    avg_loss = total_loss / max(num_batches, 1)\n"
        "    loss_history.append(avg_loss)\n"
        "    elapsed = time.time() - start_time\n\n"
        "    print(f'epoch={epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  time={elapsed:.0f}s')\n\n"
        "    if avg_loss < best_loss:\n"
        "        best_loss = avg_loss\n"
        "        torch.save({\n"
        "            'model_state': model.state_dict(),\n"
        "            'grid_size': model.grid_size,\n"
        "            'text_vocab_size': len(dataset.prompt_tokenizer.token_to_id),\n"
        "            'block_vocab_size': model.num_blocks,\n"
        "            'd_model': model.d_model, 'd_text': model.d_text, 'channels': model.channels,\n"
        "            'channel_multipliers': [int(m) for m in model.channel_multipliers],\n"
        "            'num_timesteps': model.num_timesteps,\n"
        "            'epoch': epoch,\n"
        "            'loss': avg_loss,\n"
        "        }, 'model.pt')\n"
        "        print(f'  → new best model saved (loss={avg_loss:.4f})')"
    )


def _build_diffusion_save_code():
    return ""


_ENCODER_HF_IDS = {
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "Gemma-2-2B": "google/gemma-2-2b",
    "Gemma-2-9B": "google/gemma-2-9b",
    "Gemma-2-27B": "google/gemma-2-27b-it",
    "Gemma-3-1B": "google/gemma-3-1b-it",
    "Gemma-3-4B": "google/gemma-3-4b-it",
    "Gemma-3-12B": "google/gemma-3-12b-it",
    "Gemma-3-27B": "google/gemma-3-27b-it",
    "Flan-T5-small": "google/flan-t5-small",
    "Flan-T5-base": "google/flan-t5-base",
    "Flan-T5-large": "google/flan-t5-large",
    "Flan-T5-XL": "google/flan-t5-xl",
    "Flan-T5-XXL": "google/flan-t5-xxl",
}


def _build_tf_diffusion_model_code(channels, ch_mult, d_model, ca_heads, encoder_name="Phi-3.5-mini", context_dim=768):
    ch_mult_str = ", ".join(str(m) for m in ch_mult)
    hf_id = _ENCODER_HF_IDS.get(encoder_name, "microsoft/Phi-3.5-mini-instruct")
    is_t5 = "t5" in encoder_name.lower()
    if is_t5:
        import_line = "from transformers import T5EncoderModel, AutoTokenizer\n"
        model_load_line = "encoder = T5EncoderModel.from_pretrained(encoder_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)\n"
    else:
        import_line = "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        model_load_line = "encoder = AutoModelForCausalLM.from_pretrained(encoder_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)\n"
    return (
        "# 🏗️ Create model\n"
        "from app.diffusion_model import TransformerDiffusionModel, train_transformer_diffusion_step\n"
        "model = TransformerDiffusionModel(\n"
        "    num_blocks=len(dataset.voxel_tokenizer.id_to_block),\n"
        "    grid_size=GRID_SIZE,\n"
        f"    d_model={d_model},\n"
        f"    channels={channels},\n"
        f"    channel_multipliers=({ch_mult_str}),\n"
        "    num_timesteps=50,\n"
        f"    context_dim={context_dim},  # {encoder_name} hidden dim\n"
        f"    cross_attn_heads={ca_heads},\n"
        f"    context_proj_dim={d_model * 2},\n"
        ").to(DEVICE)\n\n"
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'🧠 Trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)')\n\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)\n\n"
        "# Load frozen text encoder\n"
        + import_line
        + f"encoder_name = '{hf_id}'\n"
        "tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True, padding_side='right')\n"
        "if tokenizer.pad_token is None:\n"
        "    tokenizer.pad_token = tokenizer.eos_token or '<pad>'\n"
        + model_load_line
        + "encoder = encoder.to(DEVICE)\n"
        "encoder.eval()\n"
        "for param in encoder.parameters():\n"
        "    param.requires_grad = False\n"
        f"print(f'✅ Encoder loaded: {{encoder_name}}')\n\n"
        f"ENCODER_DISPLAY_NAME = '{encoder_name}'\n"
        f"ENCODER_HF_ID = '{hf_id}'\n"
        f"ENCODER_HIDDEN_DIM = {context_dim}\n"
    )


def _build_tf_diffusion_train_code(air_weight, encoder_name="Phi-3.5-mini", context_dim=768):
    return (
        "# 🎯 Training Loop\n"
        "best_loss = float('inf')\n"
        "loss_history = []\n"
        "start_time = time.time()\n\n"
        "for epoch in range(1, EPOCHS + 1):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    num_batches = 0\n\n"
        "    for batch in loader:\n"
        "        prompts = batch.get('prompt_text', None)\n"
        "        with torch.no_grad():\n"
        "            encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)\n"
        "            input_ids = encoded['input_ids'].to(DEVICE)\n"
        "            attention_mask = encoded['attention_mask'].to(DEVICE)\n"
        "            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)\n"
        "            context = outputs.hidden_states[-1].to(dtype=next(model.parameters()).dtype)\n"
        "        loss = train_transformer_diffusion_step(model, batch, type('FakeEncoder', (), {'__call__': lambda self, x: {'last_hidden_state': context, 'attention_mask': attention_mask}})(), optimizer, DEVICE, noise_block_prob=NOISE_BLOCK_PROB)\n"
        "        total_loss += loss\n"
        "        num_batches += 1\n\n"
        "    avg_loss = total_loss / max(num_batches, 1)\n"
        "    loss_history.append(avg_loss)\n"
        "    elapsed = time.time() - start_time\n\n"
        "    print(f'epoch={epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  time={elapsed:.0f}s')\n\n"
        "    if avg_loss < best_loss:\n"
        "        best_loss = avg_loss\n"
        "        torch.save({\n"
        "            'model_state': model.state_dict(),\n"
        "            'grid_size': GRID_SIZE,\n"
        "            'block_vocab_size': model.num_blocks,\n"
        "            'num_blocks': model.num_blocks,\n"
        "            'd_model': model.d_model,\n"
        "            'channels': model.channels,\n"
        "            'channel_multipliers': [int(m) for m in model.channel_multipliers],\n"
        "            'num_timesteps': model.num_timesteps,\n"
        "            'context_dim': model.context_dim,\n"
        "            'cross_attn_heads': model.cross_attn_heads,\n"
            "            'context_proj_dim': model.effective_context_dim,\n"
        "            'encoder_config': {\n"
        "                'display_name': ENCODER_DISPLAY_NAME,\n"
        "                'hf_id': ENCODER_HF_ID,\n"
        "                'hidden_dim': ENCODER_HIDDEN_DIM,\n"
        "            },\n"
        "            'epoch': epoch,\n"
        "            'loss': avg_loss,\n"
        "        }, 'model.pt')\n"
        "        print(f'  → new best model saved (loss={avg_loss:.4f})')"
    )


def _build_tf_diffusion_save_code(channels, ch_mult, d_model, ca_heads):
    return ""


def _create_readme(
    export_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grid_size: Tuple[int, int, int],
    model_type: str = "transformer",
    architecture: Optional[dict] = None,
    tf_unet_config: Optional[dict] = None,
    air_weight: float = 75.0,
) -> None:
    """Create a README explaining how to use the Kaggle export."""
    gx, gy, gz = grid_size

    # Build architecture description
    if model_type == "transformer" and architecture:
        arch_str = f"d_model={architecture.get('d_model', 192)}, nhead={architecture.get('nhead', 6)}, layers={architecture.get('num_layers', 5)}, FFN={architecture.get('dim_feedforward', 768)}"
    elif model_type == "diffusion":
        arch_str = "d_model=128, d_text=64, channels=64, (1,2,2)"
    elif model_type == "transformer_diffusion" and tf_unet_config:
        ch_mult = tf_unet_config.get("channel_multipliers", (1, 2, 2))
        ch_mult_str = ", ".join(str(m) for m in ch_mult)
        arch_str = f"channels={tf_unet_config.get('channels', 32)}, ch_mult=({ch_mult_str}), d_model={tf_unet_config.get('d_model', 64)}, heads={tf_unet_config.get('cross_attn_heads', 4)}"
    else:
        arch_str = "d_model=192, nhead=6, layers=5, FFN=768"

    model_desc = {
        "transformer": "Shared-Weight Voxel Transformer",
        "diffusion": "3D Voxel Diffusion Model",
        "transformer_diffusion": "Transformer Diffusion Model",
    }.get(model_type, "Transformer")

    readme = f"""# Minecraft Structure Generator — Kaggle Export

## 📁 Contents

| File | Description |
|------|-------------|
| `kaggle_notebook.ipynb` | Main training notebook (upload to Kaggle) |
| `training_data.zip` | Packaged schematic training data |
| `train.py` | Reference training script |
| `model.py` | Model definition |
| `dataset.py` | Dataset & tokenizer |
| `requirements.txt` | Python dependencies |

## 🚀 How to Use

1. **Upload to Kaggle**
   - Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
   - Click "File" → "Upload" and select all files in this folder
   - Or: upload `kaggle_notebook.ipynb` first, then add `training_data.zip` as a dataset

2. **Configure Accelerator**
   - Click the "Settings" panel (gear icon) on the right
   - Set **Accelerator** → **GPU T4 x2**
   - (Optional) Set **Persistence** → **Files only** if you want the model output to persist

3. **Run the Notebook**
   - Click "Run All" (⏩)
   - Training will run for {epochs} epochs with batch size {batch_size}
   - Grid size: {gx}×{gy}×{gz}
   - The best model checkpoint will be saved as `model.pt`

4. **Download the Result**
   - After training completes, find `model.pt` in the output
   - Download it and place it in your local `runs/` directory

## ⚙️ Training Settings

| Parameter | Value |
|-----------|-------|
| Grid Size | {gx}×{gy}×{gz} |
| Epochs | {epochs} |
| Batch Size | {batch_size} |
| Learning Rate | {learning_rate:.2e} |
| Model Type | {model_desc} |
| Architecture | {arch_str} |
| Air Weight | {air_weight} |

## 📝 Notes

- Training typically takes **2-8 hours** on 2× T4 GPUs depending on dataset size.
- The notebook uses cosine LR scheduling with 500-step warmup (transformer only).
- Loss weighting uses adaptive per-structure weights with air_weight={air_weight}.
- If you want to train a different architecture, edit the model creation cell.
"""
    readme_path = export_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    print(f"  📖 README created: {readme_path}")


# ═══════════════════════════════════════════════════════════════
# Notebook cell builders
# ═══════════════════════════════════════════════════════════════


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "vscode": {
                "languageId": "python",
            },
        },
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    }


if __name__ == "__main__":
    # Quick test / preview
    path = create_kaggle_export(
        output_dir="exports",
        epochs=5,
        batch_size=4,
        learning_rate=1.5e-3,
        aug_diversity=1,
        allow_vertical=False,
        grid_size=(16, 16, 16),
        model_type="transformer",
    )
    print(f"\n✅ Export created at: {path}")