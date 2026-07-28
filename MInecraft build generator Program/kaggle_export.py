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
        Model type — only "transformer" is currently supported.
    data_dirs : list[str] | None
        List of training data directories to include.

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
    )

    # ─── 4. Create README ───
    _create_readme(export_dir, epochs, batch_size, learning_rate, grid_size)

    return export_dir


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _copy_source_files(export_dir: Path) -> None:
    """Copy train.py, model.py, dataset.py, requirements.txt into the export dir."""
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
) -> None:
    """Create a Kaggle-compatible Jupyter notebook that runs training."""
    gx, gy, gz = grid_size
    allow_vertical_flag = "true" if allow_vertical else "false"

    notebook = {
        "cells": [
            _md_cell(
                f"# Minecraft Structure Generator — Training on Kaggle\n\n"
                f"**Settings:** Grid {gx}×{gy}×{gz}  ·  "
                f"{epochs} epochs  ·  batch {batch_size}  ·  "
                f"lr {learning_rate:.2e}\n\n"
                "This notebook trains the **Shared-Weight Voxel Transformer** "
                "using the packaged training data.\n\n"
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
                "from model import SharedWeightVoxelTransformer\n"
                "from torch.utils.data import DataLoader\n\n"
                "print('✅ Modules imported')"
            ),
            _code_cell(
                "# ⚙️ Settings (auto-filled from GUI)\n"
                f"GRID_SIZE = ({gx}, {gy}, {gz})\n"
                f"EPOCHS = {epochs}\n"
                f"BATCH_SIZE = {batch_size}\n"
                f"LR = {learning_rate}\n"
                f"AUG_DIVERSITY = {aug_diversity}\n"
                f"ALLOW_VERTICAL = {allow_vertical_flag}\n\n"
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
                ")\n\n"
                "print(f'📊 Dataset size: {len(dataset)}')\n"
                "print(f'🧱 Block vocab: {len(dataset.voxel_tokenizer.id_to_block)}')\n"
                "print(f'🔤 Prompt vocab: {len(dataset.prompt_tokenizer.token_to_id)}')\n\n"
                "loader = DataLoader(\n"
                "    dataset, batch_size=BATCH_SIZE, shuffle=True,\n"
                "    num_workers=2, pin_memory=(DEVICE.type == 'cuda'),\n"
                ")"
            ),
            _code_cell(
                "# 🏗️ Create model\n"
                "model = SharedWeightVoxelTransformer(\n"
                "    text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),\n"
                "    block_vocab_size=len(dataset.voxel_tokenizer.id_to_block),\n"
                "    grid_size=GRID_SIZE,\n"
                "    d_model=192,\n"
                "    nhead=6,\n"
                "    num_layers=5,\n"
                "    dim_feedforward=768,\n"
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
            ),
            _code_cell(
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
                "        weight_per_token = torch.where(target_flat == 0, 0.5, 1.0) * sample_weight\n"
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
                "            'd_model': 192,\n"
                "            'nhead': 6,\n"
                "            'layers': 5,\n"
                "            'dim_feedforward': 768,\n"
                "            'augmentation_diversity': AUG_DIVERSITY,\n"
                "            'allow_vertical_movement': ALLOW_VERTICAL,\n"
                "            'epoch': epoch,\n"
                "            'loss': avg_loss,\n"
                "        }, 'model.pt')\n"
                "        print(f'  → new best model saved (loss={avg_loss:.4f})')"
            ),
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


def _create_readme(
    export_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grid_size: Tuple[int, int, int],
) -> None:
    """Create a README explaining how to use the Kaggle export."""
    gx, gy, gz = grid_size
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
| Architecture | d_model=192, nhead=6, layers=5, FFN=768 |

## 📝 Notes

- Training typically takes **2-8 hours** on 2× T4 GPUs depending on dataset size.
- The notebook uses cosine LR scheduling with 500-step warmup.
- Loss weighting: air tokens count 0.5×, solid blocks 1.0×.
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
    print(f"\\n✅ Export created at: {path}")