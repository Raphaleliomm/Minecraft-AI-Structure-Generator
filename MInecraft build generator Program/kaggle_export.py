"""Kaggle Export - packt Trainingsskript, Daten und Notebook für Kaggle GPU Training."""
from __future__ import annotations

import json
import hashlib
import re
import shutil
import tempfile
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from app.paths import EXPORTS_DIR, PROJECT_ROOT, default_data_dir_strings
from app.transformer_encoder import MODEL_TO_ID as _ENCODER_HF_IDS
from app.transformer_encoder import DEFAULT_ENCODER_HIDDEN_DIM


def create_kaggle_export(
    output_dir: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 16,
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
        data_dirs = default_data_dir_strings()

    gx, gy, gz = grid_size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root = Path(output_dir) if output_dir else EXPORTS_DIR
    export_dir = export_root / f"kaggle_export_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Copy source files ───
    _copy_source_files(export_dir)

    # ─── 2. Create training data ZIP ───
    _package_source_code(export_dir)
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

    _create_kaggle_metadata(export_dir, timestamp)

    return export_dir


def _create_kaggle_metadata(export_dir: Path, timestamp: str) -> None:
    """Describe a private data package and GPU notebook for Kaggle CLI uploads."""
    stamp = timestamp.replace("_", "-")
    dataset_slug = f"minecraft-ai-training-{stamp}"
    kernel_slug = f"minecraft-ai-structure-training-{stamp}"
    human_stamp = timestamp.replace("_", " ")
    dataset_title = f"Minecraft AI Training {human_stamp}"
    kernel_title = f"Minecraft AI Structure Training {human_stamp}"

    (export_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_title,
                "id": f"raphbre/{dataset_slug}",
                "licenses": [{"name": "other"}],
                "description": (
                    "Private training package for Minecraft AI Structure Generator. "
                    "Contains creator-attributed schematic files and prompts; "
                    "underlying builds retain their respective creators' rights."
                ),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (export_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"raphbre/{kernel_slug}",
                "title": kernel_title,
                "code_file": "kaggle_notebook.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_tpu": "false",
                "enable_internet": "true",
                "machine_shape": "",
                "dataset_sources": [f"raphbre/{dataset_slug}"],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _copy_source_files(export_dir: Path) -> None:
    """Copy train.py, model.py, dataset.py, requirements.txt, and app/ into the export dir."""
    src_dir = PROJECT_ROOT

    files_to_copy = [
        "train.py",
        "model.py",
        "dataset.py",
        "requirements.txt",
        "kaggle_experiments.py",
    ]

    for fname in files_to_copy:
        src = src_dir / fname
        if not src.is_file():
            raise FileNotFoundError(
                f"Kaggle-Export unvollständig: benötigte Quelldatei fehlt: {src}"
            )
        shutil.copy2(str(src), str(export_dir / fname))

    # Copy the app/ directory (needed for diffusion_model.py, transformer_encoder.py, etc.)
    # __pycache__/.pyc are excluded: they would only bloat the upload (and may be
    # bytecode of a different Python version).
    app_src = src_dir / "app"
    app_dst = export_dir / "app"
    if app_src.exists() and app_src.is_dir():
        shutil.copytree(
            str(app_src),
            str(app_dst),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    else:
        raise FileNotFoundError(
            f"Kaggle-Export unvollständig: benötigtes app/-Paket fehlt: {app_src}"
        )


def _package_training_data(export_dir: Path, data_dirs: List[str]) -> None:
    """Create an ASCII-named ZIP with schematic files and paired prompt sidecars."""
    from dataset import discover_examples_many

    prompted_examples = discover_examples_many(data_dirs)
    if not prompted_examples:
        raise ValueError(
            "Keine prompt-annotierten .schem/.schematic-Dateien gefunden. "
            "Prüfe Datenordner und prompts.json/.txt-Dateien vor dem Kaggle-Export."
        )
    zip_path = export_dir / "training_data.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for dir_path_str in data_dirs:
            dir_path = Path(dir_path_str)
            if not dir_path.exists():
                print(f"  Data directory not found, skipping: {dir_path}")
                continue
            files = [
                path for path in sorted(dir_path.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".schem", ".schematic", ".txt"}
            ]

            def safe_component(value: str) -> str:
                ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
                ascii_value = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_value).strip(" .-")
                return ascii_value or "structure"

            safe_stems: dict[str, str] = {}
            raw_by_safe_stem: dict[str, set[str]] = {}
            for path in files:
                relative = path.relative_to(dir_path)
                safe_parent = "/".join(safe_component(part) for part in relative.parts[:-1])
                key = f"{safe_parent}/{safe_component(relative.stem)}" if safe_parent else safe_component(relative.stem)
                raw_stem = relative.with_suffix("").as_posix()
                raw_by_safe_stem.setdefault(key, set()).add(raw_stem)
                safe_stems[raw_stem] = key

            final_stems = {
                raw_stem: (
                    f"{safe_stem}--{hashlib.sha1(raw_stem.encode('utf-8')).hexdigest()[:8]}"
                    if len(raw_by_safe_stem[safe_stem]) > 1
                    else safe_stem
                )
                for raw_stem, safe_stem in safe_stems.items()
            }
            schematic_stems = {
                (path.parent.relative_to(dir_path).as_posix().casefold(), path.stem.casefold()):
                    final_stems[path.relative_to(dir_path).with_suffix("").as_posix()]
                for path in files if path.suffix.lower() in {".schem", ".schematic"}
            }

            for path in files:
                relative = path.relative_to(dir_path)
                raw_stem = relative.with_suffix("").as_posix()
                safe_stem = final_stems[raw_stem]
                if path.suffix.lower() == ".txt":
                    key = (relative.parent.as_posix().casefold(), relative.stem.casefold())
                    safe_stem = schematic_stems.get(key, safe_stem)
                arcname = f"{safe_component(dir_path.name)}/{safe_stem}{path.suffix.lower()}"
                zf.write(str(path), arcname)

            # Some sources store prompt mappings only in prompts.json. The ZIP
            # intentionally uses portable ASCII names, so preserve those prompts
            # as sidecars keyed to each schematic instead of copying a JSON map
            # whose original filenames may no longer match after sanitization.
            try:
                from dataset import discover_examples_many
                mapped: dict[Path, list[str]] = {}
                for schematic_path, prompt, _source in discover_examples_many([dir_path]):
                    mapped.setdefault(schematic_path.resolve(), []).append(prompt)
                existing_sidecars = {
                    (p.parent.resolve(), p.stem.casefold())
                    for p in files if p.suffix.lower() == ".txt"
                }
                for schematic_path, prompts in mapped.items():
                    if (schematic_path.parent, schematic_path.stem.casefold()) in existing_sidecars:
                        continue
                    relative = schematic_path.relative_to(dir_path.resolve())
                    raw_stem = relative.with_suffix("").as_posix()
                    safe_stem = final_stems.get(raw_stem)
                    if safe_stem:
                        arcname = (f"{safe_component(dir_path.name)}/{safe_stem}.txt")
                        content = "\n---\n".join(dict.fromkeys(prompts)) + "\n"
                        zf.writestr(arcname, content.encode("utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Prompts aus {dir_path} konnten nicht sicher in das Kaggle-Archiv "
                    "übernommen werden; Export wird abgebrochen, damit kein "
                    "unvollständiger Trainingsdatensatz entsteht."
                ) from exc
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        packaged_structures = sum(
            name.lower().endswith((".schem", ".schematic")) for name in zf.namelist()
        )
    if packaged_structures == 0:
        raise ValueError("Kaggle-Datenarchiv enthält keine .schem/.schematic-Dateien.")
    print(f"  Training data ZIP created: {zip_path} "
          f"({packaged_structures} Schematics, {len(prompted_examples)} Prompt-Beispiele)")


def _package_source_code(export_dir: Path) -> None:
    """Bundle Python source and app modules as a Kaggle input artifact."""
    archive_path = export_dir / "source_code.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("train.py", "model.py", "dataset.py", "requirements.txt", "kaggle_experiments.py"):
            source = export_dir / name
            if source.is_file():
                zf.write(source, name)
        app_dir = export_dir / "app"
        if app_dir.is_dir():
            for source in sorted(app_dir.rglob("*")):
                if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                    zf.write(source, source.relative_to(export_dir).as_posix())
    print(f"  Source code ZIP created: {archive_path}")


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
    allow_vertical_flag = 'true' if allow_vertical else 'false'

    if model_type == 'transformer':
        arch = architecture or {}
        d_model = int(arch.get('d_model', 192))
        nhead = int(arch.get('nhead', 6))
        num_layers = int(arch.get('num_layers', 5))
        dim_ff = int(arch.get('dim_feedforward', 768))
        recurrent_depth = max(2, int(arch.get('recurrent_depth', 4)))
        window_size = int(arch.get('window_size', 8))
        moe_experts = max(0, int(arch.get('moe_experts', 0)))
        moe_active = min(max(1, int(arch.get('moe_active', 1))), max(1, moe_experts))
    elif model_type == 'diffusion':
        d_model, nhead, num_layers, dim_ff = 128, 0, 0, 0
    elif model_type == 'transformer_diffusion':
        tf_config = tf_unet_config or {}
        channels = int(tf_config.get('channels', 32))
        ch_mult = tuple(tf_config.get('channel_multipliers', (1, 2, 2)))
        tf_d_model = int(tf_config.get('d_model', 64))
        ca_heads = int(tf_config.get('cross_attn_heads', 4))
        enc_name = encoder_name or 'Phi-3.5-mini'
        ctx_dim = int(context_dim or DEFAULT_ENCODER_HIDDEN_DIM)
    else:
        raise ValueError(f'Unsupported Kaggle model type: {model_type}')

    if model_type == 'transformer':
        model_code = _build_transformer_model_code(d_model, nhead, num_layers, dim_ff,
                                                   recurrent_depth, moe_experts, moe_active,
                                                   encoder_name if encoder_name else None,
                                                   context_dim if encoder_name else None,
                                                   window_size)
        train_code = _build_transformer_train_code(
            air_weight, d_model, nhead, num_layers, dim_ff,
            encoder_name if encoder_name else None, context_dim if encoder_name else None,
            recurrent_depth, moe_experts, moe_active, window_size,
        )
        model_desc = 'Per-Token Recurrent DiT' if recurrent_depth > 1 else 'Shared-Weight Voxel Transformer'
        if moe_experts:
            model_desc += f' with MoE ({moe_active}/{moe_experts} active experts)'
    elif model_type == 'diffusion':
        model_code = _build_diffusion_model_code()
        train_code = _build_diffusion_train_code(air_weight)
        model_desc = '3D Voxel Diffusion Model'
    else:
        model_code = _build_tf_diffusion_model_code(
            channels, ch_mult, tf_d_model, ca_heads, enc_name, ctx_dim
        )
        train_code = _build_tf_diffusion_train_code(air_weight, enc_name, ctx_dim)
        model_desc = 'Transformer Diffusion Model'

    notebook = {
        'cells': [
            _md_cell(
                f'# Minecraft Structure Generator — Training on Kaggle\n\n'
                f'**Settings:** Grid {gx}×{gy}×{gz} · {epochs} epochs · '
                f'batch {batch_size} · lr {learning_rate:.2e}\n\n'
                f'This notebook trains the **{model_desc}** on the packaged data.'
            ),
            _code_cell(
                '# Install only the missing schematic reader; Kaggle supplies its own CUDA Torch.\n'
                'import importlib.util, subprocess, sys\n'
                "if importlib.util.find_spec('nbtlib') is None:\n"
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nbtlib', '-q'])\n"
                'print("Dependencies ready")'
            ),
            _code_cell(
                '# Locate the uploaded Kaggle dataset and prepare source/data.\n'
                'import os, sys, time, zipfile\n'
                'from pathlib import Path\n'
                "input_root = Path('/kaggle/input')\n"
                "source_roots = [p.parent for p in input_root.rglob('train.py') if (p.parent/'dataset.py').is_file() and (p.parent/'app').is_dir()] if input_root.exists() else []\n"
                "source_archives = sorted(input_root.rglob('source_code.zip')) if input_root.exists() else []\n"
                "if source_roots:\n"
                "    source_root = source_roots[0]\n"
                "elif source_archives:\n"
                "    source_root = Path('/kaggle/working/minecraft-source'); source_root.mkdir(parents=True, exist_ok=True)\n"
                '    with zipfile.ZipFile(source_archives[0]) as zf:\n'
                '        bad = next((n for n in zf.namelist() if Path(n).is_absolute() or ".." in Path(n).parts), None)\n'
                '        if bad: raise ValueError(f"Unsafe path in source archive: {bad}")\n'
                '        zf.extractall(source_root)\n'
                "    matches = list(source_root.rglob('train.py'))\n"
                "    if not matches: raise FileNotFoundError('Training source archive is incomplete.')\n"
                "    source_root = matches[0].parent\n"
                "else: raise FileNotFoundError('Training source not found in Kaggle inputs.')\n"
                'sys.path.insert(0, str(source_root))\n'
                "data_archives = sorted(input_root.rglob('training_data.zip')) if input_root.exists() else []\n"
                "if data_archives:\n"
                "    data_root = Path('/kaggle/working/data'); data_root.mkdir(parents=True, exist_ok=True)\n"
                '    with zipfile.ZipFile(data_archives[0]) as zf:\n'
                '        bad = next((n for n in zf.namelist() if Path(n).is_absolute() or ".." in Path(n).parts), None)\n'
                '        if bad: raise ValueError(f"Unsafe path in training archive: {bad}")\n'
                '        zf.extractall(data_root)\n'
                "else: data_root = input_root\n"
                "schematics = list(data_root.rglob('*.schem')) + list(data_root.rglob('*.schematic'))\n"
                "if not schematics:\n"
                "    raise RuntimeError('Training archive contains no .schem/.schematic files.')\n"
                'print(f"Source: {source_root}")\n'
                'print(f"Data root: {data_root}")\n'
                'print(f"Structures found: {len(schematics)}")'
            ),
            _code_cell(
                '# Training settings\n'
                f'GRID_SIZE = ({gx}, {gy}, {gz})\n'
                f'EPOCHS = {epochs}\n'
                f'BATCH_SIZE = {batch_size}\n'
                f'LR = {learning_rate}\n'
                f'AUG_DIVERSITY = {aug_diversity}\n'
                f'ALLOW_VERTICAL = {allow_vertical_flag}\n'
                f'AIR_WEIGHT = {air_weight}\n'
                'NOISE_BLOCK_PROB = 0.20\n'
                'import torch\n'
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                'print(f"Device: {DEVICE}")\n'
                'if DEVICE.type == "cuda":\n'
                '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
                'else:\n'
                '    raise RuntimeError("Kaggle GPU is unavailable; enable GPU in the notebook settings.")'
            ),
            _code_cell(
                '# Load data and initialize the training loader.\n'
                'from dataset import MultiSourceSchematicDataset\n'
                'from torch.utils.data import DataLoader\n'
                'def report_data_progress(done, total):\n'
                '    if done % 25 == 0 or done == total:\n'
                '        print(f"Data preparation: {done}/{total} structures checked ({100*done/max(1,total):.0f}%)", flush=True)\n'
                'dataset = MultiSourceSchematicDataset(\n'
                '    [str(data_root)], target_size=GRID_SIZE, max_voxels=400_000,\n'
                '    augmentation_diversity=AUG_DIVERSITY,\n'
                '    allow_vertical_movement=ALLOW_VERTICAL, air_weight_factor=AIR_WEIGHT,\n'
                '    progress_callback=report_data_progress,\n'
                ')\n'
                "if len(dataset) == 0:\n"
                '    raise RuntimeError("No usable prompt/structure training examples were discovered.")\n'
                'print(f"Training examples: {len(dataset)}")\n'
                'print(f"Block vocab: {len(dataset.voxel_tokenizer.id_to_block)}")\n'
                'print(f"Prompt vocab: {len(dataset.prompt_tokenizer.token_to_id)}")\n'
                'loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)'
            ),
            _code_cell(model_code),
            _code_cell(train_code),
            _code_cell(
                '# Training summary\n'
                'elapsed = time.time() - start_time\n'
                'print(f"Training completed in {elapsed / 60:.1f} min")\n'
                'print(f"Best loss: {best_loss:.4f}")\n'
                'print("Model saved to model.pt in the notebook working directory")'
            ),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.0'},
            'kaggle': {'accelerator': 'GPU', 'dataSources': []},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }

    notebook_path = export_dir / 'kaggle_notebook.ipynb'
    with notebook_path.open('w', encoding='utf-8') as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
    print(f'  Notebook created: {notebook_path}')
def _build_transformer_model_code(d_model, nhead, num_layers, dim_ff, recurrent_depth=4,
                                  moe_experts=0, moe_active=1,
                                  text_encoder_name=None, context_dim=None, window_size=8):
    encoder_dim = int(context_dim or 0) if text_encoder_name else 0
    code = (
        "# Create model\n"
        "import math\n"
        "from model import SharedWeightVoxelTransformer\n"
        "model = SharedWeightVoxelTransformer(\n"
        "    text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),\n"
        "    block_vocab_size=len(dataset.voxel_tokenizer.id_to_block),\n"
        "    grid_size=GRID_SIZE,\n"
        f"    d_model={d_model},\n"
        f"    nhead={nhead},\n"
        f"    num_layers={num_layers},\n"
        f"    dim_feedforward={dim_ff},\n"
        f"    recurrent_depth={recurrent_depth}, moe_experts={moe_experts}, moe_active={moe_active},\n"
        f"    window_size={window_size},\n"
        f"    text_input_dim={encoder_dim}, text_encoder_name={text_encoder_name!r},\n"
        "    dropout=0.1,\n"
        ").to(DEVICE)\n\n"
        "from app.training_memory import plan_transformer_training, describe_training_plan\n"
        "from app.grad_accum import iter_micro_batches, per_sample_weight_mass, apply_optimizer_step\n"
        "plan = plan_transformer_training(\n"
        "    GRID_SIZE, BATCH_SIZE, len(dataset.voxel_tokenizer.id_to_block), device=DEVICE,\n"
        f"    d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_ff},\n"
        f"    window_size={window_size}, recurrent_depth={recurrent_depth}, allow_accumulation=True,\n"
        ")\n"
        "print(describe_training_plan(plan, 'en'), flush=True)\n"
        "if not plan.fits:\n"
        "    raise RuntimeError('Selected DiT/grid does not fit the available Kaggle GPU memory.')\n"
        "loader = DataLoader(dataset, batch_size=plan.effective_batch_size, shuffle=True, num_workers=2, pin_memory=True)\n\n"
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'Trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)')\n\n"
        "print(f'Active parameters per token: {model.active_parameters:,} ({model.active_parameters/1e6:.2f}M)')\n\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)\n\n"
        "# Cosine LR scheduler with warmup\n"
        "total_steps = EPOCHS * len(loader)\n"
        "warmup_steps = min(500, max(1, total_steps // 20))\n\n"
        "def lr_lambda(step):\n"
        "    if step < warmup_steps:\n"
        "        return float(step) / max(1, warmup_steps)\n"
        "    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)\n"
        "    return 0.5 * (1.0 + math.cos(progress * math.pi))\n\n"
        "scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)"
    )
    if text_encoder_name:
        encoder_setup = (
            "from app.transformer_encoder import FrozenTransformerEncoder\n"
            f"dit_encoder = FrozenTransformerEncoder(model_name={text_encoder_name!r}, device=DEVICE, "
            "dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32)\n\n"
        )
        code = code.replace(
            "from app.training_memory import plan_transformer_training, describe_training_plan\n",
            encoder_setup +
            "from app.training_memory import plan_transformer_training, describe_training_plan\n",
            1,
        )
    return code


def _build_transformer_train_code(air_weight, d_model, nhead, num_layers, dim_ff,
                                 text_encoder_name=None, context_dim=None,
                                 recurrent_depth=4, moe_experts=0, moe_active=1,
                                 window_size=8):
    prompt_forward = (
        "            with torch.no_grad():\n"
        "                encoded = dit_encoder(list(micro['prompt_text']))\n"
        "                prompt_embeddings = encoded['last_hidden_state'].to(DEVICE, dtype=next(model.parameters()).dtype)\n"
        "                prompt_mask = encoded['attention_mask'].to(DEVICE)\n"
        "            model_inputs = {'prompt_embeddings': prompt_embeddings, 'prompt_attention_mask': prompt_mask}\n"
        if text_encoder_name else "            model_inputs = {'prompt_ids': prompt_ids}\n"
    )
    return (
        "# Training Loop\n"
        "best_loss = float('inf')\n"
        "loss_history = []\n"
        "start_time = time.time()\n\n"
        "for epoch in range(1, EPOCHS + 1):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    num_batches = 0\n\n"
        "    log_interval = max(1, min(50, (len(loader) + 19) // 20))\n"
        "    for batch in loader:\n"
        "        masses = per_sample_weight_mass(batch['voxel_ids'], batch) if plan.accumulated else None\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        batch_loss = 0.0\n"
        "        for micro, share in iter_micro_batches(batch, plan.batch_size, sample_mass=masses):\n"
        "            prompt_ids = micro['prompt_ids'].to(DEVICE, non_blocking=True)\n"
        "            target = micro['voxel_ids'].to(DEVICE, non_blocking=True).reshape(prompt_ids.shape[0], -1)\n"
        "            target = model.safe_clamp_target(target)\n\n"
        + prompt_forward +
        "            sample_weight = micro['sample_weight'].to(DEVICE).view(-1, 1).expand_as(target)\n"
        "            per_block_w = micro['per_block_weight'].to(DEVICE).view(-1, 1).expand_as(target)\n"
        "            per_air_w = micro['per_air_weight'].to(DEVICE).view(-1, 1).expand_as(target)\n"
        "            weight_per_token = torch.where(target == 0, per_air_w, per_block_w) * sample_weight\n"
        "            loss = model.backward_weighted_loss(target, weight_per_token, chunk_voxels=512, grad_scale=share, **model_inputs)\n"
        "            batch_loss += float(loss) * share\n"
        "        apply_optimizer_step(model, optimizer, max_norm=1.0)\n"
        "        scheduler.step()\n\n"
        "        total_loss += batch_loss\n"
        "        num_batches += 1\n\n"
        "        if num_batches % log_interval == 0 or num_batches == len(loader):\n"
        "            print(f'  step={num_batches}/{len(loader)} ({100*num_batches/max(1,len(loader)):.0f}%)  loss_so_far={total_loss/max(1,num_batches):.4f}', flush=True)\n\n"
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
        f"            'recurrent_depth': {recurrent_depth},\n"
        f"            'window_size': {window_size},\n"
        f"            'moe_experts': {moe_experts},\n"
        f"            'moe_active': {moe_active},\n"
        f"            'text_encoder_name': {text_encoder_name!r},\n"
        f"            'text_input_dim': {int(context_dim or 0) if text_encoder_name else 0},\n"
        "            'augmentation_diversity': AUG_DIVERSITY,\n"
        "            'allow_vertical_movement': ALLOW_VERTICAL,\n"
        "            'epoch': epoch,\n"
        "            'loss': avg_loss,\n"
        "        }, 'model.pt')\n"
        "        print(f'  -> new best model saved (loss={avg_loss:.4f})')"
    ).replace("{d_model}", str(d_model)).replace("{nhead}", str(nhead)).replace("{num_layers}", str(num_layers)).replace("{dim_ff}", str(dim_ff))


def _build_diffusion_model_code():
    return (
        "# Create model\n"
        "from app.diffusion_model import VoxelDiffusionModel, train_diffusion_step\n"
        "model = VoxelDiffusionModel(\n"
        "    num_blocks=len(dataset.voxel_tokenizer.id_to_block),\n"
        "    text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),\n"
        "    grid_size=GRID_SIZE,\n"
        "    d_model=128, d_text=64, channels=64,\n"
        "    channel_multipliers=(1, 2, 2), num_timesteps=50,\n"
        ").to(DEVICE)\n\n"
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'Trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)')\n\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)"
    )


def _build_diffusion_train_code(air_weight):
    return (
        "# Training Loop\n"
        "best_loss = float('inf')\n"
        "loss_history = []\n"
        "start_time = time.time()\n\n"
        "for epoch in range(1, EPOCHS + 1):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    num_batches = 0\n\n"
        "    log_interval = max(1, min(50, (len(loader) + 19) // 20))\n"
        "    for batch in loader:\n"
        "        loss = train_diffusion_step(model, batch, optimizer, DEVICE, noise_block_prob=NOISE_BLOCK_PROB)\n"
        "        total_loss += loss\n"
        "        num_batches += 1\n\n"
        "        if num_batches % log_interval == 0 or num_batches == len(loader):\n"
        "            print(f'  step={num_batches}/{len(loader)} ({100*num_batches/max(1,len(loader)):.0f}%)  loss_so_far={total_loss/max(1,num_batches):.4f}', flush=True)\n\n"
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
        "        print(f'  -> new best model saved (loss={avg_loss:.4f})')"
    )


def _build_tf_diffusion_model_code(channels, ch_mult, d_model, ca_heads, encoder_name="Phi-3.5-mini", context_dim=DEFAULT_ENCODER_HIDDEN_DIM):
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
        "# Create model\n"
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
        "print(f'Trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)')\n\n"
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
        f"print(f'Encoder loaded: {{encoder_name}}')\n\n"
        "from app.training_memory import plan_diffusion_training, describe_training_plan\n"
        "from app.grad_accum import apply_optimizer_step, iter_micro_batches, per_sample_weight_mass\n"
        "plan = plan_diffusion_training(\n"
        "    GRID_SIZE, BATCH_SIZE, model.num_blocks, device=DEVICE,\n"
        f"    channels={channels}, channel_multipliers=({ch_mult_str}), d_model={d_model},\n"
        f"    cross_attn_heads={ca_heads}, context_dim={context_dim}, context_len=512,\n"
        ")\n"
        "print(describe_training_plan(plan, 'en'), flush=True)\n"
        "if not plan.fits:\n"
        "    raise RuntimeError('Selected TF-Diffusion/grid does not fit the available Kaggle GPU memory.')\n"
        "loader = DataLoader(dataset, batch_size=plan.effective_batch_size, shuffle=True, num_workers=2, pin_memory=True)\n"
        f"ENCODER_DISPLAY_NAME = '{encoder_name}'\n"
        f"ENCODER_HF_ID = '{hf_id}'\n"
        f"ENCODER_HIDDEN_DIM = {context_dim}\n"
    )


def _build_tf_diffusion_train_code(air_weight, encoder_name="Phi-3.5-mini", context_dim=DEFAULT_ENCODER_HIDDEN_DIM):
    is_t5 = "t5" in encoder_name.lower()
    if is_t5:
        context_extract = (
            "                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)\n"
            "                context = outputs.last_hidden_state.to(dtype=next(model.parameters()).dtype)\n"
        )
    else:
        context_extract = (
            "                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)\n"
            "                context = outputs.hidden_states[-1].to(dtype=next(model.parameters()).dtype)\n"
        )
    return (
        "# Training Loop\n"
        "best_loss = float('inf')\n"
        "loss_history = []\n"
        "start_time = time.time()\n\n"
        "for epoch in range(1, EPOCHS + 1):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    num_batches = 0\n\n"
        "    log_interval = max(1, min(50, (len(loader) + 19) // 20))\n"
        "    for batch in loader:\n"
        "        masses = per_sample_weight_mass(batch['voxel_ids'], batch) if plan.accumulated else None\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        batch_loss = 0.0\n"
        "        for micro, share in iter_micro_batches(batch, plan.batch_size, sample_mass=masses):\n"
        "            prompts = micro.get('prompt_text', None)\n"
        "            with torch.no_grad():\n"
        "                encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)\n"
        "                input_ids = encoded['input_ids'].to(DEVICE)\n"
        "                attention_mask = encoded['attention_mask'].to(DEVICE)\n"
        + context_extract +
        "            loss = train_transformer_diffusion_step(model, micro, type('FakeEncoder', (), {'__call__': lambda self, x: {'last_hidden_state': context, 'attention_mask': attention_mask}})(), optimizer, DEVICE, noise_block_prob=NOISE_BLOCK_PROB, chunk_voxels=plan.chunk_voxels, accumulate=True, grad_scale=share)\n"
        "            batch_loss += loss * share\n"
        "        apply_optimizer_step(model, optimizer, max_norm=1.0)\n"
        "        total_loss += batch_loss\n"
        "        num_batches += 1\n\n"
        "        if num_batches % log_interval == 0 or num_batches == len(loader):\n"
        "            print(f'  step={num_batches}/{len(loader)} ({100*num_batches/max(1,len(loader)):.0f}%)  loss_so_far={total_loss/max(1,num_batches):.4f}', flush=True)\n\n"
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
        "        print(f'  -> new best model saved (loss={avg_loss:.4f})')"
    )


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
    if model_type == "transformer":
        arch = architecture or {}
        arch_str = (f"d_model={arch.get('d_model', 192)}, nhead={arch.get('nhead', 6)}, "
                    f"max_depth={max(2, int(arch.get('recurrent_depth', 4)))}, "
                    f"window={arch.get('window_size', 8)}³, "
                    f"MoE={arch.get('moe_active', 1)}/{arch.get('moe_experts', 0)}, "
                    f"FFN={arch.get('dim_feedforward', 768)}")
    elif model_type == "diffusion":
        arch_str = "d_model=128, d_text=64, channels=64, (1,2,2)"
    elif model_type == "transformer_diffusion" and tf_unet_config:
        ch_mult = tf_unet_config.get("channel_multipliers", (1, 2, 2))
        ch_mult_str = ", ".join(str(m) for m in ch_mult)
        arch_str = f"channels={tf_unet_config.get('channels', 32)}, ch_mult=({ch_mult_str}), d_model={tf_unet_config.get('d_model', 64)}, heads={tf_unet_config.get('cross_attn_heads', 4)}"
    else:
        arch_str = "d_model=192, nhead=6, layers=5, FFN=768"

    model_desc = {
        "transformer": "Per-Token Recurrent DiT",
        "diffusion": "3D Voxel Diffusion Model",
        "transformer_diffusion": "Transformer Diffusion Model",
    }.get(model_type, "Transformer")

    readme = f"""# Minecraft Structure Generator — Kaggle Export

## Contents

| File | Description |
|------|-------------|
| `kaggle_notebook.ipynb` | Main training notebook (upload to Kaggle) |
| `training_data.zip` | Packaged schematic training data |
| `train.py` | Reference training script |
| `model.py` | Model definition |
| `dataset.py` | Dataset & tokenizer |
| `requirements.txt` | Python dependencies |

## How to Use

1. **Upload to Kaggle**
   - Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
   - Click "File" → "Upload" and select all files in this folder
   - Or: upload `kaggle_notebook.ipynb` first, then add `training_data.zip` as a dataset

2. **Configure Accelerator**
   - Click the "Settings" panel (gear icon) on the right
   - Set **Accelerator** → **GPU T4 x2**
   - (Optional) Set **Persistence** → **Files only** if you want the model output to persist

3. **Run the Notebook**
   - Click "Run All"
   - Training will run for {epochs} epochs with batch size {batch_size}
   - Grid size: {gx}×{gy}×{gz}
   - The best model checkpoint will be saved as `model.pt`

4. **Download the Result**
   - After training completes, find `model.pt` in the output
   - Download it and place it in your local `runs/` directory

## Training Settings

| Parameter | Value |
|-----------|-------|
| Grid Size | {gx}×{gy}×{gz} |
| Epochs | {epochs} |
| Batch Size | {batch_size} |
| Learning Rate | {learning_rate:.2e} |
| Model Type | {model_desc} |
| Architecture | {arch_str} |
| Air Weight | {air_weight} |

## Notes

- Training typically takes **2-8 hours** on 2× T4 GPUs depending on dataset size.
- The DiT notebook uses cosine LR scheduling with a warmup of up to 500 steps,
  scaled down for shorter runs.
- Loss weighting uses adaptive per-structure weights with air_weight={air_weight}.
- If you want to train a different architecture, edit the model creation cell.
"""
    readme_path = export_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    print(f"  README created: {readme_path}")


# ═══════════════════════════════════════════════════════════════
# Notebook cell builders
# ═══════════════════════════════════════════════════════════════


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"cell-{abs(hash(text)) % (16 ** 12):012x}",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"cell-{abs(hash(code)) % (16 ** 12):012x}",
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
        output_dir=str(EXPORTS_DIR),
        epochs=5,
        batch_size=4,
        learning_rate=1.5e-3,
        aug_diversity=1,
        allow_vertical=False,
        grid_size=(16, 16, 16),
        model_type="transformer",
    )
    print(f"\nExport created at: {path}")

