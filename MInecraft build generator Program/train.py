from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.constants import ALLOWED_GRID_SIZES, default_run_dir
from app.grad_accum import (
    accumulation_enabled,
    apply_optimizer_step,
    iter_micro_batches,
    per_sample_weight_mass,
)
from app.paths import default_data_dirs
from app.transformer_encoder import MODEL_NAMES, FrozenTransformerEncoder
from app.training_memory import describe_training_plan, plan_transformer_training
from dataset import MultiSourceSchematicDataset
from model import SharedWeightVoxelTransformer


def parse_size(value: str) -> tuple[int, int, int]:
    """Parse a grid size string like '16,16,16' or '16x16x16'."""
    parts = tuple(int(v) for v in value.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use X,Y,Z or XxYxZ")
    if parts not in ALLOWED_GRID_SIZES.values():
        allowed = ", ".join(f"{s[0]}x{s[1]}x{s[2]}" for s in ALLOWED_GRID_SIZES.values())
        raise argparse.ArgumentTypeError(f"Grid size must be one of: {allowed}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", default=None,
                        help="Directories with schematic data (including scraped data). "
                             "Default: the local Trainingsdaten folders (recursive)")
    parser.add_argument("--max-voxels", type=int, default=400_000)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. If not set, auto-selected based on --model-size.")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=list(ALLOWED_GRID_SIZES.keys()),
                        help="Model/Grid size: 16 = 16x16x16, 32 = 32x32x32, 48 = 48x48x48, etc.")
    parser.add_argument("--grid-size", type=parse_size, default=None,
                        help="DEPRECATED: Use --model-size instead. Grid size (16,16,16), (32,32,32) or (48,48,48).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-grad-accum", dest="grad_accum", action="store_false",
                        default=accumulation_enabled(),
                        help="Wenn der Batch nicht in den VRAM passt: verkleinern statt "
                             "die Gradienten ueber Mikro-Batches zu akkumulieren "
                             "(Standard: akkumulieren, siehe app/grad_accum.py)")
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dim-feedforward", type=int, default=768)
    parser.add_argument("--recurrent-depth", type=int, default=4, help="Maximum per-token recurrence depth for DiT (minimum 2)")
    parser.add_argument("--window-size", type=int, choices=(0, 4, 8, 16), default=8,
                        help="Self-attention: 4/8/16 voxel windows or 0 for global attention")
    parser.add_argument("--tread-ratio", type=float, default=0.25,
                        help="Fraction of voxel tokens routed around each DiT block (0 disables TREAD routing)")
    parser.add_argument("--ssa-topk", type=int, default=16,
                        help="Spatial windows selected per window for SSA global context (0 disables SSA)")
    parser.add_argument("--voxel-mask-ratio", type=float, default=0.65,
                        help="Fraction of air-target loss masked; all non-air targets are retained")
    parser.add_argument("--moe-experts", type=int, default=0, help="MoE expert count; 0 disables MoE")
    parser.add_argument("--moe-active", type=int, default=1, help="Top-k experts active per token")
    parser.add_argument("--text-encoder", choices=MODEL_NAMES, default=None,
                        help="Frozen pretrained text encoder; omit to learn prompt embeddings from scratch")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augmentation-diversity", type=int, default=1,
                        help="How densely each build is shifted/rotated for training. 0 = centered only.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (0 = load in the main process)")
    parser.add_argument("--allow-vertical-movement", action="store_true",
                        help="Allow augmented builds to move upward inside the training grid.")
    args = parser.parse_args()
    if args.recurrent_depth < 2:
        parser.error("DiT requires --recurrent-depth >= 2")

    # Grid-Größe bestimmen: --model-size hat Vorrang, sonst --grid-size (deprecated),
    # sonst Standard 16. Wichtig: --grid-size war zuvor unerreichbar, weil die "Nicht None für
    # --model-size. --grid-size löst nun die passende Modell-Größe auf.
    if args.model_size is not None:
        grid_size = ALLOWED_GRID_SIZES[args.model_size]
    elif args.grid_size is not None:
        grid_size = args.grid_size
        args.model_size = next((s for s, v in ALLOWED_GRID_SIZES.items() if v == grid_size), None)
    else:
        grid_size = ALLOWED_GRID_SIZES["16"]
        args.model_size = "16"

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        # Dynamisch basierend auf Modell-Größe (absoluter Pfad im Projektordner)
        out_dir = default_run_dir(args.model_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Grid size: {grid_size[0]}x{grid_size[1]}x{grid_size[2]}")

    data_dirs = [Path(p) for p in args.data_dir] if args.data_dir else default_data_dirs()

    print(f"Data dirs: {[str(p) for p in data_dirs]}")
    print(f"Augmentation diversity: {args.augmentation_diversity}")
    print(f"Vertical movement: {'enabled' if args.allow_vertical_movement else 'disabled'}")
    print(f"Lade Strukturen (max {args.max_voxels:,} Voxel pro Struktur) ...", flush=True)

    def report_progress(done: int, total: int) -> None:
        print(f"  {done}/{total} Beispiele geprueft", flush=True)

    dataset = MultiSourceSchematicDataset(
        data_dirs,
        target_size=grid_size,
        max_voxels=args.max_voxels,
        augmentation_diversity=args.augmentation_diversity,
        allow_vertical_movement=args.allow_vertical_movement,
        progress_callback=report_progress,
    )
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Structure categories: {len(getattr(dataset, '_category_by_path', {}))}")
    if getattr(dataset, "skipped_examples", None):
        print(f"Skipped examples: {len(dataset.skipped_examples)}")
    oversize_voxels = int(getattr(dataset, "oversize_voxels", 0))
    if oversize_voxels:
        print(f"  davon zu gross (> {args.max_voxels:,} Voxel, nicht dekodiert): "
              f"{oversize_voxels:,} Voxel gespart")
    print(f"Block vocab size: {len(dataset.voxel_tokenizer.id_to_block)}")
    print(f"Prompt vocab size: {len(dataset.prompt_tokenizer.token_to_id)}")

    dataset.prompt_tokenizer.save(out_dir / "prompt_vocab.json")
    dataset.voxel_tokenizer.save(out_dir / "block_vocab.json")

    text_encoder = None
    if args.text_encoder:
        text_encoder = FrozenTransformerEncoder(
            model_name=args.text_encoder, device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

    # ── VRAM-Guard ──
    # Die Self-Attention laeuft standardmaessig in lokalen 3D-Fenstern (8^3);
    # der Planer beruecksichtigt Fenster-Attention und Batch-Anpassungen.
    plan = plan_transformer_training(
        grid_size, args.batch_size, len(dataset.voxel_tokenizer.id_to_block),
        device=device, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.layers, dim_feedforward=args.dim_feedforward,
        recurrent_depth=max(1, args.recurrent_depth),
        window_size=args.window_size,
        allow_accumulation=args.grad_accum,
    )
    print(describe_training_plan(plan, "de"))
    if not plan.fits:
        raise SystemExit(
            "Abbruch: Diese Grid-/Batch-Kombination passt nicht in den VRAM. "
            "Bitte --model-size verkleinern (Vorschlag siehe oben).")
    if plan.accumulated:
        # Der DataLoader liefert weiterhin die gewuenschte Batch-Groesse; gerechnet
        # wird in Mikro-Batches (Gradient-Akkumulation, siehe app/grad_accum.py).
        print(f"Gradient Accumulation: Batch {plan.effective_batch_size} "
              f"= {plan.accum_steps} x {plan.batch_size} Mikro-Batches.")
    if plan.reduced:
        print(f"Batch-Groesse von {args.batch_size} auf {plan.batch_size} reduziert, "
              f"effektiv {plan.effective_batch_size}.")
    args.batch_size = plan.effective_batch_size

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = SharedWeightVoxelTransformer(
        text_vocab_size=len(dataset.prompt_tokenizer.token_to_id),
        block_vocab_size=len(dataset.voxel_tokenizer.id_to_block),
        grid_size=grid_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        recurrent_depth=max(1, args.recurrent_depth),
        window_size=args.window_size,
        moe_experts=max(0, args.moe_experts), moe_active=max(1, args.moe_active),
        text_input_dim=int(text_encoder.hidden_dim) if text_encoder else 0,
        text_encoder_name=args.text_encoder,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup
    warmup_steps = args.warmup_steps
    total_steps = args.epochs * len(loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            # Gradient-Akkumulation: der DataLoader liefert den ganzen (effektiven)
            # Batch, gerechnet wird in Mikro-Batches, die in den VRAM passen.
            # Ohne Akkumulation ist das genau ein Durchlauf (Anteil 1.0). Als
            # Anteil zaehlt die Gewichtsmasse - so sind die summierten Gradienten
            # exakt die des grossen Batches.
            masses = (per_sample_weight_mass(batch["voxel_ids"], batch)
                      if plan.accumulated else None)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for micro, share in iter_micro_batches(batch, plan.batch_size,
                                                   sample_mass=masses):
                prompt_ids = micro["prompt_ids"].to(device, non_blocking=True)
                target = micro["voxel_ids"].to(device, non_blocking=True).reshape(prompt_ids.shape[0], -1)
                target = model.safe_clamp_target(target)

                if text_encoder is not None:
                    encoded = text_encoder(list(micro["prompt_text"]))
                    model_inputs = {
                        "prompt_embeddings": encoded["last_hidden_state"].to(
                            device=device, dtype=next(model.parameters()).dtype),
                        "prompt_attention_mask": encoded["attention_mask"].to(device),
                    }
                else:
                    model_inputs = {"prompt_ids": prompt_ids}
                # Adaptive per-structure weights (consistent with GUI)
                sample_weight = micro["sample_weight"].to(device, non_blocking=True).view(-1, 1)
                sample_weight = sample_weight.expand_as(target)
                per_block_w = micro["per_block_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(target)
                per_air_w = micro["per_air_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(target)
                weight_per_token = torch.where(target == 0, per_air_w, per_block_w) * sample_weight
                loss = model.backward_weighted_loss(
                    target, weight_per_token, chunk_voxels=plan.chunk_voxels,
                    grad_scale=share, **model_inputs,
                )
                batch_loss += loss * share

            if args.grad_clip > 0:
                apply_optimizer_step(model, optimizer, max_norm=args.grad_clip)
            else:
                optimizer.step()
            scheduler.step()

            total_loss += batch_loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch={epoch:3d}/{args.epochs} "
            f"loss={avg_loss:.4f} "
            f"lr={current_lr:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "grid_size": grid_size,
                    "text_vocab_size": len(dataset.prompt_tokenizer.token_to_id),
                    "block_vocab_size": len(dataset.voxel_tokenizer.id_to_block),
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "layers": args.layers,
                    "dim_feedforward": args.dim_feedforward,
                    "recurrent_depth": model.recurrent_depth,
                    "window_size": model.window_size,
                    "tread_ratio": model.tread_ratio,
                    "ssa_topk": model.ssa_topk,
                    "voxel_mask_ratio": model.voxel_mask_ratio,
                    "moe_experts": model.moe_experts,
                    "moe_active": model.moe_active,
                    "text_encoder_name": model.text_encoder_name,
                    "text_input_dim": model.text_input_dim,
                    "data_dirs": [str(path) for path in data_dirs],
                    "max_voxels": args.max_voxels,
                    "augmentation_diversity": args.augmentation_diversity,
                    "allow_vertical_movement": args.allow_vertical_movement,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "model_size": args.model_size,
                },
                out_dir / "model.pt",
            )
            print(f"  -> new best model saved (loss={avg_loss:.4f})")

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Final model saved to {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
