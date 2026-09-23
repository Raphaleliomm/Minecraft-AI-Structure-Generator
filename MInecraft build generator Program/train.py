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
    parser.add_argument("--batch-size", type=int, default=4)
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

    # ── VRAM-Guard ──
    # Der VoxelTransformer berechnet die Self-Attention ueber ALLE Voxel
    # ([B, heads, Voxel, Voxel]) - bei Grid 32^3 sind das 6 * 32768^2 * 4 Byte
    # = 24 GiB pro Matrix und Layer und das laesst sich nicht chunken. Frueher
    # brach das Training erst mitten im ersten Batch mit "Cuda out of memory"
    # ab; hier wird vorher gerechnet und die Batch-Groesse reduziert (bzw. mit
    # klarer Meldung abgebrochen).
    plan = plan_transformer_training(
        grid_size, args.batch_size, len(dataset.voxel_tokenizer.id_to_block),
        device=device, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.layers, dim_feedforward=args.dim_feedforward,
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

                logits = model(prompt_ids)
                # Adaptive per-structure weights (consistent with GUI)
                target_flat = target.reshape(-1)
                sample_weight = micro["sample_weight"].to(device, non_blocking=True).view(-1, 1)
                sample_weight = sample_weight.expand_as(target).reshape(-1)
                per_block_w = micro["per_block_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(target).reshape(-1)
                per_air_w = micro["per_air_weight"].to(device, non_blocking=True).view(-1, 1).expand_as(target).reshape(-1)
                weight_per_token = torch.where(target_flat == 0, per_air_w, per_block_w) * sample_weight
                logp = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
                nll = torch.nn.functional.nll_loss(logp, target_flat, reduction='none')
                loss = (nll * weight_per_token).sum() / weight_per_token.sum().clamp_min(1.0)

                # Der Anteil am Gesamt-Batch skaliert nur den Gradienten; die
                # angezeigte Verlustkurve bleibt der Mittelwert.
                backward_loss = loss if share == 1.0 else loss * share
                backward_loss.backward()
                batch_loss += float(loss.detach()) * share

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