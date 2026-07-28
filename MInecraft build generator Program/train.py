from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MultiSourceSchematicDataset
from model import SharedWeightVoxelTransformer


# Erlaubte Grid-Größen
ALLOWED_GRID_SIZES = {
    "16": (16, 16, 16),
    "32": (32, 32, 32),
    "48": (48, 48, 48),
}


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
                        help="Directories with schematic data (including scraped data)")
    parser.add_argument("--priority-data-dir", action="append", default=None,
                        help="High-quality manually analyzed directories (get higher weight)")
    parser.add_argument("--priority-weight", type=float, default=5.0,
                        help="Weight multiplier for the priority (good) data dirs (default: 5.0)")
    parser.add_argument("--max-voxels", type=int, default=100_000)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. If not set, auto-selected based on --model-size.")
    parser.add_argument("--model-size", type=str, default="16",
                        choices=list(ALLOWED_GRID_SIZES.keys()),
                        help="Model/Grid size: 16 = 16x16x16, 32 = 32x32x32, 48 = 48x48x48")
    parser.add_argument("--grid-size", type=parse_size, default=None,
                        help="DEPRECATED: Use --model-size instead. Grid size (16,16,16), (32,32,32) or (48,48,48).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augmentation-diversity", type=int, default=1,
                        help="How densely each build is shifted/rotated for training. 0 = centered only.")
    parser.add_argument("--allow-vertical-movement", action="store_true",
                        help="Allow augmented builds to move upward inside the training grid.")
    args = parser.parse_args()

    # Grid-Größe bestimmen: --model-size hat Vorrang, falls nicht gesetzt dann --grid-size, sonst default 16
    if args.model_size is not None:
        grid_size = ALLOWED_GRID_SIZES[args.model_size]
    elif args.grid_size is not None:
        grid_size = args.grid_size
    else:
        grid_size = (16, 16, 16)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        # Dynamisch basierend auf Modell-Größe
        out_dir = Path(f"runs/voxel_transformer_{args.model_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Grid size: {grid_size[0]}x{grid_size[1]}x{grid_size[2]}")

    data_dirs = args.data_dir or [
        "Trainingsdaten good thoroughly analyzed",
        "Trainingsdaten zu gross vorerst ausgelagert",
    ]
    priority_dirs = args.priority_data_dir or ["Trainingsdaten good thoroughly analyzed"]

    print(f"Data dirs: {data_dirs}")
    print(f"Priority dirs (higher weight): {priority_dirs}")
    print(f"Priority weight multiplier: {args.priority_weight}")
    print(f"Augmentation diversity: {args.augmentation_diversity}")
    print(f"Vertical movement: {'enabled' if args.allow_vertical_movement else 'disabled'}")

    dataset = MultiSourceSchematicDataset(
        data_dirs,
        target_size=grid_size,
        max_voxels=args.max_voxels,
        augmentation_diversity=args.augmentation_diversity,
        allow_vertical_movement=args.allow_vertical_movement,
    )
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Structure categories: {len(getattr(dataset, '_category_by_path', {}))}")
    if getattr(dataset, "skipped_examples", None):
        print(f"Skipped without clipping: {len(dataset.skipped_examples)} examples")
    print(f"Block vocab size: {len(dataset.voxel_tokenizer.id_to_block)}")
    print(f"Prompt vocab size: {len(dataset.prompt_tokenizer.token_to_id)}")

    dataset.prompt_tokenizer.save(out_dir / "prompt_vocab.json")
    dataset.voxel_tokenizer.save(out_dir / "block_vocab.json")

    if args.priority_weight != 1.0:
        print("Priority sampler disabled: per-structure weighted loss keeps every build equally valued.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Cosine annealing with warmup
    warmup_steps = args.warmup_steps
    total_steps = args.epochs * len(loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.1415926535))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_loss_1 = 0.0
        total_loss_2 = 0.0
        num_batches = 0

        for batch in loader:
            prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
            target = batch["voxel_ids"].to(device, non_blocking=True).reshape(prompt_ids.shape[0], -1)
            target = model.safe_clamp_target(target)

            logits = model(prompt_ids)
            # Gewichtung: Air (0) = 1/16 = 0.0625, echte Blöcke = 1.0
            target_flat = target.reshape(-1)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True).view(-1, 1)
            sample_weight = sample_weight.expand_as(target).reshape(-1)
            weight_per_token = torch.where(target_flat == 0, 0.0625, 1.0) * sample_weight
            logp = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
            nll = torch.nn.functional.nll_loss(logp, target_flat, reduction='none')
            loss = (nll * weight_per_token).sum() / weight_per_token.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.detach())
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
                    "priority_dirs": [str(path) for path in priority_dirs],
                    "priority_weight": args.priority_weight,
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
