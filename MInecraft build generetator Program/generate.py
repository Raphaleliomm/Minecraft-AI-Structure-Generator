from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import PromptTokenizer, VoxelTokenizer, save_schem, trim_token_grid
from model import SharedWeightVoxelTransformer


# Erlaubte Grid-Größen (muss mit train.py übereinstimmen)
ALLOWED_GRID_SIZES = {
    "16": (16, 16, 16),
    "32": (32, 32, 32),
    "48": (48, 48, 48),
}

ALLOWED_GRID_RUN_DIRS = {
    "16": "runs/voxel_transformer_16",
    "32": "runs/voxel_transformer_32",
    "48": "runs/voxel_transformer_48",
}


def parse_size(value: str) -> tuple[int, int, int]:
    """Parse a grid size string like '16,16,16' or '16x16x16'."""
    parts = tuple(int(v) for v in value.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use X,Y,Z or XxYxZ")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--run-dir", default=None,
                        help="Run directory with model.pt. If not set, auto-selected based on --model-size.")
    parser.add_argument("--output", default="generated.schem")
    parser.add_argument("--model-size", type=str, default="16",
                        choices=list(ALLOWED_GRID_SIZES.keys()),
                        help="Model size: 16 = 16x16x16, 32 = 32x32x32, 48 = 48x48x48")
    parser.add_argument("--grid-size", type=parse_size, default=None,
                        help="DEPRECATED: Use --model-size instead.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Run-Dir bestimmen: Falls nicht gesetzt, automatisch anhand der Modell-Größe
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(ALLOWED_GRID_RUN_DIRS.get(args.model_size, "runs/voxel_transformer_scaled"))

    # Grid-Größe aus Modell-Größe oder Argument
    if args.model_size is not None:
        grid_size = ALLOWED_GRID_SIZES[args.model_size]
    elif args.grid_size is not None:
        grid_size = args.grid_size
    else:
        grid_size = None  # wird später aus checkpoint gelesen

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    prompt_tokenizer = PromptTokenizer.load(run_dir / "prompt_vocab.json")
    voxel_tokenizer = VoxelTokenizer.load(run_dir / "block_vocab.json")

    if grid_size is None:
        grid_size = tuple(checkpoint["grid_size"])

    model = SharedWeightVoxelTransformer(
        text_vocab_size=checkpoint["text_vocab_size"],
        block_vocab_size=checkpoint["block_vocab_size"],
        grid_size=grid_size,
        d_model=checkpoint["d_model"],
        nhead=checkpoint.get("nhead", 8),
        num_layers=checkpoint["layers"],
        dim_feedforward=checkpoint.get("dim_feedforward", 1024),
        dropout=0.0,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])

    prompt_ids = prompt_tokenizer.encode(args.prompt).unsqueeze(0).to(args.device)
    token_grid = trim_token_grid(model.generate(prompt_ids, temperature=args.temperature, top_k=args.top_k)[0].cpu())
    save_schem(args.output, token_grid, voxel_tokenizer.id_to_block)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
