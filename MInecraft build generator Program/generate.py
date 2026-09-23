from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.constants import ALLOWED_GRID_SIZES, default_run_dir
from app.training_memory import (
    describe_generation_plan,
    plan_generation,
    transformer_generation_bytes,
)
from dataset import PromptTokenizer, VoxelTokenizer, save_schem, trim_token_grid
from model import SharedWeightVoxelTransformer


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
    parser.add_argument("--model-size", type=str, default=None,
                        choices=list(ALLOWED_GRID_SIZES.keys()),
                        help="Model size: 16 = 16x16x16, 32 = 32x32x32, 48 = 48x48x48, 64 = 64x64x64, 96 = 96x96x96, 128 = 128x128x128, 256 = 256x256x256")
    parser.add_argument("--grid-size", type=parse_size, default=None,
                        help="DEPRECATED: Use --model-size instead.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--chunk-voxels", type=int, default=0,
                        help="Logits in Voxel-Chunks berechnen (0 = automatisch aus dem freien VRAM)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # --model-size bevorzugt; --grid-size (deprecated) wird sonst aufgelöst.
    # --grid-size war zuvor unerreichbar, weil --model-size nie None war.
    if args.model_size is None and args.grid_size is not None:
        args.model_size = next((s for s, v in ALLOWED_GRID_SIZES.items() if v == args.grid_size), "16")
    if args.model_size is None:
        args.model_size = "16"

    # Run-Dir bestimmen: Falls nicht gesetzt, wird es anhand der Modell-Größe gewählt
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = default_run_dir(args.model_size)

    # Grid-Größe aus Modell- Größe (--grid-size wird über args.model_size aufgelöst)
    grid_size = ALLOWED_GRID_SIZES[args.model_size]

    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Kein Modell gefunden: {checkpoint_path}\n"
            f"Bitte --run-dir angeben oder zuerst trainieren "
            f"(python train.py --model-size {args.model_size})."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

    # Logits im Speicher begrenzen: der Ausgabekopf kostet Voxel *
    # block_vocab_size Werte (Grid 64^3 mit 17.690 Blockzustaenden waeren das
    # 18,5 GiB) - deshalb plant das CLI die Chunk-Groesse wie die GUI.
    attn_bytes, rest_bytes = transformer_generation_bytes(
        grid_size, 1, model.block_vocab_size, checkpoint["d_model"],
        checkpoint.get("nhead", 8), checkpoint["layers"],
        checkpoint.get("dim_feedforward", 1024))
    plan = plan_generation(grid_size, model.block_vocab_size,
                           device=torch.device(args.device),
                           activation_bytes=attn_bytes + rest_bytes)
    if not plan.fits:
        raise SystemExit(describe_generation_plan(plan, "de"))
    print(describe_generation_plan(plan, "de"))
    chunk_voxels = args.chunk_voxels if args.chunk_voxels > 0 else (
        plan.chunk_voxels if plan.chunked else None)

    token_grid = trim_token_grid(model.generate(
        prompt_ids, temperature=args.temperature, top_k=args.top_k,
        chunk_voxels=chunk_voxels)[0].cpu())
    save_schem(args.output, token_grid, voxel_tokenizer.id_to_block)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
