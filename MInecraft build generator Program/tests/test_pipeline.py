from __future__ import annotations

from pathlib import Path

import torch
from nbtlib import File

from dataset import PromptTokenizer, SchematicData, VoxelTokenizer, load_schematic, save_schem
from model import SharedWeightVoxelTransformer


def test_model_single_pass_shapes():
    model = SharedWeightVoxelTransformer(
        text_vocab_size=12,
        block_vocab_size=5,
        grid_size=(4, 4, 4),
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
    )
    prompt_ids = torch.randint(0, 12, (2, 8))
    logits = model(prompt_ids)
    assert logits.shape == (2, 64, 5), f"Expected (2, 64, 5), got {logits.shape}"
    assert model.generate(prompt_ids).shape == (2, 4, 4, 4)


def test_tokenizers_and_schem_roundtrip():
    schematic = SchematicData(
        blocks=["minecraft:air", "minecraft:stone", "minecraft:oak_planks", "minecraft:air"] * 2,
        blocks_stripped=["minecraft:air", "minecraft:stone", "minecraft:oak_planks", "minecraft:air"] * 2,
        size=(2, 2, 2),
    )
    voxel_tokenizer = VoxelTokenizer()
    voxel_tokenizer.fit([schematic], target_size=(2, 2, 2))
    grid = voxel_tokenizer.encode_blocks(schematic.blocks, schematic.size, (2, 2, 2))

    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "roundtrip.schem"
    save_schem(out, grid, voxel_tokenizer.id_to_block)

    loaded = load_schematic(out)
    assert loaded.size == (2, 2, 2)
    assert set(loaded.blocks) == {"minecraft:air", "minecraft:stone", "minecraft:oak_planks"}
    assert "Schematic" in File.load(out, gzipped=True)


def test_prompt_tokenizer_padding():
    tokenizer = PromptTokenizer(max_length=5)
    tokenizer.fit(["small wooden house"])
    encoded = tokenizer.encode("small house with tower")
    assert encoded.shape == (5,)
    assert encoded[-1].item() == 0
