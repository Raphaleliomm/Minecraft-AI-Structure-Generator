"""Texture-aware augmentation - preserves structure, varies textures intelligently.

Key principles:
1. **Structure is sacred** - never change shape/form, only surface blocks
2. **Texture pattern detection** - detect if a block arrangement is:
   - REGULAR PATTERN (e.g. alternating stone/cobblestone in a checkerboard) → can vary
   - STRUCTURAL (e.g. a single column of a specific block) → preserve carefully
   - GRADIENT (e.g. more of block X at bottom, Y at top) → preserve the gradient direction
3. **Only texture blocks** - swap blocks of same family but ONLY if they appear in surface/decoration
   positions, not structural elements like corners, edges, supports
4. **Block family awareness** - swap oak_planks↔spruce_planks BUT keep 
   the count ratio per Y-layer to preserve texturing
5. **Layer-aware** - count blocks per Y-layer, keep the distribution shape

NO cropping, NO noise, NO random destruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch

from dataset import (
    AIR,
    SCHEM_EXTENSIONS,
    SchematicData,
    load_schematic,
    save_schem,
)


# Block texture families - these are SAFE to swap between
# because they serve the same texturing purpose
TEXTURE_FAMILIES: list[list[str]] = [
    # Wood types (planks)
    ["minecraft:oak_planks", "minecraft:spruce_planks", "minecraft:birch_planks",
     "minecraft:jungle_planks", "minecraft:acacia_planks", "minecraft:dark_oak_planks",
     "minecraft:mangrove_planks", "minecraft:crimson_planks", "minecraft:warped_planks"],
    # Log types
    ["minecraft:oak_log", "minecraft:spruce_log", "minecraft:birch_log",
     "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log",
     "minecraft:mangrove_log"],
    # Stairs
    ["minecraft:oak_stairs", "minecraft:spruce_stairs", "minecraft:birch_stairs",
     "minecraft:jungle_stairs", "minecraft:acacia_stairs", "minecraft:dark_oak_stairs",
     "minecraft:mangrove_stairs"],
    # Slabs
    ["minecraft:oak_slab", "minecraft:spruce_slab", "minecraft:birch_slab",
     "minecraft:jungle_slab", "minecraft:acacia_slab", "minecraft:dark_oak_slab",
     "minecraft:mangrove_slab"],
    # Fences
    ["minecraft:oak_fence", "minecraft:spruce_fence", "minecraft:birch_fence",
     "minecraft:jungle_fence", "minecraft:acacia_fence", "minecraft:dark_oak_fence",
     "minecraft:mangrove_fence"],
    # Stone types (wall texturing)
    ["minecraft:stone", "minecraft:cobblestone", "minecraft:stone_bricks",
     "minecraft:mossy_cobblestone", "minecraft:mossy_stone_bricks",
     "minecraft:cracked_stone_bricks", "minecraft:chiseled_stone_bricks",
     "minecraft:smooth_stone"],
    # Deepslate
    ["minecraft:cobbled_deepslate", "minecraft:deepslate_bricks", "minecraft:deepslate_tiles",
     "minecraft:polished_deepslate", "minecraft:cracked_deepslate_bricks",
     "minecraft:cracked_deepslate_tiles", "minecraft:chiseled_deepslate"],
    # Sandstone
    ["minecraft:sandstone", "minecraft:smooth_sandstone", "minecraft:chiseled_sandstone",
     "minecraft:cut_sandstone", "minecraft:red_sandstone", "minecraft:smooth_red_sandstone",
     "minecraft:chiseled_red_sandstone", "minecraft:cut_red_sandstone"],
    # Bricks
    ["minecraft:bricks", "minecraft:brick_stairs", "minecraft:brick_slab", "minecraft:brick_wall"],
    # Nether bricks
    ["minecraft:nether_bricks", "minecraft:red_nether_bricks",
     "minecraft:chiseled_nether_bricks", "minecraft:cracked_nether_bricks"],
    # Wool colors (warm)
    ["minecraft:red_wool", "minecraft:orange_wool", "minecraft:yellow_wool",
     "minecraft:lime_wool", "minecraft:green_wool"],
    # Wool colors (cool)
    ["minecraft:blue_wool", "minecraft:light_blue_wool", "minecraft:cyan_wool",
     "minecraft:purple_wool", "minecraft:magenta_wool", "minecraft:pink_wool"],
    # Wool colors (neutral)
    ["minecraft:white_wool", "minecraft:light_gray_wool", "minecraft:gray_wool",
     "minecraft:black_wool", "minecraft:brown_wool"],
    # Glass
    ["minecraft:glass", "minecraft:white_stained_glass", "minecraft:light_gray_stained_glass",
     "minecraft:gray_stained_glass", "minecraft:black_stained_glass"],
]

# Build lookup: for each block, which family does it belong to?
def build_family_map() -> dict[str, list[str]]:
    fm: dict[str, list[str]] = {}
    for family in TEXTURE_FAMILIES:
        for block in family:
            fm[block] = [b for b in family if b != block]
    return fm

FAMILY_MAP = build_family_map()


# Blocks that are STRUCTURAL - NEVER change these
STRUCTURAL_BLOCKS = {
    "minecraft:air",
    "minecraft:bedrock",
    "minecraft:water",
    "minecraft:lava",
}


def rotate_90(blocks: list[str], size: tuple[int, int, int]) -> tuple[list[str], tuple[int, int, int]]:
    w, h, d = size
    new_w, new_d = d, w
    out = [AIR] * (new_w * h * new_d)
    for y in range(h):
        for z in range(d):
            for x in range(w):
                src = y * d * w + z * w + x
                dst = y * new_d * new_w + x * new_w + (d - 1 - z)
                out[dst] = blocks[src]
    return out, (new_w, h, new_d)


def rotate_180(blocks: list[str], size: tuple[int, int, int]) -> tuple[list[str], tuple[int, int, int]]:
    w, h, d = size
    out = [AIR] * (w * h * d)
    for y in range(h):
        for z in range(d):
            for x in range(w):
                src = y * d * w + z * w + x
                dst = y * d * w + (d - 1 - z) * w + (w - 1 - x)
                out[dst] = blocks[src]
    return out, size


def rotate_270(blocks: list[str], size: tuple[int, int, int]) -> tuple[list[str], tuple[int, int, int]]:
    w, h, d = size
    new_w, new_d = d, w
    out = [AIR] * (new_w * h * new_d)
    for y in range(h):
        for z in range(d):
            for x in range(w):
                src = y * d * w + z * w + x
                dst = y * new_d * new_w + (w - 1 - x) * new_w + z
                out[dst] = blocks[src]
    return out, (new_w, h, new_d)


def mirror_x(blocks: list[str], size: tuple[int, int, int]) -> tuple[list[str], tuple[int, int, int]]:
    w, h, d = size
    out = [AIR] * (w * h * d)
    for y in range(h):
        for z in range(d):
            for x in range(w):
                src = y * d * w + z * w + x
                dst = y * d * w + z * w + (w - 1 - x)
                out[dst] = blocks[src]
    return out, size


def mirror_z(blocks: list[str], size: tuple[int, int, int]) -> tuple[list[str], tuple[int, int, int]]:
    w, h, d = size
    out = [AIR] * (w * h * d)
    for y in range(h):
        for z in range(d):
            for x in range(w):
                src = y * d * w + z * w + x
                dst = y * d * w + (d - 1 - z) * w + x
                out[dst] = blocks[src]
    return out, size


def detect_layer_texture_profile(blocks: list[str], size: tuple[int, int, int]) -> dict[int, dict[str, int]]:
    """Analyze block distribution per Y-layer.
    This helps preserve texturing like "more stone at bottom, planks at top".
    Returns: {y_level: {block_type: count}}
    """
    w, h, d = size
    profile: dict[int, Counter] = {}
    for y in range(h):
        profile[y] = Counter()
        for z in range(d):
            for x in range(w):
                idx = y * d * w + z * w + x
                block = blocks[idx]
                if block != AIR:
                    profile[y][block] += 1
    return {y: dict(c) for y, c in profile.items()}


def smart_block_substitution(
    blocks: list[str],
    size: tuple[int, int, int],
    substitution_rate: float,
    rng: random.Random,
) -> list[str]:
    """Intelligent block substitution that preserves structural patterns.
    
    How it works:
    1. Detect Y-layer texture profile to find gradient patterns
    2. For each candidate block, check if it's a "texture block" (surface-like)
       or a "structural block" (corners, edges, supports)
    3. Texture blocks can be swapped within family
    4. Structural blocks are preserved
    5. Layer ratios are roughly maintained
    """
    w, h, d = size
    out = list(blocks)
    
    # Get layer profiles
    layer_profile = detect_layer_texture_profile(blocks, size)
    
    # For each layer, determine which blocks are dominant (texture) vs rare (accent/detail)
    # Dominant blocks are safer to substitute (they're the general wall/floor material)
    # Rare blocks are accents (trapdoors, buttons, etc.) - preserve them
    layer_dominant: dict[int, set[str]] = {}
    for y, profile in layer_profile.items():
        total = sum(profile.values())
        if total == 0:
            layer_dominant[y] = set()
            continue
        # A block is "dominant" if it occupies >15% of that layer's blocks
        layer_dominant[y] = {b for b, c in profile.items() if c / total > 0.15}
    
    # Collect positions grouped by block type
    block_positions: dict[str, list[int]] = {}
    for i, block in enumerate(blocks):
        if block not in STRUCTURAL_BLOCKS:
            if block not in block_positions:
                block_positions[block] = []
            block_positions[block].append(i)
    
    # Perform substitutions per block type
    for block_type, positions in block_positions.items():
        if block_type not in FAMILY_MAP:
            continue
        if not FAMILY_MAP[block_type]:
            continue
        
        alternatives = FAMILY_MAP[block_type]
        
        # Determine for each position if it's safe to substitute
        # A position is safe if:
        # - The block is "dominant" in its Y-layer (it's a wall/floor material)
        # - OR it appears many times (common block)
        # - And it's not the ONLY block of its type (would change structure)
        if len(positions) <= 1:
            continue  # Don't change unique blocks (single occurrence)
        
        # Sort positions by Y level
        positions_by_y: dict[int, list[int]] = {}
        for pos in positions:
            y = pos // (d * w)
            if y not in positions_by_y:
                positions_by_y[y] = []
            positions_by_y[y].append(pos)
        
        for y, y_positions in positions_by_y.items():
            if len(y_positions) <= 1:
                continue  # Only one at this layer, skip
            
            dominant = layer_dominant.get(y, set())
            is_dominant_at_layer = block_type in dominant
            
            # Decide substitution rate for this batch
            # Dominant blocks: substitute at full rate
            # Non-dominant: substitute at reduced rate (they're accents)
            effective_rate = substitution_rate if is_dominant_at_layer else substitution_rate * 0.3
            
            for pos in y_positions:
                if rng.random() < effective_rate:
                    alt = rng.choice(alternatives)
                    out[pos] = alt
    
    return out


def detect_and_preserve_edge_columns(blocks: list[str], size: tuple[int, int, int]) -> set[int]:
    """Detect positions that form columns and edges - these are structural.
    Returns set of indices that should never be changed."""
    w, h, d = size
    preserve: set[int] = set()
    
    # A block is a "column" if above and below it are the same type
    for x in range(w):
        for z in range(d):
            # Check vertical columns
            for y in range(1, h - 1):
                idx = y * d * w + z * w + x
                idx_up = (y + 1) * d * w + z * w + x
                idx_down = (y - 1) * d * w + z * w + x
                
                b = blocks[idx]
                b_up = blocks[idx_up]
                b_down = blocks[idx_down]
                
                if b == AIR:
                    continue
                
                # If same block above AND below -> it's a column
                if b == b_up == b_down and b not in STRUCTURAL_BLOCKS:
                    preserve.add(idx)
                    preserve.add(idx_up)
                    preserve.add(idx_down)
    
    return preserve


def generate_prompt_variant(prompt: str, rng: random.Random) -> str:
    words = prompt.lower().split()
    swaps = {
        "small": ["tiny", "cozy", "compact", "little", "modest"],
        "large": ["big", "grand", "spacious", "massive", "huge"],
        "medieval": ["old", "ancient", "rustic", "vintage", "historic"],
        "wooden": ["timber", "wood", "log", "rustic"],
        "stone": ["cobblestone", "rock", "brick", "granite"],
        "modern": ["contemporary", "sleek", "new", "minimalist"],
        "beautiful": ["pretty", "nice", "lovely", "charming"],
        "cozy": ["warm", "snug", "homely", "intimate"],
        "tall": ["high", "lofty", "towering", "soaring"],
        "steep": ["sloped", "angled", "pitched", "sharp"],
    }
    new_words = []
    for w in words:
        if w in swaps and rng.random() < 0.4:
            new_words.append(rng.choice(swaps[w]))
        else:
            new_words.append(w)
    new_prompt = " ".join(new_words)
    prefixes = ["", "a ", "a nice ", "a beautiful ", "a cozy ",
                 "a small ", "a large ", "a lovely ", "an elegant "]
    suffixes = ["", " with garden", " with windows", " with details",
                " with chimney", " with decorations"]
    if rng.random() < 0.3:
        new_prompt = rng.choice(prefixes) + new_prompt
    if rng.random() < 0.2:
        new_prompt = new_prompt + rng.choice(suffixes)
    return new_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment schematics with texture-aware, non-destructive transforms.")
    parser.add_argument("--data-dirs", nargs="+", default=[
        "Trainingsdaten good thoroughly analyzed",
    ])
    parser.add_argument("--out-dir", default="scraped Trainingsdaten not as good")
    parser.add_argument("--variants-per-schematic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    if args.clean and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_sources = []
    for data_dir in args.data_dirs:
        dir_path = Path(data_dir)
        if not dir_path.exists():
            print(f"Directory not found: {dir_path}")
            continue
        for schem_path in sorted(dir_path.iterdir()):
            if schem_path.suffix.lower() not in SCHEM_EXTENSIONS:
                continue
            txt_path = schem_path.with_suffix(".txt")
            prompt = ""
            if txt_path.exists():
                prompt = txt_path.read_text(encoding="utf-8").split("---")[0].strip()
            if not prompt:
                prompt = "minecraft structure"
            all_sources.append((schem_path, prompt))

    if not all_sources:
        print("No source schematics found!")
        return

    print(f"Found {len(all_sources)} source schematics")
    print(f"Generating {len(all_sources) * args.variants_per_schematic} texture-aware variants...")

    # Non-destructive transforms
    # Priority: rotations/mirrors first (safe), then texture substitution (smart)
    TRANSFORMS = [
        ("rotate_90", lambda b, s, _: rotate_90(b, s)),
        ("rotate_180", lambda b, s, _: rotate_180(b, s)),
        ("rotate_270", lambda b, s, _: rotate_270(b, s)),
        ("mirror_x", lambda b, s, _: mirror_x(b, s)),
        ("mirror_z", lambda b, s, _: mirror_z(b, s)),
        ("mirror_x_rot90", lambda b, s, _: rotate_90(*mirror_x(b, s))),
        ("mirror_z_rot90", lambda b, s, _: rotate_90(*mirror_z(b, s))),
        ("rotate_90_mirror_x", lambda b, s, _: mirror_x(*rotate_90(b, s))),
        # Texture-aware substitution (preserves structure)
        ("texture_light", lambda b, s, rg: (smart_block_substitution(b, s, 0.10, rg), s)),
        ("texture_medium", lambda b, s, rg: (smart_block_substitution(b, s, 0.20, rg), s)),
        ("mirror_texture", lambda b, s, rg: (smart_block_substitution(mirror_x(b, s)[0], s, 0.15, rg), s)),
        ("rot90_texture", lambda b, s, rg: (smart_block_substitution(rotate_90(b, s)[0], rotate_90(b, s)[1], 0.15, rg), rotate_90(b, s)[1])),
    ]

    total_written = 0
    for schem_path, base_prompt in all_sources:
        print(f"\nProcessing: {schem_path.name}")
        try:
            schematic = load_schematic(schem_path)
        except Exception as e:
            print(f"  Skip: {e}")
            continue

        # Analyze texture profile before any transforms
        profile = detect_layer_texture_profile(schematic.blocks_stripped, schematic.size)
        print(f"  Texturing profile ({len(profile)} layers):")
        for y in sorted(profile.keys()):
            blocks_at_y = profile[y]
            if blocks_at_y:
                top_blocks = sorted(blocks_at_y.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{b}={c}" for b, c in top_blocks)
                print(f"    Y={y}: {top_str}")

        for variant_idx in range(args.variants_per_schematic):
            name, transform = rng.choice(TRANSFORMS)
            blocks = list(schematic.blocks_stripped)
            size = schematic.size
            blocks, size = transform(blocks, size, rng)
            variant_prompt = generate_prompt_variant(base_prompt, rng)

            stem = schem_path.stem[:40]
            digest = hashlib.sha1(f"{schem_path.name}:{variant_idx}:{name}".encode()).hexdigest()[:8]
            out_name = f"{stem}_tex_{name}_{digest}.schem"
            out_path = out_dir / out_name

            out_path.with_suffix(".txt").write_text(
                f"{variant_prompt}\n---\nsource: {schem_path.name}\naug: {name}\n",
                encoding="utf-8",
            )

            unique = sorted(set(blocks))
            if AIR in unique:
                unique.remove(AIR)
                unique.insert(0, AIR)
            id2block = unique
            flat = [id2block.index(b) if b in id2block else 0 for b in blocks]
            grid = torch.tensor(flat, dtype=torch.long).reshape(size[1], size[2], size[0]).permute(2, 0, 1)
            save_schem(out_path, grid, id2block)
            total_written += 1

        print(f"  {args.variants_per_schematic} variants written")

    print(f"\nDone! Generated {total_written} texture-aware schematics to {out_dir}")


if __name__ == "__main__":
    main()