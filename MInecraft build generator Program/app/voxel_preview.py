"""3D Voxel Preview - renders Minecraft structures with full orbit controls.
Supports interactive rotation (azimuth/elevation), zoom, and multiple view modes.
Uses real Minecraft block textures from Blocktextures/block/ for accurate colors.

Rendering is split in two stages so that orbiting stays interactive:

* :class:`VoxelScene` pre-digests a block grid once (positions, colours,
  neighbour occupancy). Building it is the only step that touches names/textures.
* :func:`render_scene` draws such a scene for one camera angle using numpy for
  all projection work and only visible *surface* faces (faces hidden by a solid
  neighbour are skipped). Transparent blocks are alpha blended by PIL during
  drawing (mode "RGBA" on an RGB image), so they need no extra pass.

The rendered image is mode "RGB" (the backdrop is opaque).

Interactive use should go through :class:`PreviewRenderer`, which renders on a
worker thread and drops stale frames instead of queueing them up.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.paths import TEXTURE_DIR

logger = logging.getLogger(__name__)


# Cache for loaded textures: block_name -> (avg_r, avg_g, avg_b, avg_a)
_TEXTURE_CACHE: Dict[str, Tuple[int, int, int, int]] = {}

# Cache for resolved block colours: normalized block name -> RGBA
_BLOCK_COLOR_CACHE: Dict[str, Tuple[int, int, int, int]] = {}

# Names that are invisible. Compared exactly (after stripping namespace/states):
# a substring check for "air" also matched every *_stairs block and made all
# stairs disappear from the preview and the 3D viewer.
AIR_BLOCK_NAMES = {"air", "cave_air", "void_air", "structure_void"}

# Known texture file name overrides for common blocks
_TEXTURE_MAP: Dict[str, str] = {
    "air": None,
    "stone": "stone.png",
    "cobblestone": "cobblestone.png",
    "mossy_cobblestone": "mossy_cobblestone.png",
    "stone_bricks": "stone_bricks.png",
    "stone_brick": "stone_bricks.png",
    "cracked_stone_bricks": "cracked_stone_bricks.png",
    "chiseled_stone_bricks": "chiseled_stone_bricks.png",
    "mossy_stone_bricks": "mossy_stone_bricks.png",
    "deepslate": "deepslate.png",
    "deepslate_bricks": "deepslate_bricks.png",
    "deepslate_tiles": "deepslate_tiles.png",
    "cobbled_deepslate": "cobbled_deepslate.png",
    "polished_deepslate": "polished_deepslate.png",
    "deepslate_brick_stairs": "deepslate_bricks.png",
    "deepslate_tile_stairs": "deepslate_tiles.png",
    "polished_deepslate_stairs": "polished_deepslate.png",
    "cobbled_deepslate_stairs": "cobbled_deepslate.png",
    "dirt": "dirt.png",
    "grass_block": "grass_block_side.png",
    "grass": "grass_block_side.png",
    "oak_planks": "oak_planks.png",
    "spruce_planks": "spruce_planks.png",
    "birch_planks": "birch_planks.png",
    "jungle_planks": "jungle_planks.png",
    "acacia_planks": "acacia_planks.png",
    "dark_oak_planks": "dark_oak_planks.png",
    "mangrove_planks": "mangrove_planks.png",
    "cherry_planks": "cherry_planks.png",
    "bamboo_planks": "bamboo_planks.png",
    "crimson_planks": "crimson_planks.png",
    "warped_planks": "warped_planks.png",
    "oak_log": "oak_log.png",
    "oak_log_top": "oak_log_top.png",
    "spruce_log": "spruce_log.png",
    "birch_log": "birch_log.png",
    "jungle_log": "jungle_log.png",
    "acacia_log": "acacia_log.png",
    "dark_oak_log": "dark_oak_log.png",
    "mangrove_log": "mangrove_log.png",
    "cherry_log": "cherry_log.png",
    "crimson_stem": "crimson_stem.png",
    "warped_stem": "warped_stem.png",
    "oak_wood": "oak_log.png",
    "spruce_wood": "spruce_log.png",
    "oak_stairs": "oak_planks.png",
    "spruce_stairs": "spruce_planks.png",
    "stone_stairs": "stone.png",
    "cobblestone_stairs": "cobblestone.png",
    "stone_brick_stairs": "stone_bricks.png",
    "oak_fence": "oak_planks.png",
    "cobblestone_wall": "cobblestone.png",
    "oak_door": "oak_door_top.png",
    "spruce_door": "spruce_door_top.png",
    "glass": "glass.png",
    "white_stained_glass": "white_stained_glass.png",
    "bricks": "bricks.png",
    "brick_stairs": "bricks.png",
    "nether_bricks": "nether_bricks.png",
    "red_nether_bricks": "red_nether_bricks.png",
    "sandstone": "sandstone_top.png",
    "smooth_sandstone": "sandstone_top.png",
    "red_sandstone": "red_sandstone_top.png",
    "oak_leaves": "oak_leaves.png",
    "spruce_leaves": "spruce_leaves.png",
    "water": "water_still.png",
    "sand": "sand.png",
    "snow": "snow.png",
    "gravel": "gravel.png",
    "bedrock": "bedrock.png",
    "iron_block": "iron_block.png",
    "gold_block": "gold_block.png",
    "diamond_block": "diamond_block.png",
    "emerald_block": "emerald_block.png",
    "redstone_block": "redstone_block.png",
    "coal_block": "coal_block.png",
    "quartz_block": "quartz_block_side.png",
    "pumpkin": "pumpkin_side.png",
    "hay_block": "hay_block_side.png",
    "bookshelf": "bookshelf.png",
    "lantern": "lantern.png",
    "white_wool": "white_wool.png",
    "orange_wool": "orange_wool.png",
    "magenta_wool": "magenta_wool.png",
    "light_blue_wool": "light_blue_wool.png",
    "yellow_wool": "yellow_wool.png",
    "lime_wool": "lime_wool.png",
    "pink_wool": "pink_wool.png",
    "gray_wool": "gray_wool.png",
    "light_gray_wool": "light_gray_wool.png",
    "cyan_wool": "cyan_wool.png",
    "purple_wool": "purple_wool.png",
    "blue_wool": "blue_wool.png",
    "brown_wool": "brown_wool.png",
    "green_wool": "green_wool.png",
    "red_wool": "red_wool.png",
    "black_wool": "black_wool.png",
    "white_carpet": "white_wool.png",
    "orange_carpet": "orange_wool.png",
    "magenta_carpet": "magenta_wool.png",
    "light_blue_carpet": "light_blue_wool.png",
    "yellow_carpet": "yellow_wool.png",
    "lime_carpet": "lime_wool.png",
    "pink_carpet": "pink_wool.png",
    "gray_carpet": "gray_wool.png",
    "light_gray_carpet": "light_gray_wool.png",
    "cyan_carpet": "cyan_wool.png",
    "purple_carpet": "purple_wool.png",
    "blue_carpet": "blue_wool.png",
    "brown_carpet": "brown_wool.png",
    "green_carpet": "green_wool.png",
    "red_carpet": "red_wool.png",
    "black_carpet": "black_wool.png",
    "white_concrete": "white_concrete.png",
    "orange_concrete": "orange_concrete.png",
    "magenta_concrete": "magenta_concrete.png",
    "light_blue_concrete": "light_blue_concrete.png",
    "yellow_concrete": "yellow_concrete.png",
    "lime_concrete": "lime_concrete.png",
    "pink_concrete": "pink_concrete.png",
    "gray_concrete": "gray_concrete.png",
    "light_gray_concrete": "light_gray_concrete.png",
    "cyan_concrete": "cyan_concrete.png",
    "purple_concrete": "purple_concrete.png",
    "blue_concrete": "blue_concrete.png",
    "brown_concrete": "brown_concrete.png",
    "green_concrete": "green_concrete.png",
    "red_concrete": "red_concrete.png",
    "black_concrete": "black_concrete.png",
    "white_concrete_powder": "white_concrete_powder.png",
    "blue_concrete_powder": "blue_concrete_powder.png",
    "white_terracotta": "white_terracotta.png",
    "orange_terracotta": "orange_terracotta.png",
    "red_terracotta": "red_terracotta.png",
    "blue_terracotta": "blue_terracotta.png",
    "cyan_terracotta": "cyan_terracotta.png",
    "purple_terracotta": "purple_terracotta.png",
    "potted_blue_orchid": "flower_pot.png",
    "flower_pot": "flower_pot.png",
    "blue_bed": "blue_bed_head_up.png",
    "white_bed": "white_bed_head_up.png",
    "red_bed": "red_bed_head_up.png",
    "green_bed": "green_bed_head_up.png",
    "black_bed": "black_bed_head_up.png",
    "brown_bed": "brown_bed_head_up.png",
    "cyan_bed": "cyan_bed_head_up.png",
    "gray_bed": "gray_bed_head_up.png",
    "light_blue_bed": "light_blue_bed_head_up.png",
    "lime_bed": "lime_bed_head_up.png",
    "magenta_bed": "magenta_bed_head_up.png",
    "orange_bed": "orange_bed_head_up.png",
    "pink_bed": "pink_bed_head_up.png",
    "purple_bed": "purple_bed_head_up.png",
    "yellow_bed": "yellow_bed_head_up.png",
    "oak_slab": "oak_planks.png",
    "stone_brick_slab": "stone_bricks.png",
    "smooth_stone_slab": "smooth_stone.png",
    "oak_trapdoor": "oak_trapdoor.png",
    "spruce_trapdoor": "spruce_trapdoor.png",
    "oak_sapling": "oak_sapling.png",
    "spruce_sapling": "spruce_sapling.png",
    "netherite_block": "netherite_block.png",
    "ancient_debris": "ancient_debris_side.png",
    "basalt": "basalt_side.png",
    "polished_basalt": "basalt_side.png",
    "blackstone": "blackstone.png",
    "gilded_blackstone": "blackstone.png",
    "calcite": "calcite.png",
    "tuff": "tuff.png",
    "dripstone_block": "dripstone_block.png",
    "andesite": "andesite.png",
    "diorite": "diorite.png",
    "granite": "granite.png",
    "obsidian": "obsidian.png",
    "copper_block": "copper_block.png",
    "exposed_copper": "copper_block.png",
    "weathered_copper": "copper_block.png",
    "waxed_copper_block": "copper_block.png",
    "raw_iron_block": "raw_iron_block.png",
    "raw_copper_block": "raw_copper_block.png",
    "raw_gold_block": "raw_gold_block.png",
    "amethyst_block": "amethyst_block.png",
    "budding_amethyst": "budding_amethyst.png",
    "azalea_leaves": "azalea_leaves.png",
    "flowering_azalea_leaves": "azalea_leaves.png",
    "azalea_plant": "azalea_plant.png",
    "moss_block": "moss_block.png",
    "moss_carpet": "moss_block.png",
    "soul_sand": "soul_sand.png",
    "soul_soil": "soul_soil.png",
    "crimson_nylium": "crimson_nylium.png",
    "warped_nylium": "warped_nylium.png",
    "mushroom_stem": "mushroom_stem.png",
    "brown_mushroom_block": "brown_mushroom_block.png",
    "red_mushroom_block": "red_mushroom_block.png",
    "prismarine": "prismarine.png",
    "prismarine_bricks": "prismarine_bricks.png",
    "dark_prismarine": "dark_prismarine.png",
    "sea_lantern": "sea_lantern.png",
    "purpur_block": "purpur_block.png",
    "purpur_pillar": "purpur_block.png",
    "end_stone": "end_stone.png",
    "end_stone_bricks": "end_stone_bricks.png",
    "glowstone": "glowstone.png",
    "redstone_lamp": "redstone_lamp.png",
    "shroomlight": "shroomlight.png",
    "honeycomb_block": "honeycomb_block.png",
    "honey_block": "honey_block.png",
    "slime_block": "slime_block.png",
    "magma_block": "magma_block.png",
    "sculk": "sculk.png",
    "sculk_catalyst": "sculk_catalyst.png",
    "sculk_sensor": "sculk_sensor_top.png",
    "sculk_shrieker": "sculk_shrieker.png",
    "ochre_froglight": "ochre_froglight_side.png",
    "verdant_froglight": "verdant_froglight_side.png",
    "pearlescent_froglight": "pearlescent_froglight_side.png",
    "reinforced_deepslate": "reinforced_deepslate_side.png",
}


# Runtime tints for greyscale textures (vanilla default biome colours).
# Without them water and leaves - both stored as grey masks - rendered grey.
_TEXTURE_TINTS: Dict[str, Tuple[int, int, int]] = {
    "water_still.png": (63, 118, 228),
    "water_flow.png": (63, 118, 228),
    "water_overlay.png": (63, 118, 228),
    "bubble_column.png": (63, 118, 228),
    "grass_block_top.png": (145, 189, 89),
    "tall_grass_top.png": (145, 189, 89),
    "grass.png": (145, 189, 89),
    "fern.png": (145, 189, 89),
    "vine.png": (72, 181, 24),
    "lily_pad.png": (32, 128, 48),
    "oak_leaves.png": (72, 181, 24),
    "jungle_leaves.png": (72, 181, 24),
    "acacia_leaves.png": (72, 181, 24),
    "dark_oak_leaves.png": (72, 181, 24),
    "mangrove_leaves.png": (72, 181, 24),
    "cherry_leaves.png": (72, 181, 24),
    "azalea_leaves.png": (72, 181, 24),
    "flowering_azalea_leaves.png": (72, 181, 24),
    "birch_leaves.png": (128, 167, 85),
    "spruce_leaves.png": (97, 153, 97),
}


def _load_texture_color(texture_name: str) -> Optional[Tuple[int, int, int, int]]:
    """Load a .png texture and return its average RGBA color.

    Textures Minecraft tints at runtime (water, foliage) are stored as grey
    masks; without the tint they would render as grey blocks instead of blue
    water or green leaves. :data:`_TEXTURE_TINTS` applies the vanilla defaults.

    Returns None if the texture doesn't exist.
    """
    if texture_name is None:
        return None
    cache_key = texture_name
    if cache_key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[cache_key]

    tex_path = TEXTURE_DIR / texture_name
    if not tex_path.exists():
        return None

    try:
        img = Image.open(tex_path).convert("RGBA")
        # int64 casting is required: the alpha weighted sum overflows in uint8
        # (r*255 wraps mod 256), which made every textured block render black.
        arr = np.array(img, dtype=np.int64)
        alpha = arr[:, :, 3]
        alpha_sum = int(alpha.sum())
        if alpha_sum > 0:
            avg_r = int((arr[:, :, 0] * alpha).sum() / alpha_sum)
            avg_g = int((arr[:, :, 1] * alpha).sum() / alpha_sum)
            avg_b = int((arr[:, :, 2] * alpha).sum() / alpha_sum)
        else:
            avg_r, avg_g, avg_b = (
                int(arr[:, :, 0].mean()), int(arr[:, :, 1].mean()), int(arr[:, :, 2].mean()))
        tint = _TEXTURE_TINTS.get(texture_name)
        if tint is not None:
            avg_r = min(255, int(avg_r * tint[0] / 255))
            avg_g = min(255, int(avg_g * tint[1] / 255))
            avg_b = min(255, int(avg_b * tint[2] / 255))
        # Alpha: average alpha, but at least 200 for opaque blocks
        mean_alpha = float(alpha.mean())
        avg_a = max(200, int(mean_alpha)) if mean_alpha > 100 else int(mean_alpha)
        color = (avg_r, avg_g, avg_b, avg_a)
        _TEXTURE_CACHE[cache_key] = color
        return color
    except Exception:
        return None


def _normalize_block_name(block_name: str) -> str:
    b = str(block_name).lower().strip()
    b = b.split("[", 1)[0]
    if ":" in b:
        b = b.split(":", 1)[1]
    return b


def is_air_block(block_name: str) -> bool:
    """True only for air-like blocks (exact match, not a substring test)."""
    return _normalize_block_name(block_name) in AIR_BLOCK_NAMES


def get_block_color(block_name: str) -> Tuple[int, int, int, int]:
    """Get RGBA color for a Minecraft block (memoized per block name)."""
    key = _normalize_block_name(block_name)
    cached = _BLOCK_COLOR_CACHE.get(key)
    if cached is not None:
        return cached
    color = _compute_block_color(block_name)
    _BLOCK_COLOR_CACHE[key] = color
    return color


# ── Palette lookup tables ─────────────────────────────────────────────────────
# A preview renders tens of thousands of voxels, so block names must never be
# touched per voxel. All name/texture work happens once per palette here.

_PALETTE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_PALETTE_CACHE_LIMIT = 6


def _palette_key(names: List[str]) -> str:
    digest = hashlib.md5()
    for name in names:
        digest.update(name.encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _palette_tables(id_to_block) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (colors, air, hides) tables for a block palette.

    colors: uint8 (n, 4) average texture colour per block id
    air:    bool  (n,)   block is invisible (air, cave_air, structure_void, ...)
    hides:  bool  (n,)   block is fully opaque and hides the face behind it
    """
    names = ["" if name is None else str(name) for name in id_to_block]
    key = _palette_key(names)
    cached = _PALETTE_CACHE.get(key)
    if cached is not None:
        return cached

    count = len(names)
    colors = np.zeros((count, 4), dtype=np.uint8)
    air = np.ones((count,), dtype=bool)
    hides = np.zeros((count,), dtype=bool)
    for index, name in enumerate(names):
        if not name or is_air_block(name):
            continue
        color = get_block_color(name)
        colors[index] = color
        # alpha 0 means "not visible" (air copies, void); treat it as empty too
        if color[3] <= 0:
            continue
        air[index] = False
        hides[index] = color[3] >= 255

    if len(_PALETTE_CACHE) >= _PALETTE_CACHE_LIMIT:
        _PALETTE_CACHE.clear()
    _PALETTE_CACHE[key] = (colors, air, hides)
    return colors, air, hides


def _compute_block_color(block_name: str) -> Tuple[int, int, int, int]:
    """Resolve the RGBA color for a Minecraft block, using real textures if available."""
    raw = str(block_name).lower()
    b = _normalize_block_name(raw)

    # Try texture lookup first
    tex_name = _TEXTURE_MAP.get(b)
    if tex_name:
        color = _load_texture_color(tex_name)
        if color is not None:
            return color

    # Fallback: try direct texture file (block_name.png)
    direct_tex = f"{b}.png"
    color = _load_texture_color(direct_tex)
    if color is not None:
        return color

    # Try stripping prefixes like "stripped_" or suffixes like "_stairs"
    if b.startswith("stripped_"):
        stripped = b[9:]
        tex_name = _TEXTURE_MAP.get(stripped)
        if tex_name:
            color = _load_texture_color(tex_name)
            if color is not None:
                return color

    # Try partial matches against texture map
    for key, tex in _TEXTURE_MAP.items():
        if key in b or key.replace("_", "") in b.replace("_", ""):
            color = _load_texture_color(tex)
            if color is not None:
                return color

    # Compiled dictionary fallback for blocks without texture
    FALLBACK_COLORS: Dict[str, Tuple[int, int, int, int]] = {
        "air": (0, 0, 0, 0),
        "void": (0, 0, 0, 0),
        "water": (48, 87, 214, 90),
        "lava": (238, 92, 22, 220),
        "ice": (160, 212, 235, 165),
        "vine": (58, 132, 54, 130),
        "snow": (235, 242, 247, 255),
        "cactus": (47, 126, 55, 255),
        "clay": (157, 166, 176, 255),
        "mud": (79, 67, 55, 255),
        "podzol": (103, 74, 43, 255),
        "mycelium": (112, 98, 114, 255),
        "torch": (236, 180, 61, 255),
        "soul": (74, 148, 166, 255),
        "head": (120, 96, 74, 255),
        "skull": (150, 150, 142, 255),
        "campfire": (121, 70, 43, 255),
        "chain": (89, 91, 94, 255),
        "rail": (113, 96, 72, 255),
        "anvil": (72, 72, 76, 255),
        "cauldron": (64, 66, 69, 255),
        "hopper": (70, 73, 77, 255),
        "furnace": (96, 96, 96, 255),
        "smoker": (83, 73, 62, 255),
        "blast_furnace": (91, 91, 94, 255),
        "crafting_table": (139, 95, 55, 255),
        "cartography_table": (132, 101, 72, 255),
        "smithing_table": (85, 67, 55, 255),
        "loom": (142, 119, 84, 255),
        "barrel": (124, 85, 48, 255),
        "chest": (151, 105, 48, 255),
        "lectern": (125, 83, 50, 255),
        "beehive": (188, 142, 62, 255),
        "bee_nest": (180, 137, 65, 255),
        "coral": (218, 93, 108, 255),
        "item_frame": (154, 103, 57, 190),
        "sugar_cane": (120, 164, 60, 255),
        "lily_pad": (73, 124, 53, 255),
        "sunflower": (209, 191, 61, 255),
        "lilac": (163, 119, 173, 255),
        "peony": (220, 155, 174, 255),
        "rose_bush": (185, 65, 65, 255),
    }

    for token, color in FALLBACK_COLORS.items():
        if token in b:
            return color

    # Stable fallback for unknown blocks
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(b))
    return (
        92 + seed % 80,
        76 + (seed // 7) % 80,
        58 + (seed // 17) % 80,
        255,
    )


def _draw_minecraft_backdrop(draw: ImageDraw.ImageDraw, size: Tuple[int, int]) -> None:
    """Paint a quiet Minecraft-style sky behind the model (opaque RGB colours)."""
    w, h = size
    horizon = int(h * 0.58)

    for y in range(horizon):
        t = y / max(1, horizon)
        r = int(92 + 48 * t)
        g = int(156 + 42 * t)
        b = int(217 + 24 * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    cloud = (232, 241, 246)
    cloud_y = max(12, h // 12)
    for i, x in enumerate((w // 10, w // 2 + w // 8)):
        y = cloud_y + i * 18
        block = max(10, w // 32)
        draw.rectangle([x, y, x + block * 4, y + block], fill=cloud)
        draw.rectangle([x + block, y - block, x + block * 3, y], fill=cloud)

    sun = max(18, min(w, h) // 12)
    draw.rectangle([w - sun * 2, sun, w - sun, sun * 2], fill=(248, 224, 89))


# The sky is identical for every frame of a given size, so it is drawn once.
_BACKDROP_CACHE: Dict[Tuple[int, int], Image.Image] = {}


def _backdrop_image(size: Tuple[int, int]) -> Image.Image:
    """Opaque RGB backdrop for the given size (cached, returned as a copy)."""
    size = (int(size[0]), int(size[1]))
    cached = _BACKDROP_CACHE.get(size)
    if cached is None:
        cached = Image.new("RGB", size, (112, 178, 226))
        _draw_minecraft_backdrop(ImageDraw.Draw(cached), size)
        if len(_BACKDROP_CACHE) >= 4:
            _BACKDROP_CACHE.clear()
        _BACKDROP_CACHE[size] = cached
    return cached.copy()


def _shade_array(colors: np.ndarray, factor: float) -> np.ndarray:
    """Vectorised face shading: (n, 4) uint8 in, (n, 4) uint8 out.

    Alpha is kept, RGB is scaled, clamped and truncated towards zero - the same
    result the previous per-face Python implementation produced.
    """
    scaled = colors.astype(np.float64) * factor
    return np.minimum(255.0, scaled).astype(np.uint8)


def _pack_rgba(rgba: np.ndarray) -> np.ndarray:
    """Pack (n, 4) uint8 RGBA rows into the int32 "ink" values PIL uses."""
    r = rgba[:, 0].astype(np.int64)
    g = rgba[:, 1].astype(np.int64)
    b = rgba[:, 2].astype(np.int64)
    a = rgba[:, 3].astype(np.int64)
    packed = (a << 24) | (b << 16) | (g << 8) | r
    packed = np.where(packed >= 0x80000000, packed - 0x100000000, packed)
    return packed.astype(np.int32)


def _unpack_rgba(ink: int) -> Tuple[int, int, int, int]:
    value = int(ink) & 0xFFFFFFFF
    return (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF, (value >> 24) & 0xFF)


def _low_level_polygon_ok() -> bool:
    """Check whether the low-level PIL polygon call matches the high-level one.

    ``ImageDraw.polygon`` costs two ``_getink`` calls per quad; calling
    ``ImagingDraw.draw_polygon`` with pre-packed inks is ~35 % faster. Inks are
    passed exactly as ``_getink`` would produce them, including alpha blending
    (the renderer draws with mode "RGBA" on an RGB image), and the result is
    compared against the documented API. If a future Pillow changes the
    convention the renderer silently falls back instead of drawing garbage.
    """
    sample = [(1, 1), (6, 2), (5, 6), (2, 5)]
    fills = ((0, 0, 255, 128), (0, 128, 0, 255))
    try:
        reference = Image.new("RGB", (8, 8), (200, 40, 40))
        ImageDraw.Draw(reference, "RGBA").polygon(sample, fill=fills[0], outline=fills[1])
        fast = Image.new("RGB", (8, 8), (200, 40, 40))
        batch = _PolygonBatch(ImageDraw.Draw(fast, "RGBA"), fast=True)
        inks = _pack_rgba(np.array(fills, dtype=np.uint8))
        batch.quad([list(point) for point in sample], int(inks[0]), int(inks[1]))
        return bool(np.array_equal(np.array(reference), np.array(fast)))
    except Exception:  # noqa: BLE001 - any PIL change falls back to the safe path
        return False


class _PolygonBatch:
    """Draws filled+outlined quads through the fastest available PIL path.

    Inks are always packed int32 values; when the low-level call is unavailable
    (or fails) they are unpacked back to tuples for the documented
    :mod:`ImageDraw` API.
    """

    __slots__ = ("_draw", "_fast", "_low")

    def __init__(self, draw: ImageDraw.ImageDraw, fast: bool = True):
        self._draw = draw
        self._fast = bool(fast)
        self._low = getattr(getattr(draw, "draw", None), "draw_polygon", None)

    def quad(self, points: List[List[float]], fill_ink: int, outline_ink: int) -> None:
        if self._fast and self._low is not None:
            try:
                self._low(points, fill_ink, 1)
                self._low(points, outline_ink, 0, 1)
                return
            except (TypeError, ValueError, SystemError):
                self._fast = False
        self._draw.polygon(points, fill=_unpack_rgba(fill_ink),
                           outline=_unpack_rgba(outline_ink))

    def markers(self):
        """High-level draw object (ellipse/text for unsaved blocks)."""
        return self._draw


_LOW_LEVEL_POLYGON_OK = _low_level_polygon_ok()


def _draw_rotating_build_plate(
    draw: ImageDraw.ImageDraw,
    project,
    grid_shape: Tuple[int, int, int],
) -> None:
    """Draw a grass platform in world space so it rotates with the model."""
    gx, gy, gz = grid_shape
    pad = 1.4
    thickness = 0.55
    y_top = -gy / 2 - 0.08
    y_bottom = y_top - thickness
    x0, x1 = -gx / 2 - pad, gx / 2 + pad
    z0, z1 = -gz / 2 - pad, gz / 2 + pad

    top = [
        project(x0, y_top, z0),
        project(x1, y_top, z0),
        project(x1, y_top, z1),
        project(x0, y_top, z1),
    ]
    front = [
        project(x0, y_top, z1),
        project(x1, y_top, z1),
        project(x1, y_bottom, z1),
        project(x0, y_bottom, z1),
    ]
    right = [
        project(x1, y_top, z0),
        project(x1, y_top, z1),
        project(x1, y_bottom, z1),
        project(x1, y_bottom, z0),
    ]
    back = [
        project(x1, y_top, z0),
        project(x0, y_top, z0),
        project(x0, y_bottom, z0),
        project(x1, y_bottom, z0),
    ]
    left = [
        project(x0, y_top, z1),
        project(x0, y_top, z0),
        project(x0, y_bottom, z0),
        project(x0, y_bottom, z1),
    ]

    for face, color in (
        (back, (81, 55, 34, 255)),
        (left, (92, 61, 37, 255)),
        (right, (115, 76, 43, 255)),
        (front, (102, 68, 40, 255)),
    ):
        draw.polygon(face, fill=color, outline=(50, 35, 25, 255))

    draw.polygon(top, fill=(76, 138, 48, 255), outline=(38, 83, 36, 255))

    grid_color = (45, 95, 38, 150)
    steps = max(4, min(10, max(gx, gz)))
    for i in range(1, steps):
        t = i / steps
        x = x0 + (x1 - x0) * t
        draw.line([project(x, y_top, z0), project(x, y_top, z1)], fill=grid_color, width=1)
        z = z0 + (z1 - z0) * t
        draw.line([project(x0, y_top, z), project(x1, y_top, z)], fill=grid_color, width=1)


# ── Scene preparation ─────────────────────────────────────────────────────────

# Face table: normal, the 4 corner offsets of the unit cube, brightness factor.
_FACE_TABLE = (
    ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)), 1.18),
    ((1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)), 0.92),
    ((0, 0, 1), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)), 0.82),
    ((-1, 0, 0), ((0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)), 0.70),
    ((0, 0, -1), ((1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)), 0.76),
    ((0, -1, 0), ((0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)), 0.58),
)
_FACE_DIRS = np.array([face[0] for face in _FACE_TABLE], dtype=np.int64)
_FACE_CORNERS = np.array([face[1] for face in _FACE_TABLE], dtype=np.float64)
_FACE_SHADES = np.array([face[2] for face in _FACE_TABLE], dtype=np.float64)
_UNSAVED_COLOR = (220, 38, 38, 255)


class VoxelScene:
    """Camera independent, pre-digested form of a voxel grid.

    Building a scene resolves every block name/colour once and stores compact
    numpy arrays, which makes re-rendering for a new camera angle cheap.
    """

    __slots__ = ("shape", "count", "coords", "rel", "colors", "unsaved", "occ")

    def __init__(self, shape, coords, rel, colors, unsaved, occ):
        self.shape = shape
        self.coords = coords
        self.rel = rel
        self.colors = colors
        self.unsaved = unsaved
        self.occ = occ
        self.count = int(coords.shape[0])

    @classmethod
    def build(cls, grid, id_to_block) -> "VoxelScene":
        arr = np.asarray(grid)
        if arr.ndim != 3:
            raise ValueError(f"grid must be a 3D array, got shape {arr.shape}")
        shape = tuple(int(v) for v in arr.shape)
        palette_size = len(id_to_block)

        if not np.issubdtype(arr.dtype, np.integer):
            arr = np.rint(arr)                      # generated grids may be float
        ids = np.asarray(arr, dtype=np.int64)
        unsaved_grid = (ids < 0) | (ids >= palette_size)

        if palette_size:
            colors_lut, air_lut, hides_lut = _palette_tables(id_to_block)
            safe = np.where(unsaved_grid, 0, ids)
            air_grid = np.where(unsaved_grid, False, air_lut[safe])
            hides_grid = np.where(unsaved_grid, True, hides_lut[safe])
        else:
            safe = None
            air_grid = np.zeros(shape, dtype=bool)
            hides_grid = np.ones(shape, dtype=bool)

        # Occupancy padded by one block so neighbour lookups need no bounds test.
        padded = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2), dtype=bool)
        padded[1:-1, 1:-1, 1:-1] = hides_grid

        flat_index = np.flatnonzero((~air_grid).reshape(-1))
        coords = np.stack(np.unravel_index(flat_index, shape), axis=1).astype(np.int32)
        rel = coords.astype(np.float32) - np.array(
            [shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0], dtype=np.float32)

        unsaved = unsaved_grid.reshape(-1)[flat_index]
        if palette_size and flat_index.size:
            colors = np.array(colors_lut[safe.reshape(-1)[flat_index]], dtype=np.uint8)
        else:
            colors = np.zeros((flat_index.size, 4), dtype=np.uint8)
        if unsaved.any():
            colors[unsaved] = _UNSAVED_COLOR

        return cls(shape, coords, rel, colors, unsaved, padded)


def _visible_faces(scene: VoxelScene, cam_x: float, cam_y: float, cam_z: float):
    """(face index, voxel indices) for every face that is facing and unoccluded."""
    faces = []
    for index in range(len(_FACE_TABLE)):
        nx, ny, nz = _FACE_DIRS[index]
        if nx * cam_x + ny * cam_y + nz * cam_z <= 0:
            continue                                  # back facing
        # A face is hidden when the neighbour in that direction is opaque.
        hidden = scene.occ[scene.coords[:, 0] + 1 + nx,
                           scene.coords[:, 1] + 1 + ny,
                           scene.coords[:, 2] + 1 + nz]
        selected = np.flatnonzero(~hidden)
        if selected.size:
            faces.append((index, selected))
    return faces


def render_scene(
    scene: VoxelScene,
    size: Tuple[int, int] = (420, 380),
    azimuth: float = 45.0,
    elevation: float = 30.0,
    scale: float = 3.5,
    stats: Optional[Dict[str, int]] = None,
) -> Image.Image:
    """Render a prepared :class:`VoxelScene` for one camera angle."""
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    img = _backdrop_image((width, height))
    # mode "RGBA" on an RGB image makes PIL alpha blend every polygon into the
    # image, so transparent blocks are drawn in depth order like any other face.
    draw = ImageDraw.Draw(img, "RGBA")
    batch = _PolygonBatch(draw, fast=_LOW_LEVEL_POLYGON_OK)

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    cam_x = math.cos(el_rad) * math.sin(az_rad)
    cam_y = math.sin(el_rad)
    cam_z = math.cos(el_rad) * math.cos(az_rad)

    # Right vector = cross(camera, up)
    rx = -cam_z
    ry = 0.0
    rz = cam_x
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz) or 1
    rx, ry, rz = rx / rlen, ry / rlen, rz / rlen

    # Up vector = cross(right, camera)
    ux = ry * cam_z - rz * cam_y
    uy = rz * cam_x - rx * cam_z
    uz = rx * cam_y - ry * cam_x

    def project(wx: float, wy: float, wz: float) -> Tuple[float, float]:
        sx = wx * rx + wy * ry + wz * rz
        sy = wx * ux + wy * uy + wz * uz
        return width / 2 + sx * scale, height / 2 - sy * scale

    _draw_rotating_build_plate(draw, project, scene.shape)

    if scene.count == 0:
        draw.rectangle([0, 0, width - 1, height - 1], outline=(60, 80, 120))
        draw.text((10, 10), "No blocks to display", fill=(120, 150, 200))
        if stats is not None:
            stats["voxels"] = 0
            stats["faces"] = 0
        return img

    rel = scene.rel
    cam = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
    right = np.array([rx, ry, rz], dtype=np.float32)
    up = np.array([ux, uy, uz], dtype=np.float32)
    depth = rel @ cam + 0.5 * (cam_x + cam_y + cam_z)
    base_x = width / 2 + (rel @ right) * scale
    base_y = height / 2 - (rel @ up) * scale

    faces = _visible_faces(scene, cam_x, cam_y, cam_z)
    if not faces:
        return img
    voxel_index = np.concatenate([selected for _, selected in faces])
    face_index = np.concatenate(
        [np.full(selected.size, index, dtype=np.int8) for index, selected in faces])
    if stats is not None:
        stats["voxels"] = scene.count
        stats["faces"] = int(voxel_index.size)

    # Painter's algorithm: draw far faces first so near ones cover them.
    order = np.argsort(depth[voxel_index], kind="stable")
    voxel_index = voxel_index[order]
    face_index = face_index[order]
    del order, faces

    corners = _FACE_CORNERS[face_index]
    pts = np.empty((voxel_index.size, 4, 2), dtype=np.float64)
    pts[:, :, 0] = base_x[voxel_index][:, None] + (corners @ right.astype(np.float64)) * scale
    pts[:, :, 1] = base_y[voxel_index][:, None] - (corners @ up.astype(np.float64)) * scale

    colors = scene.colors[voxel_index]
    rgb = colors[:, :3].astype(np.float64)
    shades = _FACE_SHADES[face_index][:, None]
    fills = np.empty((colors.shape[0], 4), dtype=np.uint8)
    fills[:, :3] = np.minimum(255.0, rgb * shades)
    fills[:, 3] = colors[:, 3]
    outlines = np.empty_like(fills)
    outlines[:, :3] = np.minimum(255.0, rgb * 0.48)
    outlines[:, 3] = colors[:, 3]

    points = pts.tolist()
    fill_inks = _pack_rgba(fills).tolist()
    outline_inks = _pack_rgba(outlines).tolist()
    del pts, colors, rgb, fills, outlines, corners

    # Painter's algorithm in strict depth order. Transparent blocks need alpha
    # blending, which PIL performs per polygon when the draw object is created
    # with mode "RGBA" on an RGB image. The previous renderer instead blended a
    # full size overlay once per transparent voxel (~0.3 ms each), which was by
    # far the biggest cost of a frame.
    for index in range(len(points)):
        batch.quad(points[index], fill_inks[index], outline_inks[index])

    if scene.unsaved.any():
        marker_draw = batch.markers()
        marker_font = ImageFont.load_default()
        radius = max(5, int(scale * 0.9))
        marker_dx = (0.5 * rx + 0.62 * ry + 0.5 * rz) * scale
        marker_dy = (0.5 * ux + 0.62 * uy + 0.5 * uz) * scale
        for index in np.flatnonzero(scene.unsaved).tolist():
            mx = base_x[index] + marker_dx
            my = base_y[index] - marker_dy
            marker_draw.ellipse(
                [mx - radius, my - radius, mx + radius, my + radius],
                fill=(255, 236, 153, 245),
                outline=(120, 22, 22, 255),
                width=2,
            )
            marker_draw.text((mx, my - 1), "!", fill=(120, 22, 22, 255),
                             font=marker_font, anchor="mm")

    return img


def render_orbital(
    grid: np.ndarray,
    id_to_block: List[str],
    size: Tuple[int, int] = (420, 380),
    azimuth: float = 45.0,
    elevation: float = 30.0,
    scale: float = 3.5,
    scene: Optional[VoxelScene] = None,
) -> Image.Image:
    """Render the voxel grid with full orbital camera controls.

    Args:
        grid: 3D numpy array [X, Y, Z] of block IDs
        id_to_block: mapping from ID to block name
        size: output image size (width, height)
        azimuth: camera rotation around Y axis in degrees (0-360)
        elevation: camera angle above horizon in degrees (0-90)
        scale: voxel size in pixels
        scene: optional pre-built :class:`VoxelScene` (rebuilt when omitted)

    Returns:
        PIL Image with rendered structure
    """
    if scene is None:
        scene = VoxelScene.build(grid, id_to_block)
    return render_scene(scene, size=size, azimuth=azimuth, elevation=elevation, scale=scale)


class PreviewRenderer:
    """Renders preview images on a worker thread and drops stale frames.

    While the camera is dragged, requests arrive much faster than frames can be
    rendered. Queueing them would create a growing lag, so only the newest
    request is kept and intermediate camera positions are skipped. Finishing an
    image never touches Tk - the GUI polls :meth:`take_latest` from its own loop.
    """

    def __init__(self, render: Optional[Callable[..., Image.Image]] = None):
        self._render = render or render_scene
        self._cond = threading.Condition()
        self._pending: Optional[Tuple[Any, Dict[str, Any]]] = None
        self._result: Optional[Image.Image] = None
        self._busy = False
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._rendered = 0
        self._dropped = 0

    def request(self, scene: VoxelScene, **params: Any) -> None:
        """Ask for a new frame (non blocking, replaces a pending request)."""
        with self._cond:
            if self._stopped:
                return
            if self._pending is not None:
                self._dropped += 1
            self._pending = (scene, params)
            self._busy = True
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="voxel-preview",
                                                daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                pending = self._pending
                self._pending = None
            scene, params = pending
            try:
                image = self._render(scene, **params)
            except Exception:                      # noqa: BLE001 - keep the GUI alive
                logger.exception("preview render failed")
                image = None
            with self._cond:
                if image is not None:
                    self._result = image
                    self._rendered += 1
                if self._pending is None:
                    self._busy = False
                self._cond.notify_all()

    def take_latest(self) -> Optional[Image.Image]:
        """Return the newest finished frame once, or None if there is no new one."""
        with self._cond:
            image, self._result = self._result, None
            return image

    def is_busy(self) -> bool:
        with self._cond:
            return self._busy

    def wait_idle(self, timeout: float = 30.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: not self._busy, timeout=timeout)

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return {"rendered": self._rendered, "dropped": self._dropped}

    def stop(self, timeout: float = 3.0) -> None:
        with self._cond:
            self._stopped = True
            self._pending = None
            self._busy = False
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)


# View presets: name -> (azimuth, elevation)
VIEW_ANGLES: Dict[str, Tuple[float, float]] = {
    "isometric": (45.0, 30.0),
    "topdown": (0.0, 89.0),
    "side": (0.0, 5.0),
}


def view_angles(view: str, azimuth: float = 45.0,
                elevation: float = 30.0) -> Tuple[float, float]:
    """Camera angles for a view preset; other names keep the given angles."""
    return VIEW_ANGLES.get(view, (azimuth, elevation))


def render_isometric(grid, id_to_block, size=(420, 380), scale=3.5):
    """Convenience: standard isometric view (az=45, el=30)."""
    return render_orbital(grid, id_to_block, size, azimuth=45, elevation=30, scale=scale)


def render_topdown(grid, id_to_block, size=(420, 380), scale=3.5):
    """Convenience: top-down view (az=0, el=90)."""
    return render_orbital(grid, id_to_block, size, azimuth=0, elevation=89, scale=scale)


def render_side(grid, id_to_block, size=(420, 380), scale=3.5):
    """Convenience: front side view (az=0, el=0)."""
    return render_orbital(grid, id_to_block, size, azimuth=0, elevation=5, scale=scale)


def render_preview(
    grid: np.ndarray,
    id_to_block: List[str],
    view: str = "free",
    size: Tuple[int, int] = (420, 380),
    azimuth: float = 45.0,
    elevation: float = 30.0,
    scale: float = 3.5,
    scene: Optional[VoxelScene] = None,
) -> Image.Image:
    """Render a preview with the given camera angle.

    view can be: "free" (custom az/el), "isometric", "topdown", "side"
    scene may be a pre-built :class:`VoxelScene` to skip the preparation step.
    """
    azimuth, elevation = view_angles(view, azimuth, elevation)
    return render_orbital(grid, id_to_block, size=size, azimuth=azimuth,
                          elevation=elevation, scale=scale, scene=scene)
