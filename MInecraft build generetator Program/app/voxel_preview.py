"""3D Voxel Preview - renders Minecraft structures with full orbit controls.
Supports interactive rotation (azimuth/elevation), zoom, and multiple view modes.
Uses real Minecraft block textures from Blocktextures/block/ for accurate colors."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Path to Minecraft block textures
TEXTURE_DIR = Path("Blocktextures") / "block"

# Cache for loaded textures: block_name -> (avg_r, avg_g, avg_b, avg_a)
_TEXTURE_CACHE: Dict[str, Tuple[int, int, int, int]] = {}

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


def _load_texture_color(texture_name: str) -> Optional[Tuple[int, int, int, int]]:
    """Load a .png texture and return its average RGBA color.
    Returns None if texture doesn't exist."""
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
        arr = np.array(img)
        # Average color, weighted by alpha
        if arr[:, :, 3].sum() > 0:
            avg_r = int((arr[:, :, 0] * arr[:, :, 3]).sum() / arr[:, :, 3].sum())
            avg_g = int((arr[:, :, 1] * arr[:, :, 3]).sum() / arr[:, :, 3].sum())
            avg_b = int((arr[:, :, 2] * arr[:, :, 3]).sum() / arr[:, :, 3].sum())
        else:
            avg_r, avg_g, avg_b = int(arr[:, :, 0].mean()), int(arr[:, :, 1].mean()), int(arr[:, :, 2].mean())
        # Alpha: average alpha, but at least 200 for opaque blocks
        avg_a = max(200, int(arr[:, :, 3].mean())) if arr[:, :, 3].mean() > 100 else int(arr[:, :, 3].mean())
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


def get_block_color(block_name: str) -> Tuple[int, int, int, int]:
    """Get RGBA color for a Minecraft block, using real textures if available."""
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
    """Paint a quiet Minecraft-style sky behind the model."""
    w, h = size
    horizon = int(h * 0.58)

    for y in range(horizon):
        t = y / max(1, horizon)
        r = int(92 + 48 * t)
        g = int(156 + 42 * t)
        b = int(217 + 24 * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b, 255))

    cloud = (232, 241, 246, 215)
    cloud_y = max(12, h // 12)
    for i, x in enumerate((w // 10, w // 2 + w // 8)):
        y = cloud_y + i * 18
        block = max(10, w // 32)
        draw.rectangle([x, y, x + block * 4, y + block], fill=cloud)
        draw.rectangle([x + block, y - block, x + block * 3, y], fill=cloud)

    sun = max(18, min(w, h) // 12)
    draw.rectangle([w - sun * 2, sun, w - sun, sun * 2], fill=(248, 224, 89, 255))


def _shade(color: Tuple[int, int, int, int], factor: float) -> Tuple[int, int, int, int]:
    return (
        max(0, min(255, int(color[0] * factor))),
        max(0, min(255, int(color[1] * factor))),
        max(0, min(255, int(color[2] * factor))),
        color[3],
    )


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


def render_orbital(
    grid: np.ndarray,
    id_to_block: List[str],
    size: Tuple[int, int] = (420, 380),
    azimuth: float = 45.0,
    elevation: float = 30.0,
    scale: float = 3.5,
) -> Image.Image:
    """Render the voxel grid with full orbital camera controls.
    
    Args:
        grid: 3D numpy array [X, Y, Z] of block IDs
        id_to_block: mapping from ID to block name
        size: output image size (width, height)
        azimuth: camera rotation around Y axis in degrees (0-360)
        elevation: camera angle above horizon in degrees (0-90)
        scale: voxel size in pixels
    
    Returns:
        PIL Image with rendered structure
    """
    GX, GY, GZ = grid.shape
    img = Image.new("RGBA", size, (112, 178, 226, 255))
    draw = ImageDraw.Draw(img)
    _draw_minecraft_backdrop(draw, size)

    # Convert angles to radians
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    # Camera direction vector
    cam_x = math.cos(el_rad) * math.sin(az_rad)
    cam_y = math.sin(el_rad)
    cam_z = math.cos(el_rad) * math.cos(az_rad)

    # Right vector = cross(camera, up)
    up = (0, 1, 0)
    rx = cam_y * up[2] - cam_z * up[1]
    ry = cam_z * up[0] - cam_x * up[2]
    rz = cam_x * up[1] - cam_y * up[0]
    rlen = math.sqrt(rx*rx + ry*ry + rz*rz) or 1
    rx, ry, rz = rx/rlen, ry/rlen, rz/rlen

    # Up vector = cross(right, camera)
    ux = ry * cam_z - rz * cam_y
    uy = rz * cam_x - rx * cam_z
    uz = rx * cam_y - ry * cam_x

    # Center of grid
    cx, cy, cz = GX / 2, GY / 2, GZ / 2

    def project(wx: float, wy: float, wz: float) -> Tuple[float, float]:
        sx = wx * rx + wy * ry + wz * rz
        sy = wx * ux + wy * uy + wz * uz
        return size[0] / 2 + sx * scale, size[1] / 2 - sy * scale

    _draw_rotating_build_plate(draw, project, grid.shape)

    # Collect visible voxels
    voxels = []
    for x in range(GX):
        for y in range(GY):
            for z in range(GZ):
                bid = int(grid[x, y, z])
                unsaved = bid < 0 or bid >= len(id_to_block)
                if unsaved:
                    block_name = "<nicht gespeichert>"
                    color = (220, 38, 38, 255)
                else:
                    block_name = str(id_to_block[bid])
                    if "air" in block_name.lower():
                        continue
                    color = get_block_color(block_name)
                wx, wy, wz = x - cx, y - cy, z - cz
                sd = (wx + 0.5) * cam_x + (wy + 0.5) * cam_y + (wz + 0.5) * cam_z
                voxels.append((sd, wx, wy, wz, color, unsaved))

    if not voxels:
        draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=(60, 80, 120))
        draw.text((10, 10), "No blocks to display", fill=(120, 150, 200))
        return img

    # Sort back-to-front
    voxels.sort(key=lambda v: v[0])

    faces = [
        ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)], 1.18),
        ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)], 0.92),
        ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)], 0.82),
        ((-1, 0, 0), [(0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)], 0.70),
        ((0, 0, -1), [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)], 0.76),
        ((0, -1, 0), [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)], 0.58),
    ]

    marker_font = ImageFont.load_default()

    for sd, wx, wy, wz, color, unsaved in voxels:
        alpha = color[3] if len(color) == 4 else 255
        target = img
        face_draw = draw
        if alpha < 255:
            overlay = Image.new("RGBA", size, (0, 0, 0, 0))
            face_draw = ImageDraw.Draw(overlay)
            target = overlay

        for normal, corners, shade in faces:
            nx, ny, nz = normal
            if nx * cam_x + ny * cam_y + nz * cam_z <= 0:
                continue
            pts = [project(wx + dx, wy + dy, wz + dz) for dx, dy, dz in corners]
            fill = _shade(color, shade)
            outline = _shade(color, 0.48)
            face_draw.polygon(pts, fill=fill, outline=outline)

        if alpha < 255:
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

        if unsaved:
            mx, my = project(wx + 0.5, wy + 0.62, wz + 0.5)
            radius = max(5, int(scale * 0.9))
            draw.ellipse(
                [mx - radius, my - radius, mx + radius, my + radius],
                fill=(255, 236, 153, 245),
                outline=(120, 22, 22, 255),
                width=2,
            )
            draw.text(
                (mx, my - 1),
                "!",
                fill=(120, 22, 22, 255),
                font=marker_font,
                anchor="mm",
            )

    return img


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
) -> Image.Image:
    """Render a preview with the given camera angle.
    
    view can be: "free" (custom az/el), "isometric", "topdown", "side"
    """
    if view == "isometric":
        return render_isometric(grid, id_to_block, size, scale=scale)
    elif view == "topdown":
        return render_topdown(grid, id_to_block, size, scale=scale)
    elif view == "side":
        return render_side(grid, id_to_block, size, scale=scale)
    else:
        return render_orbital(grid, id_to_block, size, azimuth=azimuth, elevation=elevation, scale=scale)