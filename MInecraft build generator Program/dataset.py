from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from nbtlib import File
from nbtlib import tag as T
from torch.utils.data import Dataset


AIR = "minecraft:air"
SCHEM_EXTENSIONS = {".schem", ".schematic"}

# ─── Simple mode: block name simplification ───

# All Minecraft wood/planks types that should map to oak
_WOOD_TYPES = {
    "oak", "spruce", "birch", "jungle", "acacia", "dark_oak",
    "crimson", "warped", "mangrove", "cherry", "bamboo",
}
# All Minecraft color types for carpet/wool
_COLOR_TYPES = {
    "white", "orange", "magenta", "light_blue", "yellow", "lime",
    "pink", "gray", "light_gray", "cyan", "purple", "blue",
    "brown", "green", "red", "black",
}


def _simplify_base(block_name: str) -> str | None:
    """If block matches a simplification rule, return the simplified block name.
    Returns None if no rule matches."""
    # Stripped/regular logs and woods -> oak
    for stripped in ("stripped_", ""):
        for wood in _WOOD_TYPES:
            if block_name == f"minecraft:{stripped}{wood}_log":
                return "minecraft:oak_log"
            if block_name == f"minecraft:{stripped}{wood}_wood":
                return "minecraft:oak_wood"
        if block_name == f"minecraft:{stripped}bamboo_block":
            return "minecraft:oak_log"

    # Planks -> oak planks
    for wood in _WOOD_TYPES:
        if block_name == f"minecraft:{wood}_planks":
            return "minecraft:oak_planks"

    # Carpets -> red carpet
    for color in _COLOR_TYPES:
        if block_name == f"minecraft:{color}_carpet":
            return "minecraft:red_carpet"

    # Wool -> white wool
    for color in _COLOR_TYPES:
        if block_name == f"minecraft:{color}_wool":
            return "minecraft:white_wool"

    # Beds -> blue bed
    for color in _COLOR_TYPES:
        if block_name == f"minecraft:{color}_bed":
            return "minecraft:blue_bed"

    # Stone brick variants -> stone_bricks
    if block_name.startswith("minecraft:") and ("stone_brick" in block_name or "stone_bricks" in block_name):
        return "minecraft:stone_bricks"

    # Cobblestone variants -> cobblestone
    if block_name.startswith("minecraft:") and "cobblestone" in block_name:
        return "minecraft:cobblestone"
    if block_name.startswith("minecraft:") and "cobbled_deepslate" in block_name:
        return "minecraft:cobblestone"

    # Deepslate stairs -> deepslate_brick_stairs
    if block_name.startswith("minecraft:") and block_name.endswith("_stairs") and "deepslate" in block_name:
        return "minecraft:deepslate_brick_stairs"

    # Potted plants -> potted blue orchid
    if block_name.startswith("minecraft:potted_"):
        return "minecraft:potted_blue_orchid"

    return None


def simplify_block(block_name: str) -> str:
    """Simplify a Minecraft block name to a canonical variant.
    All Logs/Woods -> oak, Planks -> oak planks,
    Carpets -> red carpet, Wool -> white wool,
    Beds -> blue bed, Potted plants -> potted blue orchid,
    Stone brick variants -> stone_bricks,
    Cobblestone variants -> cobblestone,
    Deepslate stairs -> deepslate_brick_stairs.
    Non-matching blocks keep their important block states (facing, axis, etc.)."""
    # First strip states to check base block name against rules
    base = strip_block_state(block_name)
    simplified = _simplify_base(base)
    if simplified is not None:
        return simplified
    # Not simplified: keep important block states
    return keep_important_states(block_name)


# ─── Block state handling ───

BLOCK_STATE_PATTERN = re.compile(r"^([a-z0-9_:-]+)(\[.*\])?$")

# Block States, die für die Architektur wichtig sind und behalten werden sollen
IMPORTANT_BLOCK_STATES = {"facing", "axis", "half", "waterlogged", "open", "type", "rotation", "shape", "part", "hanging", "lit"}

# Block States, die entfernt werden können (nur numerische Werte, Pflanzenwachstum, etc.)
UNIMPORTANT_BLOCK_STATES = {
    "age", "level", "power", "delay", "moisture", "layers",
    "stage", "distance", "persistent", "check_decay", "decayable",
    "player_placed", "triggered", "unstable", "inverted", "natural",
    "extended", "enabled", "signal_fire", "cracked",
    "hatch", "type", "facing_except_up", "east", "west", "north", "south",
    "up", "down", "snowy", "berries", "honey_level",
    "face", "attached", "powered", "locked",
    "candles", "instrument", "note", "bites", "eggs", "pickles",
    "bloom", "crackedness", "tilt",
}

def strip_block_state(block_name: str) -> str:
    """Strip ALL block states. 'minecraft:oak_stairs[facing=north,waterlogged=true]' -> 'minecraft:oak_stairs'
    Used for backward compatibility."""
    m = BLOCK_STATE_PATTERN.match(block_name)
    return m.group(1) if m else block_name

def keep_important_states(block_name: str) -> str:
    """Keep only important block states (facing, axis, half, waterlogged, open, type, rotation).
    'minecraft:oak_stairs[facing=north,waterlogged=true,age=2]' -> 'minecraft:oak_stairs[facing=north,waterlogged=true]'
    'minecraft:oak_log[axis=y]' -> 'minecraft:oak_log[axis=y]'
    'minecraft:dirt' -> 'minecraft:dirt'
    """
    m = BLOCK_STATE_PATTERN.match(block_name)
    if not m:
        return block_name
    base = m.group(1)
    state_str = m.group(2)
    if not state_str:
        return block_name

    # Parse existing states
    inner = state_str[1:-1]  # Remove [ and ]
    pairs = inner.split(",")

    kept = []
    for pair in pairs:
        if "=" in pair:
            key = pair.split("=", 1)[0].strip()
            if key in IMPORTANT_BLOCK_STATES:
                kept.append(pair.strip())

    if not kept:
        return base

    return f"{base}[{','.join(kept)}]"


def _parse_block_state(block_name: str) -> tuple[str, dict[str, str]]:
    """Parse a Minecraft block state into base name and key/value states."""
    m = BLOCK_STATE_PATTERN.match(str(block_name))
    if not m:
        return str(block_name), {}
    base = m.group(1)
    raw_state = m.group(2)
    if not raw_state:
        return base, {}
    states: dict[str, str] = {}
    for pair in raw_state[1:-1].split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        states[key.strip()] = value.strip()
    return base, states


def _format_block_state(base: str, states: dict[str, str]) -> str:
    if not states:
        return base
    # Stable order keeps tokenizer vocab deterministic.
    inner = ",".join(f"{key}={states[key]}" for key in sorted(states))
    return f"{base}[{inner}]"


_HORIZONTAL_DIRECTIONS = ("north", "east", "south", "west")


def _rotate_direction_y(value: str, quarter_turns: int) -> str:
    if value not in _HORIZONTAL_DIRECTIONS:
        return value
    idx = _HORIZONTAL_DIRECTIONS.index(value)
    return _HORIZONTAL_DIRECTIONS[(idx + quarter_turns) % 4]


def rotate_block_state_y(block_name: str, quarter_turns: int) -> str:
    """Rotate orientation-sensitive block states around Y by 90° steps.

    This keeps stairs, doors, trapdoors, signs, chiseled bookshelves/shelves,
    barrels, furnaces, ladders, torches, etc. visually aligned when the voxel
    geometry is rotated. Unknown/non-horizontal states are preserved.
    """
    quarter_turns %= 4
    if quarter_turns == 0 or "[" not in str(block_name):
        return block_name
    base, states = _parse_block_state(block_name)

    if "facing" in states:
        states["facing"] = _rotate_direction_y(states["facing"], quarter_turns)
    if "axis" in states and states["axis"] in {"x", "z"} and quarter_turns % 2 == 1:
        states["axis"] = "z" if states["axis"] == "x" else "x"
    if "rotation" in states:
        try:
            # Standing signs / skulls use 0..15, four units per quarter turn.
            states["rotation"] = str((int(states["rotation"]) + quarter_turns * 4) % 16)
        except ValueError:
            pass

    # Connector-style states are not always kept in the tokenizer, but if they
    # are present (e.g. panes/walls/fences), rotate them consistently.
    connector_values = {d: states.get(d) for d in _HORIZONTAL_DIRECTIONS if d in states}
    if connector_values:
        for d in connector_values:
            states.pop(d, None)
        for direction, value in connector_values.items():
            states[_rotate_direction_y(direction, quarter_turns)] = value

    return _format_block_state(base, states)


def _block_index(x: int, y: int, z: int, size: tuple[int, int, int]) -> int:
    sx, _sy, sz = size
    return y * sz * sx + z * sx + x


def blocks_to_grid(blocks: list[str], size: tuple[int, int, int]) -> list[list[list[str]]]:
    sx, sy, sz = size
    grid = [[[AIR for _ in range(sz)] for _ in range(sy)] for _ in range(sx)]
    for y in range(sy):
        for z in range(sz):
            for x in range(sx):
                idx = _block_index(x, y, z, size)
                if idx < len(blocks):
                    grid[x][y][z] = blocks[idx]
    return grid


def grid_to_blocks(grid: list[list[list[str]]]) -> tuple[list[str], tuple[int, int, int]]:
    sx = len(grid)
    sy = len(grid[0]) if sx else 0
    sz = len(grid[0][0]) if sx and sy else 0
    out: list[str] = []
    for y in range(sy):
        for z in range(sz):
            for x in range(sx):
                out.append(grid[x][y][z])
    return out, (sx, sy, sz)


def trim_block_grid(grid: list[list[list[str]]]) -> list[list[list[str]]]:
    """Trim empty air-only margins around real geometry without cutting blocks."""
    sx = len(grid)
    sy = len(grid[0]) if sx else 0
    sz = len(grid[0][0]) if sx and sy else 0
    coords: list[tuple[int, int, int]] = []
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                if grid[x][y][z] != AIR:
                    coords.append((x, y, z))
    if not coords:
        return [[[AIR]]]
    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
    min_z, max_z = min(c[2] for c in coords), max(c[2] for c in coords)
    return [
        [
            [grid[x][y][z] for z in range(min_z, max_z + 1)]
            for y in range(min_y, max_y + 1)
        ]
        for x in range(min_x, max_x + 1)
    ]


def rotate_block_grid_y(grid: list[list[list[str]]], quarter_turns: int) -> list[list[list[str]]]:
    """Rotate a string block grid around Y in 90° steps, including block states."""
    quarter_turns %= 4
    out = grid
    for _ in range(quarter_turns):
        sx = len(out)
        sy = len(out[0]) if sx else 0
        sz = len(out[0][0]) if sx and sy else 0
        rotated = [[[AIR for _ in range(sx)] for _ in range(sy)] for _ in range(sz)]
        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    # 90° clockwise around Y: x/z dimensions swap.
                    rotated[z][y][sx - 1 - x] = rotate_block_state_y(out[x][y][z], 1)
        out = rotated
    return out


def place_block_grid(
    source: list[list[list[str]]],
    target_size: tuple[int, int, int],
    offset: tuple[int, int, int],
) -> list[str]:
    """Place source into a target-size air grid. Raises if it would be cut."""
    tx, ty, tz = target_size
    sx = len(source)
    sy = len(source[0]) if sx else 0
    sz = len(source[0][0]) if sx and sy else 0
    ox, oy, oz = offset
    if ox < 0 or oy < 0 or oz < 0 or ox + sx > tx or oy + sy > ty or oz + sz > tz:
        raise ValueError("Augmented structure placement would be clipped")
    target = [[[AIR for _ in range(tz)] for _ in range(ty)] for _ in range(tx)]
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                target[ox + x][oy + y][oz + z] = source[x][y][z]
    return grid_to_blocks(target)[0]


@dataclass(frozen=True)
class PlacementTransform:
    rotation: int = 0
    offset: tuple[int, int, int] = (0, 0, 0)


def _axis_positions(max_offset: int, diversity: int) -> list[int]:
    if max_offset <= 0:
        return [0]
    diversity = max(1, int(diversity))
    step = max(1, int(round(max_offset / diversity)))
    values = list(range(0, max_offset + 1, step))
    if values[-1] != max_offset:
        values.append(max_offset)
    center = max_offset // 2
    if center not in values:
        values.append(center)
    return sorted(set(values))


def legal_placement_transforms(
    schematic: SchematicData,
    target_size: tuple[int, int, int],
    diversity: int = 1,
    allow_vertical_movement: bool = False,
    max_variants: int | None = 512,
) -> list[PlacementTransform]:
    """Return legal rotations/translations for a schematic without clipping.

    Smaller buildings naturally receive more possible placements because they
    have more free space in the target grid. `diversity` controls how dense the
    sampled translation grid is.
    """
    base_grid = trim_block_grid(blocks_to_grid(schematic.blocks_important, schematic.size))
    transforms: list[PlacementTransform] = []
    seen: set[tuple[int, tuple[int, int, int], tuple[int, int, int]]] = set()
    for rotation in range(4):
        rotated = rotate_block_grid_y(base_grid, rotation)
        sx = len(rotated)
        sy = len(rotated[0]) if sx else 0
        sz = len(rotated[0][0]) if sx and sy else 0
        tx, ty, tz = target_size
        if sx > tx or sy > ty or sz > tz:
            continue
        max_x, max_y, max_z = tx - sx, ty - sy, tz - sz
        xs = _axis_positions(max_x, diversity)
        ys = _axis_positions(max_y, diversity) if allow_vertical_movement else [0]
        zs = _axis_positions(max_z, diversity)
        for ox in xs:
            for oy in ys:
                for oz in zs:
                    key = (rotation, (ox, oy, oz), (sx, sy, sz))
                    if key in seen:
                        continue
                    seen.add(key)
                    transforms.append(PlacementTransform(rotation=rotation, offset=(ox, oy, oz)))
    if not transforms:
        return []
    if max_variants is not None and len(transforms) > max_variants:
        # Deterministic downsampling keeps endpoints/centers distributed without
        # exploding memory for tiny structures in large grids.
        step = (len(transforms) - 1) / max(1, max_variants - 1)
        transforms = [transforms[round(i * step)] for i in range(max_variants)]
    return transforms


def transform_blocks_to_size(
    schematic: SchematicData,
    target_size: tuple[int, int, int],
    transform: PlacementTransform,
) -> list[str]:
    base_grid = trim_block_grid(blocks_to_grid(schematic.blocks_important, schematic.size))
    rotated = rotate_block_grid_y(base_grid, transform.rotation)
    return place_block_grid(rotated, target_size, transform.offset)


def trim_token_grid(token_grid: torch.Tensor, air_id: int = 0) -> torch.Tensor:
    """Crop a generated [X,Y,Z] token grid to the minimal non-air geometry."""
    grid = token_grid.detach().cpu().long()
    if grid.ndim != 3:
        raise ValueError("token_grid must have shape (X, Y, Z)")
    occupied = grid != int(air_id)
    if not bool(occupied.any()):
        return grid[:1, :1, :1].clone()
    coords = occupied.nonzero(as_tuple=False)
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    return grid[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1].clone()


def center_token_grid(token_grid: torch.Tensor, target_size: tuple[int, int, int], air_id: int = 0) -> torch.Tensor:
    """Center trimmed geometry in a target grid; useful for preview/export UX."""
    trimmed = trim_token_grid(token_grid, air_id=air_id)
    tx, ty, tz = target_size
    sx, sy, sz = map(int, trimmed.shape)
    if sx > tx or sy > ty or sz > tz:
        return trimmed
    out = torch.full((tx, ty, tz), int(air_id), dtype=trimmed.dtype)
    ox = (tx - sx) // 2
    oy = 0
    oz = (tz - sz) // 2
    out[ox:ox + sx, oy:oy + sy, oz:oz + sz] = trimmed
    return out

def blocks_have_states(blocks: list[str]) -> bool:
    return any("[" in b for b in blocks)


# ─── Entity extraction ───

ENTITY_BLOCK_PREFIX = "entity:"
CHEST_PREFIX = "chest:"
SIGN_PREFIX = "sign:"

KNOWN_BLOCK_ENTITIES = {
    "minecraft:chest", "minecraft:trapped_chest", "minecraft:barrel",
    "minecraft:hopper", "minecraft:dispenser", "minecraft:dropper",
    "minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker",
    "minecraft:brewing_stand", "minecraft:beacon",
    "minecraft:spawner", "minecraft:command_block",
    "minecraft:jukebox", "minecraft:note_block",
    "minecraft:daylight_detector",
    "minecraft:structure_block", "minecraft:structure_void",
    "minecraft:enchanting_table", "minecraft:anvil",
    "minecraft:lectern", "minecraft:campfire",
    "minecraft:composter", "minecraft:conduit",
    "minecraft:end_portal_frame", "minecraft:end_gateway",
    "minecraft:end_rod",
    "minecraft:beehive", "minecraft:bee_nest",
    "minecraft:bell", "minecraft:respawn_anchor",
    "minecraft:lodestone",
}

def extract_block_entities(nbt_root: Any) -> List[Dict[str, Any]]:
    """Extract BlockEntity data from schematic NBT.

    Sponge schematics use:
      - 'Id' (capital I) for the block entity type (e.g. 'minecraft:chest')
      - 'Pos' as IntArray [x, y, z] for position
    """
    entities = []
    block_entities_tag = None

    if "BlockEntities" in nbt_root:
        block_entities_tag = nbt_root["BlockEntities"]
    elif "Blocks" in nbt_root and "BlockEntities" in nbt_root["Blocks"]:
        block_entities_tag = nbt_root["Blocks"]["BlockEntities"]
    elif "palette" in nbt_root and "block_entities" in nbt_root:
        block_entities_tag = nbt_root["block_entities"]

    if block_entities_tag is None:
        return entities

    for entity_tag in block_entities_tag:
        try:
            entity: Dict[str, Any] = {}
            for raw_key in entity_tag:
                key = str(raw_key)
                val = entity_tag[raw_key]
                if isinstance(val, T.String):
                    entity[key] = str(val).strip()
                elif isinstance(val, (T.Int, T.Byte, T.Short, T.Long)):
                    entity[key] = int(val)
                elif isinstance(val, (T.Float, T.Double)):
                    entity[key] = float(val)
                elif isinstance(val, T.Compound):
                    entity[key] = _compound_to_dict(val)
                elif isinstance(val, T.ByteArray):
                    entity[key] = [int(b) for b in val]
                elif isinstance(val, T.IntArray):
                    entity[key] = [int(v) for v in val]
                elif isinstance(val, T.List):
                    entity[key] = [str(v) for v in val]
                else:
                    entity[key] = str(val)

            # Normalize 'Id' -> 'id' for consistency
            if "Id" in entity and "id" not in entity:
                entity["id"] = entity["Id"]

            # Normalize 'Pos' (IntArray) -> x, y, z
            if "Pos" in entity:
                pos = entity["Pos"]
                if isinstance(pos, list) and len(pos) >= 3:
                    entity["x"] = int(pos[0])
                    entity["y"] = int(pos[1])
                    entity["z"] = int(pos[2])

            entities.append(entity)
        except Exception:
            pass
    return entities

def _compound_to_dict(comp: T.Compound) -> Dict[str, Any]:
    result = {}
    for raw_key in comp:
        key = str(raw_key)
        val = comp[raw_key]
        if isinstance(val, T.String):
            result[key] = str(val)
        elif isinstance(val, (T.Int, T.Byte, T.Short, T.Long)):
            result[key] = int(val)
        elif isinstance(val, (T.Float, T.Double)):
            result[key] = float(val)
        elif isinstance(val, T.Compound):
            result[key] = _compound_to_dict(val)
        elif isinstance(val, T.List):
            result[key] = [str(v) for v in val[:5]]
        else:
            result[key] = str(val)
    return result

def extract_entities(nbt_root: Any) -> List[Dict[str, Any]]:
    """Extract Entities (not BlockEntities) from schematic NBT."""
    ents = []
    entities_tag = None
    if "Entities" in nbt_root:
        entities_tag = nbt_root["Entities"]
    elif "entities" in nbt_root:
        entities_tag = nbt_root["entities"]
    if entities_tag is None:
        return ents
    for entity_tag in entities_tag:
        try:
            entity = {}
            for key in entity_tag:
                val = entity_tag[key]
                if isinstance(val, T.String):
                    entity[str(key)] = str(val)
                elif isinstance(val, (T.Int, T.Byte)):
                    entity[str(key)] = int(val)
                elif isinstance(val, (T.Float, T.Double)):
                    entity[str(key)] = float(val)
                elif isinstance(val, T.Compound):
                    entity[str(key)] = _compound_to_dict(val)
                elif isinstance(val, T.List):
                    entity[str(key)] = [str(v) for v in val[:5]]
                else:
                    entity[str(key)] = str(val)
            ents.append(entity)
        except Exception:
            pass
    return ents


@dataclass(frozen=True)
class SchematicData:
    blocks: list[str]
    blocks_stripped: list[str]          # Alle States entfernt (für alte Kompatibilität)
    blocks_important: list[str] = field(default_factory=list)
    size: tuple[int, int, int] = (0, 0, 0)
    block_entities: list[dict] = field(default_factory=list)  # BlockEntity data
    entities: list[dict] = field(default_factory=list)  # Entity data

    def __post_init__(self) -> None:
        if not self.blocks_important:
            object.__setattr__(self, "blocks_important", [keep_important_states(b) for b in self.blocks])


def _unwrap_root(nbt_file: File):
    # Direct Schematic key
    if "Schematic" in nbt_file and len(nbt_file) == 1:
        return nbt_file["Schematic"]
    # Handle wrapped format: {'': Compound({'Schematic': {...}})}
    if "" in nbt_file:
        inner = nbt_file[""]
        if "Schematic" in inner:
            return inner["Schematic"]
    # Handle format where Schematic is one of multiple keys
    if "Schematic" in nbt_file:
        return nbt_file["Schematic"]
    return nbt_file


def _read_varints(raw: Iterable[int], expected: int) -> list[int]:
    values: list[int] = []
    value = 0
    shift = 0
    for byte in raw:
        byte = int(byte) & 0xFF
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 35:
                raise ValueError("Invalid varint in schematic block data")
        else:
            values.append(value)
            value = 0
            shift = 0
            if len(values) == expected:
                break
    if len(values) != expected:
        raise ValueError(f"Expected {expected} block ids, decoded {len(values)}")
    return values


def _write_varints(values: Iterable[int]) -> list[int]:
    encoded: list[int] = []
    for value in values:
        value = int(value)
        while True:
            part = value & 0x7F
            value >>= 7
            if value:
                part |= 0x80
            encoded.append(part if part < 128 else part - 256)
            if not value:
                break
    return encoded


def load_schematic(path: str | Path) -> SchematicData:
    path = Path(path)
    root = _unwrap_root(File.load(path, gzipped=True))

    block_entities = extract_block_entities(root)
    entities = extract_entities(root)

    # Read palette
    palette_raw = None
    block_data_raw = None

    if "Blocks" in root and "Palette" in root["Blocks"]:
        # Sponge v3 with nested Blocks compound
        palette_raw = {int(v): str(k) for k, v in root["Blocks"]["Palette"].items()}
        block_data_raw = root["Blocks"]["Data"]
        width = int(root["Width"])
        height = int(root["Height"])
        length = int(root["Length"])
    elif "Palette" in root and "BlockData" in root:
        # Sponge v2 with flat palette
        palette_raw = {int(v): str(k) for k, v in root["Palette"].items()}
        block_data_raw = root["BlockData"]
        width = int(root["Width"])
        height = int(root["Height"])
        length = int(root["Length"])
    elif "Blocks" in root:
        # Legacy format
        width = int(root["Width"])
        height = int(root["Height"])
        length = int(root["Length"])
        blocks_raw = [f"legacy:{int(v) & 0xFF}" for v in root["Blocks"]]
        blocks_stripped = [strip_block_state(b) for b in blocks_raw]
        blocks_important = [keep_important_states(b) for b in blocks_raw]
        return SchematicData(blocks_raw, blocks_stripped, blocks_important, (width, height, length),
                           block_entities, entities)
    else:
        raise ValueError(f"Unsupported schematic format: {path}")

    ids = _read_varints(block_data_raw, width * height * length)
    blocks_raw = [palette_raw[i] for i in ids]
    blocks_stripped = [strip_block_state(b) for b in blocks_raw]
    blocks_important = [keep_important_states(b) for b in blocks_raw]
    return SchematicData(blocks_raw, blocks_stripped, blocks_important, (width, height, length),
                       block_entities, entities)


def _swap_east_west_north_south(block_name: str) -> str:
    """Swap east<->west and north<->south in block state values within a block name."""
    if "[" not in block_name:
        return block_name
    base, states_str = block_name.split("[", 1)
    states_str = states_str.rstrip("]")
    pairs = states_str.split(",")
    swapped_pairs = []
    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            if key == "facing" or key in ("east", "west", "north", "south"):
                if value == "east":
                    value = "west"
                elif value == "west":
                    value = "east"
                elif value == "north":
                    value = "south"
                elif value == "south":
                    value = "north"
            swapped_pairs.append(f"{key}={value}")
        else:
            swapped_pairs.append(pair)
    return f"{base}[{','.join(swapped_pairs)}]"


def save_schem(path: str | Path, token_grid: torch.Tensor, id_to_block: list[str],
               block_entities: list[dict] | None = None,
               swap_directions: bool = False) -> int:
    """Save a schematic using nbtlib with correct Sponge v3 format.
    Manually gzips to work around nbtlib v1.12.1 gzip bug.
    Returns the number of blocks that were converted to air due to out-of-range IDs."""
    import gzip as gzip_mod
    import io

    path = Path(path)
    grid = token_grid.detach().cpu().long()
    if grid.ndim != 3:
        raise ValueError("token_grid must have shape (X, Y, Z)")

    width, height, length = map(int, grid.shape)
    vocab_size = len(id_to_block)

    # Warn if any tokens are out of range
    out_of_range_mask = (grid < 0) | (grid >= vocab_size)
    lost_blocks = int(out_of_range_mask.sum().item())
    if lost_blocks > 0:
        import warnings
        warnings.warn(
            f"save_schem: {lost_blocks} block(s) have IDs outside the vocabulary "
            f"(0..{vocab_size-1}). These will be saved as air. "
            f"Max token ID in grid: {int(grid.max().item())}, "
            f"vocab size: {vocab_size}. "
            f"This usually means the model was trained with a different block vocabulary "
            f"than the one currently loaded."
        )

    # Hole die IDs aller tatsächlich vorkommenden Blöcke
    used_ids: set[int] = set()
    for token in grid.flatten():
        tid = int(token)
        if 0 <= tid < vocab_size:
            used_ids.add(tid)
    # Palette in der EXAKTEN Reihenfolge von id_to_block, aber nur vorhandene Blöcke
    palette_blocks: list[str] = []
    for bid in range(vocab_size):
        if bid in used_ids:
            palette_blocks.append(id_to_block[bid])
    # Fallback: air immer in der Palette
    if AIR not in palette_blocks:
        palette_blocks.insert(0, AIR)

    # Richtungen tauschen (east<->west, north<->south) wenn gewünscht
    if swap_directions:
        palette_blocks = [_swap_east_west_north_south(b) for b in palette_blocks]
        # Air in der Palette belassen (swap sollte air nicht ändern, aber sicherheitshalber)
        if AIR not in palette_blocks:
            palette_blocks.insert(0, AIR)

    # Konvertiere Grid in lineare ID-Liste (Y, Z, X Reihenfolge wie Sponge v3)
    block_ids: list[int] = []
    for y in range(height):
        for z in range(length):
            for x in range(width):
                tid = int(grid[x, y, z])
                if 0 <= tid < vocab_size:
                    block_name = id_to_block[tid]
                else:
                    block_name = AIR
                if swap_directions:
                    block_name = _swap_east_west_north_south(block_name)
                if block_name in palette_blocks:
                    block_ids.append(palette_blocks.index(block_name))
                else:
                    block_ids.append(0)

    # Baue das Schematic mit nbtlib
    palette_compound = T.Compound({})
    for block in palette_blocks:
        palette_compound[block] = T.Int(palette_blocks.index(block))

    import numpy as np
    # Data als ByteArray mit varint-kodierten Werten
    data_bytes = np.array([(b & 0xFF) if b < 0 else b for b in _write_varints(block_ids)], dtype=np.uint8)

    # BlockEntities - wenn vorhanden, korrekt einbetten
    be_list = T.List[T.Compound]([])
    if block_entities:
        for be in block_entities:
            be_compound = T.Compound({})
            for k, v in be.items():
                if isinstance(v, str):
                    be_compound[k] = T.String(v)
                elif isinstance(v, int):
                    be_compound[k] = T.Int(v)
                elif isinstance(v, (list, tuple)):
                    be_compound[k] = T.IntArray([int(x) for x in v])
                elif isinstance(v, dict):
                    inner = T.Compound({})
                    for ik, iv in v.items():
                        if isinstance(iv, str):
                            inner[ik] = T.String(iv)
                        elif isinstance(iv, (int, float)):
                            inner[ik] = T.Double(iv)
                    be_compound[k] = inner
            be_list.append(be_compound)

    blocks_compound = T.Compound({
        "Palette": palette_compound,
        "Data": T.ByteArray(data_bytes),
        "BlockEntities": be_list,
    })

    metadata_compound = T.Compound({})

    root = T.Compound({
        "Version": T.Int(3),
        "DataVersion": T.Int(3953),
        "Width": T.Short(width),
        "Height": T.Short(height),
        "Length": T.Short(length),
        "Offset": T.IntArray([0, 0, 0]),
        "Metadata": metadata_compound,
        "Blocks": blocks_compound,
    })

    # nbtlib v1.12.1 hat Bug mit gzipped=True -> selbst gzippen
    import tempfile
    wrapped = T.Compound({"Schematic": root})
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        File({"": wrapped}).save(str(tmp), gzipped=False)
        with open(tmp, 'rb') as fin:
            raw_nbt = fin.read()
        with gzip_mod.open(path, 'wb') as fout:
            fout.write(raw_nbt)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except PermissionError:
                pass

    return lost_blocks


def fit_to_size(blocks: list[str], size: tuple[int, int, int], target_size: tuple[int, int, int]) -> list[str]:
    out = [AIR] * (target_size[0] * target_size[1] * target_size[2])
    tx, ty, tz = target_size
    sx, sy, sz = size
    max_x, max_y, max_z = min(sx, tx), min(sy, ty), min(sz, tz)
    idx = 0
    for y in range(ty):
        for z in range(tz):
            for x in range(tx):
                if x < max_x and y < max_y and z < max_z:
                    out[idx] = blocks[y * sz * sx + z * sx + x]
                idx += 1
    return out


class VoxelTokenizer:
    """Tokenizes block names to IDs. Automatically keeps important block states
    (facing, axis, half, waterlogged, open) while stripping unimportant ones.
    This allows the model to learn orientation-critical details like stair directions."""

    def __init__(self, block_to_id: dict[str, int] | None = None):
        self.block_to_id = block_to_id or {AIR: 0}
        if AIR not in self.block_to_id:
            self.block_to_id[AIR] = 0
        self.id_to_block = [AIR] * len(self.block_to_id)
        for block, idx in self.block_to_id.items():
            if idx >= len(self.id_to_block):
                self.id_to_block.extend([AIR] * (idx + 1 - len(self.id_to_block)))
            self.id_to_block[idx] = block

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_block)

    def fit(self, schematics: Iterable[SchematicData], target_size: tuple[int, int, int],
            simple_mode: bool = False) -> None:
        """Build vocabulary from ALL block names across ALL schematics,
        keeping only important block states.

        If simple_mode=True, applies simplify_block() to reduce block variants
        (e.g. all wood types -> oak, all wool colors -> white).
        """
        blocks = {AIR}
        self.block_entity_types: set[str] = set()

        for schematic in schematics:
            # Include all rotated states so stairs, logs, shelves/signs etc. do
            # not disappear when augmented rotations create new orientations.
            base_grid = trim_block_grid(blocks_to_grid(schematic.blocks_important, schematic.size))
            for rotation in range(4):
                rotated = rotate_block_grid_y(base_grid, rotation)
                rotated_blocks, _ = grid_to_blocks(rotated)
                if simple_mode:
                    blocks.update(simplify_block(block) for block in rotated_blocks)
                else:
                    blocks.update(keep_important_states(block) for block in rotated_blocks)

            # Track which blocks have BlockEntities
            for be in schematic.block_entities:
                block_id = be.get("id", "")
                base = strip_block_state(block_id)
                self.block_entity_types.add(base)

        self.block_to_id = {AIR: 0}
        for block in sorted(blocks - {AIR}):
            self.block_to_id[block] = len(self.block_to_id)
        self.id_to_block = [None] * len(self.block_to_id)
        for block, idx in self.block_to_id.items():
            self.id_to_block[idx] = block

    def encode_blocks(self, blocks: list[str], size: tuple[int, int, int],
                      target_size: tuple[int, int, int],
                      simple_mode: bool = False) -> torch.Tensor:
        fitted = fit_to_size(blocks, size, target_size)
        if simple_mode:
            cleaned = [simplify_block(b) for b in fitted]
        else:
            cleaned = [keep_important_states(b) for b in fitted]
        ids = [self.block_to_id.get(block, self.block_to_id[AIR]) for block in cleaned]
        return torch.tensor(ids, dtype=torch.long).view(
            target_size[1], target_size[2], target_size[0]).permute(2, 0, 1)

    def safe_encode_blocks(self, blocks: list[str], size: tuple[int, int, int],
                           target_size: tuple[int, int, int],
                           simple_mode: bool = False) -> torch.Tensor:
        """Like encode_blocks, but falls back to AIR (0) for any unknown block."""
        fitted = fit_to_size(blocks, size, target_size)
        if simple_mode:
            cleaned = [simplify_block(b) for b in fitted]
        else:
            cleaned = [keep_important_states(b) for b in fitted]
        air_id = self.block_to_id.get(AIR, 0)
        ids = []
        for block in cleaned:
            bid = self.block_to_id.get(block)
            if bid is None or bid >= len(self.id_to_block):
                bid = air_id
            ids.append(bid)
        return torch.tensor(ids, dtype=torch.long).view(
            target_size[1], target_size[2], target_size[0]).permute(2, 0, 1)

    def safe_encode_prepared_blocks(
        self,
        blocks: list[str],
        target_size: tuple[int, int, int],
        simple_mode: bool = False,
    ) -> torch.Tensor:
        """Encode an already target-sized flat block list."""
        expected = target_size[0] * target_size[1] * target_size[2]
        if len(blocks) != expected:
            raise ValueError(f"Expected {expected} blocks, got {len(blocks)}")
        if simple_mode:
            cleaned = [simplify_block(b) for b in blocks]
        else:
            cleaned = [keep_important_states(b) for b in blocks]
        air_id = self.block_to_id.get(AIR, 0)
        ids = []
        for block in cleaned:
            bid = self.block_to_id.get(block)
            if bid is None or bid >= len(self.id_to_block):
                bid = air_id
            ids.append(bid)
        return torch.tensor(ids, dtype=torch.long).view(
            target_size[1], target_size[2], target_size[0]).permute(2, 0, 1)

    def get_block_entity_mask(self, grid: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask where blocks that could have entities are located."""
        if not hasattr(self, 'block_entity_types'):
            return torch.zeros_like(grid, dtype=torch.bool)
        mask = torch.zeros_like(grid, dtype=torch.bool)
        for be_type in self.block_entity_types:
            bid = self.block_to_id.get(be_type)
            if bid is not None:
                mask |= (grid == bid)
        return mask

    def save(self, path: str | Path) -> None:
        data = {
            "block_to_id": self.block_to_id,
            "block_entity_types": list(getattr(self, 'block_entity_types', set())),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VoxelTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Support both wrapped format ({"block_to_id": {...}}) and flat format ({block_name: id})
        if "block_to_id" in data:
            mapping = data["block_to_id"]
            block_entity_types = set(data.get("block_entity_types", []))
        else:
            mapping = data
            block_entity_types = set()
        tokenizer = cls(mapping)
        tokenizer.block_entity_types = block_entity_types
        return tokenizer


class PromptTokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None, max_length: int = 64):
        self.token_to_id = token_to_id or {"<pad>": 0, "<unk>": 1}
        self.max_length = max_length

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_:-]+", text.lower())

    def fit(self, prompts: Iterable[str]) -> None:
        vocab = set()
        for prompt in prompts:
            vocab.update(self.split(prompt))
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        for token in sorted(vocab):
            self.token_to_id[token] = len(self.token_to_id)

    def encode(self, prompt: str) -> torch.Tensor:
        ids = [self.token_to_id.get(tok, 1) for tok in self.split(prompt)[:self.max_length]]
        ids.extend([0] * (self.max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def save(self, path: str | Path) -> None:
        payload = {"token_to_id": self.token_to_id, "max_length": self.max_length}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PromptTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["token_to_id"], int(payload["max_length"]))


def discover_examples(data_dir: str | Path) -> list[tuple[Path, str]]:
    """Discover all (schematic_path, prompt) pairs in a directory.

    For .txt files with multiple prompts separated by '---', each prompt
    generates its own training example (all mapping to the same schematic).
    Lines after '---' that look like metadata (starting with 'source:', 'aug:',
    etc.) are filtered out and not treated as prompts.
    """
    data_dir = Path(data_dir)
    prompt_map_path = data_dir / "prompts.json"
    prompt_map = {}
    if prompt_map_path.exists():
        prompt_map = json.loads(prompt_map_path.read_text(encoding="utf-8"))

    META_PREFIXES = ("source:", "aug:", "author:", "name:", "url:", "id:")

    examples: list[tuple[Path, str]] = []
    for schem_path in sorted(data_dir.iterdir()):
        if schem_path.suffix.lower() not in SCHEM_EXTENSIONS:
            continue
        prompt = prompt_map.get(schem_path.name) or prompt_map.get(schem_path.stem)
        if prompt:
            examples.append((schem_path, prompt))
            continue

        txt_path = schem_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        raw = txt_path.read_text(encoding="utf-8")
        # Split on '---' to get all prompt candidates
        parts = [p.strip() for p in raw.split("---")]
        for part in parts:
            if not part:
                continue
            # Skip metadata lines (source: ..., aug: ...)
            if any(part.lower().startswith(prefix) for prefix in META_PREFIXES):
                continue
            examples.append((schem_path, part))

    return examples


def discover_examples_many(data_dirs: Iterable[str | Path]) -> list[tuple[Path, str, Path]]:
    examples: list[tuple[Path, str, Path]] = []
    for data_dir in data_dirs:
        root = Path(data_dir)
        for path, prompt in discover_examples(root):
            examples.append((path, prompt, root))
    return examples


@dataclass(frozen=True)
class AugmentedSchematicExample:
    path: Path
    prompt: str
    source_dir: Path
    schematic_index: int
    category_index: int
    transform: PlacementTransform


class MultiSourceSchematicDataset(Dataset):
    def __init__(self, data_dirs: Iterable[str | Path],
                 target_size: tuple[int, int, int] = (16, 16, 16),
                 prompt_tokenizer: PromptTokenizer | None = None,
                 voxel_tokenizer: VoxelTokenizer | None = None,
                 max_voxels: int | None = None,
                 augmentation_diversity: int = 1,
                 allow_vertical_movement: bool = False,
                 max_augmented_variants: int | None = 512,
                 simple_mode: bool = False,
                 structure_block_weight: float = 100.0,
                 air_weight_factor: float = 75.0,
                 hidden_states: torch.Tensor | None = None,
                 attention_masks: torch.Tensor | None = None,
                 hidden_index: list[int] | None = None):
        self.structure_block_weight = max(1.0, float(structure_block_weight))
        self.air_weight_factor = max(1.0, float(air_weight_factor))
        self.hidden_states = hidden_states
        self.attention_masks = attention_masks
        self.hidden_index = hidden_index  # maps each example to a row index in hidden_states
        raw_examples = discover_examples_many(data_dirs)
        if not raw_examples:
            raise ValueError("No .schem/.schematic examples with prompts found")

        self.examples: list[AugmentedSchematicExample] = []
        self.schematics: list[SchematicData] = []
        self._schematic_by_path: dict[Path, int] = {}
        self._transforms_by_path: dict[Path, list[PlacementTransform]] = {}
        self._category_by_path: dict[Path, int] = {}
        self.skipped_examples: list[Path] = []

        for path, prompt, source_dir in raw_examples:
            resolved = path.resolve()
            if resolved in self._schematic_by_path:
                schematic_index = self._schematic_by_path[resolved]
                schematic = self.schematics[schematic_index]
            else:
                schematic = load_schematic(path)
                schematic_index = len(self.schematics)
                self.schematics.append(schematic)
                self._schematic_by_path[resolved] = schematic_index

            voxels = schematic.size[0] * schematic.size[1] * schematic.size[2]
            if max_voxels is not None and voxels > max_voxels:
                self.skipped_examples.append(path)
                continue

            if resolved not in self._transforms_by_path:
                if augmentation_diversity <= 0:
                    base_grid = trim_block_grid(blocks_to_grid(schematic.blocks_important, schematic.size))
                    sx = len(base_grid)
                    sy = len(base_grid[0]) if sx else 0
                    sz = len(base_grid[0][0]) if sx and sy else 0
                    tx, ty, tz = target_size
                    if sx <= tx and sy <= ty and sz <= tz:
                        transform = PlacementTransform(
                            rotation=0,
                            offset=((tx - sx) // 2, 0, (tz - sz) // 2),
                        )
                        transforms = [transform]
                    else:
                        transforms = []
                else:
                    transforms = legal_placement_transforms(
                        schematic,
                        target_size,
                        diversity=augmentation_diversity,
                        allow_vertical_movement=allow_vertical_movement,
                        max_variants=max_augmented_variants,
                    )
                self._transforms_by_path[resolved] = transforms

            transforms = self._transforms_by_path[resolved]
            if not transforms:
                self.skipped_examples.append(path)
                continue

            if resolved not in self._category_by_path:
                self._category_by_path[resolved] = len(self._category_by_path)
            category_index = self._category_by_path[resolved]

            for transform in transforms:
                self.examples.append(AugmentedSchematicExample(
                    path=path,
                    prompt=prompt,
                    source_dir=source_dir,
                    schematic_index=schematic_index,
                    category_index=category_index,
                    transform=transform,
                ))

        if not self.examples:
            raise ValueError("All schematic examples were filtered out")

        self.target_size = target_size
        self.augmentation_diversity = int(augmentation_diversity)
        self.allow_vertical_movement = bool(allow_vertical_movement)
        self.simple_mode = bool(simple_mode)
        self.prompt_tokenizer = prompt_tokenizer or PromptTokenizer()
        if prompt_tokenizer is None:
            self.prompt_tokenizer.fit(example.prompt for example in self.examples)

        self.voxel_tokenizer = voxel_tokenizer or VoxelTokenizer()
        if voxel_tokenizer is None:
            self.voxel_tokenizer.fit(self.schematics, target_size, simple_mode=simple_mode)

        category_counts: dict[int, int] = {}
        for example in self.examples:
            category_counts[example.category_index] = category_counts.get(example.category_index, 0) + 1
        scale = len(self.examples) / max(1, len(category_counts))
        self._sample_weights = [
            scale / category_counts[example.category_index]
            for example in self.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        schematic = self.schematics[example.schematic_index]
        transformed_blocks = transform_blocks_to_size(
            schematic,
            self.target_size,
            example.transform,
        )
        voxel_ids = self.voxel_tokenizer.safe_encode_prepared_blocks(
            transformed_blocks, self.target_size, simple_mode=self.simple_mode)
        
        # ── Per-structure adaptive weight ──
        # Berechne pro-token Gewichte:
        #   block_weight = structure_block_weight / (non_air_count + eps)
        #   air_weight    = air_weight_factor / (air_count + eps)
        # So haben alle non-air tokens zusammen = structure_block_weight,
        # und alle air tokens zusammen = air_weight_factor.
        total_tokens = voxel_ids.numel()
        non_air_count = int((voxel_ids != 0).sum().item())
        air_count = total_tokens - non_air_count
        
        eps = 0.001
        self._per_block_weight_cache = self.structure_block_weight / (non_air_count + eps)
        self._per_air_weight_cache = self.air_weight_factor / (air_count + eps)
        
        result = {
            "prompt_ids": self.prompt_tokenizer.encode(example.prompt),
            "prompt_text": example.prompt,  # Raw text for pre-trained transformer encoders
            "voxel_ids": voxel_ids,
            "source_index": torch.tensor(index, dtype=torch.long),
            "category_index": torch.tensor(example.category_index, dtype=torch.long),
            "sample_weight": torch.tensor(self._sample_weights[index], dtype=torch.float32),
            "per_block_weight": torch.tensor(self._per_block_weight_cache, dtype=torch.float32),
            "per_air_weight": torch.tensor(self._per_air_weight_cache, dtype=torch.float32),
        }

        # Add pre-computed hidden states if available
        if self.hidden_states is not None and self.hidden_index is not None:
            hs_idx = self.hidden_index[index]
            result["hidden_states"] = self.hidden_states[hs_idx]
            result["attention_masks"] = self.attention_masks[hs_idx]

        return result

    def source_weights(self, priority_dirs: Iterable[str | Path], priority_weight: float) -> torch.Tensor:
        priority = {Path(path).resolve() for path in priority_dirs}
        weights = []
        for example in self.examples:
            weights.append(priority_weight if example.source_dir.resolve() in priority else 1.0)
        return torch.tensor(weights, dtype=torch.double)

    @classmethod
    def with_cache(
        cls,
        data_dirs: Iterable[str | Path],
        cache_dir: Path,
        target_size: tuple[int, int, int] = (16, 16, 16),
        **kwargs,
    ) -> "MultiSourceSchematicDataset":
        """Create dataset and load hidden states cache, matching examples to cache indices.
        
        Handles multi-prompt txt files: each prompt in a txt file (separated by '---')
        creates its own training example, and the cache has one entry per prompt.
        The matching is done by (schem_name, prompt_index) to ensure correct alignment.
        """
        from app.hidden_state_cache import load_hidden_states_raw
        cache_data = load_hidden_states_raw(cache_dir)
        hidden_states = cache_data["hidden_states"]       # [N, seq_len, hidden_dim]
        attention_masks = cache_data["attention_masks"]    # [N, seq_len]
        cache_metadata = cache_data["metadata"]
        cache_files_list = cache_metadata.get("files", [])

        # Build lookup: (schem_name, prompt_index) -> cache row index
        cache_lookup: dict[tuple[str, int], int] = {}
        for i, f in enumerate(cache_files_list):
            key = (f.get("schem", ""), f.get("prompt_index", 0))
            cache_lookup[key] = i

        # Build dataset first (will create examples)
        dataset = cls(data_dirs, target_size=target_size, **kwargs)

        # Build hidden_index: for each example, find matching cache row
        # We need to track prompt_index per schem file as we iterate examples
        prompt_counter: dict[str, int] = {}
        hidden_index = []
        for ex in dataset.examples:
            schem_name = ex.path.name
            pi = prompt_counter.get(schem_name, 0)
            prompt_counter[schem_name] = pi + 1

            key = (schem_name, pi)
            if key in cache_lookup:
                hidden_index.append(cache_lookup[key])
            else:
                # Fallback: try matching by schem name only (backward compat)
                # Find first cache entry for this schem
                found = False
                for i, f in enumerate(cache_files_list):
                    if f.get("schem") == schem_name:
                        hidden_index.append(i)
                        found = True
                        break
                if not found:
                    hidden_index.append(0)

        dataset.hidden_states = hidden_states
        dataset.attention_masks = attention_masks
        dataset.hidden_index = hidden_index
        return dataset
