from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nbtlib import File
from nbtlib import tag as T
from torch.utils.data import Dataset, default_collate


AIR = "minecraft:air"
SCHEM_EXTENSIONS = {".schem", ".schematic"}

# Separator between the prompt blocks of one sidecar (scrape_phantommarket.py
# writes sidecars in exactly this format).
PROMPT_SEPARATOR = "---"

# Metadata prefixes that are filtered out of prompt sidecars (.txt). Single
# source of truth - app/hidden_state_cache.py delegates to split_prompt_text()
# so cache and dataset always agree on the prompt list.
META_PROMPT_PREFIXES = ("source:", "aug:", "author:", "name:", "url:", "id:")

# Kept as an alias for older call sites / documentation.
META_PREFIXES = META_PROMPT_PREFIXES


def split_prompt_text(text: str) -> List[str]:
    """Split a prompt sidecar on '---' and drop pure metadata sections.

    Current format - first block is the structure name, then the description.
    Every block is one prompt, so a structure yields two training examples:

        "<name>---<description>" -> ["<name>", "<description>"]

    Legacy format of the scraper (provenance lines, no name prompt):

        "<meta lines>---<prompt>---<prompt>" -> ["<prompt>", "<prompt>"]
    """
    prompts: List[str] = []
    for part in (chunk.strip() for chunk in str(text).split(PROMPT_SEPARATOR)):
        if not part:
            continue
        if any(part.lower().startswith(prefix) for prefix in META_PROMPT_PREFIXES):
            continue
        prompts.append(part)
    return prompts

# ─── Block state handling ───
#
# Die drei Zustands-Helfer sind reine Funktionen der Blocknamen und werden fuer
# JEDEN Voxel jeder Struktur aufgerufen (Millionen Aufrufe, aber nur wenige
# hundert verschiedene Namen). Ohne Memoisierung erzeugt jeder Aufruf einen
# neuen String - das war der groesste Einzelposten beim Arbeitsspeicher.
BLOCK_STATE_CACHE_SIZE = int(os.environ.get("MINECRAFT_AI_BLOCKSTATE_CACHE", "200000"))

BLOCK_STATE_PATTERN = re.compile(r"^([a-z0-9_:-]+)(\[.*\])?$")

# Block States, die für die Architektur wichtig sind und behalten werden sollen
IMPORTANT_BLOCK_STATES = {"facing", "axis", "half", "waterlogged", "open", "type", "rotation", "shape", "part", "hanging", "lit"}

# Block States, die entfernt werden können (nur numerische Werte, Pflanzenwachstum, etc.)
#
# HINWEIS: Dieses Set wird aktuell nirgends im Code angewendet – es wird bewusst als
# tote Referenz-Liste aufbewahrt. Es dient als Vorlage, falls die KI künftig mehr
# Blockarten/Block-Zustände bauen darf und störende numerische Zustände erlaubt
# werden sollen. Falls nötig, kann hier die gewünschte Auswahl angepasst werden.
UNIMPORTANT_BLOCK_STATES = {
    "age", "level", "power", "delay", "moisture", "layers",
    "stage", "distance", "persistent", "check_decay", "decayable",
    "player_placed", "triggered", "unstable", "inverted", "natural",
    "extended", "enabled", "signal_fire", "cracked",
    "hatch", "facing_except_up", "east", "west", "north", "south",
    "up", "down", "snowy", "berries", "honey_level",
    "face", "attached", "powered", "locked",
    "candles", "instrument", "note", "bites", "eggs", "pickles",
    "bloom", "crackedness", "tilt",
}

@lru_cache(maxsize=BLOCK_STATE_CACHE_SIZE)
def strip_block_state(block_name: str) -> str:
    """Strip ALL block states. 'minecraft:oak_stairs[facing=north,waterlogged=true]' -> 'minecraft:oak_stairs'
    Used for backward compatibility."""
    m = BLOCK_STATE_PATTERN.match(block_name)
    return m.group(1) if m else block_name


@lru_cache(maxsize=BLOCK_STATE_CACHE_SIZE)
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


# Counter-clockwise order to match rotate_block_grid_y's geometry rotation
# (north -> west -> south -> east). Using clockwise here would swap
# north<->south and east<->west relative to the voxel geometry.
_HORIZONTAL_DIRECTIONS = ("north", "west", "south", "east")


def _rotate_direction_y(value: str, quarter_turns: int) -> str:
    if value not in _HORIZONTAL_DIRECTIONS:
        return value
    idx = _HORIZONTAL_DIRECTIONS.index(value)
    return _HORIZONTAL_DIRECTIONS[(idx + quarter_turns) % 4]


@lru_cache(maxsize=BLOCK_STATE_CACHE_SIZE)
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


def _axis_bounds(flags: "np.ndarray") -> tuple[int, int]:
    """Erster und letzter True-Index eines 1D-Boolean-Arrays."""
    first = int(np.argmax(flags))
    last = int(flags.shape[0] - 1 - np.argmax(flags[::-1]))
    return first, last


def trimmed_block_bounds(
    blocks: list[str],
    size: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int] | None:
    """Enger Bounding-Box der Nicht-Luft-Voxel: (min_x, min_y, min_z, max_x, max_y, max_z).

    Gleiches Ergebnis wie ``trim_block_grid(blocks_to_grid(...))``, aber ohne
    verschachtelte Python-Listen: ein Boolean-Array der Groesse
    ``Width*Height*Length`` wird per numpy in einem Durchgang pro Achse
    reduziert. Die alten Tripel-Schleifen liefen fuer jeden Voxel jeder
    Struktur viermal (einmal pro Rotation) durch Python - beim Start des
    Trainings war das der zweitgroesste Zeitfresser nach dem Blockdekodieren.

    Gibt ``None`` zurueck, wenn die Struktur nur aus Luft besteht (das alte
    ``trim_block_grid`` lieferte dafuer einen 1x1x1-Wuerfel).
    """
    sx, sy, sz = size
    if sx <= 0 or sy <= 0 or sz <= 0:
        return None
    total = sx * sy * sz
    if len(blocks) > total:
        blocks = blocks[:total]

    mask = np.fromiter((block != AIR for block in blocks), dtype=bool, count=len(blocks))
    if mask.size < total:  # kuerzere Blockdaten als der Header verspricht
        mask = np.concatenate([mask, np.zeros(total - mask.size, dtype=bool)])
    if not mask.any():
        return None

    # _block_index(x, y, z) = y*sz*sx + z*sx + x  ->  [y][z][x]
    grid = mask.reshape(sy, sz, sx)
    y_min, y_max = _axis_bounds(grid.any(axis=(1, 2)))
    z_min, z_max = _axis_bounds(grid.any(axis=(0, 2)))
    x_min, x_max = _axis_bounds(grid.any(axis=(0, 1)))
    return (x_min, y_min, z_min, x_max, y_max, z_max)


def trimmed_block_size(schematic: SchematicData) -> tuple[int, int, int]:
    """Kantenlaengen der getrimmten Struktur (1,1,1) fuer reine Luft."""
    bounds = trimmed_block_bounds(schematic.blocks_important, schematic.size)
    if bounds is None:
        return (1, 1, 1)
    x0, y0, z0, x1, y1, z1 = bounds
    return (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)


def box_size_after_rotation(size: tuple[int, int, int], rotation: int) -> tuple[int, int, int]:
    """Boxgroesse nach 90°-Schritten um Y (X und Z tauschen bei ungeraden Schritten).

    Die Form der getrimmten Box ist rotationsinvariant: das Drehen des Inhalts
    und anschliessende Trimmen ergibt genau die rotierte Box (die Extremwerte
    entlang X/Z vertauschen sich), deshalb laesst sich die Groesse jeder
    Rotation analytisch berechnen statt das Gitter zu rotieren.
    """
    sx, sy, sz = size
    return (sz, sy, sx) if int(rotation) % 2 else (sx, sy, sz)


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

    Nur die getrimmte Bounding-Box wird ausgewertet (numpy-Scan), nicht das
    komplette Gitter: fuer 3000+ Strukturen war das viermalige Rotieren des
    Gitters in Python der Hauptteil der Trainings-Startzeit.
    """
    base_size = trimmed_block_size(schematic)
    transforms: list[PlacementTransform] = []
    seen: set[tuple[int, tuple[int, int, int], tuple[int, int, int]]] = set()
    for rotation in range(4):
        sx, sy, sz = box_size_after_rotation(base_size, rotation)
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

    if _has_key(nbt_root, "BlockEntities"):
        block_entities_tag = nbt_root["BlockEntities"]
    elif _has_key(nbt_root, "Blocks") and _has_key(nbt_root["Blocks"], "BlockEntities"):
        block_entities_tag = nbt_root["Blocks"]["BlockEntities"]
    elif _has_key(nbt_root, "palette") and _has_key(nbt_root, "block_entities"):
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

def _has_key(tag: Any, key: str) -> bool:
    """``key in tag`` - aber nur fuer Compound-artige Tags.

    ``in`` auf Array-Tags (ByteArray/IntArray) landet in numpy: bei einem
    Legacy-Schematic im MCEdit-Format ist ``root["Blocks"]`` ein ByteArray, und
    ``"BlockEntities" in bytearray`` endet in
    ``TypeError: iteration over a 0-d array``. Daran brach der Dataset-Aufbau
    bei 'wutiao.schem' ab.
    """
    return isinstance(tag, dict) and key in tag


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
    if _has_key(nbt_root, "Entities"):
        entities_tag = nbt_root["Entities"]
    elif _has_key(nbt_root, "entities"):
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

    @property
    def num_voxels(self) -> int:
        return self.size[0] * self.size[1] * self.size[2]

    def __post_init__(self) -> None:
        if not self.blocks_important:
            object.__setattr__(self, "blocks_important", [keep_important_states(b) for b in self.blocks])


class SchematicTooLarge(ValueError):
    """Struktur ist groesser als das erlaubte Voxel-Limit.

    Wird aus :func:`load_schematic` geworfen, *bevor* Palette und Blockdaten
    dekodiert werden - siehe dort.
    """

    def __init__(self, path: "Path | str", size: tuple[int, int, int], max_voxels: int) -> None:
        self.path = Path(path)
        self.size = size
        self.max_voxels = int(max_voxels)
        self.num_voxels = size[0] * size[1] * size[2]
        super().__init__(
            f"{self.path.name}: {size[0]}x{size[1]}x{size[2]} = {self.num_voxels:,} Voxel "
            f"ueberschreitet das Limit von {self.max_voxels:,} Voxel"
        )


def _unwrap_root(nbt_file: File):
    # Direct Schematic key
    if _has_key(nbt_file, "Schematic") and len(nbt_file) == 1:
        return nbt_file["Schematic"]
    # Handle wrapped format: {'': Compound({'Schematic': {...}})}
    if _has_key(nbt_file, ""):
        inner = nbt_file[""]
        if _has_key(inner, "Schematic"):
            return inner["Schematic"]
    # Handle format where Schematic is one of multiple keys
    if _has_key(nbt_file, "Schematic"):
        return nbt_file["Schematic"]
    return nbt_file


def _read_varints(raw: Iterable[int], expected: int) -> list[int]:
    """Referenz-Implementierung (reines Python) fuer kleine/alte Formate."""
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


def _decode_palette_indices(raw, expected: int) -> "np.ndarray":
    """LEB128-Varints vektorisiert dekodieren (gleiches Ergebnis wie _read_varints).

    Der reine Python-Loop kostete fuer einen 400k-Voxel-Build ~0.3 s und war
    damit der groesste Einzelposten beim Trainingsstart (alle Strukturen
    zusammen mehrere Minuten). numpy erledigt dieselbe Arbeit in Millisekunden:

      * ``ends``   = Position jedes Bytes ohne Continuation-Bit (Ende eines Werts)
      * ``starts`` = jeweils ein Byte danach (abzueglich des Allererssten)
      * Beitrag von Byte k = ``(byte & 0x7F) << (7*k)``

    Faellt auf :func:`_read_varints` zurueck, wenn die Rohdaten nicht als
    Byte-Array vorliegen.
    """
    try:
        data = np.asarray(raw)
        if data.dtype != np.uint8:
            # nbtlib liefert int8-Arrays; negative Python-Bytes (-1 statt 255)
            # werden erst im Array (still) umgebrochen, dann nach uint8.
            data = data.astype(np.uint8, copy=False)
        data = data.reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return np.asarray(_read_varints(raw, expected), dtype=np.int64)

    if data.size == 0:
        if expected:
            raise ValueError(f"Expected {expected} block ids, decoded 0")
        return np.empty(0, dtype=np.int64)

    continuation = (data & 0x80) != 0
    ends = np.flatnonzero(~continuation)
    count = int(ends.size)
    if count == 0:
        raise ValueError("Invalid varint in schematic block data")
    starts = np.empty(count, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lengths = ends - starts + 1
    if int(lengths.max()) > 6:
        raise ValueError("Invalid varint in schematic block data")

    values = np.zeros(count, dtype=np.int64)
    for k in range(int(lengths.max())):
        selected = lengths > k
        values[selected] |= (data[starts[selected] + k].astype(np.int64) & 0x7F) << (7 * k)

    if values.size < expected:
        raise ValueError(f"Expected {expected} block ids, decoded {values.size}")
    return values[:expected]


def _blocks_from_palette(palette_index_to_name: dict[int, str], ids: "np.ndarray") -> list[str] | None:
    """Blocknamen aus den Palette-Indizes ziehen (vektorisiert).

    Die Palette enthaelt nur ein paar hundert Namen, deshalb werden die drei
    Ausgabelisten ueber einen Gather auf dem Namens-Array gebildet statt mit
    einem Python-Loop ueber alle Voxel. Gibt ``None`` zurueck, wenn ein Index
    ausserhalb der Palette liegt (dann uebernimmt der alte Loop, der wie bisher
    einen KeyError ausloest).
    """
    if ids.size == 0:
        return []
    if not palette_index_to_name:
        return None  # leere Palette -> wie bisher ueber den Loop (KeyError)
    if ids.min() < 0 or ids.max() >= max(palette_index_to_name) + 1:
        return None
    if max(palette_index_to_name) + 1 != len(palette_index_to_name):
        return None  # lueckenhafte Palette -> konservativ ueber den Loop gehen
    names = [palette_index_to_name[i] for i in range(len(palette_index_to_name))]
    return np.asarray(names, dtype=object)[ids].tolist()


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


def read_schematic_size(path: str | Path) -> tuple[int, int, int]:
    """Nur Width/Height/Length lesen - ohne Palette/Blockdaten zu dekodieren.

    Fuer die Groessenpruefung (max_voxels) reicht der NBT-Header: das Parsen
    eines 281-Millionen-Voxel-Builds dauert ~0.1 s, das Dekodieren der Bloecke
    dagegen Minuten und mehrere GB RAM.
    """
    path = Path(path)
    root = _unwrap_root(File.load(path, gzipped=True))
    if not all(_has_key(root, key) for key in ("Width", "Height", "Length")):
        raise ValueError(f"Unsupported schematic format: {path}")
    return int(root["Width"]), int(root["Height"]), int(root["Length"])


def load_schematic(path: str | Path, max_voxels: int | None = None) -> SchematicData:
    """Schematic laden und in Blocknamen-Listen umwandeln.

    ``max_voxels`` prueft die Groesse anhand des Headers *vor* dem Dekodieren.
    Zu grosse Strukturen werfen :class:`SchematicTooLarge` statt den Speicher zu
    sprengen: Builds mit >100 Mio. Voxeln wuerden beim Dekodieren dreistellige
    GB anfordern (drei String-Listen mit je einem Eintrag pro Voxel).
    """
    path = Path(path)
    root = _unwrap_root(File.load(path, gzipped=True))

    block_entities = extract_block_entities(root)
    entities = extract_entities(root)

    # Read palette
    palette_raw = None
    block_data_raw = None

    if _has_key(root, "Blocks") and _has_key(root["Blocks"], "Palette"):
        # Sponge v3 with nested Blocks compound
        palette_raw = {int(v): str(k) for k, v in root["Blocks"]["Palette"].items()}
        block_data_raw = root["Blocks"]["Data"]
        width = int(root["Width"])
        height = int(root["Height"])
        length = int(root["Length"])
    elif _has_key(root, "Palette") and _has_key(root, "BlockData"):
        # Sponge v2 with flat palette
        palette_raw = {int(v): str(k) for k, v in root["Palette"].items()}
        block_data_raw = root["BlockData"]
        width = int(root["Width"])
        height = int(root["Height"])
        length = int(root["Length"])
    elif _has_key(root, "Blocks"):
        # Legacy format
        width = int(root["Width"])
        height = int(root["Height"])
        length = int(root["Length"])
        size = (width, height, length)
        if max_voxels is not None and width * height * length > max_voxels:
            raise SchematicTooLarge(path, size, max_voxels)
        blocks_raw = [f"legacy:{int(v) & 0xFF}" for v in root["Blocks"]]
        blocks_stripped = [strip_block_state(b) for b in blocks_raw]
        blocks_important = [keep_important_states(b) for b in blocks_raw]
        return SchematicData(blocks_raw, blocks_stripped, blocks_important, size,
                           block_entities, entities)
    else:
        raise ValueError(f"Unsupported schematic format: {path}")

    size = (width, height, length)
    if max_voxels is not None and width * height * length > max_voxels:
        # Abbrechen, bevor eine einzige Blockzeile materialisiert wird.
        raise SchematicTooLarge(path, size, max_voxels)

    ids = _decode_palette_indices(block_data_raw, width * height * length)
    blocks_raw = _blocks_from_palette(palette_raw, ids)
    if blocks_raw is None:
        # Palette inkonsistent (Index ausserhalb) - wie bisher per KeyError/Loop.
        raw_ids = ids.tolist()
        blocks_raw = [palette_raw[i] for i in raw_ids]
    blocks_stripped = [strip_block_state(b) for b in blocks_raw]
    blocks_important = [keep_important_states(b) for b in blocks_raw]
    return SchematicData(blocks_raw, blocks_stripped, blocks_important, size,
                       block_entities, entities)


def save_schem(path: str | Path, token_grid: torch.Tensor, id_to_block: list[str],
               block_entities: list[dict] | None = None) -> int:
    """Save a schematic using nbtlib with correct Sponge v3 format.
    Manually gzips to work around nbtlib v1.12.1 gzip bug.
    Returns the number of blocks that were converted to air due to out-of-range IDs.

    The grid is written exactly as passed in - no block states are mirrored or
    swapped here, so every caller (CLI, project save, export) produces the same
    orientation. (An earlier version had a ``swap_directions`` flag that was only
    used by the GUI export button and silently mirrored east/west and north/south.)
    """
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

    # Name -> Palette-Index (erste Fundstelle gewinnt) - vermeidet O(n²) Suchen
    palette_index: dict[str, int] = {}
    for index, block in enumerate(palette_blocks):
        palette_index.setdefault(block, index)

    # Konvertiere Grid in lineare ID-Liste (Y, Z, X Reihenfolge wie Sponge v3)
    block_ids: list[int] = []
    for y in range(height):
        for z in range(length):
            for x in range(width):
                tid = int(grid[x, y, z])
                block_name = None
                if 0 <= tid < vocab_size:
                    block_name = id_to_block[tid]
                if not isinstance(block_name, str):
                    # out-of-range ID oder Lücke in der Vokabel -> air
                    block_name = AIR
                block_ids.append(palette_index.get(block_name, palette_index[AIR]))

    # Baue das Schematic mit nbtlib
    palette_compound = T.Compound({})
    for block in palette_blocks:
        palette_compound[block] = T.Int(palette_index[block])

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
        self.reset()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_block)

    def fit(self, schematics: Iterable[SchematicData],
            target_size: tuple[int, int, int] | None = None) -> None:
        """Build vocabulary from ALL block names across ALL schematics,
        keeping only important block states.

        ``target_size`` wird nur aus Kompatibilitaet akzeptiert und nicht benoetigt.
        """
        self.reset()
        for schematic in schematics:
            self.observe(schematic)
        self.finalize()

    def reset(self) -> None:
        """Vokabel-Sammlung leeren (siehe :meth:`observe` / :meth:`finalize`)."""
        self._vocab_names: set[str] = {AIR}
        self.block_entity_types: set[str] = set()

    def observe(self, schematic: SchematicData) -> None:
        """Blocknamen + BlockEntity-Typen *einer* Struktur einsammeln.

        Statt das Gitter viermal zu rotieren und jeden einzelnen Voxel durch
        Python zu schicken, werden nur die eindeutigen Blocknamen gedreht: die
        Rotation aendert ausschliesslich den Namen (facing/axis/rotation), nie
        die Menge der Namen im Gitter. Ergebnis ist identisch, der Aufwand
        sinkt von "alle Voxel" auf "alle Blocktypen".
        """
        vocab = getattr(self, "_vocab_names", None)
        if vocab is None:
            self.reset()
            vocab = self._vocab_names

        for name in set(schematic.blocks_important):
            # Include all rotated states so stairs, logs, shelves/signs etc. do
            # not disappear when augmented rotations create new orientations.
            for rotation in range(4):
                vocab.add(keep_important_states(rotate_block_state_y(name, rotation)))

        # Track which blocks have BlockEntities
        for be in schematic.block_entities:
            self.block_entity_types.add(strip_block_state(be.get("id", "")))

    def finalize(self) -> None:
        """``block_to_id`` / ``id_to_block`` aus den gesammelten Namen bauen."""
        names = getattr(self, "_vocab_names", {AIR})
        self.block_to_id = {AIR: 0}
        for block in sorted(names - {AIR}):
            self.block_to_id[block] = len(self.block_to_id)
        # Mit AIR vorbelegen statt mit None: Lücken/None wuerden sonst als
        # Blockname "None" in die Palette wandern.
        self.id_to_block = [AIR] * len(self.block_to_id)
        for block, idx in self.block_to_id.items():
            self.id_to_block[idx] = block

    def encode_blocks(self, blocks: list[str], size: tuple[int, int, int],
                      target_size: tuple[int, int, int]) -> torch.Tensor:
        fitted = fit_to_size(blocks, size, target_size)
        cleaned = [keep_important_states(b) for b in fitted]
        ids = [self.block_to_id.get(block, self.block_to_id[AIR]) for block in cleaned]
        return torch.tensor(ids, dtype=torch.long).view(
            target_size[1], target_size[2], target_size[0]).permute(2, 0, 1)

    def safe_encode_blocks(self, blocks: list[str], size: tuple[int, int, int],
                           target_size: tuple[int, int, int]) -> torch.Tensor:
        """Like encode_blocks, but falls back to AIR (0) for any unknown block."""
        fitted = fit_to_size(blocks, size, target_size)
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
    ) -> torch.Tensor:
        """Encode an already target-sized flat block list."""
        expected = target_size[0] * target_size[1] * target_size[2]
        if len(blocks) != expected:
            raise ValueError(f"Expected {expected} blocks, got {len(blocks)}")
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


def _normalize_prompts(value: Any) -> List[str]:
    """One prompts.json value -> list of prompts.

    A value may be a single prompt or a list of prompts for the same structure
    (e.g. name + description); every entry becomes its own training example.
    """
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []
    return [text for text in (str(item).strip() for item in values) if text]


def _load_prompt_map(directory: Path) -> Dict[str, List[str]]:
    """Read prompts.json of a directory (empty dict if missing/invalid)."""
    prompt_map_path = directory / "prompts.json"
    if not prompt_map_path.exists():
        return {}
    try:
        data = json.loads(prompt_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warnung: prompts.json in {directory} konnte nicht gelesen werden ({exc})")
        return {}
    if not isinstance(data, dict):
        return {}
    prompt_map: Dict[str, List[str]] = {}
    for key, value in data.items():
        prompts = _normalize_prompts(value)
        if prompts:
            prompt_map[str(key)] = prompts
    return prompt_map


def iter_schematic_files(data_dir: str | Path) -> List[Path]:
    """All .schem/.schematic files below a directory (recursive).

    Recursive discovery means nested layouts work as well, e.g. data sets split
    into ``<category>/`` sub folders (scrape_phantommarket.py --split-by-category).
    """
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SCHEM_EXTENSIONS
    )


def discover_examples(data_dir: str | Path) -> list[tuple[Path, str]]:
    """Discover all (schematic_path, prompt) pairs in a directory (recursive).

    Prompt source order: ``prompts.json`` (own folder, then the data dir root),
    otherwise a ``.txt`` sidecar. A prompt map entry may hold a single prompt or
    a list of prompts for the same structure (e.g. name + description); a .txt
    file with multiple prompts separated by '---' produces one training example
    per prompt as well. Lines of legacy metadata sections (source:, aug:, ...)
    are filtered out.

    Missing directories are skipped instead of raising FileNotFoundError.
    """
    data_dir = Path(data_dir)
    prompt_maps: Dict[Path, Dict[str, List[str]]] = {}

    def prompt_map_for(directory: Path) -> Dict[str, List[str]]:
        if directory not in prompt_maps:
            prompt_maps[directory] = _load_prompt_map(directory)
        return prompt_maps[directory]

    examples: list[tuple[Path, str]] = []
    for schem_path in iter_schematic_files(data_dir):
        prompts: List[str] = []
        for directory in (schem_path.parent, data_dir):
            mapping = prompt_map_for(directory)
            prompts = mapping.get(schem_path.name) or mapping.get(schem_path.stem) or []
            if prompts:
                break
        if prompts:
            examples.extend((schem_path, prompt) for prompt in prompts)
            continue

        txt_path = schem_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            raw = txt_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Warnung: {txt_path} konnte nicht gelesen werden ({exc})")
            continue
        for part in split_prompt_text(raw):
            examples.append((schem_path, part))

    return examples


def discover_prompt_files(data_dirs: Iterable[str | Path]) -> Tuple[List[Path], List[Path]]:
    """Return (schem_files, txt_files) for all structures with a usable prompt sidecar.

    Used by the hidden state cache so that validation and dataset discovery look
    at exactly the same structures. Structures whose prompt comes from
    ``prompts.json`` (without a .txt sidecar) cannot be cached and are therefore
    not part of this list.
    """
    schem_files: List[Path] = []
    txt_files: List[Path] = []
    seen: set[Path] = set()
    for data_dir in data_dirs:
        for schem_path in iter_schematic_files(data_dir):
            txt_path = schem_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            try:
                resolved = schem_path.resolve()
            except OSError:
                resolved = schem_path
            if resolved in seen:
                continue
            try:
                if not split_prompt_text(txt_path.read_text(encoding="utf-8")):
                    continue
            except OSError:
                continue
            seen.add(resolved)
            schem_files.append(schem_path)
            txt_files.append(txt_path)
    return schem_files, txt_files


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


# Decodierte Schematics, die im Arbeitsspeicher gehalten werden duerfen. Eine
# dekodierte Struktur kostet grob 25-30 Byte pro Voxel (drei Listen voller
# Zeiger pro Voxel), 6 Mio. Voxel entsprechen also ca. 150-200 MB.
DEFAULT_SCHEMATIC_CACHE_VOXELS = int(
    os.environ.get("MINECRAFT_AI_SCHEMATIC_CACHE_VOXELS", str(6_000_000)))


class LazySchematicStore(Sequence):
    """Laedt Schematics bei Bedarf und haelt nur das angegebene Voxel-Budget im RAM.

    Vorher hat das Dataset beim Start JEDE Struktur geladen und behalten. Bei
    mehreren tausend Strukturen plus einigen sehr grossen Builds war das einer
    der Hauptgruende fuer den hohen Speicherverbrauch (ein einzelner
    1000x1000-Build sprengt den Rahmen alleine). Ausgewertet wird jede Struktur
    beim Start einmal (Bounding-Box + Blocktypen); danach entscheidet das
    Budget, was im Cache bleibt.

    Die Anzahl der gecachten Strukturen ergibt sich aus dem Voxel-Budget, damit
    grosse Builds den Cache schneller fuellen als kleine.
    """

    def __init__(self, max_voxels: int | None = None,
                 cache_voxels: int = DEFAULT_SCHEMATIC_CACHE_VOXELS,
                 max_entries: int = 8192):
        self._paths: list[Path] = []
        self._cache: "OrderedDict[int, SchematicData]" = OrderedDict()
        self._voxels_by_index: dict[int, int] = {}
        self._cached_voxels = 0
        self.max_voxels = max_voxels
        self.cache_voxels = max(0, int(cache_voxels))
        self.max_entries = max(1, int(max_entries))
        self.load_count = 0

    # ─── Sequence-API (ersetzt die frühere Liste) ───
    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> SchematicData:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self._paths)))]
        if index < 0:
            index += len(self._paths)
        if index < 0 or index >= len(self._paths):
            raise IndexError(
                f"Schematic-Index {index} ausserhalb von 0..{len(self._paths) - 1}")
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        schematic = load_schematic(self._paths[index], max_voxels=self.max_voxels)
        self._store(index, schematic)
        return schematic

    def __iter__(self) -> Iterator[SchematicData]:
        for index in range(len(self._paths)):
            yield self[index]

    # ─── Verwaltung ───
    def path(self, index: int) -> Path:
        return self._paths[index]

    def add(self, path: Path, schematic: SchematicData | None = None) -> int:
        """Pfad registrieren und seinen Index zurückgeben.

        ``schematic`` wird direkt uebernommen (statt beim ersten Zugriff erneut
        von der Platte zu lesen); ohne Angabe laedt ``__getitem__`` bei Bedarf.
        """
        self._paths.append(Path(path))
        index = len(self._paths) - 1
        if schematic is not None:
            self._store(index, schematic)
        return index

    def release(self, index: int) -> None:
        """Struktur sofort aus dem Cache werfen."""
        voxels = self._voxels_by_index.pop(index, None)
        if voxels is not None:
            self._cached_voxels -= voxels
            self._cache.pop(index, None)

    def clear(self) -> None:
        self._cache.clear()
        self._voxels_by_index.clear()
        self._cached_voxels = 0

    @property
    def cached_voxels(self) -> int:
        return self._cached_voxels

    def _store(self, index: int, schematic: SchematicData) -> None:
        self.load_count += 1
        voxels = max(1, int(schematic.num_voxels))
        if self.cache_voxels <= 0 or voxels > self.cache_voxels:
            return  # Budget aus oder einzelne Struktur zu gross
        self._cache[index] = schematic
        self._voxels_by_index[index] = voxels
        self._cached_voxels += voxels
        self._cache.move_to_end(index)
        while (self._cached_voxels > self.cache_voxels or len(self._cache) > self.max_entries) \
                and len(self._cache) > 1:
            old_index, _ = self._cache.popitem(last=False)
            self._cached_voxels -= self._voxels_by_index.pop(old_index, 0)


class MultiSourceSchematicDataset(Dataset):
    def __init__(self, data_dirs: Iterable[str | Path],
                 target_size: tuple[int, int, int] = (16, 16, 16),
                 prompt_tokenizer: PromptTokenizer | None = None,
                 voxel_tokenizer: VoxelTokenizer | None = None,
                 max_voxels: int | None = None,
                 augmentation_diversity: int = 1,
                 allow_vertical_movement: bool = False,
                 max_augmented_variants: int | None = 512,
                 structure_block_weight: float = 100.0,
                 air_weight_factor: float = 75.0,
                 hidden_states: torch.Tensor | None = None,
                 attention_masks: torch.Tensor | None = None,
                 hidden_index: list[int] | None = None,
                 hidden_lengths: torch.Tensor | None = None,
                 encoded_cache_size: int = 1024,
                 schematic_cache_voxels: int = DEFAULT_SCHEMATIC_CACHE_VOXELS,
                 progress_callback: Callable[[int, int], None] | None = None):
        self.structure_block_weight = max(1.0, float(structure_block_weight))
        self.air_weight_factor = max(1.0, float(air_weight_factor))
        self._set_hidden_states(hidden_states, attention_masks, hidden_lengths)
        self.hidden_index = hidden_index  # maps each example to a row index in hidden_states
        raw_examples = discover_examples_many(data_dirs)
        if not raw_examples:
            raise ValueError("No .schem/.schematic examples with prompts found")
        total_examples = len(raw_examples)

        self.examples: list[AugmentedSchematicExample] = []
        self.schematics = LazySchematicStore(
            max_voxels=max_voxels, cache_voxels=schematic_cache_voxels)
        self._schematic_by_path: dict[Path, int] = {}
        self._transform_by_path: dict[Path, list[PlacementTransform]] = {}
        self._category_by_path: dict[Path, int] = {}
        self.skipped_examples: list[Path] = []
        self._oversize_paths: set[Path] = set()
        self.oversize_voxels = 0

        # Vokabular im selben Durchgang sammeln statt in einem zweiten Durchlauf
        # ueber alle Strukturen (siehe VoxelTokenizer.observe()).
        build_tokenizer = voxel_tokenizer is None
        self.voxel_tokenizer = VoxelTokenizer() if build_tokenizer else voxel_tokenizer
        if build_tokenizer:
            self.voxel_tokenizer.reset()

        for processed, (path, prompt, source_dir) in enumerate(raw_examples, start=1):
            if progress_callback and (processed % 25 == 0 or processed == total_examples):
                # Der Aufbau laeuft minutenlang (eine Struktur pro Ladevorgang),
                # ohne Rueckmeldung sieht der Nutzer nur ein "haengendes" Training.
                progress_callback(processed, total_examples)

            resolved = path.resolve()
            if resolved in self._oversize_paths:
                self.skipped_examples.append(path)
                continue

            schematic_index = self._schematic_by_path.get(resolved)
            if schematic_index is None:
                try:
                    # Laedt die Struktur und prueft die Groesse anhand des
                    # Headers, bevor Palette/Blockdaten dekodiert werden.
                    schematic = load_schematic(path, max_voxels=max_voxels)
                except SchematicTooLarge as exc:
                    self._oversize_paths.add(resolved)
                    self.oversize_voxels += exc.num_voxels
                    self.skipped_examples.append(path)
                    continue
                schematic_index = self.schematics.add(path, schematic)
                self._schematic_by_path[resolved] = schematic_index
                if build_tokenizer:
                    self.voxel_tokenizer.observe(schematic)
            else:
                schematic = self.schematics[schematic_index]

            transforms = self._transform_by_path.get(resolved)
            if transforms is None:
                if augmentation_diversity <= 0:
                    sx, sy, sz = trimmed_block_size(schematic)
                    tx, ty, tz = target_size
                    if sx <= tx and sy <= ty and sz <= tz:
                        transforms = [PlacementTransform(
                            rotation=0,
                            offset=((tx - sx) // 2, 0, (tz - sz) // 2),
                        )]
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
                self._transform_by_path[resolved] = transforms

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
        if prompt_tokenizer is None:
            self.prompt_tokenizer = PromptTokenizer()
            self.prompt_tokenizer.fit(example.prompt for example in self.examples)
        else:
            self.prompt_tokenizer = prompt_tokenizer

        if build_tokenizer:
            self.voxel_tokenizer.finalize()

        # LRU cache for encoded/placed grids. Rotation + trim + placement of the
        # source grid is the expensive part of __getitem__ and every augmentated
        # variant is requested once per epoch again.
        self.max_encoded_cache = int(encoded_cache_size)
        self._encoded_cache: "OrderedDict[tuple[int, PlacementTransform], torch.Tensor]" = OrderedDict()

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

    def _encode_example(self, example: AugmentedSchematicExample) -> torch.Tensor:
        """Encode one example to block IDs, cached per (structure, transform)."""
        key = (example.schematic_index, example.transform)
        cached = self._encoded_cache.get(key)
        if cached is not None:
            self._encoded_cache.move_to_end(key)
            return cached

        schematic = self.schematics[example.schematic_index]
        transformed_blocks = transform_blocks_to_size(
            schematic,
            self.target_size,
            example.transform,
        )
        voxel_ids = self.voxel_tokenizer.safe_encode_prepared_blocks(
            transformed_blocks, self.target_size)

        if self.max_encoded_cache > 0:
            self._encoded_cache[key] = voxel_ids
            while len(self._encoded_cache) > self.max_encoded_cache:
                self._encoded_cache.popitem(last=False)
        return voxel_ids

    def _set_hidden_states(self, hidden_states: torch.Tensor | None,
                           attention_masks: torch.Tensor | None,
                           lengths: torch.Tensor | None = None) -> None:
        """Normalise the text context cache to the flat ("ragged") layout.

        Rows are concatenated along the token axis; the start offset of row i is
        the cumulative sum of the row lengths. Storing the context unpadded is
        what keeps the cache small: padding every row to the longest prompt of
        the data set cost ~20 GB for ~6400 prompts with hidden_dim=3072 (see
        app/hidden_state_cache.py). Padding to the longest row of a batch happens
        in collate_context_batch().
        """
        if hidden_states is None:
            self.hidden_states = None
            self.attention_masks = None
            self.hidden_lengths = None
            self.hidden_offsets = None
            return

        if hidden_states.dim() == 3:  # legacy padded cache [N, seq_len, hidden_dim]
            masks = attention_masks
            if masks is None:
                masks = torch.ones(list(hidden_states.shape[:2]), dtype=torch.long)
            lengths = masks.reshape(masks.shape[0], -1).sum(dim=1).to(torch.long)
            hidden_states = torch.cat(
                [hidden_states[i, :int(lengths[i])] for i in range(hidden_states.shape[0])], dim=0)
            attention_masks = torch.ones(int(lengths.sum()), dtype=torch.long)

        if lengths is None:
            raise ValueError(
                "Fuer einen flachen Hidden-State-Cache werden die Zeilenlaengen "
                "benoetigt (load_hidden_states_raw liefert sie).")
        lengths = lengths.to(torch.long).reshape(-1)
        if attention_masks is None:
            attention_masks = torch.ones(int(hidden_states.shape[0]), dtype=torch.long)

        self.hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        self.attention_masks = attention_masks.reshape(-1)
        self.hidden_lengths = lengths
        # offsets[i] = first token of row i, offsets[num_rows] = total tokens
        self.hidden_offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        voxel_ids = self._encode_example(example)

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
        per_block_weight = self.structure_block_weight / (non_air_count + eps)
        per_air_weight = self.air_weight_factor / (air_count + eps)

        result = {
            "prompt_ids": self.prompt_tokenizer.encode(example.prompt),
            "prompt_text": example.prompt,  # Raw text for pre-trained transformer encoders
            "voxel_ids": voxel_ids,
            "source_index": torch.tensor(index, dtype=torch.long),
            "category_index": torch.tensor(example.category_index, dtype=torch.long),
            "sample_weight": torch.tensor(self._sample_weights[index], dtype=torch.float32),
            "per_block_weight": torch.tensor(per_block_weight, dtype=torch.float32),
            "per_air_weight": torch.tensor(per_air_weight, dtype=torch.float32),
        }

        # Add pre-computed hidden states if available. Rows are stored unpadded
        # (see _set_hidden_states); the batch padding happens in
        # collate_context_batch(), not here.
        if self.hidden_states is not None and self.hidden_index is not None:
            hs_idx = self.hidden_index[index]
            start = int(self.hidden_offsets[hs_idx])
            end = start + int(self.hidden_lengths[hs_idx])
            result["hidden_states"] = self.hidden_states[start:end]
            result["attention_masks"] = self.attention_masks[start:end]

        return result

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
        hidden_states = cache_data["hidden_states"]      # flat [total_tokens, hidden_dim]
        attention_masks = cache_data["attention_masks"]  # flat [total_tokens]
        hidden_lengths = cache_data["lengths"]           # [num_prompts] tokens per row
        cache_metadata = cache_data["metadata"]
        cache_files_list = cache_metadata.get("files", [])

        # Build lookup: (schem_name, prompt_index) -> cache row index
        cache_lookup: dict[tuple[str, int], int] = {}
        for i, f in enumerate(cache_files_list):
            key = (f.get("schem", ""), f.get("prompt_index", 0))
            cache_lookup[key] = i

        # Build dataset first (will create examples)
        dataset = cls(data_dirs, target_size=target_size, **kwargs)

        # Build hidden_index: for each example, find the matching cache row.
        #
        # The prompt index must be counted per (schematic file, prompt), NOT per
        # generated example: a structure with 2 prompts and 64 placement variants
        # produces 128 examples, but only 2 cache rows. The previous per-example
        # counter assigned cache rows of *other* prompts to augmented variants and
        # then aborted with "Cache unvollstaendig".
        prompt_index_by_key: dict[tuple[str, str], int] = {}
        next_prompt_index: dict[str, int] = {}
        hidden_index = []
        for ex in dataset.examples:
            schem_name = ex.path.name
            lookup_key = (schem_name, ex.prompt)
            pi = prompt_index_by_key.get(lookup_key)
            if pi is None:
                pi = next_prompt_index.get(schem_name, 0)
                prompt_index_by_key[lookup_key] = pi
                next_prompt_index[schem_name] = pi + 1

            key = (schem_name, pi)
            if key in cache_lookup:
                hidden_index.append(cache_lookup[key])
                continue

            # Backward-Kompatibilität: Falls in der Cache genau EIN Eintrag für diese
            # Datei liegt (Single-Prompt), ist die Zuordnung eindeutig und erlaubt.
            matches_for_schem = [i for i, f in enumerate(cache_files_list) if f.get("schem") == schem_name]
            if len(matches_for_schem) == 1:
                hidden_index.append(matches_for_schem[0])
                continue

            # Keine stille Fehlausrichtung: Wenn kein passender Cache-Eintrag existiert,
            # hart abbrechen, statt dem Beispiel fälschlich ein fremdes Hidden State (z.B.
            # Zeile 0) zuzuordnen.
            raise ValueError(
                f"[Cache unvollständig] Kein Hidden-State-Eintrag für '{schem_name}' "
                f"(Prompt {pi}). Bitte Cache neu berechnen oder einen neuen Cache anlegen."
            )

        dataset._set_hidden_states(hidden_states, attention_masks, hidden_lengths)
        dataset.hidden_index = hidden_index
        return dataset


def collate_context_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate with the ragged text context padded to the batch maximum.

    The hidden-state cache stores every prompt unpadded (see
    app/hidden_state_cache.py). Padding here keeps the waste minimal - the longest
    prompt of a batch is usually far below the longest prompt of the whole data
    set (median 14 tokens vs. a 512 token outlier) - and the attention mask marks
    the padded positions, so the cross-attention ignores them. Pass this as
    ``collate_fn`` for every DataLoader over a dataset built with
    ``MultiSourceSchematicDataset.with_cache()``.
    """
    if batch and batch[0].get("hidden_states") is not None:
        max_len = max(int(item["hidden_states"].shape[0]) for item in batch)
        for item in batch:
            hidden = item["hidden_states"]
            pad = max_len - int(hidden.shape[0])
            if pad <= 0:
                continue
            item["hidden_states"] = F.pad(hidden, (0, 0, 0, pad))
            masks = item.get("attention_masks")
            if masks is not None:
                item["attention_masks"] = F.pad(masks, (0, pad))
    return default_collate(batch)
