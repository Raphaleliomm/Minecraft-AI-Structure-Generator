"""Umwandlung von Litematica-Dateien und Blueprint-JSONs in Sponge-``.schem``.

Der Dataset-Loader (``dataset.py``) liest nur ``.schem``/``.schematic``. Viele
freie Quellen (Abfielder, Hugging-Face-Datensaetze) liefern aber ``.litematic``
oder ein JSON-Grid. Dieses Modul uebersetzt beide Formate ohne externe
Konverter:

* ``litematic_to_schem``  - echtes Litematica-NBT (auch mehrere Regionen mit
  negativen Groessen) -> Sponge v3 ``.schem``.
* ``blueprint_to_schem``  - das JSON-Format aus
  ``Hack337/Minecraft-Schematics`` (``size``/``palette``/``layers``) ->
  Sponge v3 ``.schem``.

Beide Funktionen schreiben ausschliesslich Vanilla-Blocknamen mit sortierten
Block-States, damit die Palette deterministisch bleibt.
"""
from __future__ import annotations

import gzip
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from nbtlib import File
from nbtlib import tag as T

AIR = "minecraft:air"
AIR_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})
DEFAULT_DATA_VERSION = 3953
MAX_DIMENSION = 32767  # Width/Height/Length sind NBT-Shorts


class StructureConversionError(ValueError):
    """Die Quelldatei ist kein unterstuetztes Litematica-/Blueprint-Format."""


class StructureTooLarge(StructureConversionError):
    """Die Struktur ueberschreitet das konfigurierte Voxel-Limit."""

    def __init__(self, size: tuple[int, int, int], max_voxels: int, source: str = "") -> None:
        self.size = size
        self.max_voxels = int(max_voxels)
        self.num_voxels = int(size[0]) * int(size[1]) * int(size[2])
        prefix = f"{source}: " if source else ""
        super().__init__(
            f"{prefix}{size[0]}x{size[1]}x{size[2]} = {self.num_voxels:,} Voxel "
            f"ueberschreitet das Limit von {self.max_voxels:,} Voxel"
        )


@dataclass(frozen=True)
class ConversionStats:
    """Kurzbericht einer Konvertierung (fuer Logs und Metadaten)."""

    size: tuple[int, int, int]
    blocks: int
    palette_size: int
    block_entities: int
    entities: int
    regions: int
    source_format: str
    output_bytes: int = 0
    name: str = ""
    author: str = ""
    unknown_chars: int = 0


# ── Block-Namen ────────────────────────────────────────────────────────────

def _format_block(entry: Any) -> str:
    """Litematica-Palette-Eintrag -> ``minecraft:name[key=value,...]``."""
    if isinstance(entry, Mapping):
        name = entry.get("Name", AIR)
        properties = entry.get("Properties")
    else:  # pragma: no cover - defensive, Litematica liefert Compounds
        name = getattr(entry, "get", lambda *_: AIR)("Name", AIR)
        properties = getattr(entry, "get", lambda *_: None)("Properties")
    name = str(name or AIR)
    if properties:
        keys = sorted(str(key) for key in properties.keys())
        state = ",".join(f"{key}={properties[key]}" for key in keys)
        if state:
            name = f"{name}[{state}]"
    return _normalise_block(name)


def _normalise_block(name: str) -> str:
    name = str(name).strip() or AIR
    return AIR if name in AIR_NAMES else name


def _palette_names(palette: Iterable[Any]) -> list[str]:
    return [_format_block(entry) for entry in palette]


# ── Litematica ─────────────────────────────────────────────────────────────

def _region_bounds(region: Any) -> tuple[list[int], list[int]]:
    """Min-Ecke und absolute Ausdehnung einer Litematica-Region.

    ``Size`` darf negativ sein; die Bloecke liegen immer im Bereich
    ``min .. min+dims-1``. Litematica indiziert die Region lokal beginnend an
    dieser Min-Ecke (nicht an ``Position``).
    """
    position = [int(region["Position"][axis]) for axis in "xyz"]
    size = [int(region["Size"][axis]) for axis in "xyz"]
    if any(value == 0 for value in size):
        raise StructureConversionError(f"Region mit leerer Groesse: {size}")
    minimum = [
        position[i] if size[i] > 0 else position[i] + size[i] + 1
        for i in range(3)
    ]
    dims = [abs(value) for value in size]
    return minimum, dims


def _decode_region_ids(region: Any) -> "np.ndarray":
    """Palette-Indizes einer Region als ``(sizeY, sizeZ, sizeX)``-Array.

    Litematica hat zwei Pack-Varianten:

    * klassisch: ``valuesPerLong = 64 // bits`` Werte pro Long, Restbits
      ungenutzt (Array-Laenge ``ceil(volume / valuesPerLong)``).
    * neuer (SubVersion >= 1, wie MC 1.16+): Bits laufen kontinuierlich ueber
      Long-Grenzen (Array-Laenge ``ceil(volume * bits / 64)``).

    Welche Variante vorliegt, verraet die tatsaechliche Laenge von
    ``BlockStates``. Der Blockindex ist in beiden Faellen
    ``(y * sizeZ + z) * sizeX + x``.
    """
    palette = list(region.get("BlockStatePalette") or [])
    if not palette:
        raise StructureConversionError("Region ohne BlockStatePalette")
    size = [abs(int(region["Size"][axis])) for axis in "xyz"]
    volume = size[0] * size[1] * size[2]
    bits = max(2, (len(palette) - 1).bit_length())
    values_per_long = 64 // bits
    mask = (1 << bits) - 1

    raw = np.asarray(region["BlockStates"], dtype=np.uint64).reshape(-1)
    padded_longs = math.ceil(volume / values_per_long)
    contiguous_longs = math.ceil(volume * bits / 64)
    if raw.size >= padded_longs:
        padded = True
    elif raw.size >= contiguous_longs:
        padded = False
    else:
        raise StructureConversionError(
            f"BlockStates zu kurz: {raw.size} Longs, erwartet "
            f"{padded_longs} (gepackt) bzw. {contiguous_longs} (kontinuierlich)"
        )

    if padded:
        flat = np.empty(volume, dtype=np.uint32)
        for slot in range(values_per_long):
            values = ((raw >> np.uint64(slot * bits)) & np.uint64(mask)).astype(np.uint32)
            target = flat[slot:volume:values_per_long]
            flat[slot:volume:values_per_long] = values[: target.size]
    else:
        # Kontinuierliches Bitpacking: jedes Value kann ueber zwei Longs laufen.
        bit_positions = np.arange(volume, dtype=np.int64) * bits
        long_index = bit_positions >> 6
        shift = bit_positions & 63
        values = (raw[long_index] >> shift.astype(np.uint64)).astype(np.uint64)
        straddle = shift + bits > 64
        if np.any(straddle):
            high = raw[long_index[straddle] + 1]
            values[straddle] |= high << (64 - shift[straddle]).astype(np.uint64)
        flat = (values & np.uint64(mask)).astype(np.uint32)

    return flat.reshape(size[1], size[2], size[0])


def _copy_nbt(value: Any) -> Any:
    """Tiefe Kopie eines nbtlib-Tags (Tags sind an ihre Datei gebunden)."""
    if isinstance(value, T.Compound):
        return T.Compound({key: _copy_nbt(item) for key, item in value.items()})
    if isinstance(value, T.List):
        subtype = value.subtype if getattr(value, "subtype", None) else T.End
        return T.List[subtype]([_copy_nbt(item) for item in value])
    if isinstance(value, T.ByteArray):
        return T.ByteArray(np.asarray(value, dtype=np.uint8).copy())
    if isinstance(value, T.IntArray):
        return T.IntArray(np.asarray(value, dtype=np.int32).copy())
    if isinstance(value, T.LongArray):
        return T.LongArray(np.asarray(value, dtype=np.int64).copy())
    return value


def _convert_block_entities(
    regions: Any,
    global_min: list[int],
    size: tuple[int, int, int],
) -> list[Any]:
    entities: list[Any] = []
    for region in regions.values():
        region_min, _ = _region_bounds(region)
        for raw in region.get("TileEntities") or []:
            try:
                world = [int(raw["x"]), int(raw["y"]), int(raw["z"])]
            except (KeyError, TypeError, ValueError):
                continue
            local = [world[i] - global_min[i] for i in range(3)]
            if not all(0 <= local[i] < size[i] for i in range(3)):
                continue
            compound = T.Compound({})
            for key, value in raw.items():
                name = str(key)
                if name in ("x", "y", "z", "id", "Id", "Pos"):
                    continue
                compound[name] = _copy_nbt(value)
            block_id = str(raw.get("id") or raw.get("Id") or "").strip()
            if block_id:
                compound["Id"] = T.String(block_id)
            compound["Pos"] = T.IntArray(local)
            entities.append(compound)
    return entities


def _convert_entities(
    regions: Any,
    global_min: list[int],
    size: tuple[int, int, int],
) -> list[Any]:
    """Entity-NBT uebernehmen und Positionen auf die Schematic-Min-Ecke ziehen.

    Litematica speichert Entity-Positionen relativ zu ``Position`` der Region
    (BlockEntities dagegen absolut). Beide werden hier auf ``0..size-1``
    umgerechnet; Entities ausserhalb der Box bleiben erhalten, weil sie
    (z.B. Bilder) an der Wand haengen koennen.
    """
    result: list[Any] = []
    for region in regions.values():
        position = [int(region["Position"][axis]) for axis in "xyz"]
        for raw in region.get("Entities") or []:
            compound = _copy_nbt(raw)
            positions = compound.get("Pos")
            if positions is not None and len(positions) >= 3:
                values = [float(positions[i]) + position[i] - global_min[i] for i in range(3)]
                compound["Pos"] = T.List[T.Double]([T.Double(value) for value in values])
            for key, axis in (("TileX", 0), ("TileY", 1), ("TileZ", 2)):
                if key in compound:
                    compound[key] = T.Int(int(compound[key]) + position[axis] - global_min[axis])
            result.append(compound)
    return result


def load_litematic(
    path: str | Path,
    *,
    max_voxels: int | None = None,
) -> tuple["np.ndarray", list[str], list[Any], list[Any], ConversionStats]:
    """Liest ein ``.litematic`` und liefert Grid, Palette, Entities, Statistik.

    Das Grid hat die Form ``(sizeX, sizeY, sizeZ)`` und enthaelt Indizes in
    ``palette``. Mehrere Regionen werden in eine gemeinsame Bounding-Box
    gelegt; spaetere Regionen ueberschreiben fruehere bei Ueberlappung.
    """
    path = Path(path)
    try:
        root = File.load(path, gzipped=True)
    except Exception as exc:  # nbtlib wirft unterschiedliche Typen
        raise StructureConversionError(f"{path.name}: kein gueltiges Litematica-NBT ({exc})") from exc

    if "Regions" not in root:
        raise StructureConversionError(f"{path.name}: kein 'Regions'-Block (kein Litematica?)")
    regions = root["Regions"]
    if not regions:
        raise StructureConversionError(f"{path.name}: Litematica ohne Regionen")

    bounds = [_region_bounds(region) for region in regions.values()]
    global_min = [min(item[0][axis] for item in bounds) for axis in range(3)]
    global_max = [
        max(item[0][axis] + item[1][axis] - 1 for item in bounds) for axis in range(3)
    ]
    size = tuple(global_max[i] - global_min[i] + 1 for i in range(3))
    if any(value <= 0 for value in size):
        raise StructureConversionError(f"{path.name}: ungueltige Groesse {size}")
    if max(size) > MAX_DIMENSION:
        raise StructureConversionError(f"{path.name}: Dimension {size} > {MAX_DIMENSION}")

    num_voxels = size[0] * size[1] * size[2]
    if max_voxels is not None and max_voxels > 0 and num_voxels > max_voxels:
        raise StructureTooLarge(size, max_voxels, path.name)

    palette: list[str] = [AIR]
    palette_index: dict[str, int] = {AIR: 0}
    grid = np.zeros(size, dtype=np.int32)

    for region in regions.values():
        region_min, dims = _region_bounds(region)
        ids = _decode_region_ids(region)
        names = _palette_names(region.get("BlockStatePalette") or [])
        mapping = np.empty(len(names), dtype=np.int32)
        for index, name in enumerate(names):
            if name not in palette_index:
                palette_index[name] = len(palette)
                palette.append(name)
            mapping[index] = palette_index[name]
        global_ids = mapping[ids]

        xs = np.arange(dims[0], dtype=np.int64) + (region_min[0] - global_min[0])
        ys = np.arange(dims[1], dtype=np.int64) + (region_min[1] - global_min[1])
        zs = np.arange(dims[2], dtype=np.int64) + (region_min[2] - global_min[2])
        # ids: (y, z, x) -> Grid: (x, y, z)
        grid[np.ix_(xs, ys, zs)] = global_ids.transpose(2, 0, 1)

    block_entities = _convert_block_entities(regions, global_min, size)
    entities = _convert_entities(regions, global_min, size)
    metadata = root.get("Metadata") or {}
    stats = ConversionStats(
        size=size,
        blocks=int(np.count_nonzero(grid)),
        palette_size=len(palette),
        block_entities=len(block_entities),
        entities=len(entities),
        regions=len(regions),
        source_format="litematic",
        name=str(metadata.get("Name") or ""),
        author=str(metadata.get("Author") or ""),
    )
    return grid, palette, block_entities, entities, stats


# ── Sponge-Schematic schreiben ─────────────────────────────────────────────

def _encode_varints(values: Iterable[int]) -> bytes:
    out = bytearray()
    for raw in values:
        value = int(raw)
        if value < 0:
            raise StructureConversionError("Negative Palette-Indizes sind ungueltig")
        while True:
            part = value & 0x7F
            value >>= 7
            if value:
                out.append(part | 0x80)
            else:
                out.append(part)
                break
    return bytes(out)


def write_sponge_schem(
    path: str | Path,
    *,
    grid: "np.ndarray",
    palette: Iterable[str],
    block_entities: Iterable[Any] = (),
    entities: Iterable[Any] = (),
    data_version: int = DEFAULT_DATA_VERSION,
    name: str = "",
    author: str = "",
) -> int:
    """Schreibt ein Sponge-v3-``.schem`` und gibt die Dateigroesse zurueck.

    ``grid`` hat die Form ``(X, Y, Z)``; die BlockData-Reihenfolge ist
    ``y, z, x`` (wie ``dataset.save_schem``).
    """
    path = Path(path)
    palette = list(palette)
    if not palette:
        raise StructureConversionError("Leere Palette")
    if len(set(palette)) != len(palette):
        raise StructureConversionError("Palette enthaelt doppelte Blocknamen")
    grid = np.asarray(grid)
    if grid.ndim != 3 or any(value <= 0 for value in grid.shape):
        raise StructureConversionError(f"Ungueltige Grid-Groesse: {grid.shape}")
    if not np.issubdtype(grid.dtype, np.integer):
        raise StructureConversionError("Grid muss ganzzahlige Palette-Indizes enthalten")
    if np.any(grid < 0) or np.any(grid >= len(palette)):
        raise StructureConversionError("Grid enthaelt ungueltige Palette-Indizes")
    if AIR not in palette:
        palette = [AIR, *palette]
        grid = grid.astype(np.int64) + 1
    width, height, length = (int(value) for value in grid.shape)
    if max(width, height, length) > MAX_DIMENSION:
        raise StructureConversionError(f"Dimension {grid.shape} > {MAX_DIMENSION}")

    flat = np.asarray(grid, dtype=np.int64).transpose(1, 2, 0).reshape(-1)
    data = _encode_varints(flat)

    palette_compound = T.Compound({name: T.Int(index) for index, name in enumerate(palette)})
    block_entities_list = T.List[T.Compound]([])
    for entity in block_entities:
        block_entities_list.append(_copy_nbt(entity))
    entity_list = T.List[T.Compound]([])
    for entity in entities:
        entity_list.append(_copy_nbt(entity))

    metadata = T.Compound({})
    if name:
        metadata["Name"] = T.String(name)
    if author:
        metadata["Author"] = T.String(author)

    root = T.Compound(
        {
            "Version": T.Int(3),
            "DataVersion": T.Int(int(data_version)),
            "Width": T.Short(width),
            "Height": T.Short(height),
            "Length": T.Short(length),
            "Offset": T.IntArray([0, 0, 0]),
            "Metadata": metadata,
            "Blocks": T.Compound(
                {
                    "Palette": palette_compound,
                    "Data": T.ByteArray(np.frombuffer(data, dtype=np.uint8)),
                    "BlockEntities": block_entities_list,
                }
            ),
            "Entities": entity_list,
        }
    )

    wrapped = T.Compound({"Schematic": root})
    buffer = io.BytesIO()
    File({"": wrapped}).write(buffer)
    raw = buffer.getvalue()

    template = path.with_name(path.name + ".part")
    try:
        with gzip.open(template, "wb", compresslevel=6) as handle:
            handle.write(raw)
        template.replace(path)
    finally:
        template.unlink(missing_ok=True)
    return path.stat().st_size


def litematic_to_schem(
    source: str | Path,
    target: str | Path,
    *,
    max_voxels: int | None = None,
    data_version: int = DEFAULT_DATA_VERSION,
) -> ConversionStats:
    """Konvertiert eine ``.litematic``-Datei in ein Sponge-``.schem``."""
    grid, palette, block_entities, entities, stats = load_litematic(
        source, max_voxels=max_voxels
    )
    output_bytes = write_sponge_schem(
        target,
        grid=grid,
        palette=palette,
        block_entities=block_entities,
        entities=entities,
        data_version=data_version,
        name=stats.name,
        author=stats.author,
    )
    return ConversionStats(
        size=stats.size,
        blocks=stats.blocks,
        palette_size=stats.palette_size,
        block_entities=stats.block_entities,
        entities=stats.entities,
        regions=stats.regions,
        source_format=stats.source_format,
        output_bytes=output_bytes,
        name=stats.name,
        author=stats.author,
    )


# ── Blueprint-JSON (Hack337/Minecraft-Schematics) ──────────────────────────

def load_blueprint(data: Mapping[str, Any]) -> tuple["np.ndarray", list[str], ConversionStats]:
    """JSON-Blueprint -> Grid/Palette/Statistik.

    Format::

        {"size": [width, height, depth],
         "palette": {"a": "minecraft:stone", ...},
         "layers": [ [ "<row z=0>", "<row z=1>", ... ], ... ]}

    ``layers[0]`` ist die unterste Ebene; jede Zeile ist ``width`` Zeichen
    lang und repraesentiert die Z-Achse.
    """
    layers = data.get("layers")
    palette_raw = data.get("palette")
    if not isinstance(layers, list) or not layers or not isinstance(palette_raw, Mapping):
        raise StructureConversionError("Blueprint ohne 'layers'/'palette'")

    height = len(layers)
    if not isinstance(layers[0], list) or not layers[0]:
        raise StructureConversionError("Blueprint-Ebene ist leer")
    depth = len(layers[0])
    width = len(str(layers[0][0]))
    if width == 0 or depth == 0:
        raise StructureConversionError("Blueprint mit leerer Breite/Tiefe")

    declared = data.get("size")
    if isinstance(declared, list) and len(declared) == 3:
        declared = tuple(int(value) for value in declared)
        if declared != (width, height, depth):
            # Das Format ist nicht konsistent dokumentiert (mal Width/Depth/Height,
            # mal Width/Height/Depth). Die tatsaechlichen Arrays gewinnen.
            pass

    if max(width, height, depth) > MAX_DIMENSION:
        raise StructureConversionError(f"Blueprint-Dimension {(width, height, depth)} zu gross")

    palette: list[str] = [AIR]
    char_to_id: dict[str, int] = {}
    for key, block in palette_raw.items():
        name = _normalise_block(str(block))
        if name not in palette:
            palette.append(name)
        char_to_id[str(key)] = palette.index(name)

    grid = np.zeros((width, height, depth), dtype=np.int32)
    unknown = 0
    for y, layer in enumerate(layers):
        if not isinstance(layer, list) or len(layer) != depth:
            raise StructureConversionError(f"Blueprint-Ebene {y} hat {len(layer)} statt {depth} Zeilen")
        for z, row in enumerate(layer):
            row = str(row)
            if len(row) != width:
                raise StructureConversionError(
                    f"Blueprint-Zeile {y}/{z} hat {len(row)} statt {width} Zeichen"
                )
            for x, char in enumerate(row):
                if char in ". ":
                    continue  # Luft
                block_id = char_to_id.get(char)
                if block_id is None:
                    unknown += 1
                    continue
                grid[x, y, z] = block_id

    stats = ConversionStats(
        size=(width, height, depth),
        blocks=int(np.count_nonzero(grid)),
        palette_size=len(palette),
        block_entities=0,
        entities=0,
        regions=1,
        source_format="blueprint",
        unknown_chars=unknown,
    )
    return grid, palette, stats


def blueprint_to_schem(
    data: Mapping[str, Any],
    target: str | Path,
    *,
    name: str = "",
    author: str = "",
) -> ConversionStats:
    grid, palette, stats = load_blueprint(data)
    output_bytes = write_sponge_schem(
        target,
        grid=grid,
        palette=palette,
        data_version=DEFAULT_DATA_VERSION,
        name=name,
        author=author,
    )
    return ConversionStats(
        size=stats.size,
        blocks=stats.blocks,
        palette_size=stats.palette_size,
        block_entities=0,
        entities=0,
        regions=1,
        source_format=stats.source_format,
        output_bytes=output_bytes,
        unknown_chars=stats.unknown_chars,
    )


# ── Format-Erkennung (fuer Downloads ohne verlaessliche Endung) ────────────

def detect_structure_format(raw: bytes) -> str:
    """Erkennt ``litematic``, ``schem``, ``schematic``, ``zip`` oder ``unknown``."""
    if not raw:
        return "unknown"
    if raw[:4] == b"PK\x03\x04":
        return "zip"
    payload = raw
    if raw[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(raw)
        except OSError:
            return "unknown"
    try:
        root = File.parse(io.BytesIO(payload))
    except Exception:
        return "unknown"
    # Sponge-Dateien sind oft als {"": {"Schematic": {...}}} gewrappt.
    if "" in root and "Schematic" in root[""]:
        root = root[""]["Schematic"]
    elif "Schematic" in root:
        root = root["Schematic"]
    if "Regions" in root:
        return "litematic"
    blocks = root.get("Blocks")
    if isinstance(blocks, Mapping) and "Palette" in blocks:
        return "schem"
    if "Blocks" in root and "Palette" in root:
        return "schem"
    if "Palette" in root and "BlockData" in root:
        return "schem"
    if "Blocks" in root:
        return "schematic"
    return "unknown"


def read_structure_size(raw: bytes) -> tuple[int, int, int] | None:
    """Liest nur Width/Height/Length aus rohen Schematic-Bytes."""
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return None
    try:
        root = File.parse(io.BytesIO(raw))
    except Exception:
        return None
    if "" in root and "Schematic" in root[""]:
        root = root[""]["Schematic"]
    elif "Schematic" in root:
        root = root["Schematic"]
    if not all(key in root for key in ("Width", "Height", "Length")):
        return None
    return int(root["Width"]), int(root["Height"]), int(root["Length"])
