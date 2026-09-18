#!/usr/bin/env python3
"""Validiert .schem-Dateien im Trainingsdaten-Ordner - ohne nbtlib.

Prüft je Datei:
  * gültiges gzip-NBT mit Root-Compound "Schematic"
  * Version / Width / Height / Length vorhanden
  * Sponge-Format v2 (Palette/BlockData) oder v3 (Blocks.Palette/Data)
  * BlockData lässt sich zu GENAU width*height*length Block-IDs dekodieren
  * jede ID liegt innerhalb der Palette (sonst scheitert dataset.load_schematic)
  * Anteil Nicht-Luft-Blöcke (Plausibilität: 0 % oder 100 % sind verdächtig)

Damit lassen sich gescrapte/konvertierte Datensätze prüfen, ohne die
komplette Torch-Umgebung zu importieren.

Beispiele:
    python validate_schematics.py                 # alle .schem im Skript-Ordner
    python validate_schematics.py "../Trainingsdaten/phantommarket"
    python validate_schematics.py datei.schem
"""
from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

NAMES = {
    0: "END", 1: "Byte", 2: "Short", 3: "Int", 4: "Long", 5: "Float", 6: "Double",
    7: "ByteArray", 8: "String", 9: "List", 10: "Compound", 11: "IntArray", 12: "LongArray",
}


class NbtReader:
    """Minimaler NBT-Leser (big-endian) für die Validierung."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, count: int) -> bytes:
        chunk = self.data[self.pos : self.pos + count]
        if len(chunk) != count:
            raise EOFError(f"unerwartetes Dateiende bei {self.pos} (+{count})")
        self.pos += count
        return chunk

    def number(self, fmt: str, size: int):
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        return self.take(self.number(">H", 2)).decode("utf-8", "replace")

    def payload(self, tag_type: int):
        if tag_type in (1, 2, 3, 4, 5, 6):
            fmt, size = {
                1: (">b", 1), 2: (">h", 2), 3: (">i", 4),
                4: (">q", 8), 5: (">f", 4), 6: (">d", 8),
            }[tag_type]
            return self.number(fmt, size)
        if tag_type == 7:
            return self.take(self.number(">i", 4))
        if tag_type == 8:
            return self.string()
        if tag_type == 9:
            item_type = self.number(">b", 1)
            length = self.number(">i", 4)
            return [self.payload(item_type) for _ in range(length)]
        if tag_type == 10:
            compound: dict[str, object] = {}
            while True:
                child = self.number(">b", 1)
                if child == 0:
                    return compound
                # Wichtig: Name zuerst lesen - bei "compound[name] = payload(child)"
                # würde Python die rechte Seite vorher auswerten.
                child_name = self.string()
                compound[child_name] = self.payload(child)
        if tag_type == 11:
            length = self.number(">i", 4)
            return [self.number(">i", 4) for _ in range(length)]
        if tag_type == 12:
            length = self.number(">i", 4)
            return [self.number(">q", 8) for _ in range(length)]
        raise ValueError(f"unbekannter Tag-Typ {NAMES.get(tag_type, tag_type)}")


def load_nbt(path: Path) -> tuple[str, dict, int]:
    """Liest das gzip-NBT und gibt (Root-Name, Root-Compound, Padding-Bytes) zurück."""
    data = gzip.open(path, "rb").read()
    reader = NbtReader(data)
    root_type = reader.number(">b", 1)
    root_name = reader.string()
    if root_type != 10:
        raise ValueError(f"Root ist kein Compound (Typ {root_type})")
    root = reader.payload(root_type)
    padding = len(data) - reader.pos
    if padding:
        # Manche Exporter hängen NUL-Bytes an; nbtlib ignoriert sie (verifiziert).
        if data[reader.pos:] != b"\x00" * padding:
            raise ValueError(f"{padding} unerwartete Bytes nach dem NBT-Baum")
    return root_name, root, padding


def decode_varints(raw: bytes, expected: int) -> list[int]:
    """Sponge-VarInts: niederwertige 7 Bit, High-Bit = Fortsetzung."""
    values: list[int] = []
    value = 0
    shift = 0
    for byte in raw:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 35:
                raise ValueError("ungueltiger VarInt in BlockData")
        else:
            values.append(value)
            value = 0
            shift = 0
            if len(values) == expected:
                break
    if len(values) != expected:
        raise ValueError(f"{len(values)} Block-IDs dekodiert, {expected} erwartet")
    return values


def check(path: Path) -> str:
    root_name, root, padding = load_nbt(path)
    node = root.get("Schematic") if isinstance(root.get("Schematic"), dict) else root
    if "Blocks" in node and isinstance(node["Blocks"], dict):  # Sponge v3
        palette = node["Blocks"].get("Palette")
        block_data = node["Blocks"].get("Data")
        layout = "Sponge v3 (Blocks.Palette/Data)"
    else:  # Sponge v2
        palette = node.get("Palette")
        block_data = node.get("BlockData")
        layout = "Sponge v2 (Palette/BlockData)"
    if not isinstance(palette, dict) or block_data is None:
        raise ValueError("weder Palette/BlockData (v2) noch Blocks (v3) gefunden")

    width, height, length = int(node["Width"]), int(node["Height"]), int(node["Length"])
    voxels = width * height * length
    ids = decode_varints(bytes(block_data), voxels)
    max_id = max(ids)
    if max_id >= len(palette):
        raise ValueError(f"Block-ID {max_id} liegt ausserhalb der Palette ({len(palette)})")
    id_to_name = {int(value): str(key) for key, value in palette.items()}
    solid = sum(1 for block_id in ids if id_to_name[block_id] != "minecraft:air")

    return (
        f"OK   {path.name}\n"
        f"       Root='{root_name}' {layout} Version={node.get('Version')} "
        f"DataVersion={node.get('DataVersion')}"
        f"{f' Padding={padding}B' if padding else ''}\n"
        f"       {width}x{height}x{length} = {voxels} Bloecke | Palette={len(palette)} | "
        f"BlockData={len(block_data)} B -> {len(ids)} IDs, max {max_id} | "
        f"nicht-Luft={solid} ({solid / voxels:.0%})"
    )


def collect_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        candidate = Path(item)
        paths.extend(sorted(candidate.glob("*.schem")) if candidate.is_dir() else [candidate])
    if not paths:
        paths = sorted(Path(__file__).parent.glob("*.schem"))
    return paths


def main() -> int:
    paths = collect_paths(sys.argv[1:])
    failures = 0
    for path in paths:
        try:
            print(check(path))
        except Exception as exc:  # noqa: BLE001 - Validierung soll jede Datei melden
            failures += 1
            print(f"FEHLER {path.name}\n       {exc!r}")
    print(f"\n{len(paths) - failures}/{len(paths)} Dateien valide")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())