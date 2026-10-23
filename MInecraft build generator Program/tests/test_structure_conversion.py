"""Tests fuer den Mehrquellen-Scraper und den Litematic-Konverter.

Keine Netzwerkzugriffe: Litematica-/Blueprint-Dateien werden synthetisch
erzeugt, das Abfielder-HTML ist ein minimales Fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from nbtlib import File
from nbtlib.tag import Compound, Int, IntArray, List, LongArray, String

from app.litematic import (
    blueprint_to_schem,
    detect_structure_format,
    litematic_to_schem,
    read_structure_size,
)
from dataset import load_schematic
from scrape_structures import (
    AbfielderSource,
    MCSchematicsSource,
    clean_description,
    sidecar_prompts,
)


def _compound(values: dict) -> Compound:
    return Compound({key: Int(value) for key, value in values.items()})


def _palette(names: list[str]) -> List:
    return List[Compound](
        [Compound({"Name": String(name)}) for name in names]
    )


def _write_litematic(
    path: Path,
    *,
    size: tuple[int, int, int],
    position: tuple[int, int, int],
    palette: list[str],
    ids: list[int],
    contiguous: bool,
    tile_entities: list[dict] | None = None,
) -> None:
    """Synthetisches Litematica-NBT inkl. Padded-/Contiguous-Bitpacking."""
    bits = max(2, (len(palette) - 1).bit_length())
    volume = abs(size[0]) * abs(size[1]) * abs(size[2])
    assert len(ids) == volume

    if contiguous:
        long_count = (volume * bits + 63) // 64
        longs = [0] * long_count
        for index, block_id in enumerate(ids):
            bit = index * bits
            long_index, shift = divmod(bit, 64)
            longs[long_index] |= block_id << shift
            if shift + bits > 64:
                longs[long_index + 1] |= block_id >> (64 - shift)
    else:
        values_per_long = 64 // bits
        longs = [0] * ((volume + values_per_long - 1) // values_per_long)
        for index, block_id in enumerate(ids):
            longs[index // values_per_long] |= block_id << ((index % values_per_long) * bits)

    def signed(value: int) -> int:
        value &= (1 << 64) - 1
        return value - (1 << 64) if value >= (1 << 63) else value

    longs = [signed(value) for value in longs]

    entity_list = List[Compound]([])
    for entity in tile_entities or []:
        entity_list.append(
            Compound(
                {
                    "id": String(entity.get("id", "minecraft:chest")),
                    "x": Int(entity["x"]),
                    "y": Int(entity["y"]),
                    "z": Int(entity["z"]),
                }
            )
        )

    region = Compound(
        {
            "Position": _compound(dict(zip("xyz", position))),
            "Size": _compound(dict(zip("xyz", size))),
            "BlockStatePalette": _palette(palette),
            "BlockStates": LongArray(longs),
            "TileEntities": entity_list,
        }
    )
    root = File(
        {
            "Version": Int(7),
            "SubVersion": Int(1 if contiguous else 0),
            "MinecraftDataVersion": Int(3953),
            "Metadata": Compound({"Name": String("Test")}),
            "Regions": Compound({"Test": region}),
        }
    )
    root.save(str(path), gzipped=True)


@pytest.mark.parametrize("contiguous", [False, True])
def test_litematic_conversion_negative_size_and_block_entity(tmp_path, contiguous):
    # Position 1/1/1 + negative Groesse -2 => Min-Ecke 0/0/0.
    # ids in (y,z,x)-Reihenfolge; Index 1 ist (x=1,y=0,z=0) = chest.
    ids = [0, 2, 1, 1, 0, 0, 0, 0]
    src = tmp_path / "test.litematic"
    _write_litematic(
        src,
        size=(-2, -2, -2),
        position=(1, 1, 1),
        palette=["minecraft:air", "minecraft:stone", "minecraft:chest"],
        ids=ids,
        contiguous=contiguous,
        tile_entities=[{"x": 1, "y": 0, "z": 0}],
    )
    out = tmp_path / "test.schem"
    stats = litematic_to_schem(src, out, max_voxels=1000)

    assert stats.size == (2, 2, 2)
    assert stats.blocks == 3
    assert stats.block_entities == 1
    assert detect_structure_format(out.read_bytes()) == "schem"
    assert read_structure_size(out.read_bytes()) == (2, 2, 2)

    data = load_schematic(out)
    assert data.size == (2, 2, 2)
    # y,z,x-Reihenfolge: Index 1 => y=0,z=0,x=1
    assert data.blocks[1] == "minecraft:chest"
    assert data.block_entities[0]["x"] == 1
    assert data.block_entities[0]["y"] == 0
    assert data.block_entities[0]["z"] == 0


def test_litematic_contiguous_packing_with_multiple_longs(tmp_path):
    # 34 Farben => 6 Bit, 32 Voxel => 3 Longs (contiguous) statt 4 (padded).
    palette = ["minecraft:air"] + [f"minecraft:test_{index}" for index in range(33)]
    ids = [(index % 34) for index in range(32)]
    src = tmp_path / "contiguous.litematic"
    _write_litematic(
        src,
        size=(2, 2, 8),
        position=(0, 0, 0),
        palette=palette,
        ids=ids,
        contiguous=True,
    )
    out = tmp_path / "contiguous.schem"
    stats = litematic_to_schem(src, out, max_voxels=1000)
    assert stats.size == (2, 2, 8)
    assert stats.blocks == 31  # ein air-Block (id 0) in den 32 Voxeln


def test_blueprint_conversion_writes_sponge_schem(tmp_path):
    blueprint = {
        "size": [3, 2, 2],
        "palette": {"a": "minecraft:stone", "b": "minecraft:oak_planks"},
        "layers": [["aaa", "aaa"], ["bbb", "bbb"]],
    }
    out = tmp_path / "blueprint.schem"
    stats = blueprint_to_schem(blueprint, out, name="Test House")
    assert stats.size == (3, 2, 2)
    assert stats.blocks == 12

    data = load_schematic(out)
    assert data.size == (3, 2, 2)
    assert all(block == "minecraft:stone" for block in data.blocks[:6])
    assert all(block == "minecraft:oak_planks" for block in data.blocks[6:])


def test_blueprint_rejects_inconsistent_rows(tmp_path):
    blueprint = {
        "size": [3, 1, 2],
        "palette": {"a": "minecraft:stone"},
        "layers": [["aaa", "aa"]],
    }
    with pytest.raises(ValueError):
        blueprint_to_schem(blueprint, tmp_path / "bad.schem")


ABFIELDER_HTML = """
<html><head>
<meta property="og:description" content="A cozy test house created by TestBuilder. Free download on abfielder.com.">
</head><body>
<h1>🏡 Test House</h1>
<div class="product-description"><p>A cozy test house with two floors.</p></div>
<a class="product-tag" href="/Products/BrowseProducts.php?tag=1">House</a>
<a class="product-tag" href="/Products/BrowseProducts.php?tag=2">Medieval</a>
<a href="/Products/ProductDownloadThankYou.php?productId=123&amp;file=456">Download</a>
<a href="/Products/ProductDownloadThankYou.php?productId=123&amp;file=456&amp;convert=schem">Convert to .schem</a>
</body></html>
"""


def test_abfielder_product_parser_extracts_prompts_and_downloads():
    parsed = AbfielderSource().parse_product(ABFIELDER_HTML)
    assert parsed["title"] == "🏡 Test House"
    assert parsed["description"] == "A cozy test house with two floors."
    assert parsed["author"] == "TestBuilder"
    assert parsed["tags"] == ["House", "Medieval"]
    assert parsed["download_links"] == [
        "https://abfielder.com/Products/ProductDownloadThankYou.php?productId=123&file=456"
    ]
    assert parsed["premium"] is False


def test_mc_schematics_parser_extracts_public_download_and_provenance():
    html = '''<html><head>
      <meta property="og:title" content="Small House Schematic">
      <meta name="description" content="A small starter house.">
    </head><body><div class="byrow"><span>by <b>BuilderOne</b></span></div>
      <a class="btn dl" href="https://files.mc-schematics.com/b/123/house.schem">Download</a>
    </body></html>'''
    parsed = MCSchematicsSource().parse_product(
        html, "https://mc-schematics.com/s/small-house-123"
    )
    assert parsed == {
        "title": "Small House",
        "description": "A small starter house.",
        "author": "BuilderOne",
        "download_url": "https://files.mc-schematics.com/b/123/house.schem",
    }


def test_clean_description_drops_credits_and_generic_lines():
    assert clean_description("credits for (ente for ukrainian)") == ""
    assert clean_description("© Abfielder Ltd 2018-2025. All rights reserved.") == ""
    assert clean_description("A small medieval house") == "A small medieval house"


def test_sidecar_prompts_name_then_description():
    assert sidecar_prompts("House", "A cozy house") == ["House", "A cozy house"]
    assert sidecar_prompts("House", "House") == ["House"]
    assert sidecar_prompts("House", "", with_name=False) == []
