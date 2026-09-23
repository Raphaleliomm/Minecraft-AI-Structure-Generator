"""Regressionstests für die Trainings-Startkosten (RAM + Startzeit).

Jeder Test dokumentiert ein vorher kaputtes Verhalten:

  * B1  Zu grosse Strukturen wurden komplett dekodiert. Ein einziger
        1000x1000-Build (281 Mio. Voxel) kostete Minuten und mehrere GB RAM -
        obwohl das Ergebnis danach wegen ``max_voxels`` verworfen wurde.
  * B2  Jede Struktur blieb für die gesamte Trainingsdauer im RAM (bei 3000+
        Strukturen und Builds bis 281 Mio. Voxeln nicht tragbar).
  * B3  Augmentierung und Vokabular rotierten/trimmten pro Voxel in Python.
  * B4  Der Hidden-State-Cache (1.7 GB) wurde komplett in den RAM kopiert.
  * B5  Ein Legacy-Schematic (Blocks als ByteArray) liess den Dataset-Aufbau
        mit "TypeError: iteration over a 0-d array" abstürzen.
"""
from __future__ import annotations

import gzip
import random

import pytest
import torch
from nbtlib import File
from nbtlib.tag import ByteArray, Compound, List, Short, String

import dataset as ds
from dataset import (
    AIR,
    LazySchematicStore,
    MultiSourceSchematicDataset,
    PlacementTransform,
    SchematicData,
    VoxelTokenizer,
    _decode_palette_indices,
    _read_varints,
    _write_varints,
    blocks_to_grid,
    legal_placement_transforms,
    load_schematic,
    read_schematic_size,
    save_schem,
    trim_block_grid,
    trimmed_block_bounds,
    trimmed_block_size,
)


# ─── Hilfsfunktionen ───

def _write_structure(data_dir, name: str, size: tuple[int, int, int],
                     block_id: int = 1, prompt: str = "small hut"):
    data_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.zeros(size, dtype=torch.long)
    grid[0, 0, 0] = block_id
    save_schem(data_dir / f"{name}.schem", grid, [AIR, "minecraft:stone"])
    (data_dir / f"{name}.txt").write_text(prompt, encoding="utf-8")
    return data_dir / f"{name}.schem"


def _write_legacy_schematic(path, size: tuple[int, int, int], blocks: list[int]):
    """MCEdit-Format: 'Blocks' ist ein ByteArray, kein Compound mit Palette."""
    root = Compound({
        "Width": Short(size[0]),
        "Height": Short(size[1]),
        "Length": Short(size[2]),
        "Blocks": ByteArray(blocks),
        "Data": ByteArray([0] * len(blocks)),
        "Materials": String("Alpha"),
        "Entities": List[Compound]([]),
        "TileEntities": List[Compound]([]),
    })
    tmp = path.with_suffix(".nbt")
    File({"": Compound({"Schematic": root})}).save(str(tmp), gzipped=False)
    with open(tmp, "rb") as fin, gzip.open(path, "wb") as fout:
        fout.write(fin.read())
    tmp.unlink()


def _old_legal_placement_transforms(schematic: SchematicData,
                                    target_size: tuple[int, int, int],
                                    diversity: int = 1) -> list[PlacementTransform]:
    """Die frühere Implementierung (Gitter trimmen und rotieren) als Referenz."""
    base_grid = trim_block_grid(blocks_to_grid(schematic.blocks_important, schematic.size))
    transforms: list[PlacementTransform] = []
    seen = set()
    for rotation in range(4):
        rotated = ds.rotate_block_grid_y(base_grid, rotation)
        sx = len(rotated)
        sy = len(rotated[0]) if sx else 0
        sz = len(rotated[0][0]) if sx and sy else 0
        tx, ty, tz = target_size
        if sx > tx or sy > ty or sz > tz:
            continue
        for ox in ds._axis_positions(tx - sx, diversity):
            for oz in ds._axis_positions(tz - sz, diversity):
                key = (rotation, (ox, 0, oz), (sx, sy, sz))
                if key in seen:
                    continue
                seen.add(key)
                transforms.append(PlacementTransform(rotation=rotation, offset=(ox, 0, oz)))
    if not transforms:
        return []
    return transforms


# ─── B1: Groessenpruefung vor dem Dekodieren ───

def test_oversize_structures_are_skipped_before_decoding(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_structure(data_dir, "big", (5, 5, 5))
    _write_structure(data_dir, "small", (2, 2, 2), prompt="tiny hut")

    decoded: list[int] = []
    original = ds._decode_palette_indices

    def spy(raw, expected):
        decoded.append(expected)
        return original(raw, expected)

    monkeypatch.setattr(ds, "_decode_palette_indices", spy)

    dataset = MultiSourceSchematicDataset([data_dir], target_size=(4, 4, 4),
                                          max_voxels=100, augmentation_diversity=1)

    # Nur die kleine Struktur wurde dekodiert (8 Voxel), nicht die 125-Voxel-Box.
    assert decoded == [8]
    assert dataset.oversize_voxels == 125
    assert [path.name for path in dataset.skipped_examples] == ["big.schem"]
    assert {example.path.name for example in dataset.examples} == {"small.schem"}


def test_read_schematic_size_needs_no_block_data(tmp_path):
    path = _write_structure(tmp_path / "data", "hut", (3, 4, 5))
    assert read_schematic_size(path) == (3, 4, 5)
    assert load_schematic(path, max_voxels=60).num_voxels == 60
    with pytest.raises(ds.SchematicTooLarge):
        load_schematic(path, max_voxels=59)


def test_schematic_too_large_is_a_value_error(tmp_path):
    path = _write_structure(tmp_path / "data", "hut", (2, 2, 2))
    with pytest.raises(ValueError) as info:
        load_schematic(path, max_voxels=4)
    assert "ueberschreitet das Limit" in str(info.value)


# ─── B2: Speicherbudget statt "alles behalten" ───

def test_lazy_schematic_store_stays_within_its_budget(tmp_path):
    data_dir = tmp_path / "data"
    for index in range(3):
        _write_structure(data_dir, f"hut{index}", (2, 2, 2), prompt=f"hut {index}")

    dataset = MultiSourceSchematicDataset(
        [data_dir], target_size=(4, 4, 4), max_voxels=1000,
        schematic_cache_voxels=8)  # Budget fuer genau eine Struktur

    assert len(dataset.schematics) == 3
    assert [schematic.size for schematic in dataset.schematics] == [(2, 2, 2)] * 3
    assert dataset.schematics.cached_voxels <= 8
    assert dataset.schematics[0].num_voxels == 8
    assert dataset.schematics[-1].blocks[0] == "minecraft:stone"


def test_lazy_schematic_store_without_cache_reloads(tmp_path):
    data_dir = tmp_path / "data"
    _write_structure(data_dir, "hut", (2, 2, 2))

    dataset = MultiSourceSchematicDataset(
        [data_dir], target_size=(4, 4, 4), max_voxels=1000, schematic_cache_voxels=0)

    assert dataset.schematics.cached_voxels == 0
    before = dataset.schematics.load_count
    assert dataset.schematics[0].num_voxels == 8
    assert dataset.schematics[0].num_voxels == 8
    assert dataset.schematics.load_count == before + 2  # kein Cache -> zweimal laden


def test_lazy_schematic_store_is_a_sequence(tmp_path):
    store = LazySchematicStore(max_voxels=1000, cache_voxels=8)
    path = _write_structure(tmp_path / "data", "hut", (2, 2, 2))
    index = store.add(path)
    assert index == 0 and len(store) == 1
    assert store.path(0) == path
    assert store[0].size == (2, 2, 2)
    assert [s.size for s in store] == [(2, 2, 2)]
    assert [s.size for s in store[0:1]] == [(2, 2, 2)]
    with pytest.raises(IndexError):
        store[5]


# ─── B3: Bounding-Box statt Gitterrotation ───

def test_trimmed_bounds_matches_grid_trim():
    rng = random.Random(2026)
    choices = [AIR, AIR, "minecraft:stone", "minecraft:oak_stairs[facing=north]"]
    for size in [(1, 1, 1), (4, 3, 5), (3, 2, 3), (2, 4, 2)]:
        for _ in range(5):
            blocks = [rng.choice(choices) for _ in range(size[0] * size[1] * size[2])]
            grid = trim_block_grid(blocks_to_grid(blocks, size))
            reference = (len(grid), len(grid[0]), len(grid[0][0]))
            schematic = SchematicData(blocks, list(blocks), size=size)
            assert trimmed_block_size(schematic) == reference
            bounds = trimmed_block_bounds(blocks, size)
            if bounds is None:
                # nur Luft -> das alte trim_block_grid lieferte einen 1x1x1-Wuerfel
                assert reference == (1, 1, 1)
                continue
            x0, y0, z0, x1, y1, z1 = bounds
            assert (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1) == reference


def test_trimmed_bounds_of_air_is_empty():
    size = (3, 3, 3)
    blocks = [AIR] * 27
    assert trimmed_block_bounds(blocks, size) is None
    assert trimmed_block_size(SchematicData(blocks, list(blocks), size=size)) == (1, 1, 1)


def test_legal_placement_transforms_match_reference():
    cases = [
        SchematicData([AIR, "minecraft:stone"], [AIR, "minecraft:stone"], size=(2, 1, 1)),
        SchematicData(["minecraft:stone"] * 6, ["minecraft:stone"] * 6, size=(3, 1, 2)),
        SchematicData(["minecraft:stone"] * 16, ["minecraft:stone"] * 16, size=(4, 1, 4)),
        SchematicData([AIR] * 8, [AIR] * 8, size=(2, 2, 2)),
        SchematicData(["minecraft:stone"] * 27, ["minecraft:stone"] * 27, size=(3, 3, 3)),
    ]
    for schematic in cases:
        for diversity in (0, 1, 2):
            assert legal_placement_transforms(schematic, (4, 4, 4), diversity=diversity) == \
                _old_legal_placement_transforms(schematic, (4, 4, 4), diversity=diversity)
        # zu gross fuer das Zielgitter -> keine Varianten
        assert legal_placement_transforms(schematic, (2, 2, 2)) == \
            _old_legal_placement_transforms(schematic, (2, 2, 2))


def test_vocabulary_contains_all_rotated_states():
    stairs = "minecraft:oak_stairs[facing=north,waterlogged=false,age=2]"
    blocks = [stairs, AIR, "minecraft:oak_log[axis=x]", AIR]
    schematic = SchematicData(blocks, list(blocks), blocks_important=list(blocks), size=(2, 2, 1))

    tokenizer = VoxelTokenizer()
    tokenizer.fit([schematic], target_size=(2, 2, 1))

    for direction in ("north", "west", "south", "east"):
        assert f"minecraft:oak_stairs[facing={direction},waterlogged=false]" in tokenizer.block_to_id
    assert "minecraft:oak_log[axis=x]" in tokenizer.block_to_id
    assert "minecraft:oak_log[axis=z]" in tokenizer.block_to_id
    assert AIR in tokenizer.block_to_id


def test_observer_api_matches_fit():
    stairs = "minecraft:oak_stairs[facing=north]"
    blocks = [stairs, stairs]
    schematic = SchematicData(blocks, list(blocks), blocks_important=list(blocks), size=(2, 1, 1))

    direct = VoxelTokenizer()
    direct.fit([schematic], (2, 1, 1))

    streaming = VoxelTokenizer()
    streaming.reset()
    streaming.observe(schematic)
    streaming.finalize()
    assert direct.block_to_id == streaming.block_to_id
    assert direct.id_to_block == streaming.id_to_block


def test_varint_decoder_matches_reference():
    rng = random.Random(7)
    cases = [
        [0] * 10,
        [1, 2, 3, 127],
        [128, 255, 256, 16383, 16384],          # mehrbyte-Varints
        [rng.randrange(0, 500) for _ in range(500)],
    ]
    for values in cases:
        raw = _write_varints(values)
        assert _read_varints(raw, len(values)) == values
        assert _decode_palette_indices(raw, len(values)).tolist() == values

    # abgeschnittene Daten werden erkannt
    with pytest.raises(ValueError):
        _decode_palette_indices([0x80, 0x80], 1)


# ─── B4: Hidden-State-Cache per mmap ───

def test_cache_loads_identically_with_and_without_mmap(tmp_path):
    from app import hidden_state_cache

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    hidden = torch.randn(12, 8, dtype=torch.float16)
    masks = torch.ones(12, dtype=torch.long)
    lengths = torch.tensor([2, 10], dtype=torch.long)
    torch.save({"hidden_states": hidden, "attention_masks": masks, "lengths": lengths,
                "layout": "ragged"}, cache_dir / "data.pt")
    (cache_dir / "metadata.json").write_text(
        '{"encoder_name": "Stub", "hidden_dim": 8, "num_samples": 2}', encoding="utf-8")

    plain = hidden_state_cache.load_hidden_states_raw(cache_dir, mmap=False)
    mapped = hidden_state_cache.load_hidden_states_raw(cache_dir, mmap=True)

    assert torch.equal(plain["hidden_states"], mapped["hidden_states"])
    assert torch.equal(plain["attention_masks"], mapped["attention_masks"])
    assert plain["lengths"].tolist() == mapped["lengths"].tolist() == [2, 10]
    assert mapped["layout"] == "ragged"

    assert hidden_state_cache._mmap_enabled(False) is False
    assert hidden_state_cache._mmap_enabled(True) is True


def test_cache_load_survives_legacy_padded_layout(tmp_path):
    """Alte Caches (gepolstert, ohne 'lengths') muessen weiter funktionieren."""
    from app import hidden_state_cache

    cache_dir = tmp_path / "legacy_cache"
    cache_dir.mkdir()
    torch.save({
        "hidden_states": torch.zeros((2, 4, 8), dtype=torch.float32),
        "attention_masks": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
    }, cache_dir / "data.pt")

    data = hidden_state_cache.load_hidden_states_raw(cache_dir)
    assert data["layout"] == "padded"
    assert data["lengths"].tolist() == [2, 4]
    assert data["hidden_states"].shape == (6, 8)


# ─── B5: Legacy-Schematic (Blocks als ByteArray) ───

def test_legacy_bytearray_schematic_loads(tmp_path):
    path = tmp_path / "legacy.schem"
    _write_legacy_schematic(path, (2, 1, 2), [1, 0, 0, 0])

    assert read_schematic_size(path) == (2, 1, 2)
    schematic = load_schematic(path, max_voxels=100)
    assert schematic.size == (2, 1, 2)
    assert schematic.blocks[0] == "legacy:1"
    assert schematic.blocks[1] == "legacy:0"

    # Ohne die _has_key-Pruefung brach der Aufbau hier mit
    # "TypeError: iteration over a 0-d array" ab.
    (tmp_path / "legacy.txt").write_text("legacy hut", encoding="utf-8")
    dataset = MultiSourceSchematicDataset([tmp_path], target_size=(2, 1, 2), max_voxels=100)
    assert len(dataset) >= 1


def test_legacy_schematic_can_be_skipped_by_size(tmp_path):
    path = tmp_path / "legacy.schem"
    _write_legacy_schematic(path, (2, 2, 2), [1] * 8)
    with pytest.raises(ds.SchematicTooLarge):
        load_schematic(path, max_voxels=7)


# ─── Rueckmeldung beim Aufbau ───

def test_dataset_reports_construction_progress(tmp_path):
    data_dir = tmp_path / "data"
    for index in range(3):
        _write_structure(data_dir, f"hut{index}", (2, 2, 2), prompt=f"hut {index}")

    calls: list[tuple[int, int]] = []
    MultiSourceSchematicDataset([data_dir], target_size=(4, 4, 4), max_voxels=1000,
                                progress_callback=lambda done, total: calls.append((done, total)))

    assert calls, "ohne Rueckmeldung sieht der Aufbau wie ein Haenger aus"
    done, total = calls[-1]
    assert done == total == 3
