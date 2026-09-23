"""Prompt-Sidecar-Format: Name + Beschreibung = zwei Trainingsbeispiele.

Abgedeckt:
  * dataset.split_prompt_text() liest das neue Format (Name zuerst) und das
    Altformat des Scrapers (name:/author:/source:/url:-Zeilen ohne Name-Prompt)
  * prompts.json darf pro Struktur eine Prompt-Liste enthalten
  * scrape_phantommarket schreibt die Sidecars (sidecar_prompts / writer)
  * --rebuild-sidecars migriert Altformat-Sidecars offline
"""
from __future__ import annotations

import json

import torch

from dataset import discover_examples, save_schem, split_prompt_text
from scrape_phantommarket import (
    rewrite_sidecars,
    sidecar_prompt_blocks,
    sidecar_prompts,
    write_prompt_sidecar,
)


NEW_SIDECAR = "Medieval House\n---\nA cozy medieval house for survival\n"
LEGACY_SIDECAR = (
    "name: Medieval House\n"
    "author: somebody\n"
    "source: phantommarket\n"
    "url: https://market.phantom-node.com/post/medieval-house\n"
    "---\n"
    "A cozy medieval house for survival\n"
)


def _grid() -> torch.Tensor:
    grid = torch.zeros((2, 2, 2), dtype=torch.long)
    grid[0, 0, 0] = 1
    return grid


# ─── Sidecar lesen (dataset) ───
def test_split_prompt_text_keeps_name_and_description():
    assert split_prompt_text(NEW_SIDECAR) == [
        "Medieval House", "A cozy medieval house for survival"]


def test_split_prompt_text_drops_legacy_metadata_section():
    assert split_prompt_text(LEGACY_SIDECAR) == ["A cozy medieval house for survival"]


# ─── Sidecar schreiben/lesen (scraper) ───
def test_sidecar_prompt_blocks_supports_both_formats():
    expected = ["Medieval House", "A cozy medieval house for survival"]
    assert sidecar_prompt_blocks(NEW_SIDECAR) == expected
    assert sidecar_prompt_blocks(LEGACY_SIDECAR) == expected


def test_sidecar_prompts_puts_name_first_and_avoids_duplicates():
    record = {"title": "Medieval House", "prompt": "A cozy medieval house"}
    assert sidecar_prompts(record) == ["Medieval House", "A cozy medieval house"]
    assert sidecar_prompts(record, with_name_prompt=False) == ["A cozy medieval house"]
    # Der erste Block ist immer der reine Name - auch wenn der Prompt ihn nennt
    long_prompt = {"title": "Medieval House", "prompt": "Medieval House near the sea"}
    assert sidecar_prompts(long_prompt) == ["Medieval House", "Medieval House near the sea"]
    # --prompt-mode title bzw. Build ohne Beschreibung: nur ein Block
    assert sidecar_prompts({"title": "Medieval House", "prompt": "Medieval House"}) == [
        "Medieval House"]


def test_written_sidecar_is_read_back_by_the_dataset(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_schem(data_dir / "house.schem", _grid(), ["minecraft:air", "minecraft:stone"])

    record = {"title": "Medieval House", "prompt": "A cozy medieval house"}
    write_prompt_sidecar(data_dir / "house.txt", sidecar_prompts(record))

    found = discover_examples(data_dir)
    assert [prompt for _, prompt in found] == ["Medieval House", "A cozy medieval house"]


def test_sidecar_name_with_separator_stays_one_prompt(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_schem(data_dir / "arena.schem", _grid(), ["minecraft:air", "minecraft:stone"])

    record = {"title": "Arena------Cobblemon", "prompt": "an aquatic arena"}
    write_prompt_sidecar(data_dir / "arena.txt", sidecar_prompts(record))

    text = (data_dir / "arena.txt").read_text(encoding="utf-8")
    assert split_prompt_text(text) == ["Arena Cobblemon", "an aquatic arena"]
    # zweiter Migrationslauf laesst die Datei unveraendert
    assert rewrite_sidecars(data_dir) == (0, 1)


# ─── prompts.json mit Prompt-Liste ───
def test_prompt_map_lists_create_one_example_per_prompt(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    blocks = ["minecraft:air", "minecraft:stone"]
    save_schem(data_dir / "hut.schem", _grid(), blocks)
    save_schem(data_dir / "tower.schem", _grid(), blocks)
    (data_dir / "prompts.json").write_text(
        json.dumps(
            {
                "hut.schem": "a cozy hut",  # Altformat: einzelner Prompt
                "tower.schem": ["Stone Tower", "a tall stone tower"],
            }
        ),
        encoding="utf-8",
    )

    found = discover_examples(data_dir)
    assert [prompt for _, prompt in found] == [
        "a cozy hut", "Stone Tower", "a tall stone tower"]
    assert [path.name for path, _ in found] == [
        "hut.schem", "tower.schem", "tower.schem"]


# ─── Migration (--rebuild-sidecars) ───
def test_rebuild_sidecars_migrates_legacy_format(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_schem(data_dir / "house.schem", _grid(), ["minecraft:air", "minecraft:stone"])
    (data_dir / "house.txt").write_text(LEGACY_SIDECAR, encoding="utf-8")
    # Sidecar ohne (noch) geladene Struktur wird ebenfalls migriert
    (data_dir / "orphan.txt").write_text(LEGACY_SIDECAR, encoding="utf-8")
    # .txt ohne Scraper-Format bleibt unangetastet
    (data_dir / "notes.txt").write_text("just my own notes\n", encoding="utf-8")

    changed, unchanged = rewrite_sidecars(data_dir)
    assert (changed, unchanged) == (2, 0)
    assert (data_dir / "house.txt").read_text(encoding="utf-8") == NEW_SIDECAR
    assert (data_dir / "orphan.txt").read_text(encoding="utf-8") == NEW_SIDECAR
    assert (data_dir / "notes.txt").read_text(encoding="utf-8") == "just my own notes\n"

    # erneuter Lauf laesst die migrierten Sidecars unveraendert
    assert rewrite_sidecars(data_dir) == (0, 1)
    assert (data_dir / "orphan.txt").read_text(encoding="utf-8") == NEW_SIDECAR
    assert [prompt for _, prompt in discover_examples(data_dir)] == [
        "Medieval House", "A cozy medieval house for survival"]
