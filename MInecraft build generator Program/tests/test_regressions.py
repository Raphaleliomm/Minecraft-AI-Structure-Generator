"""Regression tests for the bugs fixed in the review.

Each test documents one previously broken behaviour:
  * A1  "air" substring filter hid every *_stairs block
  * A2  hidden state cache was matched per example instead of per prompt
  * A3  sampling steps were not clamped to the trained timesteps
  * A4  data discovery ignored nested folders / crashed on missing folders
  * A5  save_schem mirrored block states (swap_directions) on export only
  * A6  hidden state cache padded every prompt to the longest prompt (CPU OOM)
  * B3  model renaming accepted path traversal
"""
from __future__ import annotations

import json

import torch

from app import hidden_state_cache
from app.diffusion_model import clamp_num_steps
from app.model_manager import is_valid_model_name
from app.voxel_preview import is_air_block
from dataset import (
    MultiSourceSchematicDataset,
    collate_context_batch,
    discover_examples,
    discover_prompt_files,
    save_schem,
    load_schematic,
)


# ─── A1 ───
def test_stairs_are_not_treated_as_air():
    for name in ("minecraft:oak_stairs", "minecraft:stone_stairs[facing=north]",
                 "minecraft:deepslate_tile_stairs[half=top]"):
        assert not is_air_block(name), name

    for name in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air", "air"):
        assert is_air_block(name), name


# ─── A2 ───
def test_hidden_state_cache_matches_multi_prompt_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    grid = torch.zeros((2, 2, 2), dtype=torch.long)
    grid[0, 0, 0] = 1
    save_schem(data_dir / "hut.schem", grid, ["minecraft:air", "minecraft:stone"])
    (data_dir / "hut.txt").write_text(
        "small wooden hut\n---\na large stone tower with a flag\n", encoding="utf-8")

    torch.save(
        {
            "hidden_states": torch.zeros((2, 4, 8), dtype=torch.float32),
            "attention_masks": torch.ones((2, 4), dtype=torch.long),
        },
        cache_dir / "data.pt",
    )
    (cache_dir / "metadata.json").write_text(json.dumps({
        "encoder_name": "Test",
        "hidden_dim": 8,
        "num_samples": 2,
        "files": [
            {"schem": "hut.schem", "prompt_index": 0},
            {"schem": "hut.schem", "prompt_index": 1},
        ],
    }), encoding="utf-8")

    dataset = MultiSourceSchematicDataset.with_cache(
        [str(data_dir)], cache_dir, target_size=(4, 4, 4), augmentation_diversity=2)

    assert len(dataset) > 2, "augmentation should create several examples per prompt"
    by_prompt = {}
    for example, index in zip(dataset.examples, dataset.hidden_index):
        by_prompt.setdefault(example.prompt, set()).add(index)
    # every augmented variant of a prompt shares the cache row of that prompt
    assert by_prompt["small wooden hut"] == {0}
    assert by_prompt["a large stone tower with a flag"] == {1}


# ─── A3 ───
def test_sampling_steps_are_clamped():
    assert clamp_num_steps(None, 50) == 50
    assert clamp_num_steps(200, 50) == 50
    assert clamp_num_steps(20, 50) == 20
    assert clamp_num_steps(0, 50) == 50
    assert clamp_num_steps(-5, 50) == 1


# ─── A4 ───
def test_discover_examples_is_recursive_and_tolerant(tmp_path):
    nested = tmp_path / "data" / "category one"
    nested.mkdir(parents=True)
    grid = torch.zeros((2, 2, 2), dtype=torch.long)
    save_schem(nested / "hut.schem", grid, ["minecraft:air", "minecraft:stone"])
    (nested / "hut.txt").write_text("source: test\n---\na cozy hut\n", encoding="utf-8")

    top_level = tmp_path / "data" / "tower.schem"
    save_schem(top_level, grid, ["minecraft:air", "minecraft:stone"])
    (tmp_path / "data" / "prompts.json").write_text(
        json.dumps({"tower.schem": "a tall tower"}), encoding="utf-8")

    found = discover_examples(tmp_path / "data")
    assert sorted(prompt for _, prompt in found) == ["a cozy hut", "a tall tower"]

    # missing directories are skipped instead of raising FileNotFoundError
    assert discover_examples(tmp_path / "does-not-exist") == []

    schem_files, txt_files = discover_prompt_files([tmp_path / "data"])
    assert [path.name for path in schem_files] == ["hut.schem"]
    assert [path.name for path in txt_files] == ["hut.txt"]


# ─── A5 ───
def test_save_schem_keeps_block_states(tmp_path):
    grid = torch.zeros((1, 1, 1), dtype=torch.long)
    grid[0, 0, 0] = 1
    out = tmp_path / "stairs.schem"
    save_schem(out, grid, ["minecraft:air", "minecraft:oak_stairs[facing=north,waterlogged=false]"])

    loaded = load_schematic(out)
    assert loaded.blocks == ["minecraft:oak_stairs[facing=north,waterlogged=false]"]


# ─── A6 ───
class _StubEncoder:
    """Minimal stand-in for FrozenTransformerEncoder (whitespace tokenizer)."""

    def __init__(self, hidden_dim: int = 8):
        self.hidden_dim = hidden_dim
        self.hf_id = "stub/encoder"
        self.device = torch.device("cpu")
        self.max_length = 512
        self.tokenizer = self._tokenize

    @staticmethod
    def _tokenize(texts, padding=False, truncation=True, max_length=512):
        return {"input_ids": [text.split()[:max_length] for text in texts]}

    def __call__(self, texts):
        seq_len = max(len(ids) for ids in self._tokenize(texts)["input_ids"])
        return {
            "last_hidden_state": torch.zeros((1, seq_len, self.hidden_dim), dtype=torch.float32),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
        }


def _write_two_prompt_structure(data_dir):
    """One structure with a short and a long prompt; returns both prompts."""
    short_prompt = "small hut"
    long_prompt = " ".join(f"word{i}" for i in range(10))
    data_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.zeros((2, 2, 2), dtype=torch.long)
    grid[0, 0, 0] = 1
    save_schem(data_dir / "hut.schem", grid, ["minecraft:air", "minecraft:stone"])
    (data_dir / "hut.txt").write_text(f"{short_prompt}\n---\n{long_prompt}\n", encoding="utf-8")
    return short_prompt, long_prompt


def _write_stub_cache(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    short_prompt, long_prompt = _write_two_prompt_structure(data_dir)
    monkeypatch.setattr(hidden_state_cache, "CACHE_DIR", tmp_path / "cache")
    schem_files, txt_files = discover_prompt_files([data_dir])
    cache_dir = hidden_state_cache.compute_hidden_states(
        _StubEncoder(hidden_dim=8), "Stub", schem_files, txt_files)
    return data_dir, cache_dir, short_prompt, long_prompt


def test_hidden_state_cache_is_unpadded(tmp_path, monkeypatch):
    """A6: the cache must not pad every prompt to the longest prompt of the set.

    Padding to the global maximum is what killed "Compute Hidden States" with
    "DefaultCPUAllocator: not enough memory" after a few thousand prompts: the
    array needs max_seq_len x num_prompts x hidden_dim floats - and before this
    fix three copies of it were held at the same time.
    """
    _data_dir, cache_dir, short_prompt, long_prompt = _write_stub_cache(tmp_path, monkeypatch)

    data = torch.load(cache_dir / "data.pt", map_location="cpu")
    assert data["layout"] == "ragged"
    assert data["lengths"].tolist() == [len(short_prompt.split()), len(long_prompt.split())]
    # 12 tokens stored - not two rows padded to 10 tokens each (20 tokens)
    assert data["hidden_states"].shape == (12, 8)
    assert data["attention_masks"].tolist() == [1] * 12

    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["layout"] == "ragged"
    assert metadata["total_tokens"] == 12
    assert metadata["max_seq_len"] == len(long_prompt.split())
    assert metadata["num_samples"] == 2
    assert metadata["hidden_dim"] == 8


def test_cached_rows_stay_ragged_and_collate_pads_per_batch(tmp_path, monkeypatch):
    """A6: dataset rows stay unpadded; only collate_context_batch pads a batch."""
    from torch.utils.data import DataLoader

    data_dir, cache_dir, _short, _long = _write_stub_cache(tmp_path, monkeypatch)

    dataset = MultiSourceSchematicDataset.with_cache(
        [str(data_dir)], cache_dir, target_size=(4, 4, 4), augmentation_diversity=1)

    assert dataset.hidden_lengths.tolist() == [2, 10]
    assert dataset.hidden_states.shape == (12, 8)  # flat, unpadded
    row_lengths = sorted({int(dataset[i]["hidden_states"].shape[0]) for i in range(len(dataset))})
    assert row_lengths == [2, 10]  # every example keeps its own prompt length

    # A short and a long prompt in one batch: padded to the batch maximum only
    batch = collate_context_batch([
        {"hidden_states": torch.ones((2, 8)), "attention_masks": torch.ones(2, dtype=torch.long),
         "prompt_ids": torch.tensor([1])},
        {"hidden_states": torch.ones((10, 8)), "attention_masks": torch.ones(10, dtype=torch.long),
         "prompt_ids": torch.tensor([2])},
    ])
    assert batch["hidden_states"].shape == (2, 10, 8)  # 10, not max_seq_len
    assert batch["attention_masks"][0].tolist() == [1, 1] + [0] * 8
    assert batch["attention_masks"][1].tolist() == [1] * 10
    assert int(batch["prompt_ids"].sum()) == 3

    # Same collate function the training workers pass to their DataLoader
    loader = DataLoader(dataset, batch_size=max(1, len(dataset)), collate_fn=collate_context_batch)
    first = next(iter(loader))
    assert first["hidden_states"].shape == (len(dataset), 10, 8)
    # every row contributes exactly its own (unpadded) token count
    assert int(first["attention_masks"].sum()) == sum(
        int(dataset.hidden_lengths[idx]) for idx in dataset.hidden_index)


# ─── B3 ───
def test_model_name_validation():
    assert is_valid_model_name("my_model 01")
    assert is_valid_model_name("tf_diffusion_cached_Phi-3.5-mini_20260820_193350")
    for bad in ("..", ".", "../escape", "a/b", "a\\b", "", " leading", "x" * 100):
        assert not is_valid_model_name(bad), bad


# ─── Textur-Farben (uint8 Overlauf) ───
def test_texture_colors_are_not_black():
    from app.voxel_preview import get_block_color

    color = get_block_color("minecraft:oak_planks")
    assert color[3] == 255, color
    # Before the fix the alpha weighted average overflowed in uint8 and every
    # textured block resolved to (0, 0, 0, 255).
    assert sum(color[:3]) > 150, color

    stone = get_block_color("minecraft:stone")
    assert abs(stone[0] - stone[2]) < 15 and stone[0] > 60, stone

