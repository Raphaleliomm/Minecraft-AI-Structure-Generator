"""Verify that block-state facing rotation matches the voxel geometry rotation.

The grid rotation in rotate_block_grid_y maps (dx, dz) -> (dz, -dx), which is
counter-clockwise (north -> west -> south -> east). The block-state facing
rotation must follow the same counter-clockwise order, otherwise north<->south
and east<->west get swapped relative to the geometry.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import rotate_block_grid_y, rotate_block_state_y


def test_facing_rotation_counter_clockwise():
    """facing must rotate counter-clockwise: north -> west -> south -> east."""
    assert rotate_block_state_y("minecraft:oak_stairs[facing=north]", 1) == "minecraft:oak_stairs[facing=west]"
    assert rotate_block_state_y("minecraft:oak_stairs[facing=west]", 1) == "minecraft:oak_stairs[facing=south]"
    assert rotate_block_state_y("minecraft:oak_stairs[facing=south]", 1) == "minecraft:oak_stairs[facing=east]"
    assert rotate_block_state_y("minecraft:oak_stairs[facing=east]", 1) == "minecraft:oak_stairs[facing=north]"


def test_facing_rotation_180_and_270():
    assert rotate_block_state_y("minecraft:oak_stairs[facing=north]", 2) == "minecraft:oak_stairs[facing=south]"
    assert rotate_block_state_y("minecraft:oak_stairs[facing=north]", 3) == "minecraft:oak_stairs[facing=east]"
    assert rotate_block_state_y("minecraft:oak_stairs[facing=north]", 4) == "minecraft:oak_stairs[facing=north]"


def test_geometry_and_facing_consistent():
    """A block at +X facing north must move to +Z facing west after 1 turn.

    Grid rotation: (dx, dz) -> (dz, -dx). A block at (1, 0, 0) (east side)
    becomes (0, 0, 1) (south side in +Z). Its facing must rotate the same way.
    """
    # 2x1x2 grid: block at x=1 (east), facing north
    grid = [
        [  # x=0
            [  # y=0
                "minecraft:air",  # z=0
                "minecraft:air",  # z=1
            ]
        ],
        [  # x=1
            [  # y=0
                "minecraft:air",  # z=0
                "minecraft:oak_stairs[facing=north]",  # z=1
            ]
        ],
    ]
    rotated = rotate_block_grid_y(grid, 1)
    # After rotation the block should be at x=1, z=0 facing west
    assert rotated[1][0][0] == "minecraft:oak_stairs[facing=west]", rotated
    assert rotated[0][0][0] == "minecraft:air", rotated
