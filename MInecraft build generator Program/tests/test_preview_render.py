"""Preview renderer behaviour: speed tricks must not change what is drawn."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.voxel_preview import (
    PreviewRenderer,
    VoxelScene,
    _LOW_LEVEL_POLYGON_OK,
    _PolygonBatch,
    _pack_rgba,
    _shade_array,
    get_block_color,
    is_air_block,
    render_scene,
)

PALETTE = [
    "minecraft:air", "minecraft:stone", "minecraft:oak_planks", "minecraft:glass",
    "minecraft:water", "minecraft:oak_stairs", "minecraft:stone_bricks",
]


def _scene(grid):
    return VoxelScene.build(np.asarray(grid), PALETTE)


def test_scene_skips_air_and_remembers_unoccluded_faces():
    grid = np.zeros((4, 4, 4), dtype=np.int64)
    grid[1:3, 1:3, 1:3] = 1                       # 2x2x2 stone cube
    scene = _scene(grid)
    assert scene.count == 8

    stats = {}
    image = render_scene(scene, size=(120, 100), azimuth=45, elevation=30, scale=6,
                         stats=stats)
    assert image.size == (120, 100)
    # A solid cube shows 3 sides, 4 faces each - the hidden inside faces are culled
    # (before the culling fix all 6 faces of all 8 voxels were drawn: 48).
    assert stats["faces"] == 12, stats
    assert image.mode == "RGB"


def test_solid_neighbour_hides_the_face_behind_it():
    """A face covered by an opaque neighbour must not be drawn at all."""
    grid = np.zeros((2, 1, 2), dtype=np.int64)
    grid[0, 0, 0] = 1                              # two voxels in a row (view axis = z)
    grid[0, 0, 1] = 1
    stats = {}
    render_scene(_scene(grid), size=(80, 60), azimuth=0, elevation=0, scale=8, stats=stats)
    # camera looks along +z: only the near voxel's front face is visible (1, not 2)
    assert stats["faces"] == 1, stats

    grid[0, 0, 1] = 0                              # separate them -> both faces show
    stats = {}
    render_scene(_scene(grid), size=(80, 60), azimuth=0, elevation=0, scale=8, stats=stats)
    assert stats["faces"] == 1, stats               # single voxel: its front face

    grid[0, 0, 1] = 1
    grid[1, 0, 1] = 1                              # one more voxel beside the near one
    stats = {}
    render_scene(_scene(grid), size=(80, 60), azimuth=0, elevation=0, scale=8, stats=stats)
    assert stats["faces"] == 2, stats


def test_transparent_blocks_are_blended_not_replaced():
    """Water/glass must blend with what is behind them.

    If transparency ever regresses to "draw the colour, store the alpha" the
    image would no longer depend on the blocks behind the water.
    """
    sky_only = np.zeros((4, 4, 4), dtype=np.int64)
    sky_only[1:3, 0:3, 2:3] = 4                    # water slab
    with_stone = sky_only.copy()
    with_stone[1:3, 0:3, 3:4] = 1                  # stone slab right behind it

    params = dict(size=(200, 160), azimuth=180, elevation=0, scale=10)
    over_sky = np.array(render_scene(_scene(sky_only), **params)).astype(int)
    over_stone = np.array(render_scene(_scene(with_stone), **params)).astype(int)

    changed = np.abs(over_sky - over_stone).sum(axis=-1) > 0
    assert changed.any(), "blocks behind the water did not influence the image"

    # the water-covered pixel must be neither the pure water nor the pure stone
    # fill - it has to be a mix of both
    face_shade = 0.76                              # -z face, see _FACE_TABLE
    water_fill = _shade_array(
        np.array([[*get_block_color("minecraft:water")[:3],
                   get_block_color("minecraft:water")[3]]], dtype=np.uint8), face_shade)[0][:3]
    stone_fill = _shade_array(
        np.array([[*get_block_color("minecraft:stone")[:3],
                   get_block_color("minecraft:stone")[3]]], dtype=np.uint8), face_shade)[0][:3]

    # pick a pixel in the middle of the water area (edge pixels are outlines)
    coords = np.argwhere(changed)
    centroid = coords.mean(axis=0).round().astype(int)
    row, column = coords[int(np.argmin(np.abs(coords - centroid).sum(axis=1)))]
    pixel = over_stone[row, column]
    assert not np.array_equal(pixel, water_fill), pixel
    assert not np.array_equal(pixel, stone_fill), pixel
    # it sits between the two colours: blending, not replacing
    lower = np.minimum(water_fill, stone_fill) - 3
    upper = np.maximum(water_fill, stone_fill) + 3
    assert np.all((pixel >= lower) & (pixel <= upper)), (pixel, water_fill, stone_fill)


def test_fast_polygon_path_matches_high_level_api():
    """The optimised drawing path must be pixel identical to ImageDraw.polygon."""
    quad = [(2.0, 1.0), (9.0, 3.0), (7.0, 8.0), (3.0, 6.0)]
    fills = ((0, 0, 255, 128), (0, 128, 0, 64))
    inks = _pack_rgba(np.array([fills[0], fills[1]], dtype=np.uint8))

    reference = Image.new("RGB", (12, 12), (200, 40, 40))
    ImageDraw.Draw(reference, "RGBA").polygon(list(quad), fill=fills[0], outline=fills[1])
    fast = Image.new("RGB", (12, 12), (200, 40, 40))
    _PolygonBatch(ImageDraw.Draw(fast, "RGBA")).quad([list(p) for p in quad],
                                                     int(inks[0]), int(inks[1]))
    assert np.array_equal(np.array(reference), np.array(fast))
    # and the probe that decides whether the fast path may be used agrees
    assert _LOW_LEVEL_POLYGON_OK is True


def test_renderer_handles_unknown_ids_and_empty_grids():
    grid = np.zeros((5, 5, 5), dtype=np.int64)
    scene = _scene(grid)
    assert scene.count == 0
    image = render_scene(scene, size=(60, 50))
    assert image.size == (60, 50)

    grid[0, 0, 0] = 99                             # id outside the palette
    scene = _scene(grid)
    assert scene.count == 1
    assert bool(scene.unsaved[0])
    stats = {}
    render_scene(scene, size=(60, 50), scale=8, stats=stats)
    assert stats["faces"] == 3                     # 3 of 6 faces point at the camera


def test_preview_renderer_drops_stale_frames_and_reports_latest():
    calls = []

    def fake_render(scene, **params):
        calls.append(params["azimuth"])
        return Image.new("RGB", (8, 8), (int(params["azimuth"]) % 256, 0, 0))

    renderer = PreviewRenderer(render=fake_render)
    try:
        scene = _scene(np.ones((2, 2, 2), dtype=np.int64))
        for azimuth in range(40):
            renderer.request(scene, azimuth=azimuth)
        assert renderer.wait_idle(10.0)
        stats = renderer.stats()
        assert stats["rendered"] < 40, stats          # stale requests were dropped
        assert stats["rendered"] + stats["dropped"] >= 40
        assert calls[-1] == 39                        # the newest frame won

        image = renderer.take_latest()
        assert image is not None and image.getpixel((0, 0)) == (39, 0, 0)
        assert renderer.take_latest() is None          # only delivered once
    finally:
        renderer.stop()


def test_preview_renderer_survives_render_errors():
    def broken_render(scene, **params):
        raise RuntimeError("boom")

    renderer = PreviewRenderer(render=broken_render)
    try:
        renderer.request(_scene(np.ones((2, 2, 2), dtype=np.int64)))
        assert renderer.wait_idle(10.0)
        assert renderer.take_latest() is None
        assert renderer.stats()["rendered"] == 0
    finally:
        renderer.stop()


def test_preview_renderer_view_ignores_requests_after_stop():
    renderer = PreviewRenderer(render=lambda scene, **params: Image.new("RGB", (4, 4)))
    renderer.stop()
    renderer.request(_scene(np.ones((2, 2, 2), dtype=np.int64)))
    assert not renderer.is_busy()
    assert renderer.take_latest() is None


def test_render_preview_view_presets_and_scene_reuse():
    """The public helper must map view names and accept a prepared scene."""
    from app.voxel_preview import render_preview, view_angles

    grid = np.zeros((6, 6, 6), dtype=np.int64)
    grid[1:5, 1:5, 1:5] = 1
    scene = _scene(grid)

    free = render_preview(grid, PALETTE, view="free", size=(120, 100), azimuth=10,
                          elevation=20, scale=6)
    reused = render_preview(grid, PALETTE, view="free", size=(120, 100), azimuth=10,
                            elevation=20, scale=6, scene=scene)
    assert np.array_equal(np.array(free), np.array(reused))

    for view, angles in (("isometric", (45.0, 30.0)), ("topdown", (0.0, 89.0)),
                         ("side", (0.0, 5.0))):
        assert view_angles(view, 1.0, 2.0) == angles
        image = render_preview(grid, PALETTE, view=view, size=(120, 100), scale=6)
        assert image.size == (120, 100)
    assert view_angles("free", 12.0, 34.0) == (12.0, 34.0)


def test_drag_resolution_frame_matches_full_frame():
    """The cheap drag frame must show the same picture, only smaller."""
    grid = np.zeros((12, 10, 12), dtype=np.int64)
    grid[2:10, 1:6, 2:10] = 1
    grid[4:8, 6:7, 4:8] = 5
    scene = _scene(grid)

    full = np.array(render_scene(scene, size=(420, 380), azimuth=35, elevation=30,
                                 scale=3.5))
    factor = 0.6
    small = np.array(render_scene(scene, size=(int(420 * factor), int(380 * factor)),
                                  azimuth=35, elevation=30, scale=3.5 * factor))

    def content_box(pixels, background):
        mask = np.abs(pixels.astype(int) - background).sum(axis=-1) > 12
        rows, columns = np.nonzero(mask)
        return rows.min(), rows.max(), columns.min(), columns.max()

    background = np.array(render_scene(_scene(np.zeros_like(grid)), size=(8, 8)))[0, 0]
    full_box = np.array(content_box(full, background), dtype=float)
    small_box = np.array(content_box(small, background), dtype=float) / factor
    # the structure occupies the same area of the frame in both renders
    assert np.abs(full_box - small_box).max() <= 3.0, (full_box, small_box)


def test_app_preview_scene_is_cached_and_invalidated():
    """main_app must rebuild the render scene only when grid or palette change."""
    from types import SimpleNamespace

    from app.main_app import MinecraftStructureApp

    fake = SimpleNamespace(
        generated_np=np.ones((2, 2, 2), dtype=np.int64),
        id_to_block=["minecraft:air", "minecraft:stone"],
        _preview_scene_cache=None, _preview_scene_source=None, _preview_scene_palette=None,
    )
    first = MinecraftStructureApp._preview_scene(fake)
    assert first is not None and first.count == 8
    assert MinecraftStructureApp._preview_scene(fake) is first          # cached

    fake.generated_np = np.ones((3, 3, 3), dtype=np.int64)               # new grid
    second = MinecraftStructureApp._preview_scene(fake)
    assert second is not first and second.count == 27

    fake.id_to_block = ["minecraft:air", "minecraft:stone"]              # new palette
    assert MinecraftStructureApp._preview_scene(fake) is not second

    fake.generated_np = None
    assert MinecraftStructureApp._preview_scene(fake) is None


def test_app_preview_poll_delivers_frames_and_stops_when_idle():
    """The Tk poll callback must push frames into the label and then stop."""
    from types import SimpleNamespace

    import app.main_app as main_app
    from app.voxel_preview import PreviewRenderer
    from app.main_app import MinecraftStructureApp

    class FakeLabel:
        def __init__(self):
            self.options = {}

        def configure(self, **kwargs):
            self.options.update(kwargs)

    scheduled = []
    fake = SimpleNamespace(
        preview_label=FakeLabel(),
        current_preview_img=None,
        preview_renderer=PreviewRenderer(
            render=lambda scene, **params: Image.new("RGB", (4, 4), (7, 8, 9))),
        _preview_poll_job=None,
        _poll_preview=lambda: None,
        after=lambda ms, callback: scheduled.append((ms, callback)) or "job",
    )
    original_image_class = main_app.ctk.CTkImage
    main_app.ctk.CTkImage = lambda **kwargs: SimpleNamespace(size=kwargs.get("size"))
    try:
        fake.preview_renderer.request(_scene(np.ones((2, 2, 2), dtype=np.int64)))
        assert fake.preview_renderer.wait_idle(10.0)

        MinecraftStructureApp._poll_preview(fake)
        assert fake.current_preview_img is not None
        assert "image" in fake.preview_label.options
        assert fake._preview_poll_job is None
        assert scheduled == []                      # nothing left to wait for

        # a busy renderer keeps the poll loop alive
        fake.preview_renderer.request(_scene(np.ones((2, 2, 2), dtype=np.int64)))
        MinecraftStructureApp._schedule_preview_poll(fake)
        assert fake._preview_poll_job == "job"
        assert scheduled and scheduled[0][0] == 20
    finally:
        main_app.ctk.CTkImage = original_image_class
        fake.preview_renderer.stop()


def test_app_preview_angles_follow_view_mode():
    from types import SimpleNamespace

    from app.main_app import MinecraftStructureApp

    fake = SimpleNamespace(current_view="free", azimuth=12.5, elevation=41.0)
    assert MinecraftStructureApp._preview_angles(fake) == (12.5, 41.0)
    fake.current_view = "topdown"
    assert MinecraftStructureApp._preview_angles(fake) == (0.0, 89.0)
    fake.current_view = "side"
    assert MinecraftStructureApp._preview_angles(fake) == (0.0, 5.0)
    fake.current_view = "isometric"
    assert MinecraftStructureApp._preview_angles(fake) == (45.0, 30.0)
