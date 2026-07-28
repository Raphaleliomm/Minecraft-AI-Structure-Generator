"""3D Voxel Viewer - Real-time 3D Minecraft structure viewer.
Uses pyglet + OpenGL for true 3D rendering with:
- Procedural Minecraft textures
- Orbit camera (mouse drag = rotate, scroll = zoom)
- Fullscreen mode (F11)
- Lighting for depth effect

Runs in its own process to avoid OpenGL thread/main-thread conflicts.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.voxel_preview import get_block_color

# Set BEFORE any pyglet import to avoid ARB_pixel_format error
os.environ["PYGLET_WIN32_DISABLE_ARB_PIXEL_FORMAT"] = "1"

HAS_PYGLET = False
try:
    import pyglet
    from pyglet import gl, window, clock
    from pyglet.window import key, mouse
    import ctypes

    import ctypes

    # OpenGL 1.x compatibility functions (missing in pyglet 2.x)
    # We load them directly from opengl32.dll to bypass pyglet's missing wrappers
    _gl_dll = ctypes.CDLL("opengl32.dll")
    _glu_dll = ctypes.CDLL("glu32.dll")

    def _make_gl(name: str, dll=_gl_dll, restype=None):
        """Get an OpenGL function from the DLL and set its return type."""
        try:
            f = getattr(dll, name)
            f.restype = restype
            return f
        except AttributeError:
            return None

    # Legacy GL 1.x functions needed for fixed-function pipeline
    # Loaded directly from DLLs since pyglet 2.x removed them
    _gl_funcs = {}
    for _name in (
        "glMatrixMode", "glLoadIdentity", "glLoadMatrixf",
        "glPushMatrix", "glPopMatrix",
        "glVertexPointer", "glTexCoordPointer", "glNormalPointer",
        "glEnableClientState", "glDisableClientState",
        "glColorMaterial", "glLightfv",
    ):
        _f = _make_gl(_name, _gl_dll)
        if _f is not None:
            _gl_funcs[_name] = _f

    # Legacy fixed-function constants removed from pyglet.gl in some pyglet 2.x
    # builds. Values are from the OpenGL 1.x specification.
    GL_MODELVIEW = 0x1700
    GL_PROJECTION = 0x1701
    GL_TEXTURE_2D = 0x0DE1
    GL_LIGHTING = 0x0B50
    GL_LIGHT0 = 0x4000
    GL_LIGHT1 = 0x4001
    GL_POSITION = 0x1203
    GL_AMBIENT = 0x1200
    GL_DIFFUSE = 0x1201
    GL_COLOR_MATERIAL = 0x0B57
    GL_FRONT_AND_BACK = 0x0408
    GL_AMBIENT_AND_DIFFUSE = 0x1602
    GL_VERTEX_ARRAY = 0x8074
    GL_NORMAL_ARRAY = 0x8075
    GL_TEXTURE_COORD_ARRAY = 0x8078
    GL_QUADS = 0x0007

    HAS_PYGLET = True
except Exception:
    pass


# ─── Procedural Minecraft textures (16x16 pixel) ───

TEXTURE_SIZE = 16


def _noise(x, y, seed=0):
    return hash((x, y, seed)) % 256 / 255.0


def _generate_texture(block_name: str) -> Image.Image:
    """Generate a Minecraft-like 16x16 RGBA texture procedurally."""
    b = block_name.lower()
    arr = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.uint8)
    arr[:, :] = get_block_color(block_name)

    # Stone / Cobblestone
    if "stone" in b or "cobble" in b:
        base_r, base_g, base_b = 128, 128, 128
        if "mossy" in b:
            base_r, base_g, base_b = 100, 120, 80
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 1) * 0.3 + _noise(x, y, 2) * 0.2
                crack = 1.0 if _noise(x, y, 3) > 0.7 else 0.9
                brightness = 0.7 + n * 0.6
                arr[y, x] = [
                    int(base_r * brightness * crack),
                    int(base_g * brightness * crack),
                    int(base_b * brightness * crack),
                    255,
                ]

    # Wood Planks
    elif "planks" in b:
        plank_map = {
            "oak": (160, 130, 80), "spruce": (100, 70, 40),
            "birch": (180, 170, 130), "jungle": (140, 100, 70),
            "acacia": (170, 100, 60), "dark_oak": (60, 40, 20),
            "mangrove": (130, 60, 40), "crimson": (100, 40, 50),
            "warped": (50, 120, 110),
        }
        base = (160, 130, 80)
        for name, col in plank_map.items():
            if name in b: base = col; break
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                strip = 0.9 + 0.1 * math.sin(x * 0.8) * math.cos(y * 0.8)
                n = _noise(x, y, 5) * 0.1
                arr[y, x] = [
                    int(base[0] * (strip + n)),
                    int(base[1] * (strip + n)),
                    int(base[2] * (strip + n)),
                    255,
                ]

    # Log
    elif "log" in b:
        base = (120, 90, 50)
        if "spruce" in b: base = (60, 40, 20)
        if "birch" in b: base = (180, 170, 140)
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                dist = abs(x - 7.5) + abs(y - 7.5)
                ring = 0.7 + 0.3 * math.sin(dist * 1.5)
                n = _noise(x, y, 7) * 0.08
                arr[y, x] = [
                    int(base[0] * (ring + n)),
                    int(base[1] * (ring + n)),
                    int(base[2] * (ring + n)),
                    255,
                ]

    # Grass / Leaves
    elif "grass" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                g = 50 + int(_noise(x, y, 9) * 30)
                arr[y, x] = [80, 100 + g, 50, 255]
    elif "leaves" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 11)
                g = 80 + int(n * 40)
                arr[y, x] = [40, g, 30, 180]

    # Dirt / Sand
    elif "dirt" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 13) * 20
                arr[y, x] = [140, 100 + int(n), 60, 255]
    elif "sand" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 15) * 15
                arr[y, x] = [210, 190 - int(n), 140, 255]

    # Water / Glass (transparent)
    elif "water" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                wave = 0.8 + 0.2 * math.sin(x * 0.5 + y * 0.3)
                arr[y, x] = [30, 60, int(180 * wave), 100]
    elif "glass" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = 0.95 + _noise(x, y, 17) * 0.05
                arr[y, x] = [int(200 * n), int(210 * n), int(230 * n), 100]

    # Wool
    elif "wool" in b:
        wool_colors = {
            "white": (220, 220, 220), "orange": (220, 140, 40),
            "magenta": (180, 60, 140), "light_blue": (120, 170, 220),
            "yellow": (220, 200, 50), "lime": (100, 210, 60),
            "pink": (210, 130, 150), "gray": (100, 100, 100),
            "light_gray": (160, 160, 160), "cyan": (50, 150, 160),
            "purple": (120, 50, 160), "blue": (50, 80, 180),
            "brown": (100, 70, 40), "green": (60, 140, 40),
            "red": (180, 50, 50), "black": (30, 30, 30),
        }
        base = (220, 220, 220)
        for name, col in wool_colors.items():
            if name in b: base = col; break
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 19) * 15 - 7
                arr[y, x] = [
                    max(0, min(255, base[0] + int(n))),
                    max(0, min(255, base[1] + int(n))),
                    max(0, min(255, base[2] + int(n))),
                    255,
                ]

    # Bricks
    elif "brick" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                mortar = 0.8 if ((x // 4) + (y // 4)) % 2 == 0 else 0.9
                if x % 4 == 0 or y % 4 == 0:
                    arr[y, x] = [80, 60, 50, 255]
                else:
                    n = _noise(x, y, 21) * 15
                    arr[y, x] = [
                        int((140 + n) * mortar),
                        int((70 + n) * mortar),
                        int((50 + n) * mortar),
                        255,
                    ]

    # Deepslate
    elif "deepslate" in b or "stone" in b and "brick" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 25) * 15
                spec = 0.9 + 0.1 * math.sin(x * 2 + y * 3)
                arr[y, x] = [
                    int((70 + n) * spec),
                    int((70 + n) * spec),
                    int((80 + n) * spec),
                    255,
                ]

    # Concrete
    elif "concrete" in b:
        base = (180, 180, 180)
        for name, col in {"white":(200,200,200),"red":(140,50,50),"blue":(50,70,150)}.items():
            if name in b: base = col; break
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 27) * 8
                arr[y, x] = [
                    max(0, min(255, base[0] + int(n))),
                    max(0, min(255, base[1] + int(n))),
                    max(0, min(255, base[2] + int(n))),
                    255,
                ]

    # Iron / Gold / Diamond blocks
    elif "iron_block" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 29) * 20
                arr[y, x] = [180 + int(n), 180 + int(n), 180 + int(n), 255]
    elif "gold_block" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 31) * 15
                arr[y, x] = [220, 190 + int(n), 50, 255]
    elif "diamond_block" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 33) * 15
                arr[y, x] = [80 + int(n), 200, 200 + int(n), 255]

    # Pumpkin
    elif "pumpkin" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 37) * 15
                arr[y, x] = [200, 140 + int(n), 40, 255]
                if 4 < x < 12 and 6 < y < 10:
                    arr[y, x] = [150, 100, 50, 255]

    # Snow
    elif "snow" in b:
        for x in range(TEXTURE_SIZE):
            for y in range(TEXTURE_SIZE):
                n = _noise(x, y, 35) * 10
                arr[y, x] = [
                    min(255, 230 + int(n)),
                    min(255, 240 + int(n)),
                    min(255, 250 + int(n)),
                    255,
                ]

    return Image.fromarray(arr, "RGBA")


class TextureAtlas:
    """Texture atlas for OpenGL rendering."""

    def __init__(self):
        self._atlas_id: Optional[int] = None
        self._block_to_uv: Dict[str, Tuple[float, float, float, float]] = {}
        self._rows = 1
        self._cols = 1

    def build(self, id_to_block: List[str]) -> None:
        block_names = sorted(set(str(b) for b in id_to_block))
        block_names = [b for b in block_names if "air" not in b.lower()]
        n = len(block_names)
        self._cols = min(16, n)
        self._rows = (n + self._cols - 1) // self._cols
        atlas_w = self._cols * TEXTURE_SIZE
        atlas_h = self._rows * TEXTURE_SIZE
        atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
        for i, name in enumerate(block_names):
            col = i % self._cols
            row = i // self._cols
            tex = _generate_texture(name)
            atlas.paste(tex, (col * TEXTURE_SIZE, row * TEXTURE_SIZE))
            u0 = col / self._cols; v0 = row / self._rows
            u1 = (col + 1) / self._cols; v1 = (row + 1) / self._rows
            self._block_to_uv[name] = (u0, v0, u1, v1)
        img_data = atlas.tobytes()
        self._atlas_id = gl.GLuint(0)
        gl.glGenTextures(1, self._atlas_id)
        gl.glBindTexture(GL_TEXTURE_2D, self._atlas_id)
        gl.glTexImage2D(GL_TEXTURE_2D, 0, gl.GL_RGBA, atlas_w, atlas_h,
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
        gl.glTexParameteri(GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    def get_uv(self, block_name: str) -> Tuple[float, float, float, float]:
        b = str(block_name).lower()
        for name, uv in self._block_to_uv.items():
            if name.lower() == b: return uv
        for name, uv in self._block_to_uv.items():
            if any(part in b for part in name.lower().split(":")):
                return uv
        return next(iter(self._block_to_uv.values()), (0, 0, 1, 1))

    def bind(self) -> None:
        if self._atlas_id is not None:
            gl.glBindTexture(GL_TEXTURE_2D, self._atlas_id)

    def delete(self) -> None:
        if self._atlas_id is not None:
            gl.glDeleteTextures(1, [self._atlas_id])


# ─── Matrix helpers for modern OpenGL ───

def _perspective(fov_y: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Create a perspective projection matrix (column-major, OpenGL style)."""
    f = 1.0 / math.tan(math.radians(fov_y) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = -1.0
    m[3, 2] = 2.0 * zfar * znear / (znear - zfar)
    return m

def _look_at(eye: Tuple[float, ...], target: Tuple[float, ...], up: Tuple[float, ...]) -> np.ndarray:
    """Create a view matrix (column-major, OpenGL style)."""
    f = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
    f = f / np.linalg.norm(f)
    u = np.array(up, dtype=np.float32)
    u = u / np.linalg.norm(u)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s; m[1, 0:3] = u; m[2, 0:3] = -f
    m[0:3, 3] = -np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)
    return m

def _load_matrix(mat: np.ndarray) -> None:
    """Load a 4x4 numpy matrix (column-major) as current OpenGL matrix."""
    # OpenGL expects column-major float array
    arr = (gl.GLfloat * 16)(*mat.T.ravel())
    f = _gl_funcs.get("glLoadMatrixf")
    if f is not None:
        f(arr)

def _glu_look_at(ex: float, ey: float, ez: float,
                  tx: float, ty: float, tz: float,
                  ux: float, uy: float, uz: float) -> None:
    _load_matrix(_look_at((ex, ey, ez), (tx, ty, tz), (ux, uy, uz)))

def _glu_perspective(fov: float, aspect: float, znear: float, zfar: float) -> None:
    _load_matrix(_perspective(fov, aspect, znear, zfar))

def _ortho(left: float, right: float, bottom: float, top: float,
           znear: float, zfar: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (zfar - znear)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(zfar + znear) / (zfar - znear)
    return m


# Make GL functions available globally (pyglet 2 compat)
def _gl(func_name, *args):
    """Call a legacy OpenGL 1.x function loaded from DLL."""
    f = _gl_funcs.get(func_name)
    if f is not None:
        f(*args)


def _drain_gl_errors() -> None:
    """Clear pending OpenGL errors from optional legacy calls."""
    try:
        while gl.glGetError() != 0:
            pass
    except Exception:
        pass


def _safe_enable(cap: int) -> None:
    try:
        gl.glEnable(cap)
    except Exception:
        _drain_gl_errors()


def _safe_disable(cap: int) -> None:
    try:
        gl.glDisable(cap)
    except Exception:
        _drain_gl_errors()

# Face definitions: (verts_relative, normal)
_FACES = {
    "+z": ([(0,0,1),(1,0,1),(1,1,1),(0,1,1)], (0,0,1)),
    "-z": ([(1,0,0),(0,0,0),(0,1,0),(1,1,0)], (0,0,-1)),
    "+x": ([(1,0,0),(1,0,1),(1,1,1),(1,1,0)], (1,0,0)),
    "-x": ([(0,0,1),(0,0,0),(0,1,0),(0,1,1)], (-1,0,0)),
    "+y": ([(0,1,0),(1,1,0),(1,1,1),(0,1,1)], (0,1,0)),
    "-y": ([(0,0,1),(1,0,1),(1,0,0),(0,0,0)], (0,-1,0)),
}


class VoxelViewer3D:
    """Real-time 3D voxel viewer with OpenGL."""

    def __init__(self, grid: np.ndarray, id_to_block: List[str],
                 title: str = "Minecraft 3D Viewer"):
        self.grid = grid
        self.id_to_block = id_to_block
        self.GX, self.GY, self.GZ = grid.shape
        self.azimuth = 45.0
        self.elevation = 30.0
        self.distance = max(self.GX, self.GY, self.GZ) * 2.8
        self.target_distance = self.distance
        self._drag_start_x = 0; self._drag_start_y = 0
        self._orbit_start_az = 45.0; self._orbit_start_el = 30.0
        self._dragging = False
        self.window: Optional[window.Window] = None
        self.fullscreen = False
        self.title = title
        self.atlas = TextureAtlas()
        # VBO data: [v3f, t2f, n3f] interleaved
        self._vbo_id: Optional[int] = None
        self._vbo_count: int = 0

    def start(self) -> None:
        if not HAS_PYGLET:
            print("pyglet not available")
            return
        try:
            config = gl.Config(
                double_buffer=True,
                depth_size=24,
                major_version=2,
                minor_version=1,
            )
            self.window = window.Window(960, 720, self.title, resizable=True, config=config)
        except Exception:
            config = gl.Config(double_buffer=True, depth_size=24)
            self.window = window.Window(960, 720, self.title, resizable=True, config=config)
        self.window.set_minimum_size(640, 480)
        self.atlas.build(self.id_to_block)
        self._build_geometry()

        @self.window.event
        def on_draw(): self._render()

        @self.window.event
        def on_resize(w, h): gl.glViewport(0, 0, w, h)

        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            if button == mouse.LEFT:
                self._drag_start_x = x; self._drag_start_y = y
                self._orbit_start_az = self.azimuth
                self._orbit_start_el = self.elevation
                self._dragging = True

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & mouse.LEFT:
                self.azimuth = (self._orbit_start_az - (x - self._drag_start_x) * 0.5) % 360
                self.elevation = max(0.1, min(89.9, self._orbit_start_el + (y - self._drag_start_y) * 0.5))

        @self.window.event
        def on_mouse_release(x, y, button, modifiers): self._dragging = False

        @self.window.event
        def on_mouse_scroll(x, y, sx, sy):
            self.target_distance = max(2.0, min(50.0, self.target_distance - sy * 1.5))

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.F11:
                self.fullscreen = not self.fullscreen
                self.window.set_fullscreen(self.fullscreen)
            elif symbol == key.ESCAPE and self.fullscreen:
                self.fullscreen = False; self.window.set_fullscreen(False)

        @self.window.event
        def on_close(): self.atlas.delete()

        clock.schedule_interval(lambda dt: setattr(self, 'distance',
            self.distance + (self.target_distance - self.distance) * 0.15), 1/60.0)
        pyglet.app.run()

    def _build_geometry(self):
        """Build interleaved vertex data and upload to VBO."""
        verts = []
        for x in range(self.GX):
            for y in range(self.GY):
                for z in range(self.GZ):
                    bid = int(self.grid[x, y, z])
                    if bid < 0 or bid >= len(self.id_to_block):
                        continue
                    block_name = str(self.id_to_block[bid])
                    if "air" in block_name.lower():
                        continue
                    u0, v0, u1, v1 = self.atlas.get_uv(block_name)
                    # Check neighbors
                    neigh = {}
                    for dx, dy, dz in [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if nx<0 or nx>=self.GX or ny<0 or ny>=self.GY or nz<0 or nz>=self.GZ:
                            neigh[(dx,dy,dz)] = True
                        else:
                            b = int(self.grid[nx, ny, nz])
                            if b>=0 and b<len(self.id_to_block) and "air" in str(self.id_to_block[b]).lower():
                                neigh[(dx,dy,dz)] = True
                            else:
                                neigh[(dx,dy,dz)] = False
                    face_keys = [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]
                    face_names = ["+z","-z","+x","-x","+y","-y"]
                    for fk, fn in zip(face_keys, face_names):
                        if neigh.get(fk):
                            fverts, fnorm = _FACES[fn]
                            for vx, vy, vz in fverts:
                                verts.extend([x+vx-self.GX/2, y+vy-self.GY/2, z+vz-self.GZ/2])
                                verts.extend([u0, v0])
                                verts.extend(fnorm)
        if not verts:
            return
        self._vbo_count = len(verts) // 8  # 3 pos + 2 tex + 3 normal = 8 floats per vertex
        arr = (gl.GLfloat * len(verts))(*verts)
        vbo = gl.GLuint(0)
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(arr), arr, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self._vbo_id = vbo

    def _render(self):
        if self.window is None: return
        self.window.clear()
        gl.glClearColor(0.05, 0.08, 0.15, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # ── Projection matrix (perspective) ──
        aspect = max(1, self.window.width) / max(1, self.window.height)
        proj = _perspective(45.0, aspect, 0.1, 100.0)
        _gl("glMatrixMode", GL_PROJECTION)
        _gl("glLoadIdentity")
        _load_matrix(proj)
        _drain_gl_errors()

        # ── ModelView matrix (camera) ──
        az_rad = math.radians(self.azimuth); el_rad = math.radians(self.elevation)
        cx = self.distance * math.cos(el_rad) * math.sin(az_rad)
        cy = self.distance * math.sin(el_rad)
        cz = self.distance * math.cos(el_rad) * math.cos(az_rad)
        view = _look_at((cx, cy, cz), (0, 0, 0), (0, 1, 0))
        _gl("glMatrixMode", GL_MODELVIEW)
        _gl("glLoadIdentity")
        _load_matrix(view)
        _drain_gl_errors()

        # ── Lighting and rendering ──
        _safe_enable(gl.GL_DEPTH_TEST)
        _safe_enable(gl.GL_CULL_FACE)
        _safe_enable(GL_TEXTURE_2D)
        _safe_enable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        _safe_enable(GL_LIGHTING)
        _safe_enable(GL_LIGHT0)
        _safe_enable(GL_LIGHT1)
        _safe_enable(GL_COLOR_MATERIAL)
        _gl("glColorMaterial", GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Light positions
        _gl("glLightfv", GL_LIGHT0, GL_POSITION, (gl.GLfloat * 4)(15, 20, 10, 1))
        _gl("glLightfv", GL_LIGHT0, GL_DIFFUSE, (gl.GLfloat * 4)(0.9, 0.9, 0.85, 1))
        _gl("glLightfv", GL_LIGHT0, GL_AMBIENT, (gl.GLfloat * 4)(0.3, 0.3, 0.35, 1))
        _gl("glLightfv", GL_LIGHT1, GL_POSITION, (gl.GLfloat * 4)(-5, -10, 5, 1))
        _gl("glLightfv", GL_LIGHT1, GL_DIFFUSE, (gl.GLfloat * 4)(0.2, 0.2, 0.3, 1))
        _gl("glLightfv", GL_LIGHT1, GL_AMBIENT, (gl.GLfloat * 4)(0.1, 0.1, 0.15, 1))
        _drain_gl_errors()

        self.atlas.bind()
        if self._vbo_id is not None:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_id)
            stride = ctypes.sizeof(gl.GLfloat) * 8
            _gl("glVertexPointer", 3, gl.GL_FLOAT, stride, None)
            _gl("glEnableClientState", GL_VERTEX_ARRAY)
            offset = ctypes.c_void_p(ctypes.sizeof(gl.GLfloat) * 3)
            _gl("glTexCoordPointer", 2, gl.GL_FLOAT, stride, offset)
            _gl("glEnableClientState", GL_TEXTURE_COORD_ARRAY)
            offset = ctypes.c_void_p(ctypes.sizeof(gl.GLfloat) * 5)
            _gl("glNormalPointer", gl.GL_FLOAT, stride, offset)
            _gl("glEnableClientState", GL_NORMAL_ARRAY)
            _drain_gl_errors()
            try:
                gl.glDrawArrays(GL_QUADS, 0, self._vbo_count)
            except Exception:
                _drain_gl_errors()
            _gl("glDisableClientState", GL_VERTEX_ARRAY)
            _gl("glDisableClientState", GL_TEXTURE_COORD_ARRAY)
            _gl("glDisableClientState", GL_NORMAL_ARRAY)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        _safe_disable(GL_LIGHTING)
        _safe_disable(GL_TEXTURE_2D)
        _safe_disable(gl.GL_DEPTH_TEST)

        # ── HUD overlay (ortho) ──
        _gl("glMatrixMode", GL_PROJECTION)
        _gl("glLoadIdentity")
        _load_matrix(_ortho(0, self.window.width, 0, self.window.height, -1, 1))
        _drain_gl_errors()
        _gl("glMatrixMode", GL_MODELVIEW)
        _gl("glLoadIdentity")
        _drain_gl_errors()

        lines = [
            f"Minecraft 3D Viewer - {self.GX}×{self.GY}×{self.GZ} Voxels",
            "Drag=Rotate  Scroll=Zoom  F11=Fullscreen  ESC=Exit",
            f"Az:{self.azimuth:.0f}° El:{self.elevation:.0f}° Zoom:{self.distance:.1f}",
        ]
        for i, text in enumerate(lines):
            pyglet.text.Label(text, font_name='Segoe UI', font_size=13 if i == 0 else 10,
                x=10, y=self.window.height - 20 - 16 * i,
                color=(200, 220, 255, 255) if i == 0 else
                      ((150, 170, 200, 200) if i == 1 else (120, 140, 180, 180))
            ).draw()


def open_3d_viewer(grid: np.ndarray, id_to_block: List[str],
                   title: str = "Minecraft 3D Viewer") -> None:
    """Open 3D viewer in a subprocess to avoid thread/OpenGL conflicts."""
    if not HAS_PYGLET:
        print("pyglet not available. Install with: pip install pyglet")
        return
    viewer = VoxelViewer3D(grid, id_to_block, title)
    viewer.start()
