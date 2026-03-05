"""
rendering/renderer.py — OpenGL Renderer

Handles all GPU interaction:
  - Shader compilation and linking
  - VAO/VBO creation and upload
  - Chunk mesh rendering
  - Selection highlight rendering
  - Sky rendering
  - Shader uniform management

This class has NO world logic — it only knows about GPU buffers and shaders.
The World class can run without this (headless server).
"""

from __future__ import annotations
import os
import ctypes
import array
import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

try:
    import OpenGL.GL as gl
    from OpenGL.GL import shaders as gl_shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

if TYPE_CHECKING:
    from world.chunk import Chunk
    from world.world import World


# ─── Shader Utilities ─────────────────────────────────────────────────────────

def _load_shader_source(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def _compile_shader(vert_path: str, frag_path: str) -> int:
    """Compile and link a GLSL shader program. Returns program ID."""
    vert_src = _load_shader_source(vert_path)
    frag_src = _load_shader_source(frag_path)

    vert_shader = gl_shaders.compileShader(vert_src, gl.GL_VERTEX_SHADER)
    frag_shader = gl_shaders.compileShader(frag_src, gl.GL_FRAGMENT_SHADER)
    program     = gl_shaders.compileProgram(vert_shader, frag_shader)
    return program


# ─── GPU Chunk Mesh ───────────────────────────────────────────────────────────

class GPUMesh:
    """Holds a VAO and VBO for a single chunk mesh."""
    __slots__ = ('vao', 'vbo', 'vertex_count')

    def __init__(self):
        self.vao          = gl.glGenVertexArrays(1)
        self.vbo          = gl.glGenBuffers(1)
        self.vertex_count = 0

    def upload(self, vertex_data: array.array):
        """Upload float vertex data to GPU."""
        self.vertex_count = len(vertex_data) // 12  # 12 floats per vertex

        raw = (ctypes.c_float * len(vertex_data))(*vertex_data)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        ctypes.sizeof(raw), raw,
                        gl.GL_DYNAMIC_DRAW)

        stride = 12 * ctypes.sizeof(ctypes.c_float)
        # attr 0: position (xyz)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # attr 1: UV (uv)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)
        # attr 2: normal (xyz)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(5 * 4))
        gl.glEnableVertexAttribArray(2)
        # attr 3: AO
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(8 * 4))
        gl.glEnableVertexAttribArray(3)
        # attr 4: face_id
        gl.glVertexAttribPointer(4, 1, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(9 * 4))
        gl.glEnableVertexAttribArray(4)
        # attr 5: block_id
        gl.glVertexAttribPointer(5, 1, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(10 * 4))
        gl.glEnableVertexAttribArray(5)
        # attr 6: sky_light
        gl.glVertexAttribPointer(6, 1, gl.GL_FLOAT, False, stride,
                                  ctypes.c_void_p(11 * 4))
        gl.glEnableVertexAttribArray(6)

        gl.glBindVertexArray(0)

    def draw(self):
        if self.vertex_count == 0:
            return
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)
        gl.glBindVertexArray(0)

    def destroy(self):
        gl.glDeleteBuffers(1, [self.vbo])
        gl.glDeleteVertexArrays(1, [self.vao])


# ─── Renderer ─────────────────────────────────────────────────────────────────

class Renderer:
    """
    Central OpenGL renderer.

    Owns:
      - Shader programs
      - Texture atlas
      - Per-chunk GPUMesh objects
      - Selection highlight VAO
    """

    SHADER_DIR = os.path.join(os.path.dirname(__file__), '..', 'shaders')

    def __init__(self):
        if not OPENGL_AVAILABLE:
            raise RuntimeError("PyOpenGL not installed. Run: pip install PyOpenGL PyOpenGL_accelerate")

        # Shader programs
        self.voxel_shader     = _compile_shader(
            os.path.join(self.SHADER_DIR, 'voxel.vert'),
            os.path.join(self.SHADER_DIR, 'voxel.frag'))
        self.selection_shader = _compile_shader(
            os.path.join(self.SHADER_DIR, 'selection.vert'),
            os.path.join(self.SHADER_DIR, 'selection.frag'))

        # Atlas texture
        self.atlas_texture = self._create_placeholder_atlas()

        # Chunk GPU meshes: (cx, cz) → GPUMesh
        self.chunk_meshes: Dict[Tuple[int,int], GPUMesh] = {}

        # Selection highlight cube VAO
        self.selection_vao = self._build_selection_vao()

        # OpenGL state
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Cached uniforms
        self._uniforms: Dict[str, int] = {}

    # ── Texture ───────────────────────────────────────────────────────────────

    def _create_placeholder_atlas(self) -> int:
        """
        Create a procedural placeholder atlas (colored squares per block type).
        In a real game, this would load a PNG texture atlas.
        """
        ATLAS_SIZE = 256    # 16×16 grid of 16×16 tiles = 256×256 pixels
        TILE       = 16

        tex_data = []
        for row in range(16):
            for ty in range(TILE):
                for col in range(16):
                    for tx in range(TILE):
                        # Each tile gets a distinct color based on its atlas position
                        block_idx = row * 16 + col
                        r = int(((block_idx * 37 + 50)  % 200) + 30)
                        g = int(((block_idx * 61 + 80)  % 200) + 30)
                        b = int(((block_idx * 89 + 110) % 200) + 30)
                        # Checkerboard pattern within tile for visual debugging
                        checker = (((tx // 4) + (ty // 4)) % 2) * 30
                        tex_data.extend([r + checker, g + checker, b + checker, 255])

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        raw = (ctypes.c_ubyte * len(tex_data))(*tex_data)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
                         ATLAS_SIZE, ATLAS_SIZE, 0,
                         gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, raw)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        return texture

    def load_atlas_png(self, path: str):
        """Load a real texture atlas from a PNG file."""
        try:
            from PIL import Image
            img  = Image.open(path).convert("RGBA")
            data = img.tobytes()
            w, h = img.size
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.atlas_texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0,
                             gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        except ImportError:
            print("[Renderer] Pillow not installed — using placeholder atlas")

    # ── Chunk Mesh Upload ─────────────────────────────────────────────────────

    def upload_chunk_mesh(self, chunk: 'Chunk', vertex_data: array.array):
        """Upload (or re-upload) a chunk's mesh to GPU."""
        key = (chunk.chunk_x, chunk.chunk_z)
        if key not in self.chunk_meshes:
            self.chunk_meshes[key] = GPUMesh()
        self.chunk_meshes[key].upload(vertex_data)

    def remove_chunk(self, cx: int, cz: int):
        """Free GPU memory for an unloaded chunk."""
        key = (cx, cz)
        if key in self.chunk_meshes:
            self.chunk_meshes[key].destroy()
            del self.chunk_meshes[key]

    # ── Main Render ───────────────────────────────────────────────────────────

    def render_world(self, mvp: list, model: list,
                      camera_pos: Tuple[float,float,float],
                      sun_dir: Tuple[float,float,float],
                      sky_color: Tuple[float,float,float],
                      sun_intensity: float,
                      game_time: float,
                      underwater: bool):
        """Render all loaded chunk meshes."""
        gl.glUseProgram(self.voxel_shader)

        # Set uniforms
        self._set_uniform_mat4('u_mvp',    mvp)
        self._set_uniform_mat4('u_model',  model)
        self._set_uniform_vec3('u_sun_dir', sun_dir)
        self._set_uniform_vec3('u_sky_color', sky_color)
        self._set_uniform_vec3('u_camera_pos', camera_pos)
        self._set_uniform_float('u_time',          game_time)
        self._set_uniform_float('u_sun_intensity',  sun_intensity)
        self._set_uniform_float('u_fog_start',      200.0)
        self._set_uniform_float('u_fog_end',        260.0)
        self._set_uniform_int('u_atlas', 0)
        self._set_uniform_bool('u_underwater', underwater)

        # Bind atlas
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.atlas_texture)

        # Draw all chunks
        for mesh in self.chunk_meshes.values():
            mesh.draw()

        gl.glUseProgram(0)

    def render_selection(self, mvp: list,
                          hit_x: int, hit_y: int, hit_z: int):
        """Render a wireframe/darkened highlight on the selected block."""
        gl.glUseProgram(self.selection_shader)

        # Offset MVP to place selection cube at block position
        sel_mvp = _translate_mvp(mvp, hit_x, hit_y, hit_z)
        self._set_uniform_mat4('u_mvp', sel_mvp)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(2.0)

        gl.glBindVertexArray(self.selection_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
        gl.glBindVertexArray(0)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glUseProgram(0)

    # ── Selection Cube ────────────────────────────────────────────────────────

    def _build_selection_vao(self) -> int:
        """Build a 1×1×1 cube VAO slightly expanded for outline visibility."""
        E = 0.005  # expand by 0.5%
        verts = _unit_cube_vertices(-E, 1 + E)
        raw = (ctypes.c_float * len(verts))(*verts)

        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(raw), raw,
                         gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False,
                                  3 * ctypes.sizeof(ctypes.c_float),
                                  ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)
        return vao

    # ── Uniform Helpers ───────────────────────────────────────────────────────

    def _get_loc(self, name: str) -> int:
        prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        key = (prog, name)
        if key not in self._uniforms:
            self._uniforms[key] = gl.glGetUniformLocation(prog, name)
        return self._uniforms[key]

    def _set_uniform_mat4(self, name: str, mat: list):
        loc = self._get_loc(name)
        if loc >= 0:
            flat = [float(x) for row in mat for x in row] if isinstance(mat[0], (list, tuple)) else mat
            gl.glUniformMatrix4fv(loc, 1, False, (ctypes.c_float * 16)(*flat))

    def _set_uniform_vec3(self, name: str, v: tuple):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    def _set_uniform_float(self, name: str, v: float):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform1f(loc, float(v))

    def _set_uniform_int(self, name: str, v: int):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform1i(loc, int(v))

    def _set_uniform_bool(self, name: str, v: bool):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform1i(loc, int(v))

    def clear(self, sky_color: Tuple[float,float,float]):
        r, g, b = sky_color
        gl.glClearColor(r, g, b, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


# ─── Math Helpers ─────────────────────────────────────────────────────────────

def _unit_cube_vertices(lo: float = 0.0, hi: float = 1.0):
    """Return 36 vertices (12 triangles) for a unit cube."""
    faces = [
        # TOP
        [(lo,hi,lo),(hi,hi,lo),(hi,hi,hi),(lo,hi,lo),(hi,hi,hi),(lo,hi,hi)],
        # BOTTOM
        [(lo,lo,hi),(hi,lo,hi),(hi,lo,lo),(lo,lo,hi),(hi,lo,lo),(lo,lo,lo)],
        # NORTH
        [(hi,hi,lo),(lo,hi,lo),(lo,lo,lo),(hi,hi,lo),(lo,lo,lo),(hi,lo,lo)],
        # SOUTH
        [(lo,hi,hi),(hi,hi,hi),(hi,lo,hi),(lo,hi,hi),(hi,lo,hi),(lo,lo,hi)],
        # EAST
        [(hi,hi,hi),(hi,hi,lo),(hi,lo,lo),(hi,hi,hi),(hi,lo,lo),(hi,lo,hi)],
        # WEST
        [(lo,hi,lo),(lo,hi,hi),(lo,lo,hi),(lo,hi,lo),(lo,lo,hi),(lo,lo,lo)],
    ]
    verts = []
    for face in faces:
        for v in face:
            verts.extend(v)
    return verts


def _translate_mvp(mvp: list, tx: float, ty: float, tz: float) -> list:
    """Apply a translation to the MVP matrix (for selection cube placement)."""
    # Create translation matrix
    t = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[tx,ty,tz,1]]
    # Multiply mvp × t (column-major OpenGL order)
    result = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += mvp[i][k] * t[k][j]
    return result
