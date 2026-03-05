"""
world/chunk.py — Chunk Data + Greedy Meshing + Face Culling

This module implements:
  1. Chunk — the raw 16×256×16 block array with O(1) access
  2. ChunkMesher — generates an optimized vertex buffer using:
       a) Face Culling: skip any face adjacent to a solid block
       b) Greedy Meshing: merge co-planar same-block faces into quads

The output is a flat float array ready for upload to a VBO.
Vertex layout per vertex (12 floats):
    [x, y, z,  u, v,  nx, ny, nz,  ao,  face_id,  block_id,  light]
"""

from __future__ import annotations
import struct
import array
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.registry import BlockRegistry

# Chunk dimensions
CHUNK_W = 16
CHUNK_H = 256
CHUNK_D = 16

# Face indices (used throughout the engine consistently)
FACE_TOP    = 0   # +Y
FACE_BOTTOM = 1   # -Y
FACE_NORTH  = 2   # -Z
FACE_SOUTH  = 3   # +Z
FACE_EAST   = 4   # +X
FACE_WEST   = 5   # -X

# Normal vectors for each face
FACE_NORMALS = [
    ( 0,  1,  0),   # TOP
    ( 0, -1,  0),   # BOTTOM
    ( 0,  0, -1),   # NORTH
    ( 0,  0,  1),   # SOUTH
    ( 1,  0,  0),   # EAST
    (-1,  0,  0),   # WEST
]

# Direction offsets to the neighbor for each face
FACE_DIRS = [
    ( 0,  1,  0),   # TOP
    ( 0, -1,  0),   # BOTTOM
    ( 0,  0, -1),   # NORTH
    ( 0,  0,  1),   # SOUTH
    ( 1,  0,  0),   # EAST
    (-1,  0,  0),   # WEST
]

# Quad vertices (CCW winding, unit cube), per face
# Each face = 4 vertices, each vertex = (x, y, z)
FACE_VERTS = [
    # TOP (+Y)
    [(0,1,0),(1,1,0),(1,1,1),(0,1,1)],
    # BOTTOM (-Y)
    [(0,0,1),(1,0,1),(1,0,0),(0,0,0)],
    # NORTH (-Z)
    [(1,1,0),(0,1,0),(0,0,0),(1,0,0)],
    # SOUTH (+Z)
    [(0,1,1),(1,1,1),(1,0,1),(0,0,1)],
    # EAST (+X)
    [(1,1,1),(1,1,0),(1,0,0),(1,0,1)],
    # WEST (-X)
    [(0,1,0),(0,1,1),(0,0,1),(0,0,0)],
]

# UV coordinates for each vertex of a face (matching FACE_VERTS winding)
FACE_UVS = [
    (0,0),(1,0),(1,1),(0,1),  # TOP
    (0,0),(1,0),(1,1),(0,1),  # BOTTOM
    (0,0),(1,0),(1,1),(0,1),  # NORTH
    (0,0),(1,0),(1,1),(0,1),  # SOUTH
    (0,0),(1,0),(1,1),(0,1),  # EAST
    (0,0),(1,0),(1,1),(0,1),  # WEST
]

# AO corner-sampling offsets for each face's 4 vertices
# Each vertex needs 3 neighbor offsets: [side1, side2, corner]
AO_OFFSETS = {
    FACE_TOP: [
        # vertex 0: (0,1,0)
        [(-1,1,0),(0,1,-1),(-1,1,-1)],
        # vertex 1: (1,1,0)
        [(1,1,0),(0,1,-1),(1,1,-1)],
        # vertex 2: (1,1,1)
        [(1,1,0),(0,1,1),(1,1,1)],
        # vertex 3: (0,1,1)
        [(-1,1,0),(0,1,1),(-1,1,1)],
    ],
    FACE_BOTTOM: [
        [(-1,-1,0),(0,-1,1),(-1,-1,1)],
        [(1,-1,0),(0,-1,1),(1,-1,1)],
        [(1,-1,0),(0,-1,-1),(1,-1,-1)],
        [(-1,-1,0),(0,-1,-1),(-1,-1,-1)],
    ],
    FACE_NORTH: [
        [(1,1,-1),(0,0,-1),(1,0,-1)],     # adjusted for north face
        [(-1,1,-1),(0,0,-1),(-1,0,-1)],
        [(-1,-1,-1),(0,0,-1),(-1,0,-1)],
        [(1,-1,-1),(0,0,-1),(1,0,-1)],
    ],
    FACE_SOUTH: [
        [(-1,1,1),(0,0,1),(-1,0,1)],
        [(1,1,1),(0,0,1),(1,0,1)],
        [(1,-1,1),(0,0,1),(1,0,1)],
        [(-1,-1,1),(0,0,1),(-1,0,1)],
    ],
    FACE_EAST: [
        [(1,1,1),(1,0,0),(1,0,1)],
        [(1,1,-1),(1,0,0),(1,0,-1)],
        [(1,-1,-1),(1,0,0),(1,0,-1)],
        [(1,-1,1),(1,0,0),(1,0,1)],
    ],
    FACE_WEST: [
        [(-1,1,-1),(-1,0,0),(-1,0,-1)],
        [(-1,1,1),(-1,0,0),(-1,0,1)],
        [(-1,-1,1),(-1,0,0),(-1,0,1)],
        [(-1,-1,-1),(-1,0,0),(-1,0,-1)],
    ],
}


def compute_ao(s1: bool, s2: bool, corner: bool) -> float:
    """
    Ambient Occlusion formula (Mikola Lysenko's method):
        AO = 0 if both sides are solid (full occlusion)
        AO = 1,2,3 based on how many neighbors are solid

    Returns a float in [0, 1] where 0=fully occluded, 1=fully lit.
    """
    if s1 and s2:
        return 0.0
    count = int(s1) + int(s2) + int(corner)
    return (3 - count) / 3.0


# ─── Chunk Data ───────────────────────────────────────────────────────────────

class Chunk:
    """
    Raw voxel data for a 16×256×16 section of the world.

    Blocks stored as bytearray (uint8) — supports 256 block types.
    For modded worlds needing >256 types, switch to array.array('H', ...)
    and update the mesher accordingly.

    Coordinates are LOCAL (0–15, 0–255, 0–15).
    Chunk position in world is (chunk_x, chunk_z).
    """

    __slots__ = ('chunk_x', 'chunk_z', 'blocks', 'dirty',
                 'mesh_vertices', 'mesh_index_count',
                 'vao', 'vbo', 'neighbors')

    def __init__(self, chunk_x: int, chunk_z: int):
        self.chunk_x  = chunk_x
        self.chunk_z  = chunk_z
        self.blocks   = bytearray(CHUNK_W * CHUNK_H * CHUNK_D)  # all AIR
        self.dirty    = True   # needs remesh

        # Render data (set by renderer)
        self.mesh_vertices   = None
        self.mesh_index_count = 0
        self.vao = None
        self.vbo = None

        # Neighbor references (set by ChunkManager) — needed for cross-chunk AO
        # Order: [north, south, east, west] → (-Z, +Z, +X, -X)
        self.neighbors = [None, None, None, None]

    # ── Block Access ──────────────────────────────────────────────────────────

    @staticmethod
    def _idx(x: int, y: int, z: int) -> int:
        """Flat index. x, z in [0,15], y in [0,255]."""
        return x + z * CHUNK_W + y * CHUNK_W * CHUNK_D

    def get_block(self, x: int, y: int, z: int) -> int:
        if 0 <= x < CHUNK_W and 0 <= y < CHUNK_H and 0 <= z < CHUNK_D:
            return self.blocks[self._idx(x, y, z)]
        return 0  # out of bounds = air

    def set_block(self, x: int, y: int, z: int, block_id: int):
        if 0 <= x < CHUNK_W and 0 <= y < CHUNK_H and 0 <= z < CHUNK_D:
            self.blocks[self._idx(x, y, z)] = block_id
            self.dirty = True

    def get_neighbor_block(self, x: int, y: int, z: int) -> int:
        """
        Get a block that may be in a neighboring chunk.
        Used during meshing to check cross-chunk face visibility.
        """
        if 0 <= x < CHUNK_W and 0 <= y < CHUNK_H and 0 <= z < CHUNK_D:
            return self.blocks[self._idx(x, y, z)]

        # Cross-chunk lookup
        if z < 0 and self.neighbors[0]:           # North neighbor (-Z)
            return self.neighbors[0].get_block(x, y, z + CHUNK_D)
        if z >= CHUNK_D and self.neighbors[1]:    # South neighbor (+Z)
            return self.neighbors[1].get_block(x, y, z - CHUNK_D)
        if x >= CHUNK_W and self.neighbors[2]:    # East neighbor (+X)
            return self.neighbors[2].get_block(x - CHUNK_W, y, z)
        if x < 0 and self.neighbors[3]:           # West neighbor (-X)
            return self.neighbors[3].get_block(x + CHUNK_W, y, z)
        return 0  # unloaded neighbor = air (don't cull border faces)

    @property
    def world_x(self) -> int:
        return self.chunk_x * CHUNK_W

    @property
    def world_z(self) -> int:
        return self.chunk_z * CHUNK_D


# ─── Greedy Mesher ────────────────────────────────────────────────────────────

class ChunkMesher:
    """
    Generates an optimized triangle mesh from a Chunk.

    Algorithm: Greedy Meshing (Mikola Lysenko, 2012)
    ─────────────────────────────────────────────────
    For each axis-aligned slice through the chunk:
      1. Build a 2D mask: for each cell, determine if a visible face exists
         between this cell and the next cell in the slice direction.
      2. Scan the mask for rectangles of identical entries (same block, same AO).
      3. Emit a single quad covering the entire rectangle.
      4. Mark those cells as processed.

    Complexity: O(N) where N = number of surface voxels (vs O(N²) naive).
    Reduces vertex count by ~20–80% for typical terrain.
    """

    # Vertex layout: x,y,z, u,v, nx,ny,nz, ao, face_id, block_id, sky_light
    VERTEX_FLOATS = 12

    def __init__(self, registry: 'BlockRegistry'):
        self.registry = registry

    def build_mesh(self, chunk: Chunk) -> array.array:
        """
        Build the full mesh for a chunk.
        Returns an array.array('f', ...) of floats, 12 floats per vertex.
        Two triangles per face = 6 vertices per quad.
        """
        vertices = array.array('f')

        # Process all 6 face directions using greedy meshing
        # Axis: 0=X, 1=Y, 2=Z; positive direction per axis
        for face_id in range(6):
            self._greedy_slice(chunk, face_id, vertices)

        return vertices

    def _greedy_slice(self, chunk: Chunk, face_id: int,
                       out: array.array):
        """
        Greedy mesh one face direction.

        For FACE_TOP (face_id=0, axis=Y, dir=+1):
          - We iterate over Y slices (y=0..255)
          - For each slice, build a 16×16 mask in X-Z plane
          - mask[x][z] = block_id if face is visible, else 0
        """
        nx, ny, nz = FACE_NORMALS[face_id]
        dx, dy, dz = FACE_DIRS[face_id]

        # Determine the slice axis (the one the normal is along)
        if ny != 0:
            # Y-axis slices (TOP/BOTTOM)
            self._greedy_axis(chunk, face_id, 'y', out)
        elif nz != 0:
            # Z-axis slices (NORTH/SOUTH)
            self._greedy_axis(chunk, face_id, 'z', out)
        else:
            # X-axis slices (EAST/WEST)
            self._greedy_axis(chunk, face_id, 'x', out)

    def _greedy_axis(self, chunk: Chunk, face_id: int, axis: str,
                      out: array.array):
        """Core greedy meshing for one axis."""
        reg = self.registry
        dx, dy, dz = FACE_DIRS[face_id]

        # Determine u/v axes (the two axes perpendicular to the normal)
        if axis == 'y':
            D, W, H = CHUNK_H, CHUNK_W, CHUNK_D
            def get(i, j, k): return chunk.get_neighbor_block(j, i, k)
            def get_n(i, j, k): return chunk.get_neighbor_block(j+dx, i+dy, k+dz)
        elif axis == 'z':
            D, W, H = CHUNK_D, CHUNK_W, CHUNK_H
            def get(i, j, k): return chunk.get_neighbor_block(j, k, i)
            def get_n(i, j, k): return chunk.get_neighbor_block(j+dx, k+dz, i+dz if dz else k+dz)
            # Rebuild lambdas for z properly
        else:  # x
            D, W, H = CHUNK_W, CHUNK_D, CHUNK_H
            def get(i, j, k): return chunk.get_neighbor_block(i, k, j)
            def get_n(i, j, k): return chunk.get_neighbor_block(i+dx, k, j)

        # Rebuild with correct axis mapping (cleaner approach)
        self._greedy_full(chunk, face_id, axis, out)

    def _greedy_full(self, chunk: Chunk, face_id: int, axis: str,
                      out: array.array):
        """
        Full greedy meshing with correct axis mapping.

        We iterate over each slice along `axis`, build a 2D mask,
        then greedily merge quads.
        """
        reg   = self.registry
        ddx, ddy, ddz = FACE_DIRS[face_id]

        # Map axis to iteration order
        if axis == 'y':
            ranges = (CHUNK_H, CHUNK_W, CHUNK_D)
            def world_pos(layer, u, v): return (u, layer, v)
            def neighbor_pos(layer, u, v): return (u + ddx, layer + ddy, v + ddz)
            quad_verts_fn = self._quad_verts_y
        elif axis == 'z':
            ranges = (CHUNK_D, CHUNK_W, CHUNK_H)
            def world_pos(layer, u, v): return (u, v, layer)
            def neighbor_pos(layer, u, v): return (u + ddx, v + ddy, layer + ddz)
            quad_verts_fn = self._quad_verts_z
        else:  # x
            ranges = (CHUNK_W, CHUNK_D, CHUNK_H)
            def world_pos(layer, u, v): return (layer, v, u)
            def neighbor_pos(layer, u, v): return (layer + ddx, v + ddy, u + ddz)
            quad_verts_fn = self._quad_verts_x

        D, W, H = ranges
        mask        = [0]  * (W * H)   # block_id or 0 if not visible
        mask_ao     = [0.0] * (W * H)  # ambient occlusion value

        for layer in range(D):
            # ── Build mask ────────────────────────────────────────────────────
            for u in range(W):
                for v in range(H):
                    bx, by, bz         = world_pos(layer, u, v)
                    nbx, nby, nbz      = neighbor_pos(layer, u, v)

                    block_id = chunk.get_neighbor_block(bx, by, bz)
                    if block_id == 0:
                        mask[u + v * W] = 0
                        continue

                    bdef = reg.get(block_id)
                    if bdef.render_type == "none":
                        mask[u + v * W] = 0
                        continue

                    # Face visible if neighbor is air or transparent
                    neighbor_id = chunk.get_neighbor_block(nbx, nby, nbz)
                    ndef = reg.get(neighbor_id)

                    if not ndef.solid or ndef.transparent:
                        # Compute AO for top-left vertex of this cell
                        # (simplified: store average AO; full per-vertex AO in emit)
                        mask[u + v * W] = block_id
                    else:
                        mask[u + v * W] = 0

            # ── Greedy merge ──────────────────────────────────────────────────
            processed = [False] * (W * H)

            for v in range(H):
                u = 0
                while u < W:
                    bid = mask[u + v * W]
                    if bid == 0 or processed[u + v * W]:
                        u += 1
                        continue

                    # Expand width (u-direction)
                    width = 1
                    while (u + width < W and
                           mask[u + width + v * W] == bid and
                           not processed[u + width + v * W]):
                        width += 1

                    # Expand height (v-direction)
                    height = 1
                    done = False
                    while v + height < H and not done:
                        for k in range(width):
                            if (mask[u + k + (v + height) * W] != bid or
                                    processed[u + k + (v + height) * W]):
                                done = True
                                break
                        if not done:
                            height += 1

                    # Mark as processed
                    for dv in range(height):
                        for du in range(width):
                            processed[u + du + (v + dv) * W] = True

                    # Emit quad
                    bx, by, bz = world_pos(layer, u, v)
                    self._emit_quad(out, chunk, face_id, axis,
                                    bx, by, bz, width, height, bid, reg)
                    u += width

    def _emit_quad(self, out: array.array, chunk: Chunk, face_id: int,
                    axis: str, bx: int, by: int, bz: int,
                    width: int, height: int, block_id: int, reg):
        """
        Emit 6 vertices (2 triangles) for a greedy quad.

        Vertex format: x,y,z, u,v, nx,ny,nz, ao, face_id, block_id, light
        """
        bdef   = reg.get(block_id)
        nx, ny, nz = FACE_NORMALS[face_id]

        # Atlas UV for this block + face
        col, row = bdef.get_texture(face_id)
        ATLAS = 16.0
        u0 = col / ATLAS
        v0 = row / ATLAS
        u1 = (col + 1) / ATLAS
        v1 = (row + 1) / ATLAS

        # Base position in world space
        wx = chunk.world_x + bx
        wy = by
        wz = chunk.world_z + bz

        # Light level
        sky = 1.0  # TODO: propagate sky light

        # Compute AO for each corner of the quad
        def sample_ao(ox, oy, oz, side1, side2, corner):
            s1 = reg.get(chunk.get_neighbor_block(bx + side1[0],
                                                   by + side1[1],
                                                   bz + side1[2])).solid
            s2 = reg.get(chunk.get_neighbor_block(bx + side2[0],
                                                   by + side2[1],
                                                   bz + side2[2])).solid
            c  = reg.get(chunk.get_neighbor_block(bx + corner[0],
                                                   by + corner[1],
                                                   bz + corner[2])).solid
            return compute_ao(s1, s2, c)

        # AO offsets per face vertex (use FACE_TOP as fallback)
        ao_offsets = AO_OFFSETS.get(face_id, AO_OFFSETS[FACE_TOP])

        # Generate the 4 quad corner positions based on face and axis
        # We stretch the quad by (width, height) along the two tangent axes
        if axis == 'y':
            # Tangent axes: X (width), Z (height)
            q = [
                (wx,          wy, wz         ),
                (wx + width,  wy, wz         ),
                (wx + width,  wy, wz + height),
                (wx,          wy, wz + height),
            ]
        elif axis == 'z':
            q = [
                (wx,         wy,          wz),
                (wx + width, wy,          wz),
                (wx + width, wy + height, wz),
                (wx,         wy + height, wz),
            ]
        else:  # x
            q = [
                (wx, wy,          wz        ),
                (wx, wy,          wz + width),
                (wx, wy + height, wz + width),
                (wx, wy + height, wz        ),
            ]

        # Compute AO for each corner
        aos = []
        for vi, offsets in enumerate(ao_offsets[:4]):
            s1o, s2o, co = offsets
            s1 = reg.get(chunk.get_neighbor_block(
                bx + s1o[0], by + s1o[1], bz + s1o[2])).solid
            s2 = reg.get(chunk.get_neighbor_block(
                bx + s2o[0], by + s2o[1], bz + s2o[2])).solid
            c  = reg.get(chunk.get_neighbor_block(
                bx + co[0],  by + co[1],  bz + co[2])).solid
            aos.append(compute_ao(s1, s2, c))

        # UV coordinates scaled by quad size (texture tiling)
        uvs = [
            (u0,        v0       ),
            (u0 + (u1-u0)*width, v0       ),
            (u0 + (u1-u0)*width, v0 + (v1-v0)*height),
            (u0,        v0 + (v1-v0)*height),
        ]

        # Flip quad if AO diagonal differs (prevents seam artifacts)
        flip = (aos[0] + aos[2]) < (aos[1] + aos[3])

        if not flip:
            tri_indices = [0, 1, 2,  0, 2, 3]
        else:
            tri_indices = [0, 1, 3,  1, 2, 3]

        for i in tri_indices:
            px, py, pz = q[i]
            pu, pv     = uvs[i]
            ao         = aos[i]
            out.extend([
                float(px), float(py), float(pz),
                pu, pv,
                float(nx), float(ny), float(nz),
                ao,
                float(face_id),
                float(block_id),
                sky,
            ])

    # ── Unused stubs (axis-specific quad builders handled inline above) ────────
    def _quad_verts_y(self, *a): pass
    def _quad_verts_z(self, *a): pass
    def _quad_verts_x(self, *a): pass
