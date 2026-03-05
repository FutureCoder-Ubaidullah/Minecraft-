"""
world/world.py — World Manager

Manages the infinite chunk grid:
  - Chunk loading / unloading based on player position
  - O(1) chunk lookup via dict keyed on (chunk_x, chunk_z)
  - Background terrain generation (threading)
  - Block get/set across chunk boundaries
  - Save/load via region files (Anvil-inspired)
  - Headless-safe: no OpenGL references
"""

from __future__ import annotations
import os
import struct
import gzip
import threading
import queue
import time
from typing import Dict, Optional, List, Tuple, Set

from world.chunk import Chunk, CHUNK_W, CHUNK_H, CHUNK_D
from world.noise import TerrainGenerator
from engine.registry import BLOCK_REGISTRY


# ─── World ────────────────────────────────────────────────────────────────────

class World:
    """
    Infinite voxel world.

    The world is organized as a flat dictionary:
        chunks: Dict[(cx, cz), Chunk]

    This gives O(1) average-case lookup for any chunk. The dict never
    fills memory because chunks are unloaded when out of render distance.

    Thread model:
        - Main thread: reads chunks, sets blocks, triggers mesh rebuilds
        - Generator thread: runs terrain generation, pushes to ready_queue
        - (Future) Lighting thread: propagates light changes
    """

    RENDER_DISTANCE = 8    # chunks in each direction (diameter = 17 chunks)
    UNLOAD_DISTANCE = 12   # chunks beyond this are unloaded

    def __init__(self, seed: int = 12345, world_dir: str = "saves/world"):
        self.seed       = seed
        self.world_dir  = world_dir
        self.chunks: Dict[Tuple[int,int], Chunk] = {}

        self.generator  = TerrainGenerator(seed)
        self.registry   = BLOCK_REGISTRY

        # Threading infrastructure
        self._gen_queue:   queue.Queue = queue.Queue()   # (cx, cz) → generate
        self._ready_queue: queue.Queue = queue.Queue()   # Chunk → main thread
        self._generating:  Set[Tuple[int,int]] = set()
        self._lock = threading.Lock()

        # Start background generator thread
        self._running = True
        self._gen_thread = threading.Thread(target=self._generation_worker,
                                             daemon=True, name="ChunkGen")
        self._gen_thread.start()

        # Track which chunks need mesh rebuilds (set by main thread)
        self.dirty_chunks: Set[Tuple[int,int]] = set()

        os.makedirs(world_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_chunk(self, cx: int, cz: int) -> Optional[Chunk]:
        """O(1) chunk lookup. Returns None if not loaded."""
        return self.chunks.get((cx, cz))

    def get_or_load_chunk(self, cx: int, cz: int) -> Optional[Chunk]:
        """Get chunk, scheduling generation if not present."""
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            key = (cx, cz)
            if key not in self._generating:
                self._generating.add(key)
                self._gen_queue.put(key)
        return chunk

    def get_block(self, wx: int, wy: int, wz: int) -> int:
        """Get block ID at world coordinates."""
        if wy < 0 or wy >= CHUNK_H:
            return 0
        cx, lx = divmod(wx, CHUNK_W)
        cz, lz = divmod(wz, CHUNK_D)
        # Handle negative coordinates
        if wx < 0 and lx != 0:
            cx -= 1; lx = CHUNK_W + lx
        if wz < 0 and lz != 0:
            cz -= 1; lz = CHUNK_D + lz
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return 0
        return chunk.get_block(lx, wy, lz)

    def set_block(self, wx: int, wy: int, wz: int, block_id: int):
        """Set block at world coordinates. Marks chunk (and neighbors) dirty."""
        if wy < 0 or wy >= CHUNK_H:
            return
        cx, lx = divmod(wx, CHUNK_W)
        cz, lz = divmod(wz, CHUNK_D)
        if wx < 0 and lx != 0:
            cx -= 1; lx = CHUNK_W + lx
        if wz < 0 and lz != 0:
            cz -= 1; lz = CHUNK_D + lz

        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return
        chunk.set_block(lx, wy, lz, block_id)
        self.dirty_chunks.add((cx, cz))

        # If block is on chunk border, neighboring chunk needs remesh
        if lx == 0:           self.dirty_chunks.add((cx - 1, cz))
        if lx == CHUNK_W - 1: self.dirty_chunks.add((cx + 1, cz))
        if lz == 0:           self.dirty_chunks.add((cx, cz - 1))
        if lz == CHUNK_D - 1: self.dirty_chunks.add((cx, cz + 1))

    def update(self, player_wx: float, player_wz: float):
        """
        Called every frame. Processes ready chunks and schedules
        load/unload based on player position.
        """
        pcx = int(player_wx) // CHUNK_W
        pcz = int(player_wz) // CHUNK_D

        # Absorb newly-generated chunks from background thread
        chunks_absorbed = 0
        while not self._ready_queue.empty() and chunks_absorbed < 4:
            chunk = self._ready_queue.get_nowait()
            self.chunks[(chunk.chunk_x, chunk.chunk_z)] = chunk
            self._link_neighbors(chunk.chunk_x, chunk.chunk_z)
            self.dirty_chunks.add((chunk.chunk_x, chunk.chunk_z))
            chunks_absorbed += 1

        # Schedule missing chunks within render distance
        rd = self.RENDER_DISTANCE
        for dz in range(-rd, rd + 1):
            for dx in range(-rd, rd + 1):
                cx, cz = pcx + dx, pcz + dz
                if (cx, cz) not in self.chunks and (cx, cz) not in self._generating:
                    self._generating.add((cx, cz))
                    self._gen_queue.put((cx, cz))

        # Unload distant chunks
        unload_dist_sq = self.UNLOAD_DISTANCE ** 2
        to_unload = [
            key for key in list(self.chunks.keys())
            if (key[0] - pcx)**2 + (key[1] - pcz)**2 > unload_dist_sq
        ]
        for key in to_unload:
            chunk = self.chunks.pop(key)
            self.dirty_chunks.discard(key)
            self._save_chunk(chunk)

    def get_loaded_chunks(self) -> List[Chunk]:
        return list(self.chunks.values())

    # ── Physics helpers ───────────────────────────────────────────────────────

    def is_solid(self, wx: int, wy: int, wz: int) -> bool:
        bid = self.get_block(wx, wy, wz)
        return self.registry.get(bid).solid

    def get_surface_y(self, wx: int, wz: int) -> int:
        """Scan downward to find topmost solid block."""
        for y in range(CHUNK_H - 1, -1, -1):
            if self.is_solid(wx, y, wz):
                return y
        return 0

    # ── Neighbor linking ─────────────────────────────────────────────────────

    def _link_neighbors(self, cx: int, cz: int):
        """Connect a chunk to its loaded neighbors for cross-chunk AO/meshing."""
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return
        # neighbors order: [north(-Z), south(+Z), east(+X), west(-X)]
        neighbor_keys = [(cx, cz-1), (cx, cz+1), (cx+1, cz), (cx-1, cz)]
        for i, nkey in enumerate(neighbor_keys):
            nbr = self.chunks.get(nkey)
            if nbr:
                chunk.neighbors[i] = nbr
                # Reverse link
                rev = [1, 0, 3, 2]
                nbr.neighbors[rev[i]] = chunk
                self.dirty_chunks.add(nkey)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _chunk_path(self, cx: int, cz: int) -> str:
        # Region files: group 32×32 chunks per file (Anvil-inspired)
        rx, rz = cx >> 5, cz >> 5
        return os.path.join(self.world_dir, f"r.{rx}.{rz}.bin")

    def _save_chunk(self, chunk: Chunk):
        """
        Serialize chunk to disk.
        Format: 4-byte header (cx, cz as int16) + gzip-compressed block data.
        """
        path = self._chunk_path(chunk.chunk_x, chunk.chunk_z)
        try:
            # Read existing region or create empty
            region = {}
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    region = self._read_region(f)
            key = (chunk.chunk_x & 31, chunk.chunk_z & 31)
            region[key] = gzip.compress(bytes(chunk.blocks), compresslevel=1)
            self._write_region(path, region)
        except Exception as e:
            pass  # Non-fatal — chunk will regenerate on next load

    def _load_chunk_from_disk(self, cx: int, cz: int) -> Optional[Chunk]:
        path = self._chunk_path(cx, cz)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                region = self._read_region(f)
            key = (cx & 31, cz & 31)
            if key not in region:
                return None
            chunk = Chunk(cx, cz)
            chunk.blocks = bytearray(gzip.decompress(region[key]))
            return chunk
        except Exception:
            return None

    def _read_region(self, f) -> dict:
        """Simple region format: 4-byte count, then (key_x, key_z, data_len, data)*."""
        data = f.read()
        if len(data) < 4:
            return {}
        region = {}
        count = struct.unpack_from('<I', data, 0)[0]
        offset = 4
        for _ in range(count):
            if offset + 6 > len(data):
                break
            kx, kz, dlen = struct.unpack_from('<bbI', data, offset)
            offset += 6
            region[(kx, kz)] = data[offset:offset + dlen]
            offset += dlen
        return region

    def _write_region(self, path: str, region: dict):
        parts = [struct.pack('<I', len(region))]
        for (kx, kz), blob in region.items():
            parts.append(struct.pack('<bbI', kx, kz, len(blob)))
            parts.append(blob)
        with open(path, 'wb') as f:
            f.write(b''.join(parts))

    # ── Background Generation Thread ─────────────────────────────────────────

    def _generation_worker(self):
        """
        Background thread: consumes (cx, cz) from _gen_queue,
        generates terrain, pushes Chunk to _ready_queue.
        """
        while self._running:
            try:
                cx, cz = self._gen_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Try disk first
            chunk = self._load_chunk_from_disk(cx, cz)
            if chunk is None:
                # Generate fresh
                chunk = Chunk(cx, cz)
                chunk.blocks = self.generator.generate_chunk(cx, cz)

                # Decorate with trees
                for lx, base_y, lz in self.generator.get_tree_positions(cx, cz):
                    self.generator.place_tree(chunk.blocks, lx, base_y, lz)

            self._ready_queue.put(chunk)
            self._generating.discard((cx, cz))

    def shutdown(self):
        """Save all loaded chunks and stop threads."""
        self._running = False
        self._gen_thread.join(timeout=2.0)
        for chunk in self.chunks.values():
            self._save_chunk(chunk)
