"""
world/noise.py — 3D Perlin Noise & Terrain Generation

Full implementation of:
  - Improved Perlin Noise (Ken Perlin's 2002 reference implementation)
  - Fractal Brownian Motion (fBm) for layered detail
  - Height-map terrain with biome weighting
  - 3D cave carving via 3D Perlin threshold
  - Ore vein generation
  - Tree/structure placement
"""

from __future__ import annotations
import math
import random
from typing import Tuple


# ─── Permutation Table ────────────────────────────────────────────────────────

class PerlinNoise:
    """
    Ken Perlin's Improved Noise (2002).
    Reference: https://cs.nyu.edu/~perlin/noise/

    This is a pure-Python implementation suitable for world generation
    (called once per chunk, not per frame). For runtime performance,
    the C extension or numpy vectorized version should be used.
    """

    def __init__(self, seed: int = 0):
        # Build permutation table seeded deterministically
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        # Double to avoid index overflow
        self._perm = p * 2
        # Gradient table: 16 gradient vectors in 3D
        self._grad3 = [
            (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
            (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
            (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
            (1,1,0),(0,-1,1),(-1,1,0),(0,-1,-1),
        ]

    # ── Core math ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fade(t: float) -> float:
        """Quintic ease curve: 6t⁵ - 15t⁴ + 10t³"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(t: float, a: float, b: float) -> float:
        return a + t * (b - a)

    def _grad(self, h: int, x: float, y: float, z: float) -> float:
        """Compute dot product of gradient vector (indexed by h) with (x,y,z)."""
        gx, gy, gz = self._grad3[h & 15]
        return gx * x + gy * y + gz * z

    # ── 2D Noise ──────────────────────────────────────────────────────────────

    def noise2(self, x: float, y: float) -> float:
        """Returns value in [-1, 1]."""
        return self.noise3(x, y, 0.0)

    # ── 3D Noise ──────────────────────────────────────────────────────────────

    def noise3(self, x: float, y: float, z: float) -> float:
        """
        Classic 3D Perlin noise.
        Returns value in approximately [-1, 1] (practical range ~[-0.87, 0.87]).
        """
        p = self._perm

        # Unit cube corner
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        zi = int(math.floor(z)) & 255

        # Relative position in cube
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        zf = z - math.floor(z)

        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        # Hash coordinates of the 8 cube corners
        A  = p[xi]   + yi;  AA = p[A] + zi;  AB = p[A+1] + zi
        B  = p[xi+1] + yi;  BA = p[B] + zi;  BB = p[B+1] + zi

        # Trilinear interpolation of gradients
        return self._lerp(w,
            self._lerp(v,
                self._lerp(u, self._grad(p[AA  ], xf,   yf,   zf  ),
                              self._grad(p[BA  ], xf-1, yf,   zf  )),
                self._lerp(u, self._grad(p[AB  ], xf,   yf-1, zf  ),
                              self._grad(p[BB  ], xf-1, yf-1, zf  ))),
            self._lerp(v,
                self._lerp(u, self._grad(p[AA+1], xf,   yf,   zf-1),
                              self._grad(p[BA+1], xf-1, yf,   zf-1)),
                self._lerp(u, self._grad(p[AB+1], xf,   yf-1, zf-1),
                              self._grad(p[BB+1], xf-1, yf-1, zf-1))))


# ─── Fractal Brownian Motion ─────────────────────────────────────────────────

class FBM:
    """
    Fractal Brownian Motion — sums multiple octaves of Perlin noise.

    Each octave doubles frequency (lacunarity) and halves amplitude (gain).
    This creates the characteristic natural-looking layered terrain.

    Height formula:
        H(x,z) = Σᵢ amplitude_i * noise(x * frequency_i, z * frequency_i)
        where amplitude_i = gain^i,  frequency_i = lacunarity^i
    """

    def __init__(self, noise: PerlinNoise, octaves: int = 6,
                 lacunarity: float = 2.0, gain: float = 0.5):
        self.noise      = noise
        self.octaves    = octaves
        self.lacunarity = lacunarity
        self.gain       = gain

    def sample2(self, x: float, z: float) -> float:
        """Returns fBm value, normalized to [0, 1]."""
        total = 0.0
        freq  = 1.0
        amp   = 1.0
        max_v = 0.0
        for _ in range(self.octaves):
            total += self.noise.noise2(x * freq, z * freq) * amp
            max_v += amp
            amp  *= self.gain
            freq *= self.lacunarity
        return (total / max_v + 1.0) * 0.5   # remap [-1,1] → [0,1]

    def sample3(self, x: float, y: float, z: float) -> float:
        """3D fBm — used for cave carving."""
        total = 0.0
        freq  = 1.0
        amp   = 1.0
        max_v = 0.0
        for _ in range(self.octaves):
            total += self.noise.noise3(x * freq, y * freq, z * freq) * amp
            max_v += amp
            amp  *= self.gain
            freq *= self.lacunarity
        return total / max_v   # stays in [-1, 1]


# ─── Biome System ─────────────────────────────────────────────────────────────

class BiomeType:
    OCEAN      = 0
    PLAINS     = 1
    DESERT     = 2
    FOREST     = 3
    MOUNTAINS  = 4
    TUNDRA     = 5
    SWAMP      = 6


def classify_biome(temperature: float, humidity: float, height: float) -> int:
    """
    Map (temperature, humidity, height) all in [0,1] → BiomeType.

    Uses Whittaker biome classification adapted for Minecraft-style worlds.
    """
    if height < 0.35:
        return BiomeType.OCEAN
    if height > 0.80:
        return BiomeType.MOUNTAINS
    if temperature < 0.20:
        return BiomeType.TUNDRA
    if temperature > 0.75 and humidity < 0.25:
        return BiomeType.DESERT
    if humidity > 0.65:
        return BiomeType.SWAMP if height < 0.45 else BiomeType.FOREST
    return BiomeType.PLAINS


# ─── Terrain Generator ────────────────────────────────────────────────────────

CHUNK_W = 16   # X width
CHUNK_H = 256  # Y height
CHUNK_D = 16   # Z depth

# Block IDs (avoid circular import — use raw ints, registry resolves names)
AIR_ID        = 0
STONE_ID      = 1
GRASS_ID      = 2
DIRT_ID       = 3
SAND_ID       = 12
GRAVEL_ID     = 13
BEDROCK_ID    = 7
SNOW_ID       = 80
ICE_ID        = 79
WATER_ID      = 9
LOG_ID        = 17
LEAVES_ID     = 18


class TerrainGenerator:
    """
    Generates chunk block data using layered noise.

    Pipeline per chunk:
      1. Sample 2D heightmap for terrain surface
      2. Sample temperature / humidity for biome
      3. Apply 3D cave noise to carve caves
      4. Place ores via 3D threshold noise
      5. Decorate surface: trees, flowers, snow caps
    """

    SEA_LEVEL = 62

    def __init__(self, seed: int):
        self.seed = seed

        # Terrain shape noise (large-scale mountains)
        self.terrain_noise    = FBM(PerlinNoise(seed),         octaves=6, lacunarity=2.0, gain=0.5)
        # Fine detail on top of terrain
        self.detail_noise     = FBM(PerlinNoise(seed + 1),     octaves=4, lacunarity=2.0, gain=0.45)
        # Temperature map (large scale)
        self.temp_noise       = FBM(PerlinNoise(seed + 2),     octaves=3, lacunarity=2.0, gain=0.6)
        # Humidity map
        self.humid_noise      = FBM(PerlinNoise(seed + 3),     octaves=3, lacunarity=2.0, gain=0.6)
        # 3D cave noise (two separate fields; caves appear at intersections)
        self.cave_noise1      = FBM(PerlinNoise(seed + 4),     octaves=3, lacunarity=2.0, gain=0.5)
        self.cave_noise2      = FBM(PerlinNoise(seed + 5),     octaves=3, lacunarity=2.0, gain=0.5)
        # Ore placement
        self.ore_noise_coal   = FBM(PerlinNoise(seed + 10),    octaves=2, lacunarity=2.0, gain=0.5)
        self.ore_noise_iron   = FBM(PerlinNoise(seed + 11),    octaves=2, lacunarity=2.0, gain=0.5)
        self.ore_noise_gold   = FBM(PerlinNoise(seed + 12),    octaves=2, lacunarity=2.0, gain=0.5)
        self.ore_noise_diamond= FBM(PerlinNoise(seed + 13),    octaves=2, lacunarity=2.0, gain=0.5)
        # Tree placement
        self.tree_noise       = PerlinNoise(seed + 20)

    # ── Height Sampling ───────────────────────────────────────────────────────

    def get_surface_height(self, world_x: int, world_z: int) -> int:
        """
        Compute the Y coordinate of the topmost solid block at (x, z).

        Uses multi-octave fBm with domain-warping for more natural shapes.

        Domain warping:
            Offset the sample coordinates by another noise field to break
            the grid-aligned look:
                x' = x + warp_scale * warp_noise(x, z)
                z' = z + warp_scale * warp_noise(x + 5.2, z + 1.3)
        """
        # Coordinate scaling
        TERRAIN_SCALE = 0.003   # Large-scale continent shape
        DETAIL_SCALE  = 0.015   # Fine terrain detail

        sx = world_x * TERRAIN_SCALE
        sz = world_z * TERRAIN_SCALE

        # Domain warp (cheap version: use detail noise as offset)
        warp_x = self.detail_noise.sample2(sx + 1.7, sz + 9.2) * 2.0 - 1.0
        warp_z = self.detail_noise.sample2(sx + 8.3, sz + 2.8) * 2.0 - 1.0
        warp_strength = 0.5

        # Continent base (very large bumps)
        base = self.terrain_noise.sample2(sx + warp_x * warp_strength,
                                           sz + warp_z * warp_strength)
        # Local terrain detail
        detail = self.detail_noise.sample2(world_x * DETAIL_SCALE,
                                            world_z * DETAIL_SCALE) * 0.3

        # Combined height: remap to [4, 200]
        combined = base * 0.7 + detail
        height = int(4 + combined * 196)
        return max(4, min(CHUNK_H - 1, height))

    def get_biome(self, world_x: int, world_z: int) -> int:
        BIOME_SCALE = 0.002
        temp  = self.temp_noise.sample2( world_x * BIOME_SCALE, world_z * BIOME_SCALE)
        humid = self.humid_noise.sample2(world_x * BIOME_SCALE, world_z * BIOME_SCALE)
        h_norm = (self.get_surface_height(world_x, world_z) - 4) / 196.0
        return classify_biome(temp, humid, h_norm)

    # ── Cave Carving ──────────────────────────────────────────────────────────

    def is_cave(self, world_x: int, world_y: int, world_z: int) -> bool:
        """
        3D Perlin caves using the "two-noise intersection" technique.

        Two independent 3D noise fields are sampled. A cave voxel occurs
        when BOTH fields are near zero (i.e., near the zero-isosurface of
        each field). This produces natural tube-shaped caves.

        Threshold: |n1| < T AND |n2| < T
        Typical T = 0.08–0.12 gives good cave density.
        """
        CAVE_SCALE = 0.04
        CAVE_THRESH = 0.10

        # Don't carve the top 8 blocks (preserve surface)
        if world_y > self.get_surface_height(world_x, world_z) - 8:
            return False
        # Don't carve below y=2 (keep bedrock layer)
        if world_y <= 2:
            return False

        n1 = self.cave_noise1.sample3(
            world_x * CAVE_SCALE,
            world_y * CAVE_SCALE * 0.5,   # caves stretch horizontally
            world_z * CAVE_SCALE
        )
        n2 = self.cave_noise2.sample3(
            world_x * CAVE_SCALE + 100,
            world_y * CAVE_SCALE * 0.5 + 100,
            world_z * CAVE_SCALE + 100
        )
        return abs(n1) < CAVE_THRESH and abs(n2) < CAVE_THRESH

    # ── Ore Placement ─────────────────────────────────────────────────────────

    def get_ore(self, world_x: int, world_y: int, world_z: int) -> int:
        """
        Returns ore block ID if this position should have ore, else 0.
        Uses 3D Perlin threshold noise (similar to cave carving but tighter).

        Ore spawn tables (Y range, threshold, ID):
          Coal:    y 5–128,  thresh 0.04, id 16
          Iron:    y 5–64,   thresh 0.05, id 15
          Gold:    y 5–32,   thresh 0.06, id 14
          Diamond: y 5–16,   thresh 0.07, id 56
        """
        ORE_SCALE = 0.08

        def check(noise_fbm, y_min, y_max, threshold, ore_id):
            if not (y_min <= world_y <= y_max):
                return 0
            n = noise_fbm.sample3(
                world_x * ORE_SCALE,
                world_y * ORE_SCALE,
                world_z * ORE_SCALE
            )
            return ore_id if abs(n) < threshold else 0

        # Diamond first (rarest, should override others if overlapping)
        for y_min, y_max, thr, ore_id in [
            (5, 16,  0.03, 56),   # diamond
            (5, 32,  0.05, 14),   # gold
            (5, 64,  0.06, 15),   # iron
            (5, 128, 0.07, 16),   # coal
        ]:
            if not (y_min <= world_y <= y_max):
                continue
            # Re-use the appropriate noise field
            noise_map = {56: self.ore_noise_diamond, 14: self.ore_noise_gold,
                         15: self.ore_noise_iron,    16: self.ore_noise_coal}
            n = noise_map[ore_id].sample3(
                world_x * ORE_SCALE,
                world_y * ORE_SCALE,
                world_z * ORE_SCALE
            )
            if abs(n) < thr:
                return ore_id
        return 0

    # ── Main Chunk Generation ─────────────────────────────────────────────────

    def generate_chunk(self, chunk_x: int, chunk_z: int) -> bytearray:
        """
        Generate a complete 16×256×16 chunk.

        Returns a flat bytearray of size CHUNK_W * CHUNK_H * CHUNK_D
        indexed as: data[x + z * CHUNK_W + y * CHUNK_W * CHUNK_D]

        Note: For blocks requiring IDs > 255 (future modding), switch to
        array('H', ...) (unsigned short). This implementation uses bytes
        for cache efficiency.
        """
        data   = bytearray(CHUNK_W * CHUNK_H * CHUNK_D)
        origin_x = chunk_x * CHUNK_W
        origin_z = chunk_z * CHUNK_D

        for lx in range(CHUNK_W):
            for lz in range(CHUNK_D):
                wx = origin_x + lx
                wz = origin_z + lz

                surface_y = self.get_surface_height(wx, wz)
                biome     = self.get_biome(wx, wz)

                # Choose surface blocks by biome
                if biome == BiomeType.DESERT:
                    surface_block = SAND_ID
                    sub_block     = SAND_ID
                    sub_depth     = 4
                elif biome == BiomeType.TUNDRA:
                    surface_block = SNOW_ID
                    sub_block     = DIRT_ID
                    sub_depth     = 4
                elif biome == BiomeType.OCEAN or surface_y < self.SEA_LEVEL:
                    surface_block = GRAVEL_ID
                    sub_block     = GRAVEL_ID
                    sub_depth     = 3
                else:
                    surface_block = GRASS_ID
                    sub_block     = DIRT_ID
                    sub_depth     = 4

                for ly in range(CHUNK_H):
                    idx = lx + lz * CHUNK_W + ly * CHUNK_W * CHUNK_D

                    # ── Bedrock floor (bottom 4 layers, randomized) ───────────
                    if ly == 0:
                        data[idx] = BEDROCK_ID
                        continue
                    if ly <= 3:
                        # Bedrock thins out probabilistically toward y=4
                        rng_val = PerlinNoise(self.seed + ly).noise2(wx * 0.5, wz * 0.5)
                        data[idx] = BEDROCK_ID if rng_val > (ly - 1) * 0.25 else STONE_ID
                        continue

                    # ── Air above surface ────────────────────────────────────
                    if ly > surface_y:
                        if ly <= self.SEA_LEVEL and biome == BiomeType.OCEAN:
                            data[idx] = WATER_ID
                        # else stays AIR (0)
                        continue

                    # ── Cave carving ─────────────────────────────────────────
                    if self.is_cave(wx, ly, wz):
                        # Flood caves below sea level with water
                        if ly < self.SEA_LEVEL:
                            data[idx] = WATER_ID
                        # else AIR (0)
                        continue

                    # ── Surface layers ────────────────────────────────────────
                    depth = surface_y - ly
                    if depth == 0:
                        data[idx] = surface_block
                    elif depth <= sub_depth:
                        data[idx] = sub_block
                    else:
                        # Ore check before defaulting to stone
                        ore = self.get_ore(wx, ly, wz)
                        data[idx] = ore if ore else STONE_ID

        return data

    # ── Tree Spawning ─────────────────────────────────────────────────────────

    def get_tree_positions(self, chunk_x: int, chunk_z: int):
        """
        Return list of (local_x, surface_y, local_z) for tree placement.
        Uses Poisson-disk-like spacing via noise thresholding.
        """
        TREE_SCALE = 0.3
        TREE_THRESH = 0.85
        MARGIN = 2   # Don't spawn trees at chunk edge (bleed-over handled separately)

        positions = []
        origin_x = chunk_x * CHUNK_W
        origin_z = chunk_z * CHUNK_D

        for lx in range(MARGIN, CHUNK_W - MARGIN):
            for lz in range(MARGIN, CHUNK_D - MARGIN):
                wx = origin_x + lx
                wz = origin_z + lz
                biome = self.get_biome(wx, wz)
                if biome not in (BiomeType.FOREST, BiomeType.PLAINS):
                    continue
                # Noise-based placement (reproducible, no RNG state issues)
                n = self.tree_noise.noise2(wx * TREE_SCALE, wz * TREE_SCALE)
                if n > TREE_THRESH:
                    sy = self.get_surface_height(wx, wz)
                    if sy > self.SEA_LEVEL:
                        positions.append((lx, sy + 1, lz))
        return positions

    def place_tree(self, data: bytearray, lx: int, base_y: int, lz: int,
                   trunk_height: int = 5):
        """
        Carve a simple oak tree into chunk data.
        Clamps coordinates to chunk bounds (cross-chunk foliage handled by
        the chunk manager after all neighbors exist).
        """
        def set_block(x, y, z, block_id):
            if 0 <= x < CHUNK_W and 0 < y < CHUNK_H and 0 <= z < CHUNK_D:
                data[x + z * CHUNK_W + y * CHUNK_W * CHUNK_D] = block_id

        # Trunk
        for dy in range(trunk_height):
            set_block(lx, base_y + dy, lz, LOG_ID)

        # Foliage crown (3 layers)
        top = base_y + trunk_height
        # Top cap (1 block)
        set_block(lx, top, lz, LEAVES_ID)
        # Second layer (3x3)
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                set_block(lx + dx, top - 1, lz + dz, LEAVES_ID)
        # Third layer (5x5, skip corners)
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if abs(dx) == 2 and abs(dz) == 2:
                    continue  # skip corners
                set_block(lx + dx, top - 2, lz + dz, LEAVES_ID)
        # Fourth layer (same as third, one below)
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if abs(dx) == 2 and abs(dz) == 2:
                    continue
                if data[(lx + dx) + (lz + dz) * CHUNK_W +
                        (top - 3) * CHUNK_W * CHUNK_D] == AIR_ID:
                    set_block(lx + dx, top - 3, lz + dz, LEAVES_ID)
