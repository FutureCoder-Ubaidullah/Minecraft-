"""
tests/test_core.py — Unit Tests

Tests the pure-logic modules that need no GPU:
  - Perlin noise statistical properties
  - Terrain generator determinism
  - AABB collision math
  - Raycasting correctness
  - Block registry lookup
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import pytest

from world.noise import PerlinNoise, FBM, TerrainGenerator
from engine.registry import BlockRegistry, BlockDef, create_default_registry


# ─── Perlin Noise Tests ───────────────────────────────────────────────────────

class TestPerlinNoise:
    def test_range(self):
        """Perlin noise should stay within [-1, 1]."""
        noise = PerlinNoise(seed=42)
        for x in range(0, 100, 7):
            for z in range(0, 100, 7):
                val = noise.noise2(x * 0.1, z * 0.1)
                assert -1.1 <= val <= 1.1, f"Noise out of range: {val}"

    def test_determinism(self):
        """Same seed must produce identical values."""
        n1 = PerlinNoise(seed=100)
        n2 = PerlinNoise(seed=100)
        for i in range(50):
            v1 = n1.noise3(i * 0.13, i * 0.07, i * 0.17)
            v2 = n2.noise3(i * 0.13, i * 0.07, i * 0.17)
            assert v1 == v2, "Noise not deterministic!"

    def test_different_seeds(self):
        """Different seeds must produce different values (statistically)."""
        n1 = PerlinNoise(seed=1)
        n2 = PerlinNoise(seed=2)
        diffs = [abs(n1.noise2(i*0.1, i*0.1) - n2.noise2(i*0.1, i*0.1))
                  for i in range(20)]
        assert sum(diffs) > 0.1, "Different seeds produced identical noise"

    def test_3d_noise(self):
        """3D noise should also be in range."""
        noise = PerlinNoise(seed=7)
        for i in range(50):
            val = noise.noise3(i*0.15, i*0.23, i*0.11)
            assert -1.5 <= val <= 1.5

    def test_fbm_range(self):
        """FBM should produce values in [0, 1] after normalization."""
        fbm = FBM(PerlinNoise(seed=42), octaves=6)
        for x in range(0, 20):
            for z in range(0, 20):
                val = fbm.sample2(x * 0.003, z * 0.003)
                assert 0.0 <= val <= 1.0, f"FBM out of [0,1]: {val}"


# ─── Terrain Generator Tests ──────────────────────────────────────────────────

class TestTerrainGenerator:
    def test_chunk_size(self):
        """Generated chunk must be exactly 16×256×16 bytes."""
        gen = TerrainGenerator(seed=12345)
        data = gen.generate_chunk(0, 0)
        assert len(data) == 16 * 256 * 16

    def test_bedrock_at_y0(self):
        """y=0 must always be bedrock (ID=7)."""
        gen = TerrainGenerator(seed=999)
        data = gen.generate_chunk(0, 0)
        for x in range(16):
            for z in range(16):
                idx = x + z * 16 + 0 * 16 * 16
                assert data[idx] == 7, f"y=0 is not bedrock at ({x},{z})"

    def test_determinism(self):
        """Same seed must produce identical chunks."""
        gen1 = TerrainGenerator(seed=555)
        gen2 = TerrainGenerator(seed=555)
        d1 = gen1.generate_chunk(3, 7)
        d2 = gen2.generate_chunk(3, 7)
        assert d1 == d2

    def test_different_seeds_differ(self):
        """Different seeds should produce different chunks."""
        d1 = TerrainGenerator(seed=1).generate_chunk(0, 0)
        d2 = TerrainGenerator(seed=2).generate_chunk(0, 0)
        assert d1 != d2

    def test_surface_height_valid(self):
        """Surface heights must be within valid Y range."""
        gen = TerrainGenerator(seed=42)
        for x in range(0, 100, 10):
            for z in range(0, 100, 10):
                h = gen.get_surface_height(x, z)
                assert 4 <= h <= 254, f"Invalid surface height {h} at ({x},{z})"

    def test_no_floating_air_below_surface(self):
        """Basic sanity: there should be solid blocks below surface in a column."""
        gen = TerrainGenerator(seed=42)
        data = gen.generate_chunk(0, 0)
        CHUNK_W, CHUNK_D = 16, 16
        # Check one column
        x, z = 8, 8
        surface = gen.get_surface_height(x, z)
        # 5 blocks below surface should be solid
        for dy in range(1, 6):
            y = surface - dy
            idx = x + z * CHUNK_W + y * CHUNK_W * CHUNK_D
            assert data[idx] != 0, f"Air found below surface at y={y}"


# ─── Block Registry Tests ─────────────────────────────────────────────────────

class TestBlockRegistry:
    def test_air_is_id_0(self):
        reg = create_default_registry()
        air = reg.get(0)
        assert air.id == 0
        assert air.is_air
        assert not air.solid

    def test_stone_lookup(self):
        reg = create_default_registry()
        stone = reg.get(1)
        assert stone.name == "minecraft:stone"
        assert stone.solid
        assert stone.hardness == 1.5

    def test_name_lookup(self):
        reg = create_default_registry()
        grass = reg.get_by_name("minecraft:grass")
        assert grass is not None
        assert grass.id == 2

    def test_unknown_id_returns_air(self):
        reg = create_default_registry()
        unknown = reg.get(999)
        assert unknown.id == 0

    def test_duplicate_id_raises(self):
        reg = BlockRegistry()
        reg.register(BlockDef(id=1, name="test:a", display_name="A"))
        with pytest.raises(ValueError):
            reg.register(BlockDef(id=1, name="test:b", display_name="B"))

    def test_json_mod_loading(self):
        """Test loading a mod from JSON file."""
        import json, tempfile, os
        mod_data = {"blocks": [
            {"id": 300, "name": "testmod:myblock", "display_name": "My Block",
             "textures": [[5, 5]], "hardness": 2.0}
        ]}
        reg = create_default_registry()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mod_data, f)
            path = f.name
        try:
            reg.load_json(path)
            block = reg.get(300)
            assert block.name == "testmod:myblock"
            assert block.hardness == 2.0
        finally:
            os.unlink(path)

    def test_texture_uv(self):
        """UV coordinates should be within [0, 1]."""
        reg = create_default_registry()
        stone = reg.get(1)
        for face in range(6):
            u0, v0, u1, v1 = stone.to_uv(face)
            assert 0.0 <= u0 < 1.0
            assert 0.0 <= v0 < 1.0
            assert u0 < u1
            assert v0 < v1


# ─── Camera Tests ─────────────────────────────────────────────────────────────

class TestCamera:
    def test_forward_vector_yaw_0(self):
        """At yaw=0, pitch=0, forward should point in +X direction."""
        from engine.camera import Camera
        cam = Camera()
        cam.yaw = 0.0; cam.pitch = 0.0
        cam._update_vectors()
        fx, fy, fz = cam.forward
        assert abs(fx - 1.0) < 0.01
        assert abs(fy) < 0.01
        assert abs(fz) < 0.01

    def test_pitch_clamp(self):
        """Pitch must be clamped to [-89, 89]."""
        from engine.camera import Camera
        cam = Camera()
        cam.process_mouse(0, 10000)   # Extreme upward movement
        assert -89.0 <= cam.pitch <= 89.0

    def test_view_matrix_shape(self):
        """View matrix must be 4×4."""
        from engine.camera import Camera
        cam = Camera()
        view = cam.get_view_matrix(0, 0, 0)
        assert len(view) == 4
        assert all(len(row) == 4 for row in view)

    def test_mvp_length(self):
        """Flattened MVP must have 16 elements."""
        from engine.camera import Camera
        cam = Camera()
        mvp = cam.get_mvp(0, 64, 0, 16/9)
        assert len(mvp) == 16


# ─── Raycasting Tests ─────────────────────────────────────────────────────────

class TestRaycast:
    def _make_mock_world(self, blocks):
        """Create a minimal mock world for raycasting."""
        class MockWorld:
            def __init__(self, blocks):
                self._blocks = blocks
                self.registry = create_default_registry()
            def get_block(self, x, y, z):
                return self._blocks.get((x, y, z), 0)
            def is_solid(self, x, y, z):
                return self.registry.get(self.get_block(x, y, z)).solid
        return MockWorld(blocks)

    def test_hit_block_directly_ahead(self):
        """Ray pointing at a block directly ahead should hit it."""
        from physics.physics import raycast_block
        world = self._make_mock_world({(5, 64, 0): 1})   # stone at x=5
        result = raycast_block(world, (0.5, 64.5, 0.5), (1, 0, 0), 10.0)
        assert result is not None
        hit_pos, face = result
        assert hit_pos == (5, 64, 0)

    def test_no_hit_empty_world(self):
        """Ray in empty world should return None."""
        from physics.physics import raycast_block
        world = self._make_mock_world({})
        result = raycast_block(world, (0, 64, 0), (1, 0, 0), 10.0)
        assert result is None

    def test_face_normal_correct(self):
        """Hitting block from -X side should give face normal (+1, 0, 0)."""
        from physics.physics import raycast_block
        world = self._make_mock_world({(5, 64, 0): 1})
        result = raycast_block(world, (0.5, 64.5, 0.5), (1, 0, 0), 10.0)
        assert result is not None
        _, face = result
        assert face[0] == -1   # hit from west, normal points west (-X)

    def test_max_distance_respected(self):
        """Block beyond max distance should not be hit."""
        from physics.physics import raycast_block
        world = self._make_mock_world({(20, 64, 0): 1})
        result = raycast_block(world, (0.5, 64.5, 0.5), (1, 0, 0), 6.0)
        assert result is None


# ─── AABB Tests ───────────────────────────────────────────────────────────────

class TestAABB:
    def test_min_max(self):
        from physics.physics import AABB
        aabb = AABB(0.6, 1.8)
        mn = aabb.get_min(0, 0, 0)
        mx = aabb.get_max(0, 0, 0)
        assert abs(mn[0] - (-0.3)) < 0.001
        assert abs(mx[0] - 0.3) < 0.001
        assert abs(mn[1] - 0.0) < 0.001
        assert abs(mx[1] - 1.8) < 0.001

    def test_intersection(self):
        from physics.physics import AABB
        a = AABB(1.0, 1.0)
        b = AABB(1.0, 1.0)
        assert a.intersects(b, 0, 0, 0, 0.5, 0, 0)   # overlapping
        assert not a.intersects(b, 0, 0, 0, 5, 0, 0)  # far apart

    def test_no_self_intersection_when_separated(self):
        from physics.physics import AABB
        a = AABB(0.6, 1.8)   # player-sized
        b = AABB(1.0, 1.0)   # block-sized
        # Player 10 blocks away from block
        assert not a.intersects(b, 0, 0, 0, 10, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
