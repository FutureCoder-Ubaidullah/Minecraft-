# Project Minecraft — Voxel Engine v1.0

A production-architecture Minecraft clone framework implementing all core
engine systems with emphasis on correctness, performance, and extensibility.

```
╔══════════════════════════════════════════════════════════╗
║  ALL 23 TESTS PASSING  |  Pure Python, no C extensions  ║
╚══════════════════════════════════════════════════════════╝
```

---

## Architecture Overview

```
voxel_engine/
├── main.py                   Entry point (client or --headless server)
│
├── engine/
│   ├── registry.py           ★ Block Registry (Modding API)
│   ├── camera.py             ★ 6-DOF First Person Camera
│   ├── player.py             ★ Player (physics + inventory + stats)
│   ├── game.py               ★ Main Loop (GLFW + DeltaTime)
│   └── server.py             Headless dedicated server
│
├── world/
│   ├── noise.py              ★ 3D Perlin + fBm + Terrain Generation
│   ├── chunk.py              ★ Chunk Data + Greedy Mesher + Face Culling
│   └── world.py              ★ Infinite World (dict-based O(1) lookup)
│
├── physics/
│   └── physics.py            ★ AABB Collision + Swept Resolution + DDA Raycast
│
├── rendering/
│   └── renderer.py           OpenGL Renderer (VAO/VBO/Shaders)
│
├── shaders/
│   ├── voxel.vert            ★ Vertex shader (AO + sun lighting + fog)
│   ├── voxel.frag            ★ Fragment shader (atlas + fog + underwater)
│   ├── selection.vert        Block highlight
│   └── selection.frag
│
├── audio/
│   └── audio.py              3D Positional Audio (OpenAL)
│
├── data/
│   └── blocks/
│       └── example_mod.json  ★ Example mod (3 new blocks via JSON)
│
└── tests/
    └── test_core.py          23 unit tests (all passing)
```

---

## Key Technical Systems

### 1. Perlin Noise (world/noise.py)
Ken Perlin's 2002 Improved Noise — deterministic, seeded permutation table:
```
Hash function:   h = perm[perm[perm[xi] + yi] + zi]
Gradient:        g = grad3[h & 15]
Fade curve:      f(t) = 6t⁵ - 15t⁴ + 10t³   (quintic, C² continuous)
Trilinear lerp:  result = lerp(w, lerp(v, lerp(u, g000, g100), ...), ...)
```

**Fractal Brownian Motion** (layered noise for terrain):
```
H(x,z) = Σᵢ₌₀ⁿ  gain^i · noise(x · lacunarity^i,  z · lacunarity^i)
```
Octaves=6, lacunarity=2.0, gain=0.5 → 6 layers of detail.

**Domain Warping** applied for non-grid-aligned continent shapes:
```
x' = x + warpStrength · warpNoise(x, z)
z' = z + warpStrength · warpNoise(x + 5.2, z + 1.3)
```

**3D Cave Carving** — two-noise intersection method:
```
isCave(x,y,z) = |n1(x,y,z)| < T  AND  |n2(x,y,z)| < T
```
Produces natural tube-shaped cave networks. T=0.10 gives realistic density.

---

### 2. Greedy Meshing (world/chunk.py)
**Face Culling** first: skip any face adjacent to a solid, opaque block.
This alone reduces draw calls by ~80% in typical terrain.

**Greedy Meshing** (Mikola Lysenko, 2012) merges co-planar identical faces:
```
For each axis-aligned slice:
  1. Build 2D mask: cell = block_id if face visible, else 0
  2. Scan mask for maximal rectangles of identical cells
  3. Emit ONE quad per rectangle (vs one per face naively)
  4. Mark cells processed
```
Reduces vertex count by 20–80% vs naive meshing.

**Ambient Occlusion** per vertex (Mikola Lysenko's method):
```
AO(v) = 0            if side1 AND side2 are solid  (full occlusion)
AO(v) = (3 - count) / 3    otherwise, count = solid neighbors
```
Applied as a smooth per-vertex float; quads are diagonally flipped to prevent
the "seam" artifact when AO values differ across a quad.

---

### 3. Swept AABB Collision (physics/physics.py)
Resolves entity movement against world geometry in 3 passes (Y → X → Z):

```
For each solid block in broad-phase region:
    tEntry = gap / velocity      (time until first contact)
    tExit  = gap2 / velocity     (time until separation)
    
    Verify overlap on other two axes at time tEntry
    Track earliest tEntry across all blocks
    
Move to: position + velocity × (earliest_t - ε)
```

Prevents tunneling (sweep, not teleport). Separates axes to allow
wall-sliding (blocking X doesn't block Z).

---

### 4. DDA Raycasting (physics/physics.py — `raycast_block`)
Amanatides & Woo (1987) — marches a ray through the integer voxel grid:
```
step     = sign(direction)
tMax     = distance to first grid line crossing per axis
tDelta   = distance per block step per axis

while t < maxDistance:
    advance along axis with smallest tMax
    check block at new integer position
```
O(N) where N = number of blocks traversed. No per-block AABB tests needed.

---

### 5. Block Registry / Modding API (engine/registry.py)
All block properties live in `BlockDef` data objects — **zero hard-coded logic**:

```python
@dataclass
class BlockDef:
    id: int              # Numeric block ID
    name: str            # "minecraft:stone"
    textures: list       # Atlas UV offsets per face
    solid: bool          # Physics collision?
    transparent: bool    # Light pass-through?
    friction: float      # Movement friction
    hardness: float      # Break time multiplier
    light_level: int     # Emitted light 0–15
    sound_group: str     # Audio category
    render_type: str     # "cube" | "cross" | "fluid" | "none"
```

Mods add blocks via JSON — no Python required:
```json
{
  "name": "mymod:glowing_ore",
  "textures": [[6, 2]],
  "hardness": 3.0,
  "light_level": 8,
  "sound_group": "stone"
}
```

---

### 6. GLSL Shaders (shaders/)
**Vertex shader** computes per-vertex lighting:
```glsl
// Face base brightness (Minecraft-style constant per face direction)
float face_brightness = [1.0, 0.5, 0.8, 0.8, 0.7, 0.7][face_id];

// Sun contribution
float sun_dot = max(0.0, dot(a_normal, u_sun_dir));

// AO quadratic curve (looks more natural than linear)
float ao_curved = a_ao * a_ao;

// Combined
v_light = face_brightness * (0.9 + 0.1 * sun_dot) * mix(0.2, 1.0, ao_curved) * sky;
```

**Fragment shader** applies fog and underwater effects:
```glsl
// Distance fog
vec3 final = mix(lit_color, u_sky_color, v_fog);

// Underwater caustics + exponential fog
float caustic = sin(pos.x * 2.3 + time * 1.5) * sin(pos.z * 1.9 + time * 1.2);
float uw_fog  = 1.0 - exp(-density * distance);
```

---

### 7. Day/Night Cycle (engine/game.py — `DayNightCycle`)
```
Sun angle:     θ(t) = t × 2π    (full rotation per day)
Sun direction: (cos θ, sin θ, 0.3)
Sun intensity: max(0, sin θ)     (0 at night, 1 at noon)

Sky color: keyframe interpolation at:
  t=0.00 → dawn orange     t=0.25 → noon blue
  t=0.50 → dusk orange     t=0.75 → midnight black
```

---

## Installation & Running

```bash
# Install dependencies
pip install glfw PyOpenGL PyOpenGL_accelerate Pillow

# Run the game (client)
python main.py

# Run headless server
python main.py --headless --port 25565 --seed 12345

# Optional: place a Minecraft-format terrain.png in data/ for real textures
```

### Controls
| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| Space | Jump |
| Ctrl | Toggle sprint |
| Shift | Crouch |
| F | Toggle fly mode |
| LMB (hold) | Break block |
| RMB | Place block |
| 1–9 | Hotbar slot |
| Scroll | Change slot |
| F3 | Debug overlay |
| Esc | Release mouse |
| Q | Quit |

---

## Adding Mods

Create a JSON file in `data/blocks/mymod.json`:
```json
{
  "mod_id": "mymod",
  "blocks": [
    {
      "name": "mymod:magic_stone",
      "display_name": "Magic Stone",
      "textures": [[8, 4]],
      "hardness": 1.0,
      "light_level": 5,
      "sound_group": "stone"
    }
  ]
}
```

Then load it at startup:
```python
from engine.registry import BLOCK_REGISTRY
BLOCK_REGISTRY.load_directory("data/blocks")
```

---

## Performance Notes

| System | Target | Notes |
|--------|--------|-------|
| Chunk gen | <50ms/chunk | Background thread |
| Mesh build | <5ms/chunk | Greedy meshing |
| Face culling | ~80% reduction | vs no culling |
| Chunk render | O(surface voxels) | Not O(total voxels) |
| Block lookup | O(1) | Dict + array index |
| Raycast | O(distance) | DDA algorithm |
| Collision | O(nearby blocks) | Swept AABB |

For Python performance, consider:
- **Cython** or **PyPy** for the noise generator
- **numpy** vectorized noise for batch chunk generation
- **ctypes** or C extension for the mesher inner loop
