"""
engine/registry.py — Block Registry (Modding API Core)

Implements the Registry Pattern. Every block is a pure data object.
External mods define blocks via JSON; the engine never hard-codes block
properties. This file is the single source of truth for all block IDs.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple


# ─── Block Data Object ────────────────────────────────────────────────────────

@dataclass
class BlockDef:
    """
    Complete definition of a block type. Loaded from JSON or registered
    programmatically. No game logic lives here — pure data.
    """
    id: int                          # Numeric ID used in chunk arrays
    name: str                        # e.g. "minecraft:stone"
    display_name: str                # "Stone"

    # Texture atlas UV offsets (u, v) per face [top, bottom, north, south, east, west]
    # Each is a (col, row) index into a 16×16 texture atlas (256 textures)
    textures: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 0)] * 6)

    # Physics
    solid: bool = True               # Does it block movement?
    transparent: bool = False        # Does light pass through?
    fluid: bool = False              # Is it a fluid (water/lava)?
    friction: float = 0.6            # Movement friction (0=ice, 1=sticky)
    hardness: float = 1.0            # Break time multiplier
    blast_resistance: float = 1.0    # Explosion resistance

    # Lighting
    light_level: int = 0             # 0–15 emitted light
    light_filter: int = 15           # How much light is absorbed (15 = opaque)

    # Drops
    drop_id: Optional[int] = None    # Block ID dropped when broken (None = self)
    drop_count: int = 1

    # Sound group (maps to AudioManager sound keys)
    sound_group: str = "stone"       # "stone", "grass", "wood", "gravel", etc.

    # Render hints
    render_type: str = "cube"        # "cube", "cross" (flowers), "fluid", "none"

    @property
    def is_air(self) -> bool:
        return self.id == 0

    def get_texture(self, face: int) -> Tuple[int, int]:
        """face: 0=top 1=bottom 2=north 3=south 4=east 5=west"""
        if len(self.textures) == 1:
            return self.textures[0]
        if len(self.textures) == 3:
            # [top, bottom, sides]
            return self.textures[[0, 1, 2, 2, 2, 2][face]]
        return self.textures[face % len(self.textures)]

    def to_uv(self, face: int, atlas_size: int = 16) -> Tuple[float, float, float, float]:
        """Return (u0, v0, u1, v1) in normalized UV coordinates for atlas sampling."""
        col, row = self.get_texture(face)
        s = 1.0 / atlas_size
        return col * s, row * s, (col + 1) * s, (row + 1) * s


# ─── Block Registry ───────────────────────────────────────────────────────────

class BlockRegistry:
    """
    Global registry of all block definitions.

    Supports:
      - Programmatic registration (built-in blocks)
      - JSON file loading (mod support)
      - O(1) lookup by ID or name
    """

    def __init__(self):
        self._by_id:   Dict[int,  BlockDef] = {}
        self._by_name: Dict[str,  BlockDef] = {}
        self._next_dynamic_id = 256  # Mods start here; 0–255 are vanilla

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, block: BlockDef) -> BlockDef:
        if block.id in self._by_id:
            raise ValueError(f"Block ID {block.id} already registered as "
                             f"'{self._by_id[block.id].name}'")
        if block.name in self._by_name:
            raise ValueError(f"Block name '{block.name}' already registered")
        self._by_id[block.id]     = block
        self._by_name[block.name] = block
        return block

    def register_from_dict(self, data: dict) -> BlockDef:
        """Register a block from a plain dictionary (parsed from JSON)."""
        textures_raw = data.get("textures", [[0, 0]])
        textures = [tuple(t) for t in textures_raw]
        block = BlockDef(
            id            = data["id"],
            name          = data["name"],
            display_name  = data.get("display_name", data["name"]),
            textures      = textures,
            solid         = data.get("solid", True),
            transparent   = data.get("transparent", False),
            fluid         = data.get("fluid", False),
            friction      = data.get("friction", 0.6),
            hardness      = data.get("hardness", 1.0),
            blast_resistance = data.get("blast_resistance", 1.0),
            light_level   = data.get("light_level", 0),
            light_filter  = data.get("light_filter", 15),
            drop_id       = data.get("drop_id"),
            drop_count    = data.get("drop_count", 1),
            sound_group   = data.get("sound_group", "stone"),
            render_type   = data.get("render_type", "cube"),
        )
        return self.register(block)

    def load_json(self, path: str):
        """Load block definitions from a JSON file (mod support)."""
        with open(path, "r") as f:
            data = json.load(f)
        blocks = data if isinstance(data, list) else data.get("blocks", [])
        for block_data in blocks:
            # Assign dynamic ID if not specified
            if "id" not in block_data:
                block_data["id"] = self._next_dynamic_id
                self._next_dynamic_id += 1
            self.register_from_dict(block_data)

    def load_directory(self, directory: str):
        """Load all *.json files from a mods directory."""
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".json"):
                self.load_json(os.path.join(directory, fname))

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, block_id: int) -> BlockDef:
        return self._by_id.get(block_id, self._by_id[0])   # fall back to AIR

    def get_by_name(self, name: str) -> Optional[BlockDef]:
        return self._by_name.get(name)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._by_id

    def all_blocks(self) -> List[BlockDef]:
        return list(self._by_id.values())


# ─── Vanilla Block Definitions (loaded at engine startup) ─────────────────────

VANILLA_BLOCKS = [
    # id  name                        display     textures(col,row)       solid  trans  fluid  fric   hard  blast  light lf    drop  dc  sound    render
    # AIR
    {"id": 0,  "name": "minecraft:air",       "display_name": "Air",
     "textures": [[0,0]], "solid": False, "transparent": True, "light_filter": 0,
     "hardness": 0, "sound_group": "none", "render_type": "none"},

    # STONE
    {"id": 1,  "name": "minecraft:stone",     "display_name": "Stone",
     "textures": [[1,0]], "hardness": 1.5, "blast_resistance": 6.0,
     "drop_id": 4, "sound_group": "stone"},                       # drops cobblestone

    # GRASS
    {"id": 2,  "name": "minecraft:grass",     "display_name": "Grass Block",
     "textures": [[0,0],[2,0],[3,0],[3,0],[3,0],[3,0]],           # top, bottom, sides
     "hardness": 0.6, "sound_group": "grass", "friction": 0.6},

    # DIRT
    {"id": 3,  "name": "minecraft:dirt",      "display_name": "Dirt",
     "textures": [[2,0]], "hardness": 0.5, "sound_group": "gravel", "friction": 0.55},

    # COBBLESTONE
    {"id": 4,  "name": "minecraft:cobblestone","display_name": "Cobblestone",
     "textures": [[0,1]], "hardness": 2.0, "blast_resistance": 6.0, "sound_group": "stone"},

    # WOOD PLANKS
    {"id": 5,  "name": "minecraft:planks",    "display_name": "Oak Planks",
     "textures": [[4,0]], "hardness": 2.0, "blast_resistance": 3.0, "sound_group": "wood"},

    # SAND
    {"id": 12, "name": "minecraft:sand",      "display_name": "Sand",
     "textures": [[2,1]], "hardness": 0.5, "friction": 0.4, "sound_group": "gravel"},

    # GRAVEL
    {"id": 13, "name": "minecraft:gravel",    "display_name": "Gravel",
     "textures": [[3,1]], "hardness": 0.6, "friction": 0.4, "sound_group": "gravel"},

    # LOG
    {"id": 17, "name": "minecraft:log",       "display_name": "Oak Log",
     "textures": [[5,1],[5,1],[4,1],[4,1],[4,1],[4,1]],
     "hardness": 2.0, "sound_group": "wood"},

    # LEAVES
    {"id": 18, "name": "minecraft:leaves",    "display_name": "Oak Leaves",
     "textures": [[4,3]], "transparent": True, "light_filter": 1,
     "hardness": 0.2, "sound_group": "grass"},

    # GLASS
    {"id": 20, "name": "minecraft:glass",     "display_name": "Glass",
     "textures": [[1,3]], "transparent": True, "light_filter": 0,
     "hardness": 0.3, "blast_resistance": 0.3, "sound_group": "stone"},

    # WATER
    {"id": 9,  "name": "minecraft:water",     "display_name": "Water",
     "textures": [[13,12]], "solid": False, "transparent": True, "fluid": True,
     "light_filter": 3, "friction": 0.8, "sound_group": "none", "render_type": "fluid"},

    # LAVA
    {"id": 11, "name": "minecraft:lava",      "display_name": "Lava",
     "textures": [[13,14]], "solid": False, "transparent": False, "fluid": True,
     "light_level": 15, "light_filter": 15, "sound_group": "none", "render_type": "fluid"},

    # BEDROCK
    {"id": 7,  "name": "minecraft:bedrock",   "display_name": "Bedrock",
     "textures": [[1,1]], "hardness": -1, "blast_resistance": 3600000, "sound_group": "stone"},

    # SNOW
    {"id": 80, "name": "minecraft:snow",      "display_name": "Snow Block",
     "textures": [[2,4]], "hardness": 0.2, "friction": 0.35, "sound_group": "snow"},

    # ICE
    {"id": 79, "name": "minecraft:ice",       "display_name": "Ice",
     "textures": [[3,4]], "transparent": True, "light_filter": 1,
     "hardness": 0.5, "friction": 0.02, "sound_group": "stone"},

    # GLOWSTONE
    {"id": 89, "name": "minecraft:glowstone", "display_name": "Glowstone",
     "textures": [[9,6]], "light_level": 15, "hardness": 0.3, "sound_group": "stone"},

    # FLOWER (cross render)
    {"id": 37, "name": "minecraft:dandelion", "display_name": "Dandelion",
     "textures": [[13,0]], "solid": False, "transparent": True, "light_filter": 0,
     "hardness": 0, "sound_group": "grass", "render_type": "cross"},
]


def create_default_registry() -> BlockRegistry:
    reg = BlockRegistry()
    for block_data in VANILLA_BLOCKS:
        reg.register_from_dict(block_data)
    return reg


# Global singleton — import this everywhere
BLOCK_REGISTRY: BlockRegistry = create_default_registry()
AIR   = BLOCK_REGISTRY.get(0)
STONE = BLOCK_REGISTRY.get(1)
GRASS = BLOCK_REGISTRY.get(2)
DIRT  = BLOCK_REGISTRY.get(3)
