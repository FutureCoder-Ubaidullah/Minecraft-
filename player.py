"""
engine/player.py — Player Entity

Combines:
  - PhysicsBody (movement + collision)
  - Camera (looking direction)
  - Inventory system (hotbar + backpack)
  - Player stats (health, hunger, XP)
  - Block interaction (place/break with raycasting)
  - Input handling
"""

from __future__ import annotations
import math
import time
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING

from physics.physics import PhysicsBody, raycast_block, get_place_position
from engine.camera import Camera

if TYPE_CHECKING:
    from world.world import World
    from engine.registry import BlockRegistry


# ─── Inventory ────────────────────────────────────────────────────────────────

class ItemStack:
    """A stack of one item type."""
    __slots__ = ('item_id', 'count', 'meta')

    def __init__(self, item_id: int, count: int = 1, meta: int = 0):
        self.item_id = item_id
        self.count   = count
        self.meta    = meta    # Damage / variant data

    def is_empty(self) -> bool:
        return self.count <= 0 or self.item_id == 0

    def split(self) -> 'ItemStack':
        """Take half the stack."""
        half = max(1, self.count // 2)
        self.count -= half
        return ItemStack(self.item_id, half, self.meta)


class Inventory:
    """
    Player inventory.
    Slots 0–8: hotbar (displayed at bottom of screen)
    Slots 9–35: main inventory
    Slots 36–39: armor (head, chest, legs, feet)
    Slot 40: offhand
    """

    HOTBAR_SIZE    = 9
    INVENTORY_SIZE = 27
    ARMOR_SIZE     = 4
    TOTAL_SIZE     = HOTBAR_SIZE + INVENTORY_SIZE + ARMOR_SIZE + 1  # 41

    MAX_STACK = 64

    def __init__(self):
        self.slots: List[Optional[ItemStack]] = [None] * self.TOTAL_SIZE
        self.selected_slot = 0   # Hotbar index 0–8

    # ── Hotbar ────────────────────────────────────────────────────────────────

    @property
    def held_item(self) -> Optional[ItemStack]:
        return self.slots[self.selected_slot]

    def select_slot(self, index: int):
        self.selected_slot = max(0, min(self.HOTBAR_SIZE - 1, index))

    def scroll_slot(self, delta: int):
        self.selected_slot = (self.selected_slot + delta) % self.HOTBAR_SIZE

    # ── Item Management ───────────────────────────────────────────────────────

    def add_item(self, item_id: int, count: int = 1) -> int:
        """
        Add items to inventory. Returns leftover count if inventory full.
        First tries to fill existing stacks, then finds empty slots.
        """
        remaining = count

        # Try to fill existing stacks
        for i, slot in enumerate(self.slots[:self.HOTBAR_SIZE + self.INVENTORY_SIZE]):
            if slot and slot.item_id == item_id and slot.count < self.MAX_STACK:
                can_add = min(remaining, self.MAX_STACK - slot.count)
                slot.count += can_add
                remaining  -= can_add
                if remaining == 0:
                    return 0

        # Fill empty slots
        for i in range(self.HOTBAR_SIZE + self.INVENTORY_SIZE):
            if remaining <= 0:
                break
            if self.slots[i] is None:
                stack_size     = min(remaining, self.MAX_STACK)
                self.slots[i]  = ItemStack(item_id, stack_size)
                remaining     -= stack_size

        return remaining  # leftovers

    def remove_item(self, item_id: int, count: int = 1) -> bool:
        """Remove items from inventory. Returns True if successful."""
        available = sum(s.count for s in self.slots if s and s.item_id == item_id)
        if available < count:
            return False
        remaining = count
        for slot in self.slots:
            if slot and slot.item_id == item_id and remaining > 0:
                taken       = min(slot.count, remaining)
                slot.count -= taken
                remaining  -= taken
                if slot.count == 0:
                    slot.item_id = 0
        return True

    def count_item(self, item_id: int) -> int:
        return sum(s.count for s in self.slots if s and s.item_id == item_id)


# ─── Crafting ─────────────────────────────────────────────────────────────────

class CraftingRecipe:
    """
    A shaped or shapeless crafting recipe.
    """
    def __init__(self, inputs: Dict[int, int], output_id: int, output_count: int = 1):
        self.inputs       = inputs         # {item_id: count}
        self.output_id    = output_id
        self.output_count = output_count

    def can_craft(self, inv: Inventory) -> bool:
        return all(inv.count_item(iid) >= count for iid, count in self.inputs.items())

    def craft(self, inv: Inventory) -> Optional[ItemStack]:
        if not self.can_craft(inv):
            return None
        for iid, count in self.inputs.items():
            inv.remove_item(iid, count)
        return ItemStack(self.output_id, self.output_count)


# Built-in recipes
RECIPES = [
    CraftingRecipe({17: 1}, 5, 4),    # 1 log → 4 planks
    CraftingRecipe({5: 4}, 58, 1),    # 4 planks → crafting table
    CraftingRecipe({4: 8}, 61, 1),    # 8 cobblestone → furnace
]


# ─── Player Stats ─────────────────────────────────────────────────────────────

class PlayerStats:
    MAX_HEALTH = 20.0   # 10 hearts
    MAX_HUNGER = 20.0   # 10 drumsticks

    def __init__(self):
        self.health       = self.MAX_HEALTH
        self.hunger       = self.MAX_HUNGER
        self.saturation   = 5.0    # Hunger saturation (buffer before hunger depletes)
        self.xp_points    = 0
        self.xp_level     = 0

        # Timers
        self._health_regen_timer = 0.0
        self._hunger_timer       = 0.0
        self._damage_cooldown    = 0.0

    def update(self, dt: float, on_ground: bool, sprinting: bool):
        self._damage_cooldown = max(0.0, self._damage_cooldown - dt)

        # Hunger depletion (sprinting costs more)
        exhaust_rate = 0.01 if not sprinting else 0.02
        self._hunger_timer += dt
        if self._hunger_timer >= 4.0:   # every 4 seconds
            self._hunger_timer = 0.0
            if self.saturation > 0:
                self.saturation -= 1.0
            elif self.hunger > 0:
                self.hunger = max(0.0, self.hunger - 1.0)

        # Natural health regeneration (requires hunger >= 18)
        if self.hunger >= 18.0 and self.health < self.MAX_HEALTH:
            self._health_regen_timer += dt
            regen_interval = 1.0 if self.hunger >= 20 else 4.0
            if self._health_regen_timer >= regen_interval:
                self._health_regen_timer = 0.0
                self.health = min(self.MAX_HEALTH, self.health + 1.0)

        # Starvation damage (hunger = 0)
        if self.hunger <= 0.0:
            self._health_regen_timer += dt
            if self._health_regen_timer >= 4.0:
                self._health_regen_timer = 0.0
                self.take_damage(1.0)

    def take_damage(self, amount: float) -> bool:
        """Returns True if player died."""
        if self._damage_cooldown > 0:
            return False
        self.health -= amount
        self._damage_cooldown = 0.5
        if self.health <= 0:
            self.health = 0.0
            return True
        return False

    def heal(self, amount: float):
        self.health = min(self.MAX_HEALTH, self.health + amount)

    def eat(self, hunger_restore: float, saturation_restore: float):
        self.hunger     = min(self.MAX_HUNGER, self.hunger + hunger_restore)
        self.saturation = min(self.hunger, self.saturation + saturation_restore)

    def add_xp(self, amount: int):
        self.xp_points += amount
        # Level-up threshold (increases per level)
        threshold = self._xp_threshold(self.xp_level)
        while self.xp_points >= threshold:
            self.xp_points -= threshold
            self.xp_level  += 1
            threshold = self._xp_threshold(self.xp_level)

    @staticmethod
    def _xp_threshold(level: int) -> int:
        if level < 16:
            return 2 * level + 7
        if level < 31:
            return 5 * level - 38
        return 9 * level - 158

    @property
    def is_alive(self) -> bool:
        return self.health > 0.0


# ─── Player ───────────────────────────────────────────────────────────────────

class Player:
    """
    Complete player entity.

    Owns: PhysicsBody, Camera, Inventory, PlayerStats
    """

    REACH           = 6.0   # Block interaction range (blocks)
    BREAK_TIME      = 0.5   # Default break time (seconds)
    SPRINT_HUNGER_THRESHOLD = 6.0  # Can't sprint below this hunger

    def __init__(self, x: float = 0.0, y: float = 100.0, z: float = 0.0):
        self.body       = PhysicsBody(x, y, z)
        self.camera     = Camera()
        self.inventory  = Inventory()
        self.stats      = PlayerStats()

        # Interaction state
        self.target_block:  Optional[Tuple[int,int,int]] = None
        self.target_face:   Optional[Tuple[int,int,int]] = None
        self.breaking_block: Optional[Tuple[int,int,int]] = None
        self.break_progress = 0.0   # 0 → 1
        self.break_start    = 0.0

        # Input state
        self.move_forward  = False
        self.move_back     = False
        self.move_left     = False
        self.move_right    = False
        self.sprinting     = False
        self.crouching     = False
        self.wants_jump    = False

        # Give starting items
        self._give_starter_items()

    def _give_starter_items(self):
        self.inventory.add_item(5,  64)   # planks
        self.inventory.add_item(4,  32)   # cobblestone
        self.inventory.add_item(17, 16)   # logs
        self.inventory.add_item(1,  64)   # stone

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def x(self) -> float: return self.body.x
    @property
    def y(self) -> float: return self.body.y
    @property
    def z(self) -> float: return self.body.z

    @property
    def eye_pos(self) -> Tuple[float, float, float]:
        return self.camera.get_eye_position(self.x, self.y, self.z)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, world: 'World', dt: float):
        """Full player update: movement, physics, stats, interaction."""
        self._apply_input(dt)
        self.body.update(world, dt)
        self.stats.update(dt, self.body.on_ground, self.sprinting)
        self._update_target(world)

        # FOV
        target_fov = Camera.FOV_SPRINT if self.sprinting else Camera.FOV_DEFAULT
        self.camera.lerp_fov(target_fov, dt)

    def _apply_input(self, dt: float):
        """Convert input flags to velocity on the physics body."""
        speed = (PhysicsBody.SPRINT_SPEED if self.sprinting
                 else PhysicsBody.CROUCH_SPEED if self.crouching
                 else PhysicsBody.WALK_SPEED)

        if self.body.flying:
            speed = PhysicsBody.FLY_SPEED

        # Compute move direction in world space (using camera yaw only)
        yaw_r  = math.radians(self.camera.yaw)
        fwd_x  =  math.cos(yaw_r)
        fwd_z  =  math.sin(yaw_r)
        right_x =  math.sin(yaw_r)
        right_z = -math.cos(yaw_r)

        move_x = 0.0; move_z = 0.0
        if self.move_forward: move_x += fwd_x;   move_z += fwd_z
        if self.move_back:    move_x -= fwd_x;   move_z -= fwd_z
        if self.move_right:   move_x += right_x; move_z += right_z
        if self.move_left:    move_x -= right_x; move_z -= right_z

        # Normalize diagonal movement
        mag = math.sqrt(move_x**2 + move_z**2)
        if mag > 0.01:
            move_x /= mag; move_z /= mag

        # Apply to body velocity directly (we set target velocity, drag handles decel)
        if self.body.on_ground or self.body.flying:
            self.body.vx = move_x * speed
            self.body.vz = move_z * speed
        else:
            # Air control (reduced)
            self.body.vx += move_x * speed * dt * 4.0
            self.body.vz += move_z * speed * dt * 4.0

        if self.wants_jump:
            self.body.jump()
            self.wants_jump = False

        # Can't sprint if hungry
        if self.stats.hunger <= self.SPRINT_HUNGER_THRESHOLD:
            self.sprinting = False

    def _update_target(self, world: 'World'):
        """Update which block the player is looking at."""
        result = raycast_block(world, self.eye_pos,
                                self.camera.forward, self.REACH)
        if result:
            self.target_block, self.target_face = result
        else:
            self.target_block = None
            self.target_face  = None

    # ── Block Interaction ─────────────────────────────────────────────────────

    def start_breaking(self, world: 'World'):
        """Begin breaking the targeted block."""
        if self.target_block is None:
            return
        bx, by, bz = self.target_block
        bid = world.get_block(bx, by, bz)
        bdef = world.registry.get(bid)
        if bdef.hardness < 0:
            return  # unbreakable (bedrock)
        self.breaking_block = self.target_block
        self.break_progress = 0.0
        self.break_start    = time.time()

    def update_breaking(self, world: 'World', dt: float) -> bool:
        """
        Continue breaking the currently targeted block.
        Returns True if the block was broken this frame.
        """
        if self.breaking_block is None:
            return False

        # Abort if looking away
        if self.target_block != self.breaking_block:
            self.breaking_block = None
            self.break_progress = 0.0
            return False

        bx, by, bz = self.breaking_block
        bid  = world.get_block(bx, by, bz)
        bdef = world.registry.get(bid)

        break_time = self.BREAK_TIME * bdef.hardness
        # TODO: apply tool efficiency multiplier here

        self.break_progress += dt / max(break_time, 0.05)

        if self.break_progress >= 1.0:
            self._break_block(world, bx, by, bz, bid, bdef)
            self.breaking_block = None
            self.break_progress = 0.0
            return True

        return False

    def _break_block(self, world, bx, by, bz, bid, bdef):
        world.set_block(bx, by, bz, 0)
        # Drop item
        drop_id = bdef.drop_id if bdef.drop_id is not None else bid
        if drop_id != 0:
            self.inventory.add_item(drop_id, bdef.drop_count)
        self.stats.add_xp(1)

    def place_block(self, world: 'World'):
        """Place the held block at the targeted face."""
        if self.target_block is None or self.target_face is None:
            return
        held = self.inventory.held_item
        if held is None or held.is_empty():
            return

        place_pos = get_place_position(self.target_block, self.target_face)
        px, py, pz = place_pos

        # Prevent placing inside player
        from world.chunk import CHUNK_H
        if not (0 <= py < CHUNK_H):
            return

        # Check player isn't occupying this space
        # (simplified: check if place pos overlaps player AABB)
        player_min = self.body.aabb.get_min(self.x, self.y, self.z)
        player_max = self.body.aabb.get_max(self.x, self.y, self.z)
        block_max  = (px + 1.0, py + 1.0, pz + 1.0)
        if not (px >= player_max[0] or px + 1 <= player_min[0] or
                py >= player_max[1] or py + 1 <= player_min[1] or
                pz >= player_max[2] or pz + 1 <= player_min[2]):
            return  # would intersect player

        world.set_block(px, py, pz, held.item_id)
        held.count -= 1
        if held.count <= 0:
            self.inventory.slots[self.inventory.selected_slot] = None
