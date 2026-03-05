"""
engine/game.py — Main Game Loop

Integrates all subsystems:
  - GLFW window creation and input
  - Delta time (fixed update + render)
  - World update + chunk meshing
  - Player update
  - Renderer calls
  - Day/night cycle
  - HUD (crosshair, hotbar, debug overlay)
"""

from __future__ import annotations
import math
import time
import os
import sys
from typing import Tuple

# ─── GLFW ─────────────────────────────────────────────────────────────────────
try:
    import glfw
    GLFW_AVAILABLE = True
except ImportError:
    GLFW_AVAILABLE = False
    print("[Game] glfw not installed. Run: pip install glfw")

try:
    import OpenGL.GL as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from world.world import World
from world.chunk import ChunkMesher
from engine.player import Player
from engine.camera import Camera
from engine.registry import BLOCK_REGISTRY


# ─── Day/Night Cycle ─────────────────────────────────────────────────────────

class DayNightCycle:
    """
    Controls sun direction, sky color, and ambient light over a 20-minute day.

    Time is normalized to [0, 1]:
      0.0 = dawn    0.25 = noon    0.5 = dusk    0.75 = midnight
    """

    DAY_LENGTH = 20 * 60.0  # 20 real minutes per Minecraft day

    SKY_COLORS = {
        0.00: (0.97, 0.58, 0.24),   # Dawn — orange horizon
        0.10: (0.53, 0.81, 0.98),   # Morning — light blue
        0.25: (0.47, 0.77, 0.98),   # Noon — blue sky
        0.40: (0.53, 0.81, 0.98),   # Afternoon
        0.50: (0.97, 0.45, 0.18),   # Dusk — orange
        0.60: (0.05, 0.05, 0.15),   # Early night — dark blue
        0.75: (0.01, 0.01, 0.08),   # Midnight — nearly black
        0.90: (0.03, 0.03, 0.12),   # Late night
        1.00: (0.97, 0.58, 0.24),   # Back to dawn
    }

    def __init__(self):
        self.time_of_day = 0.25   # Start at noon

    def update(self, dt: float):
        self.time_of_day = (self.time_of_day + dt / self.DAY_LENGTH) % 1.0

    @property
    def sun_angle(self) -> float:
        """Sun angle in radians (0=sunrise, π=sunset, 2π=next sunrise)."""
        return self.time_of_day * 2.0 * math.pi

    @property
    def sun_direction(self) -> Tuple[float, float, float]:
        """Normalized sun direction vector (light source direction)."""
        a = self.sun_angle
        return (math.cos(a), math.sin(a), 0.3)  # slight Z offset avoids flat noon

    @property
    def sun_intensity(self) -> float:
        """How bright the sun is [0, 1]. Night=0, noon=1."""
        return max(0.0, math.sin(self.sun_angle))

    @property
    def sky_color(self) -> Tuple[float, float, float]:
        """Interpolate sky color based on time of day."""
        t = self.time_of_day
        keys = sorted(self.SKY_COLORS.keys())

        # Find surrounding keyframes
        for i in range(len(keys) - 1):
            if keys[i] <= t <= keys[i + 1]:
                f = (t - keys[i]) / (keys[i + 1] - keys[i])
                c0 = self.SKY_COLORS[keys[i]]
                c1 = self.SKY_COLORS[keys[i + 1]]
                return (c0[0] + f*(c1[0]-c0[0]),
                        c0[1] + f*(c1[1]-c0[1]),
                        c0[2] + f*(c1[2]-c0[2]))
        return self.SKY_COLORS[0.25]


# ─── Game ─────────────────────────────────────────────────────────────────────

class Game:
    """
    Top-level game object.

    Window → Input → Update → Render loop runs at uncapped FPS.
    Physics is stepped at fixed 60 Hz to ensure determinism.
    """

    WINDOW_W    = 1280
    WINDOW_H    = 720
    WINDOW_TITLE = "Project Minecraft — Voxel Engine v1.0"

    FIXED_DT    = 1.0 / 60.0   # Fixed physics timestep
    MAX_FRAME_DT = 0.05         # Clamp frame delta (prevents spiral of death)

    def __init__(self, seed: int = 12345, world_name: str = "world"):
        self.seed = seed
        self.world_name = world_name
        self.running = False

        # Subsystems (initialized in _init())
        self.window   = None
        self.world    = None
        self.player   = None
        self.renderer = None
        self.mesher   = None
        self.day_night = DayNightCycle()

        # Timing
        self.last_time      = 0.0
        self.accumulator    = 0.0
        self.game_time      = 0.0   # total elapsed seconds
        self.frame_count    = 0
        self.fps            = 0.0
        self._fps_timer     = 0.0
        self._fps_frames    = 0

        # Input state
        self.mouse_captured = True
        self.last_mouse_x   = 0.0
        self.last_mouse_y   = 0.0
        self.first_mouse    = True

        # Mouse button state
        self.lmb_held = False
        self.rmb_held = False
        self.lmb_break_timer = 0.0

        # Debug
        self.show_debug = False

    # ── Initialization ────────────────────────────────────────────────────────

    def _init(self):
        """Initialize all subsystems."""
        if not GLFW_AVAILABLE:
            raise RuntimeError("GLFW not available")
        if not OPENGL_AVAILABLE:
            raise RuntimeError("PyOpenGL not available")

        # ── GLFW Window ───────────────────────────────────────────────────────
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)  # macOS
        glfw.window_hint(glfw.SAMPLES, 0)   # No MSAA for voxels (wasteful)

        self.window = glfw.create_window(
            self.WINDOW_W, self.WINDOW_H, self.WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)   # Disable VSync for max FPS measurement

        # ── Input callbacks ───────────────────────────────────────────────────
        glfw.set_key_callback(self.window,          self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window,       self._scroll_callback)
        glfw.set_cursor_pos_callback(self.window,   self._cursor_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

        # Capture mouse
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        # ── Subsystems ────────────────────────────────────────────────────────
        self.world    = World(seed=self.seed, world_dir=f"saves/{self.world_name}")
        self.player   = Player(0.0, 100.0, 0.0)
        self.mesher   = ChunkMesher(BLOCK_REGISTRY)

        from rendering.renderer import Renderer
        self.renderer = Renderer()

        # Try loading a real texture atlas
        atlas_path = os.path.join("data", "terrain.png")
        if os.path.exists(atlas_path):
            self.renderer.load_atlas_png(atlas_path)

        print(f"[Game] Initialized. OpenGL version: {gl.glGetString(gl.GL_VERSION).decode()}")
        print(f"[Game] World seed: {self.seed}")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        """The main game loop."""
        self._init()
        self.running    = True
        self.last_time  = time.perf_counter()

        while self.running and not glfw.window_should_close(self.window):
            # ── Delta time ────────────────────────────────────────────────────
            now       = time.perf_counter()
            frame_dt  = min(now - self.last_time, self.MAX_FRAME_DT)
            self.last_time = now
            self.game_time += frame_dt

            # ── Fixed physics update ──────────────────────────────────────────
            self.accumulator += frame_dt
            while self.accumulator >= self.FIXED_DT:
                self._fixed_update(self.FIXED_DT)
                self.accumulator -= self.FIXED_DT

            # ── Variable update ───────────────────────────────────────────────
            self._update(frame_dt)

            # ── Render ────────────────────────────────────────────────────────
            self._render()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            # ── FPS counter ───────────────────────────────────────────────────
            self._fps_frames += 1
            self._fps_timer  += frame_dt
            if self._fps_timer >= 1.0:
                self.fps        = self._fps_frames / self._fps_timer
                self._fps_timer = 0.0
                self._fps_frames = 0
                glfw.set_window_title(
                    self.window,
                    f"{self.WINDOW_TITLE}  |  FPS: {self.fps:.0f}  |  "
                    f"Pos: ({self.player.x:.1f}, {self.player.y:.1f}, {self.player.z:.1f})"
                )

        self._shutdown()

    # ── Update Passes ─────────────────────────────────────────────────────────

    def _fixed_update(self, dt: float):
        """Fixed timestep: physics, breaking logic."""
        self.player.body.update(self.world, dt)

        # Block breaking (hold LMB)
        if self.lmb_held:
            if self.player.breaking_block is None:
                self.player.start_breaking(self.world)
            self.player.update_breaking(self.world, dt)

    def _update(self, dt: float):
        """Variable timestep: world streaming, meshing, day/night."""
        # World streaming
        self.world.update(self.player.x, self.player.z)

        # Process dirty chunks (mesh rebuild)
        rebuilt = 0
        for key in list(self.world.dirty_chunks):
            if rebuilt >= 2:   # Limit per-frame to avoid stutter
                break
            chunk = self.world.chunks.get(key)
            if chunk:
                mesh = self.mesher.build_mesh(chunk)
                self.renderer.upload_chunk_mesh(chunk, mesh)
                chunk.dirty = False
                rebuilt += 1
            self.world.dirty_chunks.discard(key)

        # Remove unloaded chunk meshes
        loaded_keys = set(self.world.chunks.keys())
        for key in list(self.renderer.chunk_meshes.keys()):
            if key not in loaded_keys:
                self.renderer.remove_chunk(key[0], key[1])

        # Day/night
        self.day_night.update(dt)

        # Player non-physics update (target, stats, FOV)
        self.player._update_target(self.world)
        self.player.stats.update(dt, self.player.body.on_ground, self.player.sprinting)
        self.player.camera.lerp_fov(
            Camera.FOV_SPRINT if self.player.sprinting else Camera.FOV_DEFAULT, dt)

    def _render(self):
        """Render everything."""
        sky = self.day_night.sky_color
        self.renderer.clear(sky)

        w, h = glfw.get_framebuffer_size(self.window)
        if h == 0: h = 1
        gl.glViewport(0, 0, w, h)
        aspect = w / h

        # MVP matrix
        ex, ey, ez = self.player.eye_pos
        mvp_flat  = self.player.camera.get_mvp(self.player.x, self.player.y,
                                                 self.player.z, aspect)
        model_flat = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]  # identity

        # Check if underwater
        eye_block = self.world.get_block(int(ex), int(ey), int(ez))
        underwater = (eye_block == 9)

        self.renderer.render_world(
            mvp    = mvp_flat,
            model  = model_flat,
            camera_pos   = (ex, ey, ez),
            sun_dir      = self.day_night.sun_direction,
            sky_color    = sky,
            sun_intensity = self.day_night.sun_intensity,
            game_time    = self.game_time,
            underwater   = underwater,
        )

        # Selection highlight
        if self.player.target_block:
            bx, by, bz = self.player.target_block
            self.renderer.render_selection(mvp_flat, bx, by, bz)

    # ── Input Callbacks ───────────────────────────────────────────────────────

    def _key_callback(self, window, key, scancode, action, mods):
        p = self.player

        # Movement keys
        if key == glfw.KEY_W:
            p.move_forward = (action != glfw.RELEASE)
        elif key == glfw.KEY_S:
            p.move_back    = (action != glfw.RELEASE)
        elif key == glfw.KEY_A:
            p.move_left    = (action != glfw.RELEASE)
        elif key == glfw.KEY_D:
            p.move_right   = (action != glfw.RELEASE)
        elif key == glfw.KEY_LEFT_CONTROL and action != glfw.RELEASE:
            p.sprinting = not p.sprinting

        elif key == glfw.KEY_SPACE:
            if action == glfw.PRESS:
                p.wants_jump = True
            elif action == glfw.RELEASE:
                p.wants_jump = False

        elif key == glfw.KEY_LEFT_SHIFT:
            p.crouching = (action != glfw.RELEASE)

        # Fly toggle (double-tap space in real MC; here use F)
        elif key == glfw.KEY_F and action == glfw.PRESS:
            p.body.flying = not p.body.flying

        # Hotbar
        elif glfw.KEY_1 <= key <= glfw.KEY_9 and action == glfw.PRESS:
            p.inventory.select_slot(key - glfw.KEY_1)

        # Debug overlay
        elif key == glfw.KEY_F3 and action == glfw.PRESS:
            self.show_debug = not self.show_debug

        # Mouse capture toggle
        elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.mouse_captured = not self.mouse_captured
            mode = glfw.CURSOR_DISABLED if self.mouse_captured else glfw.CURSOR_NORMAL
            glfw.set_input_mode(window, glfw.CURSOR, mode)

        elif key == glfw.KEY_Q and action == glfw.PRESS:
            self.running = False

    def _mouse_button_callback(self, window, button, action, mods):
        if not self.mouse_captured:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.lmb_held = (action == glfw.PRESS)
            if action == glfw.PRESS:
                self.player.start_breaking(self.world)
            else:
                self.player.breaking_block = None
                self.player.break_progress = 0.0
        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            self.player.place_block(self.world)

    def _scroll_callback(self, window, xoff, yoff):
        self.player.inventory.scroll_slot(-int(yoff))

    def _cursor_callback(self, window, x, y):
        if not self.mouse_captured:
            self.first_mouse = True
            return
        if self.first_mouse:
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.first_mouse  = False
            return
        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        self.last_mouse_x = x
        self.last_mouse_y = y
        self.player.camera.process_mouse(dx, dy)

    def _resize_callback(self, window, width, height):
        gl.glViewport(0, 0, width, height)

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self):
        print("[Game] Shutting down...")
        self.world.shutdown()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("[Game] Goodbye.")
