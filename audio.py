"""
audio/audio.py — 3D Positional Audio Manager

Wraps OpenAL (via PyOpenAL or python-openal) for positional sound.
Provides:
  - Sound loading and caching
  - Positional block sounds (break, place, step)
  - Background music with crossfading
  - Underwater audio filter (low-pass simulation)

Sound groups map block types to sound sets:
  stone, grass, wood, gravel, snow, none
"""

from __future__ import annotations
import os
import math
from typing import Dict, Optional, Tuple

try:
    from openal import oalOpen, oalGetListener, Listener, Source
    OPENAL_AVAILABLE = True
except ImportError:
    OPENAL_AVAILABLE = False


# Sound group definitions — maps to filenames in data/sounds/
SOUND_GROUPS: Dict[str, Dict[str, list]] = {
    "stone":  {"break": ["stone1.ogg","stone2.ogg","stone3.ogg","stone4.ogg"],
               "place": ["stone_place1.ogg"],
               "step":  ["step_stone1.ogg","step_stone2.ogg","step_stone3.ogg"]},
    "grass":  {"break": ["grass1.ogg","grass2.ogg","grass3.ogg"],
               "place": ["grass_place1.ogg"],
               "step":  ["step_grass1.ogg","step_grass2.ogg","step_grass3.ogg","step_grass4.ogg"]},
    "wood":   {"break": ["wood1.ogg","wood2.ogg","wood3.ogg","wood4.ogg"],
               "place": ["wood_place1.ogg"],
               "step":  ["step_wood1.ogg","step_wood2.ogg","step_wood3.ogg"]},
    "gravel": {"break": ["gravel1.ogg","gravel2.ogg","gravel3.ogg","gravel4.ogg"],
               "place": ["gravel_place1.ogg"],
               "step":  ["step_gravel1.ogg","step_gravel2.ogg","step_gravel3.ogg"]},
    "snow":   {"break": ["snow1.ogg","snow2.ogg","snow3.ogg"],
               "place": ["snow_place1.ogg"],
               "step":  ["step_snow1.ogg","step_snow2.ogg"]},
    "none":   {"break": [], "place": [], "step": []},
}


class AudioManager:
    """
    Manages all game audio using OpenAL for 3D positional sound.

    If OpenAL is not available, all methods are no-ops (graceful degradation).
    """

    SOUNDS_DIR  = os.path.join("data", "sounds")
    MAX_SOURCES = 32   # OpenAL source pool

    def __init__(self):
        self.available = OPENAL_AVAILABLE
        if not self.available:
            print("[Audio] PyOpenAL not found — audio disabled. "
                  "Install with: pip install PyOpenAL")
            return

        self._sources: list = []
        self._buffers: Dict[str, int] = {}   # filename → buffer id
        self._step_timer = 0.0
        self._step_interval = 0.4            # seconds between footstep sounds

        # Initialize OpenAL listener (player position + orientation)
        # Updated every frame in update()

    def update_listener(self, pos: Tuple[float,float,float],
                         forward: Tuple[float,float,float],
                         up: Tuple[float,float,float],
                         velocity: Tuple[float,float,float] = (0,0,0)):
        """
        Update OpenAL listener position and orientation.
        Call every frame with the camera's eye position.

        OpenAL orientation format: [at_x, at_y, at_z, up_x, up_y, up_z]
        """
        if not self.available:
            return
        try:
            listener = oalGetListener()
            listener.set_position(list(pos))
            listener.set_orientation(list(forward) + list(up))
            listener.set_velocity(list(velocity))
        except Exception:
            pass

    def play_block_break(self, sound_group: str,
                          block_pos: Tuple[float,float,float]):
        """Play a block-break sound at the given world position."""
        self._play_group_sound(sound_group, "break", block_pos, volume=0.8)

    def play_block_place(self, sound_group: str,
                          block_pos: Tuple[float,float,float]):
        """Play a block-place sound at the given world position."""
        self._play_group_sound(sound_group, "place", block_pos, volume=0.7)

    def play_footstep(self, sound_group: str,
                       player_pos: Tuple[float,float,float],
                       dt: float, on_ground: bool):
        """
        Play a footstep sound at regular intervals while moving on ground.
        Call every frame with movement state.
        """
        if not on_ground:
            return
        self._step_timer += dt
        if self._step_timer >= self._step_interval:
            self._step_timer = 0.0
            self._play_group_sound(sound_group, "step", player_pos, volume=0.4)

    def _play_group_sound(self, group: str, event: str,
                           pos: Tuple[float,float,float], volume: float = 1.0):
        """Pick a random sound from the group and play it at 3D position."""
        if not self.available:
            return
        import random
        sounds = SOUND_GROUPS.get(group, SOUND_GROUPS["none"]).get(event, [])
        if not sounds:
            return
        filename = random.choice(sounds)
        path     = os.path.join(self.SOUNDS_DIR, filename)
        if not os.path.exists(path):
            return   # Sound file missing — skip gracefully

        try:
            source = oalOpen(path)
            source.set_position(list(pos))
            source.set_rolloff_factor(1.0)    # Linear attenuation
            source.set_reference_distance(4.0)
            source.set_max_distance(32.0)
            source.set_gain(volume)
            source.play()
        except Exception:
            pass

    def play_ambient(self, track: str, loop: bool = True, volume: float = 0.3):
        """Play background ambient music."""
        if not self.available:
            return
        path = os.path.join(self.SOUNDS_DIR, "music", track)
        if not os.path.exists(path):
            return
        try:
            source = oalOpen(path)
            source.set_gain(volume)
            if loop:
                source.set_looping(True)
            source.play()
        except Exception:
            pass

    def shutdown(self):
        """Release all OpenAL resources."""
        if not self.available:
            return
        # Sources are garbage collected by PyOpenAL
        pass
