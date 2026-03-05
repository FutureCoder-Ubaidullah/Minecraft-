"""
engine/camera.py — First-Person Camera

Implements a 6-DOF first-person camera:
  - Pitch/yaw via mouse delta
  - View matrix via Euler angles (no Quaternion overhead needed for FPS)
  - Projection matrix (configurable FOV)
  - Direction vector extraction (for raycasting)

Math notes:
  - We use a right-handed coordinate system: +X=east, +Y=up, +Z=south
  - Pitch clamped to [-89, +89] degrees to prevent gimbal flip
  - View matrix computed as: V = Rotation × Translation (no parent transform)
"""

from __future__ import annotations
import math
from typing import Tuple


# ─── Matrix Math (pure Python, no numpy dependency for core engine) ───────────

def _mat4_identity() -> list:
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def _mat4_multiply(A: list, B: list) -> list:
    C = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C

def _mat4_perspective(fov_deg: float, aspect: float,
                       near: float, far: float) -> list:
    """
    Standard perspective projection matrix (OpenGL convention, column-major).

    P[0][0] = 1 / (aspect * tan(fov/2))
    P[1][1] = 1 / tan(fov/2)
    P[2][2] = -(far + near) / (far - near)
    P[2][3] = -1
    P[3][2] = -(2 * far * near) / (far - near)
    """
    f   = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf  = 1.0 / (near - far)
    m   = [[0.0]*4 for _ in range(4)]
    m[0][0] = f / aspect
    m[1][1] = f
    m[2][2] = (far + near) * nf
    m[2][3] = -1.0
    m[3][2] = 2.0 * far * near * nf
    return m

def _mat4_flatten(m: list) -> list:
    """Flatten 4×4 nested list to 16-float column-major list for OpenGL."""
    return [m[col][row] for row in range(4) for col in range(4)]


# ─── Camera ───────────────────────────────────────────────────────────────────

class Camera:
    """
    First-person camera with pitch/yaw control.

    The camera does NOT own physics — it just tracks where the player is
    looking. The PhysicsBody in player.py owns x/y/z position.
    """

    EYE_HEIGHT  = 1.62   # Eye level above feet (Minecraft: 1.62 blocks)
    FOV_DEFAULT = 70.0
    FOV_SPRINT  = 75.0   # Field of view widens slightly when sprinting
    NEAR_PLANE  = 0.05
    FAR_PLANE   = 500.0

    def __init__(self, fov: float = FOV_DEFAULT):
        self.fov    = fov
        self.pitch  = 0.0   # Degrees, clamped [-89, 89]
        self.yaw    = -90.0 # Degrees, starts facing -Z (north)

        # Derived vectors (updated each frame from pitch/yaw)
        self.forward = (0.0, 0.0, -1.0)
        self.right   = (1.0, 0.0,  0.0)
        self.up      = (0.0, 1.0,  0.0)

    # ── Mouse Input ───────────────────────────────────────────────────────────

    def process_mouse(self, dx: float, dy: float, sensitivity: float = 0.15):
        """
        Update yaw and pitch from mouse delta.

        dx: horizontal mouse movement (pixels)
        dy: vertical mouse movement (pixels, positive = down on screen)
        sensitivity: degrees per pixel
        """
        self.yaw   += dx * sensitivity
        self.pitch -= dy * sensitivity  # invert Y: up mouse → look up

        # Clamp pitch to prevent camera flip
        self.pitch = max(-89.0, min(89.0, self.pitch))

        # Keep yaw in [0, 360) for clarity (optional)
        self.yaw %= 360.0

        self._update_vectors()

    # ── Vector Computation ────────────────────────────────────────────────────

    def _update_vectors(self):
        """
        Recompute forward/right/up from yaw and pitch.

        Forward vector from Euler angles:
            fx = cos(pitch) * cos(yaw)
            fy = sin(pitch)
            fz = cos(pitch) * sin(yaw)

        Right = normalize(cross(forward, world_up))
        Up    = cross(right, forward)
        """
        yaw_r   = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)

        cp = math.cos(pitch_r)
        fx = cp * math.cos(yaw_r)
        fy = math.sin(pitch_r)
        fz = cp * math.sin(yaw_r)
        self.forward = _normalize3(fx, fy, fz)

        # Right: cross(forward, world_up=(0,1,0))
        rx = fz  # simplified cross with (0,1,0)
        ry = 0.0
        rz = -fx
        self.right = _normalize3(rx, ry, rz)

        # Up: cross(right, forward)
        ux = self.right[1]*fz - self.right[2]*fy
        uy = self.right[2]*fx - self.right[0]*fz
        uz = self.right[0]*fy - self.right[1]*fx
        self.up = _normalize3(ux, uy, uz)

    # ── Matrices ──────────────────────────────────────────────────────────────

    def get_view_matrix(self, eye_x: float, eye_y: float, eye_z: float) -> list:
        """
        Compute view matrix for camera at (eye_x, eye_y, eye_z).

        View matrix = lookAt(eye, eye+forward, up)

        LookAt construction:
            f = normalize(center - eye)   [forward]
            s = normalize(cross(f, up))   [right]
            u = cross(s, f)               [actual up]

            V = [ s.x  s.y  s.z  -dot(s,eye) ]
                [ u.x  u.y  u.z  -dot(u,eye) ]
                [-f.x -f.y -f.z   dot(f,eye) ]
                [  0    0    0       1        ]
        """
        fx, fy, fz = self.forward
        sx, sy, sz = self.right
        ux, uy, uz = self.up

        m = [[0.0]*4 for _ in range(4)]
        m[0][0] = sx;  m[1][0] = sy;  m[2][0] = sz
        m[0][1] = ux;  m[1][1] = uy;  m[2][1] = uz
        m[0][2] = -fx; m[1][2] = -fy; m[2][2] = -fz
        m[3][0] = -(sx*eye_x + sy*eye_y + sz*eye_z)
        m[3][1] = -(ux*eye_x + uy*eye_y + uz*eye_z)
        m[3][2] =  (fx*eye_x + fy*eye_y + fz*eye_z)
        m[3][3] = 1.0
        return m

    def get_projection_matrix(self, aspect: float) -> list:
        return _mat4_perspective(self.fov, aspect, self.NEAR_PLANE, self.FAR_PLANE)

    def get_mvp(self, pos_x: float, pos_y: float, pos_z: float,
                 aspect: float) -> list:
        """Return flattened MVP matrix (Model=Identity for world geometry)."""
        V   = self.get_view_matrix(pos_x, pos_y + self.EYE_HEIGHT, pos_z)
        P   = self.get_projection_matrix(aspect)
        VP  = _mat4_multiply(P, V)
        return _mat4_flatten(VP)

    def get_eye_position(self, px: float, py: float, pz: float
                          ) -> Tuple[float, float, float]:
        return (px, py + self.EYE_HEIGHT, pz)

    def get_look_direction(self) -> Tuple[float, float, float]:
        return self.forward

    # ── FOV Animation ────────────────────────────────────────────────────────

    def lerp_fov(self, target: float, dt: float, speed: float = 8.0):
        """Smooth FOV transition (sprint zoom, portal effect, etc.)."""
        self.fov += (target - self.fov) * min(1.0, dt * speed)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize3(x: float, y: float, z: float) -> Tuple[float, float, float]:
    length = math.sqrt(x*x + y*y + z*z)
    if length < 1e-9:
        return (0.0, 0.0, -1.0)
    return (x / length, y / length, z / length)
