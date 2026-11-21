"""Cartesian tool paths: poses, straight-line segments and their timing.

    path = cartesian_path.plan([start, goal1, goal2], max_linear_velocity=0.1)
    path.sample(t).position, path.samples(dt=0.01)

Each segment moves the tool on the straight line between two poses, turning
about a fixed axis at the same time (slerp), on a quintic (minimum-jerk) time
scaling: it starts and ends at rest, so the path stops at every waypoint. A
segment takes the shortest time within the linear and angular speed limits.

Poses are positions (m) and unit quaternions (x, y, z, w), of the tool in the
robot's base frame. No ROS dependencies.
"""

import dataclasses
import math
from collections.abc import Sequence

from src.trajectory_generator import point_to_point

Quaternion = tuple[float, float, float, float]


def _normalize(q: Sequence[float]) -> Quaternion:
    n = math.sqrt(sum(v * v for v in q))
    if n == 0:
        raise ValueError("zero quaternion")
    return tuple(v / n for v in q)


def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """R_z(yaw) R_y(pitch) R_x(roll), as in URDF."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_to_rpy(q: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Rotation angle (rad, in [0, pi]) from orientation a to b."""
    dot = abs(sum(x * y for x, y in zip(_normalize(a), _normalize(b))))
    return 2 * math.acos(min(1.0, dot))


def slerp(a: Sequence[float], b: Sequence[float], s: float) -> Quaternion:
    """Rotation from a towards b by fraction s, the short way round."""
    a, b = _normalize(a), _normalize(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-v for v in b), -dot
    if dot > 1 - 1e-9:
        return _normalize([x + s * (y - x) for x, y in zip(a, b)])
    theta = math.acos(dot)
    wa, wb = math.sin((1 - s) * theta) / math.sin(theta), math.sin(s * theta) / math.sin(theta)
    return tuple(wa * x + wb * y for x, y in zip(a, b))


@dataclasses.dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion: Quaternion = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(float(v) for v in self.position))
        object.__setattr__(self, "quaternion", _normalize([float(v) for v in self.quaternion]))
        if len(self.position) != 3 or len(self.quaternion) != 4:
            raise ValueError("a pose is a position (x, y, z) and a quaternion (x, y, z, w)")

    @staticmethod
    def from_xyz_rpy(x: float, y: float, z: float, roll: float = 0, pitch: float = 0, yaw: float = 0) -> "Pose":
        return Pose((x, y, z), quaternion_from_rpy(roll, pitch, yaw))

    @property
    def rpy(self) -> tuple[float, float, float]:
        return quaternion_to_rpy(self.quaternion)

    def translated(self, dx: float, dy: float, dz: float) -> "Pose":
        """Moved by (dx, dy, dz) in the base frame, same orientation."""
        x, y, z = self.position
        return Pose((x + dx, y + dy, z + dz), self.quaternion)

    def rotated(self, roll: float, pitch: float, yaw: float) -> "Pose":
        """Turned in place about the tool's own axes (R = R_self * R_rpy)."""
        return Pose(self.position, quaternion_multiply(self.quaternion, quaternion_from_rpy(roll, pitch, yaw)))

    def distance(self, other: "Pose") -> float:
        return math.dist(self.position, other.position)

    def angle(self, other: "Pose") -> float:
        return angle_between(self.quaternion, other.quaternion)

    def to_json(self) -> dict:
        return {"position": list(self.position), "quaternion": list(self.quaternion)}

    @staticmethod
    def from_json(data: dict) -> "Pose":
        return Pose(tuple(data["position"]), tuple(data["quaternion"]))


@dataclasses.dataclass(frozen=True)
class Segment:
    start: Pose
    goal: Pose
    duration: float

    def __post_init__(self):
        if not self.duration > 0:
            raise ValueError(f"duration must be positive, got {self.duration}")

    def sample(self, time: float) -> Pose:
        """Pose at `time` (clamped to [0, duration])."""
        tau = min(max(time, 0.0), self.duration) / self.duration
        s = tau**3 * (10 - 15 * tau + 6 * tau**2)
        position = tuple(a + s * (b - a) for a, b in zip(self.start.position, self.goal.position))
        return Pose(position, slerp(self.start.quaternion, self.goal.quaternion, s))


@dataclasses.dataclass(frozen=True)
class CartesianPath:
    segments: tuple[Segment, ...]

    def __post_init__(self):
        if not self.segments:
            raise ValueError("a path needs at least one segment")

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def waypoints(self) -> list[Pose]:
        return [self.segments[0].start] + [s.goal for s in self.segments]

    def sample(self, time: float) -> Pose:
        t = min(max(time, 0.0), self.duration)
        offset = 0.0
        for segment in self.segments:
            if t <= offset + segment.duration or segment is self.segments[-1]:
                return segment.sample(t - offset)
            offset += segment.duration
        raise AssertionError("unreachable")

    def samples(self, dt: float) -> list[tuple[float, Pose]]:
        """(time, pose) every `dt` s or less within each segment, including every waypoint once."""
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}")
        out, offset = [], 0.0
        for i, segment in enumerate(self.segments):
            n = max(1, math.ceil(segment.duration / dt - 1e-9))
            for k in range(0 if i == 0 else 1, n + 1):
                t = segment.duration * k / n
                out.append((offset + t, segment.sample(t)))
            offset += segment.duration
        return out


def segment_duration(start: Pose, goal: Pose, max_linear_velocity: float, max_angular_velocity: float,
                     min_duration: float = 0.1) -> float:
    """Shortest quintic segment within the speed limits (peak speed is 15/8 x the average)."""
    if not (max_linear_velocity > 0 and max_angular_velocity > 0):
        raise ValueError("speed limits must be positive")
    k = point_to_point.PEAK_VELOCITY_FACTOR
    return max(min_duration, k * start.distance(goal) / max_linear_velocity,
               k * start.angle(goal) / max_angular_velocity)


def plan(
    waypoints: Sequence[Pose],
    max_linear_velocity: float = 0.1,
    max_angular_velocity: float = 0.5,
    min_segment_duration: float = 0.1,
) -> CartesianPath:
    """Straight-line segments through `waypoints` (at least 2), each as fast as the limits allow (m/s, rad/s)."""
    if len(waypoints) < 2:
        raise ValueError(f"need at least 2 waypoints, got {len(waypoints)}")
    return CartesianPath(tuple(
        Segment(a, b, segment_duration(a, b, max_linear_velocity, max_angular_velocity, min_segment_duration))
        for a, b in zip(waypoints, waypoints[1:])
    ))
