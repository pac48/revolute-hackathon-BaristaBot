"""Point-to-point joint trajectories with a quintic (minimum-jerk) time scaling.

Every joint moves from `start` to `goal` along the same normalized profile

    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,   tau = t / duration

so all joints start and finish together, with zero velocity and acceleration at
both ends. Peak speed is 15/8 * |goal - start| / duration (at tau = 1/2) and peak
acceleration is 10/sqrt(3) * |goal - start| / duration^2, which is how
`minimum_duration` turns per-joint limits into a duration.

No ROS dependencies; see msgs.to_joint_trajectory for the ROS message.
"""

import dataclasses
import math
from collections.abc import Sequence

PEAK_VELOCITY_FACTOR = 15.0 / 8.0
PEAK_ACCELERATION_FACTOR = 10.0 / math.sqrt(3.0)


@dataclasses.dataclass(frozen=True)
class Sample:
    time: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class PointToPoint:
    start: tuple[float, ...]
    goal: tuple[float, ...]
    duration: float

    def __post_init__(self):
        if len(self.start) != len(self.goal):
            raise ValueError(f"start has {len(self.start)} joints, goal has {len(self.goal)}")
        if not self.duration > 0:
            raise ValueError(f"duration must be positive, got {self.duration}")

    @property
    def dof(self) -> int:
        return len(self.start)

    def sample(self, time: float) -> Sample:
        """State at `time` (clamped to [0, duration])."""
        t = min(max(time, 0.0), self.duration)
        tau = t / self.duration
        s = tau**3 * (10 - 15 * tau + 6 * tau**2)
        ds = 30 * tau**2 * (1 - tau) ** 2 / self.duration
        dds = 60 * tau * (1 - 3 * tau + 2 * tau**2) / self.duration**2
        delta = [g - q for q, g in zip(self.start, self.goal)]
        return Sample(
            time=t,
            positions=tuple(q + d * s for q, d in zip(self.start, delta)),
            velocities=tuple(d * ds for d in delta),
            accelerations=tuple(d * dds for d in delta),
        )

    def samples(self, dt: float) -> list[Sample]:
        """Samples every `dt` seconds from 0 to duration; the last one is exactly at `duration`."""
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}")
        n = max(1, math.ceil(self.duration / dt - 1e-9))
        return [self.sample(self.duration * i / n) for i in range(n + 1)]


def _per_joint(value: float | Sequence[float] | None, n: int, name: str) -> list[float | None]:
    if value is None or isinstance(value, (int, float)):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name} has {len(value)} entries for {n} joints")
    return list(value)


def minimum_duration(
    start: Sequence[float],
    goal: Sequence[float],
    max_velocity: float | Sequence[float],
    max_acceleration: float | Sequence[float] | None = None,
) -> float:
    """Shortest duration for which every joint stays within its velocity/acceleration limit."""
    n = len(start)
    duration = 0.0
    for q, g, v, a in zip(
        start,
        goal,
        _per_joint(max_velocity, n, "max_velocity"),
        _per_joint(max_acceleration, n, "max_acceleration"),
    ):
        distance = abs(g - q)
        if v is not None:
            if not v > 0:
                raise ValueError(f"max_velocity must be positive, got {v}")
            duration = max(duration, PEAK_VELOCITY_FACTOR * distance / v)
        if a is not None:
            if not a > 0:
                raise ValueError(f"max_acceleration must be positive, got {a}")
            duration = max(duration, math.sqrt(PEAK_ACCELERATION_FACTOR * distance / a))
    return duration


def plan(
    start: Sequence[float],
    goal: Sequence[float],
    duration: float | None = None,
    max_velocity: float | Sequence[float] = 1.0,
    max_acceleration: float | Sequence[float] | None = None,
    min_duration: float = 0.1,
) -> PointToPoint:
    """A point-to-point move from `start` to `goal`.

    With `duration`, the move takes exactly that long. Otherwise it takes the
    shortest time within `max_velocity` / `max_acceleration` (scalars or one per
    joint), but at least `min_duration`.
    """
    if duration is None:
        duration = max(min_duration, minimum_duration(start, goal, max_velocity, max_acceleration))
    return PointToPoint(tuple(map(float, start)), tuple(map(float, goal)), float(duration))
