"""Joint trajectories given as dense samples (e.g. IK along a Cartesian path).

    trajectory = SampledTrajectory(times, positions).limit_velocity(1.0)

Velocities are finite differences of the positions (zero at both ends, where
the paths here start and finish at rest); accelerations are left out, so
joint_trajectory_controller interpolates between the points with cubic splines
through the positions and velocities. Dense samples (every few cm or degrees of
joint motion) keep that close to the sampled path.

No ROS dependencies; msgs.to_joint_trajectory turns it into a ROS message.
"""

import dataclasses
from collections.abc import Sequence

from src.trajectory_generator.point_to_point import Sample


@dataclasses.dataclass(frozen=True)
class SampledTrajectory:
    times: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "times", tuple(float(t) for t in self.times))
        object.__setattr__(self, "positions", tuple(tuple(float(v) for v in q) for q in self.positions))
        if len(self.times) != len(self.positions) or len(self.times) < 2:
            raise ValueError(f"need matching times and positions, at least 2 ({len(self.times)}, {len(self.positions)})")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("times must increase strictly")
        if len({len(q) for q in self.positions}) != 1:
            raise ValueError("every sample needs the same number of joints")

    @property
    def dof(self) -> int:
        return len(self.positions[0])

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]

    def velocities(self) -> list[tuple[float, ...]]:
        """Central differences (weighted for uneven spacing), zero at both ends."""
        t, q = self.times, self.positions
        out = [tuple(0.0 for _ in q[0])]
        for i in range(1, len(t) - 1):
            h0, h1 = t[i] - t[i - 1], t[i + 1] - t[i]
            out.append(tuple(((b - a) / h1 * h0 + (a - p) / h0 * h1) / (h0 + h1)
                             for p, a, b in zip(q[i - 1], q[i], q[i + 1])))
        out.append(tuple(0.0 for _ in q[0]))
        return out

    def peak_velocities(self) -> list[float]:
        """Largest |velocity| per joint, from consecutive samples."""
        t, q = self.times, self.positions
        return [max(abs(b[j] - a[j]) / (t1 - t0) for a, b, t0, t1 in zip(q, q[1:], t, t[1:])) for j in range(self.dof)]

    def scaled(self, factor: float) -> "SampledTrajectory":
        """The same path `factor` times slower."""
        if not factor > 0:
            raise ValueError(f"factor must be positive, got {factor}")
        t0 = self.times[0]
        return SampledTrajectory(tuple(t0 + (t - t0) * factor for t in self.times), self.positions)

    def limit_velocity(self, max_velocity: float | Sequence[float]) -> "SampledTrajectory":
        """Slowed down uniformly (if needed) so no joint exceeds its speed limit."""
        limits = [max_velocity] * self.dof if isinstance(max_velocity, (int, float)) else list(max_velocity)
        if len(limits) != self.dof or any(not v > 0 for v in limits):
            raise ValueError(f"need {self.dof} positive velocity limits, got {max_velocity}")
        factor = max(v / limit for v, limit in zip(self.peak_velocities(), limits))
        return self.scaled(factor) if factor > 1 else self

    def samples(self, dt: float | None = None) -> list[Sample]:
        """The samples themselves (`dt` is ignored: they are already dense), times from 0."""
        t0 = self.times[0]
        return [Sample(t - t0, q, v, ()) for t, q, v in zip(self.times, self.positions, self.velocities())]
