"""Time-parameterize a multi-waypoint joint path (e.g. from a motion planner).

Each segment between consecutive waypoints is a point_to_point.PointToPoint:
all joints move along the straight joint-space line together, starting and
ending at rest. That keeps the motion exactly on the segments a planner
checked for collisions, at the cost of stopping at every waypoint.

No ROS dependencies; msgs.to_joint_trajectory turns it into a ROS message.
"""

import dataclasses
from collections.abc import Sequence

from src.trajectory_generator import point_to_point


@dataclasses.dataclass(frozen=True)
class WaypointPath:
    segments: tuple[point_to_point.PointToPoint, ...]

    def __post_init__(self):
        if not self.segments:
            raise ValueError("a path needs at least one segment")

    @property
    def dof(self) -> int:
        return len(self.segments[0].start)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def waypoints(self) -> list[tuple[float, ...]]:
        return [self.segments[0].start] + [s.goal for s in self.segments]

    def sample(self, time: float) -> point_to_point.Sample:
        """State at `time` (clamped to [0, duration])."""
        t = min(max(time, 0.0), self.duration)
        offset = 0.0
        for segment in self.segments:
            if t <= offset + segment.duration or segment is self.segments[-1]:
                s = segment.sample(t - offset)
                return point_to_point.Sample(t, s.positions, s.velocities, s.accelerations)
            offset += segment.duration
        raise AssertionError("unreachable")

    def samples(self, dt: float) -> list[point_to_point.Sample]:
        """Samples every `dt` s or less within each segment, including every waypoint once."""
        out = []
        offset = 0.0
        for i, segment in enumerate(self.segments):
            for s in segment.samples(dt)[(1 if i else 0):]:  # each waypoint once
                out.append(point_to_point.Sample(offset + s.time, s.positions, s.velocities, s.accelerations))
            offset += segment.duration
        return out


def plan(
    waypoints: Sequence[Sequence[float]],
    max_velocity: float | Sequence[float] = 1.0,
    max_acceleration: float | Sequence[float] | None = None,
    min_segment_duration: float = 0.1,
) -> WaypointPath:
    """Timed path through `waypoints` (at least 2): each segment as fast as the limits allow."""
    if len(waypoints) < 2:
        raise ValueError(f"need at least 2 waypoints, got {len(waypoints)}")
    return WaypointPath(
        tuple(
            point_to_point.plan(a, b, max_velocity=max_velocity, max_acceleration=max_acceleration,
                                min_duration=min_segment_duration)
            for a, b in zip(waypoints, waypoints[1:])
        )
    )
