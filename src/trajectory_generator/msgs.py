"""Convert trajectories to ROS messages for joint_trajectory_controller."""

from collections.abc import Sequence

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from src.trajectory_generator.point_to_point import PointToPoint
from src.trajectory_generator.sampled_trajectory import SampledTrajectory
from src.trajectory_generator.waypoint_path import WaypointPath


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def to_joint_trajectory(
    joint_names: Sequence[str], trajectory: PointToPoint | WaypointPath | SampledTrajectory, dt: float = 0.05
) -> JointTrajectory:
    """`trajectory` sampled every `dt` s, with positions, velocities and accelerations.

    The controller interpolates between points with quintic splines (it has all
    three), so `dt` only needs to be fine enough to follow the profile's shape.
    A SampledTrajectory gives its own points, without accelerations (cubic splines).
    """
    if len(joint_names) != trajectory.dof:
        raise ValueError(f"{len(joint_names)} joint names for a {trajectory.dof}-joint trajectory")
    msg = JointTrajectory(joint_names=list(joint_names))
    for s in trajectory.samples(dt):
        msg.points.append(
            JointTrajectoryPoint(
                positions=list(s.positions),
                velocities=list(s.velocities),
                accelerations=list(s.accelerations),
                time_from_start=_duration(s.time),
            )
        )
    return msg
