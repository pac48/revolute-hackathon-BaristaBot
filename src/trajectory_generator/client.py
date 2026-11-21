"""Send point-to-point moves to a joint_trajectory_controller.

    client = TrajectoryClient(node)  # an rclpy node
    result = client.move_to({"joint1": 0.5, "joint2": -0.3}, max_velocity=1.0)

The move starts from the current positions on /joint_states and is sent as a
control_msgs/FollowJointTrajectory goal to <controller>/follow_joint_trajectory.
Every wait is bounded by `timeout`, so a missing controller fails fast.
"""

import dataclasses
import time
from collections.abc import Mapping, Sequence

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

from src.trajectory_generator import (
    cartesian_path,
    kinematics_client,
    point_to_point,
    sampled_trajectory,
    waypoint_path,
)
from src.trajectory_generator.msgs import to_joint_trajectory


class TrajectoryError(RuntimeError):
    """`stage` says what failed: "server" (not found), "accept", "result" or "joint_states"."""

    def __init__(self, message: str, stage: str = ""):
        super().__init__(message)
        self.stage = stage


@dataclasses.dataclass(frozen=True)
class ExecutionTiming:
    """Wall-clock seconds for one execute() call (see TrajectoryClient.last_timing)."""

    server_wait: float  # until the action server was available
    accept: float  # goal sent -> accepted
    execution: float  # accepted -> result
    planned: float  # trajectory duration

    @property
    def overhead(self) -> float:
        """Time beyond the planned duration between acceptance and the result."""
        return self.execution - self.planned


@dataclasses.dataclass(frozen=True)
class CartesianMove:
    """What move_cartesian planned: the tool path, its IK solutions and the joint trajectory."""

    path: cartesian_path.CartesianPath
    ik: kinematics_client.IkPath
    trajectory: sampled_trajectory.SampledTrajectory

    @property
    def slowdown(self) -> float:
        """How much longer than the Cartesian timing the joint limits made it (1 = not at all)."""
        return self.trajectory.duration / self.path.duration


class TrajectoryClient:
    def __init__(self, node: Node, controller: str = "joint_trajectory_controller", timeout: float = 10.0):
        self.node = node
        self.timeout = timeout
        self._joint_states: dict[str, float] = {}
        node.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        # The controller's current setpoint (it holds the last trajectory's end).
        self._references: dict[str, float] = {}
        node.create_subscription(JointTrajectoryControllerState, f"/{controller}/controller_state",
                                 self._on_controller_state, 10)
        self._action = ActionClient(node, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        self.last_timing: ExecutionTiming | None = None

    def _on_joint_states(self, msg: JointState):
        self._joint_states.update(zip(msg.name, msg.position))

    def _on_controller_state(self, msg: JointTrajectoryControllerState):
        self._references.update(zip(msg.joint_names, msg.reference.positions))

    def _spin_until(self, done, what: str, stage: str = ""):
        deadline = time.monotonic() + self.timeout
        while not done():
            if time.monotonic() > deadline:
                raise TrajectoryError(f"timed out after {self.timeout}s waiting for {what}", stage)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_for_server(self):
        """Wait until the controller's action server is available."""
        if not self._action.wait_for_server(timeout_sec=self.timeout):
            raise TrajectoryError(f"action server {self._action._action_name} not available", "server")

    def current_positions(self, joints: Sequence[str]) -> list[float]:
        """Latest /joint_states positions of `joints`, waiting for them if needed."""
        self._spin_until(
            lambda: all(j in self._joint_states for j in joints), f"/joint_states with {list(joints)}", "joint_states"
        )
        return [self._joint_states[j] for j in joints]

    def commanded_positions(self, joints: Sequence[str]) -> list[float]:
        """The controller's setpoint for `joints` (what it holds or tracks), waiting for it if needed.

        Differs from current_positions by the tracking error (e.g. sag under
        gravity). Starting a trajectory here avoids a jump in the setpoint.
        """
        self._spin_until(
            lambda: all(j in self._references for j in joints), f"controller_state with {list(joints)}", "joint_states"
        )
        return [self._references[j] for j in joints]

    def wait_until_at(
        self,
        goal: Mapping[str, float],
        tolerance: float | Mapping[str, float] = 0.01,
        timeout: float = 2.0,
    ) -> dict[str, float]:
        """Spin until every joint in `goal` is within `tolerance` of it (scalar or per joint), or `timeout`.

        Returns the latest measured positions either way; compare them to decide.
        The controller reports success when the trajectory ends, which can be a
        little before the joints settle on the goal.
        """
        joints = list(goal)

        def tol(joint):
            return tolerance.get(joint, 0.01) if isinstance(tolerance, Mapping) else tolerance

        deadline = time.monotonic() + timeout
        positions = dict(zip(joints, self.current_positions(joints)))
        while any(abs(positions[j] - goal[j]) > tol(j) for j in joints) and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            positions = {j: self._joint_states[j] for j in joints}
        return positions

    def move_to(
        self,
        goal: Mapping[str, float],
        duration: float | None = None,
        max_velocity: float | Sequence[float] = 1.0,
        max_acceleration: float | Sequence[float] | None = None,
    ) -> FollowJointTrajectory.Result:
        """Move the joints in `goal` there with a quintic point-to-point trajectory; wait for the result.

        See point_to_point.plan for `duration` / `max_velocity` / `max_acceleration`.
        Raises TrajectoryError unless the controller reports success.
        """
        joints = list(goal)
        trajectory = point_to_point.plan(
            self.current_positions(joints),
            [goal[j] for j in joints],
            duration=duration,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
        return self.execute(joints, trajectory)

    def current_pose(self, model: str, commanded: bool = False) -> cartesian_path.Pose:
        """`model`'s tip pose (in its root frame) at the measured joint states, or at the
        controller's setpoint with `commanded`."""
        joints = kinematics_client.chain(model).joint_names
        positions = self.commanded_positions(joints) if commanded else self.current_positions(joints)
        return kinematics_client.forward(model, dict(zip(joints, positions)))

    def move_cartesian(
        self,
        model: str,
        waypoints: Sequence[cartesian_path.Pose],
        max_linear_velocity: float = 0.1,
        max_angular_velocity: float = 0.5,
        max_joint_velocity: float | Sequence[float] = 1.0,
        dt: float = 0.01,
        max_joint_step: float = 0.2,
        check_collisions: bool = True,
    ) -> tuple["CartesianMove", FollowJointTrajectory.Result]:
        """Move `model`'s tip through `waypoints` on straight lines, and wait.

        The Cartesian path (cartesian_path.plan: straight segments from the pose at
        the controller's current setpoint, so the setpoint doesn't jump; quintic timing within the linear/angular speed limits, stopping at
        each waypoint) is sampled every `dt` s and solved with IK in order
        (kinematics_client.ik_path), each from the previous solution. The joint
        trajectory goes through those solutions at the sample times, slowed down
        uniformly if a joint would exceed `max_joint_velocity`. Raises
        kinematics_client.KinematicsError, before moving, if a pose has no solution
        or the solution jumps (see ik_path), or, with `check_collisions`, puts the model's bodies in contact.
        """
        joints = kinematics_client.chain(model).joint_names
        start = dict(zip(joints, self.commanded_positions(joints)))
        start_pose = kinematics_client.forward(model, start)
        path = cartesian_path.plan([start_pose, *waypoints], max_linear_velocity, max_angular_velocity)
        samples = path.samples(dt)
        ik = kinematics_client.ik_path(model, start, [pose for _, pose in samples], max_joint_step=max_joint_step,
                                       check_collisions=check_collisions)
        trajectory = sampled_trajectory.SampledTrajectory(
            [t for t, _ in samples], ik.positions
        ).limit_velocity(max_joint_velocity)
        move = CartesianMove(path, ik, trajectory)
        return move, self.execute(ik.joint_names, trajectory)

    def execute(
        self,
        joints: Sequence[str],
        trajectory: point_to_point.PointToPoint | waypoint_path.WaypointPath | sampled_trajectory.SampledTrajectory,
    ) -> FollowJointTrajectory.Result:
        """Send `trajectory` for `joints` and wait until it finishes (timings in last_timing)."""
        self.last_timing = None
        t0 = time.monotonic()
        self.wait_for_server()
        goal = FollowJointTrajectory.Goal(trajectory=to_joint_trajectory(joints, trajectory))

        t1 = time.monotonic()
        send = self._action.send_goal_async(goal)
        self._spin_until(send.done, "the goal to be accepted", "accept")
        handle = send.result()
        if not handle.accepted:
            raise TrajectoryError("goal rejected by the controller", "accept")

        t2 = time.monotonic()
        result_future = handle.get_result_async()
        timeout, self.timeout = self.timeout, trajectory.duration + self.timeout
        try:
            self._spin_until(result_future.done, "the trajectory to finish", "result")
        finally:
            self.timeout = timeout
        self.last_timing = ExecutionTiming(t1 - t0, t2 - t1, time.monotonic() - t2, trajectory.duration)
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise TrajectoryError(
                f"trajectory failed (error_code {result.error_code}): {result.error_string}", "result"
            )
        return result
