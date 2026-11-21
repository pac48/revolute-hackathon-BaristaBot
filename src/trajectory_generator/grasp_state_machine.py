"""A state machine of grasping episodes: the cube is put somewhere reachable at random, picked up and lifted.

    machine = GraspStateMachine(arm, GraspConfig(episodes=10), rng=np.random.default_rng(0))
    machine.run()  # -> State.DONE (or State.FAILED if no reachable cube pose could be found)
    machine.successes, machine.failures

The robot is behind `Arm` (grasp_cube's RosArm: TrajectoryClient moves, the sim's /mujoco_state and
/set_mujoco_state), so this module has no ROS and is tested with a fake arm. Every episode starts with a
planned (collision-free) move to the home pose, for the most room to move from; then straight tool lines
(Cartesian, dense IK) above and down to the cube, the tool pointing straight down and the fingers closing
on two of the cube's faces.

    HOME -> RANDOMIZE -> OPEN -> APPROACH -> DESCEND -> CLOSE -> LIFT -> HOME -> ... -> DONE

RANDOMIZE samples a cube pose in front of the robot (GraspConfig's region, any yaw) until the tool can reach
the poses above and at it (Arm.reachable: IK), and puts the cube there (Arm.set_cube). An episode succeeds
when the lifted cube has come up with the gripper. CLOSE checks the gripper stopped on something and LIFT
that the cube came up; either failing opens, backs up and tries again (RECOVER), up to `max_attempts`; then
the episode failed. A move the tool cannot make (Unreachable, e.g. IK failing along a line) also fails the
episode. Each episode, success or not, is followed by a new random cube pose until `episodes` have run.

The episode proper (Arm.begin_episode .. Arm.end_episode, e.g. what a recorder keeps) is from the cube being
placed to it being lifted; HOME and RANDOMIZE are the reset. Only a successful episode is ended.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Callable, Protocol, Sequence

import numpy as np

from src.trajectory_generator.cartesian_path import Pose


class State(enum.Enum):
    HOME = "home"
    RANDOMIZE = "randomize"
    OPEN = "open"
    APPROACH = "approach"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    RECOVER = "recover"
    DONE = "done"
    FAILED = "failed"


class Unreachable(RuntimeError):
    """The arm cannot make a move (no IK solution along it)."""


class Arm(Protocol):
    def go_home(self) -> None:
        """A planned (collision-free) move to the home pose, the tool above the workspace."""

    def cube_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The cube's center (m, base frame) and orientation (w, x, y, z) now, or None if not known."""

    def set_cube(self, position: Sequence[float], quat: Sequence[float]) -> None:
        """Puts the cube at `position` with orientation `quat` (w, x, y, z), at rest."""

    def reachable(self, above: Pose, at: Pose) -> bool:
        """Whether the tool can reach `above` the cube from where the arm is, and then `at` it (IK)."""

    def move_tool(self, poses: Sequence[Pose], speed: float, near_cube: bool = False) -> None:
        """The tool (tcp) through `poses` on straight lines at up to `speed` m/s; raises Unreachable.
        `near_cube`: the move goes down to, holds or leaves the cube (contact with it is expected)."""

    def gripper(self, position: float) -> float:
        """Moves the gripper (joint7, m: 0 closed .. 0.05 open) toward `position`; returns where it stopped."""

    def begin_episode(self) -> None:
        """The reset is done (home, cube placed): the episode begins (e.g. start recording)."""

    def end_episode(self) -> None:
        """The episode succeeded (e.g. stop recording, keeping it). Never called for a failed episode: the next
        begin_episode starts over (a recorder discards what it had)."""


GRIPPER_OPEN = 0.05
GRIPPER_CLOSED = 0.0  # past the cube's width: the gripper stops on it and squeezes


@dataclasses.dataclass
class GraspConfig:
    approach: float = 0.08  # m above the cube's center to come down from (higher, joint4 nears its limit)
    lift: float = 0.08  # m to lift it by
    grasp_offset: float = 0.0  # m: the tcp (the fingertips) this far above the cube's center when closing
    approach_speed: float = 0.10  # m/s, in free space
    contact_speed: float = 0.03  # m/s, going down to the cube and lifting it
    # Where the cube is put (its center, base frame): x and y ranges in front of the robot, any yaw.
    x_range: tuple[float, float] = (0.22, 0.38)
    y_range: tuple[float, float] = (-0.15, 0.15)
    cube_z: float = 0.02  # its center resting on the floor (half its 4 cm size)
    max_samples: int = 50  # random poses tried for a reachable one before FAILED
    held_opening: float = 0.01  # joint7 (m) above which the closed gripper holds something
    lifted: float = 0.03  # m the cube must rise by when lifted
    moved: float = 0.01  # m the cube may move between APPROACH and DESCEND before approaching again
    episodes: int = 1  # 0: forever
    max_attempts: int = 3  # grasp tries per episode


def finger_angle(position: Sequence[float], quat: Sequence[float] | None) -> float:
    """The yaw (rad) of the direction the fingers close along: across the line from the base to the cube by
    default, or, knowing the cube's orientation (w, x, y, z), along one of its horizontal face normals, the
    one nearest that default (the gripper is symmetric: 180 degrees apart is the same grasp)."""
    default = math.atan2(position[1], position[0]) + math.pi / 2
    if quat is None:
        return default
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    axis = R.T[np.argmin(np.abs(R[2]))]  # the cube's most horizontal axis
    yaw = math.atan2(axis[1], axis[0])
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    best = min((yaw + k * math.pi / 2 for k in range(4)), key=lambda c: abs(wrap(2 * (c - default))))
    return default + wrap(2 * (best - default)) / 2


def top_down_pose(position: Sequence[float], angle: float) -> Pose:
    """The tool at `position` pointing straight down (tcp x at -z), the fingers closing along yaw `angle`."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(angle), math.sin(angle), 0.0])
    R = np.column_stack([x, y, np.cross(x, y)])
    return Pose(tuple(position), _quaternion(R))


def yaw_quat(yaw: float) -> np.ndarray:
    """(w, x, y, z) of a turn by `yaw` about z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


class GraspStateMachine:
    def __init__(self, arm: Arm, config: GraspConfig | None = None, rng: np.random.Generator | None = None,
                 log: Callable[[str], None] = print):
        self.arm = arm
        self.config = config or GraspConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.log = log
        self.state = State.HOME
        self.episodes_done = 0
        self.successes = 0
        self.failures = 0
        self.attempts = 0
        self.pick: np.ndarray | None = None  # the cube's center when it was gone for
        self.angle = 0.0  # the fingers' yaw for this grasp

    def _at(self, center, angle=None) -> Pose:
        return top_down_pose(np.asarray(center) + [0.0, 0.0, self.config.grasp_offset],
                             self.angle if angle is None else angle)

    def _above(self, center, height: float, angle=None) -> Pose:
        return top_down_pose(np.asarray(center) + [0.0, 0.0, self.config.grasp_offset + height],
                             self.angle if angle is None else angle)

    def step(self) -> State:
        """Runs the current state; returns (and moves to) the next."""
        handler = getattr(self, f"_{self.state.value}")
        try:
            next_state = handler()
        except Unreachable as e:
            self.log(f"{self.state.value}: cannot reach ({e})")
            next_state = self._end_episode(success=False)
        if next_state is not self.state:
            self.log(f"{self.state.value} -> {next_state.value}")
        self.state = next_state
        return next_state

    def run(self) -> State:
        while self.state not in (State.DONE, State.FAILED):
            self.step()
        return self.state

    def _home(self) -> State:
        self.arm.go_home()
        return State.RANDOMIZE

    def _randomize(self) -> State:
        c = self.config
        for _ in range(c.max_samples):
            position = np.array([self.rng.uniform(*c.x_range), self.rng.uniform(*c.y_range), c.cube_z])
            quat = yaw_quat(self.rng.uniform(-math.pi, math.pi))
            angle = finger_angle(position, quat)
            if self.arm.reachable(self._above(position, c.approach, angle), self._at(position, angle)):
                self.arm.set_cube(position, quat)
                self.arm.begin_episode()
                self.attempts = 0
                self.log(f"episode {self.episodes_done + 1}: cube at {np.round(position, 3).tolist()}")
                return State.OPEN
        self.log(f"no reachable cube pose in {c.max_samples} samples")
        return State.FAILED

    def _open(self) -> State:
        self.arm.gripper(GRIPPER_OPEN)
        return State.APPROACH

    def _approach(self) -> State:
        seen = self.arm.cube_pose()
        if seen is None:
            self.log("no cube")
            return State.FAILED
        position, quat = seen
        self.pick = np.asarray(position, dtype=float)
        self.angle = finger_angle(self.pick, quat)
        self.arm.move_tool([self._above(self.pick, self.config.approach)], self.config.approach_speed)
        return State.DESCEND

    def _descend(self) -> State:
        seen = self.arm.cube_pose()
        if seen is not None and np.linalg.norm(np.asarray(seen[0]) - self.pick) > self.config.moved:
            self.log(f"the cube moved to {np.round(seen[0], 3).tolist()}: approaching again")
            return State.APPROACH
        self.arm.move_tool([self._at(self.pick)], self.config.contact_speed, near_cube=True)
        return State.CLOSE

    def _close(self) -> State:
        opening = self.arm.gripper(GRIPPER_CLOSED)
        if opening < self.config.held_opening:
            self.log(f"the gripper closed on nothing (joint7 {opening:.3f})")
            return self._retry()
        return State.LIFT

    def _lift(self) -> State:
        c = self.config
        self.arm.move_tool([self._above(self.pick, c.lift)], c.contact_speed, near_cube=True)
        seen = self.arm.cube_pose()
        if seen is None or seen[0][2] - self.pick[2] < c.lifted:
            self.log(f"the cube did not come up ({'not seen' if seen is None else np.round(seen[0], 3).tolist()})")
            return self._retry()
        return self._end_episode(success=True)

    def _retry(self) -> State:
        self.attempts += 1
        if self.attempts >= self.config.max_attempts:
            self.log(f"giving up on this cube after {self.attempts} attempts")
            return self._end_episode(success=False)
        return State.RECOVER

    def _recover(self) -> State:
        self.arm.gripper(GRIPPER_OPEN)
        self.arm.move_tool([self._above(self.pick, self.config.approach)], self.config.contact_speed, near_cube=True)
        return State.APPROACH

    def _end_episode(self, success: bool) -> State:
        self.episodes_done += 1
        if success:
            self.arm.end_episode()
            self.successes += 1
        else:
            self.failures += 1
        self.log(f"episode {self.episodes_done} {'lifted the cube' if success else 'failed'} "
                 f"({self.successes} lifted, {self.failures} failed)")
        if self.config.episodes and self.episodes_done >= self.config.episodes:
            return State.DONE
        return State.HOME
