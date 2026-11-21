"""Reaching with sampling MPC (DIAL-MPC, //src/sampling_mpc): the reBot's tool cycles through poses.

ROS-free parts of //src/trajectory_generator:mpc_reach: the waypoints (a square, or picking up the scene's
cube from above and putting it back, with the gripper's targets: `grasp_waypoints`), switching between them, the
MPC for the model, and turning a plan into a joint trajectory. The MPC plans joint position targets
for the controller (a joint impedance controller: tau = kp (q* - q) + kd (v* - v)), so its model has
the arm's actuators as position servos with the controller's gains (sampling_mpc.servo). What is sent
to the controller is not the plan (its targets are jagged at the scale of its control nodes, and every replan
differs from the last) but a quintic setpoint (`QuinticSetpoint`) that moves smoothly to where the plan ends
(`plan_end`) and, at every replan, starts from the position, velocity and acceleration the last one had: it never
jumps, whatever the plans do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco
import numpy as np

from src.sampling_mpc.costs import EffortCost, GroundCost, SitePoseCost, StateCost, SumCost, VelocityLimitCost
from src.sampling_mpc.dial_mpc import DialConfig, DialMpc
from src.sampling_mpc.servo import position_servos

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
VELOCITY_LIMIT = 1.0  # rad/s, every arm joint: the MPC's cost (VelocityLimitCost) and the sent setpoint's end (plan_end)
TOOL_SITE = "tcp"
BASE_BODY = "base_link"  # the arm's root body: GroundCost keeps what moves under it above the floor
# The pose the waypoints keep the tool's orientation of: the wrist bent, tool at (0.40, 0, 0.34) m
# (as in cartesian_demo, whose waypoints were checked for reachability and contacts).
START = {"joint1": 0.0, "joint2": 1.2, "joint3": 1.4, "joint4": -0.7, "joint5": 0.0, "joint6": 0.0}
# The corners of a 30 x 18 cm square in the vertical plane in front of the robot (base frame).
SQUARE = ((0.40, -0.15, 0.24), (0.40, 0.15, 0.24), (0.40, 0.15, 0.42), (0.40, -0.15, 0.42))


GRIPPER_JOINT = "joint7"  # m: 0 closed .. 0.05 open
GRIPPER_OPEN = 0.05
GRIPPER_CLOSED = 0.0  # past the cube's width (joint7 ~ 0.024 on the 4 cm cube): the servo squeezes it
# Grasping from above at the scene cube's place (0.3 m out): 8 cm above it joint4 is at -1.31 (limit -1.57);
# from 15 cm up the pose is out of reach.
GRASP_APPROACH = 0.08


@dataclass(frozen=True)
class Waypoint:
    position: np.ndarray  # (3,) m, base frame
    xmat: np.ndarray  # (9,) the tool's rotation matrix, row-major
    gripper: float | None = None  # joint7 target while going there (None: not commanded)
    dwell: float | None = None  # seconds there before the next (None: the cycler's)


def tool_pose(mjm: mujoco.MjModel, joints: Mapping[str, float]) -> Waypoint:
    """The tool site's pose with the named joints at the given positions (the others at zero)."""
    data = mujoco.MjData(mjm)
    for name, value in joints.items():
        data.qpos[mjm.joint(name).qposadr[0]] = value
    mujoco.mj_kinematics(mjm, data)
    site = mjm.site(TOOL_SITE).id
    return Waypoint(data.site_xpos[site].copy(), data.site_xmat[site].copy())


def square_waypoints(mjm: mujoco.MjModel, corners: Sequence[Sequence[float]] = SQUARE) -> list[Waypoint]:
    """`corners` with the tool's orientation at START."""
    xmat = tool_pose(mjm, START).xmat
    return [Waypoint(np.asarray(c, dtype=float), xmat.copy()) for c in corners]


def default_finger_angle(position: Sequence[float]) -> float:
    """The direction (yaw, rad) the fingers close along by default: across the line from the base to `position`."""
    return float(np.arctan2(position[1], position[0]) + np.pi / 2)


def top_down_xmat(position: Sequence[float], finger_angle: float | None = None) -> np.ndarray:
    """The tool pointing straight down (the tcp's x, along the fingers, at -z), the fingers closing along the
    horizontal direction `finger_angle` (yaw of the tcp's y; default `default_finger_angle`: the start pose's
    orientation pitched on down)."""
    phi = default_finger_angle(position) if finger_angle is None else finger_angle
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(phi), np.sin(phi), 0.0])
    return np.column_stack([x, y, np.cross(x, y)]).flatten()


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def cube_finger_angle(position: Sequence[float], quat: Sequence[float]) -> float:
    """The finger direction that closes on two opposite faces of a cube at `position` with orientation `quat`
    (w x y z): along its most horizontal axis, or that turned by a multiple of 90 degrees, whichever is
    nearest the default (the gripper is symmetric: 180 degrees apart are the same grasp)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    axes = R.reshape(3, 3).T  # the cube's axes, in world
    axis = axes[np.argmin(np.abs(axes[:, 2]))]
    yaw = np.arctan2(axis[1], axis[0])
    default = default_finger_angle(position)
    candidates = [yaw + k * np.pi / 2 for k in range(4)]
    # Nearest modulo pi, but as an angle near the default (so joint6 turns the least).
    best = min(candidates, key=lambda c: abs(_wrap(2 * (c - default))) / 2)
    return default + _wrap(2 * (best - default)) / 2


def body_position(mjm: mujoco.MjModel, body: str = "cube") -> np.ndarray:
    """Where `body` is in the model's default state (world frame = the arm's base frame)."""
    data = mujoco.MjData(mjm)
    mujoco.mj_kinematics(mjm, data)
    return data.xpos[mjm.body(body).id].copy()


def grasp_waypoints(position: Sequence[float], approach: float = GRASP_APPROACH, close_dwell: float = 1.0,
                    open_dwell: float = 0.8, finger_angle: float | None = None) -> list[Waypoint]:
    """Picking up an object at `position` (its center) from above and putting it back, as one lap: above it
    (open), down to it, close, lift, down again, open, back up. The tcp is at the fingertips, so at `position`
    the fingers hold its upper part."""
    at = np.asarray(position, dtype=float)
    above = at + [0.0, 0.0, approach]
    xmat = top_down_xmat(at, finger_angle)
    steps = [
        (above, GRIPPER_OPEN, None),
        (at, GRIPPER_OPEN, None),
        (at, GRIPPER_CLOSED, close_dwell),  # the same pose: stays while the gripper closes
        (above, GRIPPER_CLOSED, None),  # lifted
        (at, GRIPPER_CLOSED, None),  # put back
        (at, GRIPPER_OPEN, open_dwell),
    ]
    return [Waypoint(p.copy(), xmat.copy(), gripper, dwell) for p, gripper, dwell in steps]


class GraspTask:
    """`grasp_waypoints` around an object that may move: until the gripper closes (the steps above it and down
    to it) they follow the object's latest pose (`track`); from the close on they stay where it was grasped,
    so lifting it does not move the targets; after the release (the next lap) they follow it again."""

    TRACKED = (0, 1)  # the steps going to the object

    def __init__(self, position: Sequence[float], quat: Sequence[float] | None = None,
                 approach: float = GRASP_APPROACH, dwell: float = 0.3, timeout: float = 10.0):
        self.approach = approach
        self.cycler = WaypointCycler(self._waypoints(position, quat), dwell=dwell, timeout=timeout)

    def _waypoints(self, position, quat) -> list[Waypoint]:
        angle = None if quat is None else cube_finger_angle(position, quat)
        return grasp_waypoints(position, approach=self.approach, finger_angle=angle)

    @property
    def target(self) -> Waypoint:
        return self.cycler.target

    @property
    def index(self) -> int:
        return self.cycler.index

    def track(self, position: Sequence[float], quat: Sequence[float] | None = None) -> None:
        """The object's pose now (world = base frame); moves the targets if it is being gone to."""
        if self.cycler.index in self.TRACKED:
            self.cycler.waypoints = self._waypoints(position, quat)

    def update(self, now: float, position: Sequence[float], xmat: Sequence[float]) -> bool:
        """WaypointCycler.update: True when the step changed."""
        return self.cycler.update(now, position, xmat)


class GripperRamp:
    """The gripper's setpoint: from where it is toward its target at `speed` (m/s; the real gripper's slew,
    config/rebot_arm_shared.yaml), so the sim's stiff gripper servo is not stepped across the stroke."""

    def __init__(self, position: float, speed: float = 0.08):
        self.speed = speed
        self.target = position
        self._p0 = position
        self._t0 = 0.0

    def at(self, t: float) -> float:
        step = self.speed * max(t - self._t0, 0.0)
        return float(self._p0 + np.clip(self.target - self._p0, -step, step))

    def set_target(self, now: float, target: float) -> None:
        self._p0 = self.at(now)
        self._t0 = now
        self.target = target


def set_joint_state(mjm: mujoco.MjModel, data: mujoco.MjData, states: Mapping[str, tuple[float, float]],
                    required: Sequence[str]) -> list[str]:
    """Writes measured joint states (joint -> (position, velocity)) into `data` for the hinge and slide joints
    of the model. Joints missing from `states` (or with a non-finite position) keep their value; the
    names of those in `required` that were missing are returned."""
    scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    missing = []
    for j in range(mjm.njnt):
        if int(mjm.jnt_type[j]) not in scalar:
            continue
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name not in states or not np.isfinite(states[name][0]):
            if name in required:
                missing.append(name)
            continue
        data.qpos[mjm.jnt_qposadr[j]] = states[name][0]
        data.qvel[mjm.jnt_dofadr[j]] = states[name][1] if np.isfinite(states[name][1]) else 0.0
    return missing


def rotation_error(xmat: Sequence[float], target: Sequence[float]) -> float:
    """|R - R*|_F^2 (0 aligned; 8 sin^2(angle / 2))."""
    return float(np.sum((np.asarray(xmat) - np.asarray(target)) ** 2))


class WaypointCycler:
    """Which waypoint the tool is going to: the next one once it has stayed at it for `dwell` seconds
    (within the tolerances), or `timeout` seconds after it was first the target."""

    def __init__(self, waypoints: Sequence[Waypoint], position_tolerance: float = 0.025,
                 rotation_tolerance: float = 0.15, dwell: float = 0.3, timeout: float = 10.0):
        if not waypoints:
            raise ValueError("no waypoints")
        self.waypoints = list(waypoints)
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.dwell = dwell
        self.timeout = timeout
        self.index = 0
        self.laps = 0
        self._since: float | None = None  # when the tool got to the waypoint
        self._started: float | None = None  # when the waypoint became the target

    @property
    def target(self) -> Waypoint:
        return self.waypoints[self.index]

    def update(self, now: float, position: Sequence[float], xmat: Sequence[float]) -> bool:
        """Feeds the tool's pose at time `now` (s); True if the target changed."""
        if self._started is None:
            self._started = now
        target = self.target
        there = (np.linalg.norm(np.asarray(position) - target.position) < self.position_tolerance
                 and rotation_error(xmat, target.xmat) < self.rotation_tolerance)
        if there and self._since is None:
            self._since = now
        elif not there:
            self._since = None
        dwell = self.dwell if target.dwell is None else target.dwell
        if (self._since is not None and now - self._since >= dwell) or now - self._started >= self.timeout:
            self.index = (self.index + 1) % len(self.waypoints)
            self.laps += self.index == 0
            self._since = None
            self._started = now
            return True
        return False


def servo_gains(controller_params: Mapping, joints: Sequence[str] = ARM_JOINTS) -> dict[str, tuple[float, float]]:
    """Joint -> (kp, kd) from a joint impedance controller's parameters (its `gains`: stiffness, damping)."""
    gains = controller_params["gains"]
    return {j: (float(gains[j]["stiffness"]), float(gains[j]["damping"])) for j in joints}


# The noise on the last control node, in units of half the joint range. Position targets: the reference's 1.0
# is for actions normalized around a default pose, and explores far more than a servo can follow in the horizon.
# With `update_gain` 2 (below), 0.05 reached the first corner in 0.84 s in simulation; 0.015 took 1.8 s, and
# a gain of 6 or a lower temperature were worse. (0.15 did not keep the fingers off the floor either.)
DEFAULT_SIGMA_SCALE = 0.05
# The horizon: 32 control steps of 40 ms (1.28 s; it was 16, 0.64 s, which did not see far enough ahead to keep
# the fingers off the floor), with 8 control nodes - 1 (0.16 s apart, as before).
HORIZON_STEPS = 32
HORIZON_NODES = 8


def reaching_config(**overrides) -> DialConfig:
    """DialConfig for reaching with position targets: little noise, annealed over two iterations (to 0.3 of
    it the second time, so the plan that is sent is the least noisy), a softer softmax, and a horizon of
    HORIZON_STEPS steps of 40 ms (1.28 s)."""
    values = dict(Nsample=1024, Hsample=HORIZON_STEPS, Hnode=HORIZON_NODES, ctrl_dt=0.04, sigma_scale=DEFAULT_SIGMA_SCALE, Ndiffuse=2,
                  traj_diffuse_factor=0.3, temp_sample=0.1, update_gain=2.0)
    values.update(overrides)
    return DialConfig(**values)


def actuator_positions(mjm: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """The position of the joint each actuator drives (hinge or slide), in actuator order."""
    return np.array([data.qpos[mjm.jnt_qposadr[int(mjm.actuator_trnid[a, 0])]] for a in range(mjm.nu)])


@dataclass
class ReachingWeights:
    position: float = 100.0  # per m^2
    rotation: float = 5.0  # per |R - R*|_F^2
    velocity: float = 0.1  # per (rad/s)^2, every joint: smooth motion
    effort: float = 0.0  # per (servo force / its limit)^2: minimum effort (off: it stops the tool reaching)
    ctrl_rate: float = 0.0  # per (change of a position target per control step, rad)^2
    terminal_scale: float = 5.0  # the terminal pose cost, times the stage cost's weights
    # Per (rad/s)^2 an arm joint is faster than VELOCITY_LIMIT: 0.1 rad/s over costs 10, far more than missing
    # the target by 5 cm (0.25); nothing within the limit.
    velocity_limit: float = 1000.0
    # Per m^2 any of the arm's collision geoms is below the floor (z = 0; the rollouts have no contacts): 1 cm
    # below costs 10, like 0.1 rad/s over the velocity limit; nothing above it. (1e7 made no difference.)
    ground: float = 1e5


def make_reaching_mpc(mjm: mujoco.MjModel, gains: Mapping[str, tuple[float, float]], waypoint: Waypoint,
                      config: DialConfig | None = None, weights: ReachingWeights | None = None,
                      device=None) -> tuple[DialMpc, SitePoseCost]:
    """An MPC reaching `waypoint` with the tool (and the cost, to `set_target` as the waypoint changes).

    Turns the arm's actuators of `mjm` into position servos with `gains` (in place; contacts off).
    Call `mpc.hold(actuator_positions(mjm, data))` before the first plan, so the plan starts from the
    measured pose, not the middle of the joint ranges. The first mpc.plan() compiles the GPU kernels
    (tens of seconds).
    """
    w = weights or ReachingWeights()
    position_servos(mjm, gains)
    pose = SitePoseCost(mjm, TOOL_SITE, waypoint.position, waypoint.xmat, pos_weight=w.position,
                        rot_weight=w.rotation, terminal_scale=w.terminal_scale, device=device)
    # The gripper is not part of the task: it stays where it is (closed).
    cost = SumCost(pose, StateCost(mjm, qvel_weight=w.velocity, device=device),
                   EffortCost(mjm, force_weight=w.effort, rate_weight=w.ctrl_rate, device=device),
                   VelocityLimitCost(mjm, joints=ARM_JOINTS, limit=VELOCITY_LIMIT, weight=w.velocity_limit,
                                     terminal_scale=w.terminal_scale, device=device),
                   GroundCost(mjm, BASE_BODY, weight=w.ground, terminal_scale=w.terminal_scale, device=device))
    return DialMpc(mjm, cost, config or reaching_config(), device=device), pose


def quintic(p0: np.ndarray, v0: np.ndarray, a0: np.ndarray, p1: np.ndarray, v1: np.ndarray, duration: float) -> np.ndarray:
    """Coefficients (6, J) of the quintic p(t) = sum c_k t^k from (p0, v0, a0) at t = 0 to (p1, v1, a = 0) at `duration`."""
    T = duration
    c = np.zeros((6,) + np.shape(p0))
    c[0], c[1], c[2] = p0, v0, a0 / 2
    # The remaining three from the end conditions: position, velocity, acceleration (0).
    M = np.array([[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]])
    rhs = np.stack([p1 - (c[0] + c[1] * T + c[2] * T**2), v1 - (c[1] + 2 * c[2] * T), -2 * c[2]])
    c[3:] = np.linalg.solve(M, rhs)
    return c


class QuinticSetpoint:
    """The setpoint sent to the controller: one quintic after another, each starting from the position, velocity and
    acceleration the last one had at that moment, so the setpoint never jumps (positions, velocities or accelerations)
    however the plans change. A plan only chooses where the next quintic ends (`retarget`)."""

    def __init__(self, initial: Sequence[float]):
        self._c = np.zeros((6, len(initial)))
        self._c[0] = initial
        self._t0 = 0.0
        self._duration = 0.0

    def state(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and acceleration of the setpoint at `now` (it holds the end of its segment after)."""
        tau = min(max(now - self._t0, 0.0), self._duration)
        k = np.arange(6)
        c = self._c
        p = sum(c[i] * tau**i for i in k)
        v = sum(i * c[i] * tau ** (i - 1) for i in range(1, 6))
        a = sum(i * (i - 1) * c[i] * tau ** (i - 2) for i in range(2, 6))
        return p, v, a

    def retarget(self, now: float, target: Sequence[float], target_velocity: Sequence[float], duration: float) -> None:
        """From the setpoint's state at `now`, a new quintic reaching `target` (with `target_velocity`) in `duration` s."""
        p0, v0, a0 = self.state(now)
        self._c = quintic(p0, v0, a0, np.asarray(target, dtype=float), np.asarray(target_velocity, dtype=float), duration)
        self._t0 = now
        self._duration = duration

    def points(self, now: float, dt: float) -> list[tuple[float, list[float]]]:
        """(time from now, positions) every dt up to the end of the current segment."""
        n = int(np.floor((self._duration - (now - self._t0)) / dt + 1e-9))
        return [((k + 1) * dt, self.state(now + (k + 1) * dt)[0].tolist()) for k in range(n)]


def plan_end(targets: np.ndarray, dt: float, fit_steps: int = 8, max_velocity: float = VELOCITY_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """Where a plan (position targets (K, J), one per dt) ends and how fast it is moving there: a line fit to its last
    `fit_steps` targets, so a noisy last node does not set the end position and velocity; the velocity is limited."""
    n = min(fit_steps, len(targets))
    t = dt * np.arange(n)
    last = np.asarray(targets)[-n:]
    slope = np.array([np.polyfit(t, last[:, j], 1)[0] for j in range(last.shape[1])])
    end = last.mean(axis=0) + slope * (t[-1] - t.mean())
    return end, np.clip(slope, -max_velocity, max_velocity)


def joint_positions(mjm: mujoco.MjModel, qpos: np.ndarray, joints: Sequence[str] = ARM_JOINTS) -> np.ndarray:
    """The positions of `joints` (hinge or slide) from qpos (..., nq), as (..., len(joints))."""
    return np.asarray(qpos)[..., [mjm.joint(j).qposadr[0] for j in joints]]
