"""Grasping episodes: the cube is put somewhere reachable at random, picked up and lifted (grasp_state_machine).

Usage: bazel run //src/trajectory_generator:grasp_cube -- [--model rebot_arm_gemini_305] [--episodes 10]
           [--seed 0] [--x 0.22 0.38] [--y -0.15 0.15] [--approach 0.08] [--lift 0.08] [--speed 0.1]
           [--contact-speed 0.03]

Needs the simulation running (`bazel run //src/ros2_control_sim:mujoco_arm_sim -- --model rebot_arm_gemini_305`):
the cube's pose is read from /mujoco_state (the sim's whole MuJoCo state) and set with /set_mujoco_state.
Every episode moves (a straight joint-space move) back to the home pose, then puts the cube at
a random pose in front of the robot
(--x, --y, any yaw) that the tool can reach above and at (IK; else another sample), opens the gripper,
comes above the cube and down to it on straight lines (TrajectoryClient.move_cartesian: dense IK), the
tool pointing straight down with the fingers closing on two of its faces, closes and lifts it. The episode
succeeds when the cube came up; a missed grasp is tried again (up to 3 times), and a move the arm cannot
make ends the episode. --episodes of them (0: until stopped). Exits 0 when they have run, 1 if no reachable
cube pose was found.
"""

import argparse
import sys
import time

import mujoco
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy

from mujoco_state_msgs.msg import MujocoState
from mujoco_state_msgs.srv import SetMujocoState
from std_srvs.srv import Trigger
from src.robot_model.active_model import active_model
from src.robot_model.rebot_arm_gemini_305_data import mjcf as gemini_mjcf
from src.trajectory_generator import grasp_state_machine as gsm
from src.trajectory_generator import kinematics_client
from src.trajectory_generator.client import TrajectoryClient, TrajectoryError
from src.trajectory_generator.kinematics_client import KinematicsError

MODELS = {"rebot_arm_gemini_305": gemini_mjcf}  # the models with the cube in their scene
# The home pose: the wrist bent, the tool at (0.40, 0, 0.34) m, clear of the table (as cartesian_demo).
HOME = {"joint1": 0.0, "joint2": 1.2, "joint3": 1.4, "joint4": -0.7, "joint5": 0.0, "joint6": 0.0}
GRIPPER = "joint7"


class RosArm:
    """grasp_state_machine.Arm on a TrajectoryClient, with the cube's pose from /mujoco_state and set by
    /set_mujoco_state."""

    def __init__(self, node, model: str, args):
        self.node = node
        self.model = model
        self.args = args
        self.client = TrajectoryClient(node, controller=args.controller)
        mjm = mujoco.MjModel.from_xml_path(MODELS[model].path())
        body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if body < 0 or mjm.body_jntnum[body] != 1 or mjm.jnt_type[mjm.body_jntadr[body]] != mujoco.mjtJoint.mjJNT_FREE:
            raise SystemExit(f"{model}'s scene has no free body 'cube'")
        joint = mjm.body_jntadr[body]
        self.cube_qpos, self.cube_qvel = int(mjm.jnt_qposadr[joint]), int(mjm.jnt_dofadr[joint])
        self.nq, self.nv = mjm.nq, mjm.nv
        self.latest = None  # (receipt time, qpos, qvel)
        node.create_subscription(MujocoState, "/mujoco_state", self._on_state,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.set_state = node.create_client(SetMujocoState, "/set_mujoco_state")
        if self._wait_for_state(5.0) is None or not self.set_state.wait_for_service(timeout_sec=5.0):
            raise SystemExit("no /mujoco_state or /set_mujoco_state: is the simulation running (with this model)?")
        self.recorder = None  # (start, stop) clients of the data recorder, with --record
        if args.record:
            self.recorder = tuple(node.create_client(Trigger, f"{args.record}/{name}") for name in ("start", "stop"))
            if not all(c.wait_for_service(timeout_sec=5.0) for c in self.recorder):
                raise SystemExit(f"no {args.record}/start and stop: is the recorder running with --episodic?")

    def _on_state(self, msg: MujocoState):
        if len(msg.qpos) == self.nq and len(msg.qvel) == self.nv:
            self.latest = (time.monotonic(), np.array(msg.qpos), np.array(msg.qvel))

    def _wait_for_state(self, timeout: float, newer_than: float = 0.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.latest is not None and self.latest[0] > newer_than:
                return self.latest
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return None

    def go_home(self) -> None:
        # A straight joint-space move (there is no motion planner): nothing checks it for collisions on the way.
        self.client.move_to(HOME, max_velocity=self.args.max_joint_velocity)
        self.client.wait_until_at(HOME, 0.02, 2.0)

    def cube_pose(self):
        state = self._wait_for_state(1.0, newer_than=time.monotonic())  # one taken after now
        if state is None:
            return None
        a = self.cube_qpos
        return state[1][a:a + 3].copy(), state[1][a + 3:a + 7].copy()

    def set_cube(self, position, quat) -> None:
        _, qpos, qvel = self._wait_for_state(1.0, newer_than=time.monotonic())
        qpos[self.cube_qpos:self.cube_qpos + 3] = position
        qpos[self.cube_qpos + 3:self.cube_qpos + 7] = quat
        qvel[self.cube_qvel:self.cube_qvel + 6] = 0.0
        future = self.set_state.call_async(SetMujocoState.Request(qpos=qpos.tolist(), qvel=qvel.tolist()))
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if not future.done() or not future.result().success:
            raise TrajectoryError(f"/set_mujoco_state failed: {future.result().message if future.done() else 'timed out'}",
                                  "set_cube")
        time.sleep(0.2)  # it settles on the floor

    def reachable(self, above, at) -> bool:
        joints = kinematics_client.chain(self.model).joint_names
        start = dict(zip(joints, self.client.current_positions(joints)))
        try:
            # Far from the arm's pose: no limit on the joint step (that is for densely sampled paths).
            path = kinematics_client.ik_path(self.model, start, [above], max_joint_step=10.0)
            # At the cube without the collision check: the kinematics' model has the cube at its default pose,
            # not where it is (and at a grasp the fingers are around it).
            kinematics_client.ik_path(self.model, dict(zip(path.joint_names, path.positions[-1])), [at],
                                      max_joint_step=10.0, check_collisions=False)
            return True
        except KinematicsError:
            return False

    def move_tool(self, poses, speed: float, near_cube: bool = False) -> None:
        try:
            # Near the cube without the collision check (see reachable).
            move, _ = self.client.move_cartesian(self.model, poses, max_linear_velocity=speed, max_angular_velocity=0.5,
                                                 max_joint_velocity=self.args.max_joint_velocity,
                                                 check_collisions=not near_cube)
        except KinematicsError as e:
            raise gsm.Unreachable(str(e)) from e
        self.client.wait_until_at(dict(zip(move.ik.joint_names, move.ik.positions[-1])), 0.005, 1.0)

    def _call(self, client, what: str) -> str:
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 30.0  # stopping writes what is queued
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if not future.done() or not future.result().success:
            raise TrajectoryError(f"recorder {what}: {future.result().message if future.done() else 'timed out'}", what)
        return future.result().message

    def begin_episode(self) -> None:
        if self.recorder:
            self.node.get_logger().info(f"recording to {self._call(self.recorder[0], 'start')}")

    def end_episode(self) -> None:
        if self.recorder:
            self.node.get_logger().info(f"recorded {self._call(self.recorder[1], 'stop')}")

    def gripper(self, position: float) -> float:
        self.client.move_to({GRIPPER: position}, duration=self.args.gripper_time)
        time.sleep(0.3)  # the fingers settle (on the cube, if they close on it)
        return self.client.current_positions([GRIPPER])[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    active_model.add_model_argument(ap, MODELS)
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--record", nargs="?", const="/data_recorder", metavar="RECORDER",
                    help="mark the episodes for a data recorder run with --episodic (its node, default /data_recorder): "
                         "~/start once the cube is placed, ~/stop once it is lifted; a failed episode is not stopped, "
                         "so the recorder discards it at the next start")
    defaults = gsm.GraspConfig()
    ap.add_argument("--episodes", type=int, default=10, help="grasping episodes (default 10; 0: until stopped)")
    ap.add_argument("--seed", type=int, help="random seed of the cube poses (default: a new one each run)")
    ap.add_argument("--x", type=float, nargs=2, default=defaults.x_range, metavar=("MIN", "MAX"),
                    help=f"the cube's x range, m (default {defaults.x_range[0]} {defaults.x_range[1]})")
    ap.add_argument("--y", type=float, nargs=2, default=defaults.y_range, metavar=("MIN", "MAX"),
                    help=f"the cube's y range, m (default {defaults.y_range[0]} {defaults.y_range[1]})")
    ap.add_argument("--approach", type=float, default=defaults.approach, help=f"m above the cube to come down from (default {defaults.approach})")
    ap.add_argument("--lift", type=float, default=defaults.lift, help=f"m to lift it (default {defaults.lift})")
    ap.add_argument("--speed", type=float, default=defaults.approach_speed, help=f"tool speed in free space, m/s (default {defaults.approach_speed})")
    ap.add_argument("--contact-speed", type=float, default=defaults.contact_speed,
                    help=f"tool speed near the cube, m/s (default {defaults.contact_speed})")
    ap.add_argument("--max-joint-velocity", type=float, default=1.0, help="rad/s (default 1)")
    ap.add_argument("--gripper-time", type=float, default=0.8, help="s to open or close the gripper (default 0.8)")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    rclpy.init(args=ros_args)
    node = rclpy.create_node("grasp_cube")
    try:
        arm = RosArm(node, args.model, args)
        config = gsm.GraspConfig(approach=args.approach, lift=args.lift, approach_speed=args.speed,
                                 contact_speed=args.contact_speed, x_range=tuple(args.x), y_range=tuple(args.y),
                                 episodes=args.episodes)
        machine = gsm.GraspStateMachine(arm, config, rng=np.random.default_rng(args.seed), log=node.get_logger().info)
        state = machine.run()
        node.get_logger().info(f"{state.value}: {machine.successes} of {machine.episodes_done} episodes lifted the cube")
        return 0 if state is gsm.State.DONE else 1
    except (TrajectoryError, KinematicsError) as e:
        node.get_logger().error(f"{e}")
        return 1
    except KeyboardInterrupt:
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
