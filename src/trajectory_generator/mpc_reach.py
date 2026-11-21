"""Sampling MPC (DIAL-MPC on the GPU) drives the reBot's tool through poses, streaming to the controller.

Usage: bazel run //src/trajectory_generator:mpc_reach -- [--model rebot_arm|rebot_arm_gemini_305] [--rate 2]
                                                         [--task grasp|square] [--cube X Y Z]
                                                         [--samples 1024] [--no-command]

Needs the simulation (or robot) running, e.g. `bazel run //src/ros2_control_sim:mujoco_arm_sim`,
and an NVIDIA GPU. --task grasp (the default) picks up the cube from above, lifts it, puts it back, lets go
and goes again (mpc_reaching.GraspTask). In the simulation it follows the cube: the sim publishes its whole
MuJoCo state (/mujoco_state, mit_system_interface), and until the gripper closes the targets move with the
cube's pose (the fingers closing on two of its faces); the lift and the put-back are where it was grasped.
Without /mujoco_state (the real arm) the cube is where the model has it (rebot_arm_gemini_305's scene.xml),
or at --cube (m, base frame), and must be back there for the next lap.
--task square cycles the tool through the corners of a 30 x 18 cm square in front of the robot, with the
orientation of the cartesian_demo's start pose. The first plan compiles the GPU kernels (tens of seconds);
the arm starts moving after that.

Every cycle (--rate, default 2 Hz) it plans from the latest /mujoco_state (every joint, the cube's too) if the
sim publishes it for this model, else from /joint_states: the MPC's model is the arm with position servos of the
controller's gains, rolled out from the measured state; a quintic setpoint moves to where the plan ends (what is left of its 1.28 s horizon after the time it took to plan) and is
sent to <controller>/joint_trajectory as a JointTrajectory of positions 20 ms apart (joint1-joint6, and joint7 when the
waypoint sets the gripper: it ramps to that at 8 cm/s; the MPC does not plan it). Each replan starts its quintic from the position, velocity and acceleration the setpoint has then, so the setpoint
never jumps, however the plans change. The MPC's model has no
contacts (it doesn't avoid self-collisions: the waypoints are in the region checked by cartesian_demo).

Publishes /mpc_reach/target_pose (geometry_msgs/PoseStamped, the current waypoint, frame base_link). With
--no-command it plans and publishes the target but sends nothing to the controller.
"""

import argparse
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from mujoco_state_msgs.msg import MujocoState
from src.robot_model.active_model import active_model
from src.robot_model.rebot_arm_data import rs_mjcf
from src.robot_model.rebot_arm_gemini_305_data import mjcf as gemini_mjcf
from src.ros2_control_sim import controller_config, rebot_arm_controllers
from src.trajectory_generator import mpc_reaching

MODELS = {"rebot_arm": rs_mjcf, "rebot_arm_gemini_305": gemini_mjcf}


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


class MpcReach(Node):
    def __init__(self, args):
        super().__init__("mpc_reach")
        self.args = args
        self.mjm = mujoco.MjModel.from_xml_path(MODELS[args.model].path())
        controller_params = controller_config.load(rebot_arm_controllers.path())[args.controller]["ros__parameters"]
        self.gains = mpc_reaching.servo_gains(controller_params)
        # The cube's free joint (its qpos address), to follow it in /mujoco_state; None if the model has none.
        self.cube_qpos = None
        cube_body = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_body >= 0 and self.mjm.body_jntnum[cube_body] == 1:
            joint = self.mjm.body_jntadr[cube_body]
            if self.mjm.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                self.cube_qpos = int(self.mjm.jnt_qposadr[joint])
        self.task = None  # mpc_reaching.GraspTask for --task grasp
        if args.task == "square":
            self.cycler = mpc_reaching.WaypointCycler(mpc_reaching.square_waypoints(self.mjm), dwell=args.dwell,
                                                      timeout=args.timeout)
        else:
            if args.cube is not None:
                cube = np.asarray(args.cube)
            elif cube_body >= 0:
                cube = mpc_reaching.body_position(self.mjm, "cube")
            else:
                raise SystemExit(f"model {args.model} has no body 'cube': give its position with --cube X Y Z")
            self.task = mpc_reaching.GraspTask(cube, approach=args.approach, dwell=args.dwell, timeout=args.timeout)
            self.cycler = self.task.cycler
            self.get_logger().info(f"grasping the cube, at first at {np.round(cube, 3).tolist()}")
        config = mpc_reaching.reaching_config(Nsample=args.samples, Hsample=args.horizon, Hnode=args.nodes,
                                              seed=args.seed, sigma_scale=args.noise)
        self.get_logger().info("building the MPC; the first plan compiles the GPU kernels (tens of seconds)...")
        # Reads the MJCF's actuator -> joint map before the servos replace the motors.
        self.actuator_joints = [
            mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, int(self.mjm.actuator_trnid[a, 0]))
            for a in range(self.mjm.nu)
        ]
        self.mpc, self.pose_cost = mpc_reaching.make_reaching_mpc(self.mjm, self.gains, self.cycler.target, config)
        self.data = mujoco.MjData(self.mjm)  # for the tool's pose by forward kinematics
        self.site = self.mjm.site(mpc_reaching.TOOL_SITE).id

        self.lock = threading.Lock()
        self.state = None  # (stamp s, {joint: (q, v)})
        self.mujoco_state = None  # (stamp s, qpos, qvel): the sim's whole state, when it matches this model
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, qos)
        self.create_subscription(MujocoState, "/mujoco_state", self.on_mujoco_state, qos)
        self.command_pub = None
        if not args.no_command:
            self.command_pub = self.create_publisher(JointTrajectory, f"/{args.controller}/joint_trajectory", 10)
        self.target_pub = self.create_publisher(PoseStamped, "/mpc_reach/target_pose", 10)
        self.stats = []  # per plan: acc RMS within the sent trajectory, start jump rate, max control step
        self.last_plan_time = None  # None: the next plan starts from the measured pose
        self.setpoint = None  # mpc_reaching.QuinticSetpoint, once there is a measurement
        self.gripper = None  # mpc_reaching.GripperRamp, likewise
        # Plan once (compiles the kernels) before the loop, from the model's default state.
        t0 = time.time()
        self.mpc.plan(self.data.qpos, self.data.qvel)
        self.mpc.reset()
        self.get_logger().info(f"MPC ready ({time.time() - t0:.0f} s); waiting for /mujoco_state or /joint_states")
        self.create_timer(1.0 / args.rate, self.cycle)

    def on_joint_states(self, msg: JointState):
        values = {}
        for i, name in enumerate(msg.name):
            q = msg.position[i] if i < len(msg.position) else float("nan")
            v = msg.velocity[i] if i < len(msg.velocity) else 0.0
            values[name] = (q, v if np.isfinite(v) else 0.0)
        with self.lock:
            self.state = (time.monotonic(), values)

    def on_mujoco_state(self, msg: MujocoState):
        if len(msg.qpos) != self.mjm.nq or len(msg.qvel) != self.mjm.nv:
            self.get_logger().warning(f"/mujoco_state has nq {len(msg.qpos)}, nv {len(msg.qvel)}; the model "
                                      f"{self.mjm.nq}, {self.mjm.nv}: using /joint_states", throttle_duration_sec=10.0)
            return
        with self.lock:
            self.mujoco_state = (time.monotonic(), np.array(msg.qpos), np.array(msg.qvel))

    def cycle(self):
        with self.lock:
            snapshot, full = self.state, self.mujoco_state
        fresh = lambda s: s is not None and time.monotonic() - s[0] <= 0.5
        if fresh(full):  # the whole state: the arm's and the cube's
            stamp, qpos, qvel = full
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            if self.task is not None and self.cube_qpos is not None:
                self.task.track(qpos[self.cube_qpos:self.cube_qpos + 3], qpos[self.cube_qpos + 3:self.cube_qpos + 7])
        elif fresh(snapshot):
            stamp, values = snapshot
            # The passive gripper fingers are not required (the equality constraints keep them with joint7).
            missing = mpc_reaching.set_joint_state(self.mjm, self.data, values, required=self.actuator_joints)
            if missing:
                self.get_logger().warning(f"joint_states misses {missing}", throttle_duration_sec=5.0)
                return
        else:
            return
        mujoco.mj_kinematics(self.mjm, self.data)
        now = time.monotonic()
        if self.gripper is None:
            self.gripper = mpc_reaching.GripperRamp(float(self.data.qpos[self.mjm.joint(mpc_reaching.GRIPPER_JOINT).qposadr[0]]))
            if self.cycler.target.gripper is not None:
                self.gripper.set_target(now, self.cycler.target.gripper)
        changed = self.cycler.update(now, self.data.site_xpos[self.site], self.data.site_xmat[self.site])
        target = self.cycler.target
        if changed or self.task is not None:  # a followed cube moves the target between steps too
            self.pose_cost.set_target(target.position, target.xmat)
        if changed:
            if target.gripper is not None:
                self.gripper.set_target(now, target.gripper)
            gripper = "" if target.gripper is None else f", gripper {target.gripper:.3f}"
            self.get_logger().info(f"waypoint {self.cycler.index} at {np.round(target.position, 2).tolist()}{gripper}")
        self.publish_target()

        if self.last_plan_time is None:
            self.mpc.hold(mpc_reaching.actuator_positions(self.mjm, self.data))
        else:  # warm start: the plan moves on by the time since the last one
            self.mpc.shift(max(1, round((now - self.last_plan_time) / self.mpc.config.ctrl_dt)))
        self.last_plan_time = now
        plan = self.mpc.plan(self.data.qpos, self.data.qvel)
        # The setpoint is a quintic from where it is now (position, velocity, acceleration) to where the plan ends,
        # not the plan itself: the plan is jagged at the scale of its control nodes and differs at every replan, and
        # the setpoint must not jump. The plan is for the state measured at `stamp`; `latency` has passed since (plus
        # --extra-latency for the controller receiving the trajectory), so what is left of it is its second part.
        columns = [self.actuator_joints.index(j) for j in mpc_reaching.ARM_JOINTS]
        sent_at = time.monotonic()
        latency = sent_at - stamp + self.args.extra_latency
        remaining = plan.controls.shape[0] * plan.ctrl_dt - latency
        if self.setpoint is None:
            self.setpoint = mpc_reaching.QuinticSetpoint(mpc_reaching.joint_positions(self.mjm, self.data.qpos))
        points = []
        if remaining >= 0.1:
            end, end_velocity = mpc_reaching.plan_end(plan.controls[:, columns], plan.ctrl_dt)
            self.setpoint.retarget(sent_at, end, end_velocity, remaining)
            points = self.setpoint.points(sent_at, 0.02)
        error = np.linalg.norm(self.data.site_xpos[self.site] - self.cycler.target.position)
        if len(self.stats) >= 5:
            acc, jump, step = np.mean(self.stats, axis=0)
            self.get_logger().info(f"sent: acc {acc:.1f} rad/s^2, max speed {jump:.2f} rad/s; plan: max step {step:.3f} rad",
                                   throttle_duration_sec=4.0)
            self.stats = self.stats[-20:]
        cube = ""
        if fresh(full) and self.cube_qpos is not None:
            cube = f", cube {np.round(self.data.qpos[self.cube_qpos:self.cube_qpos + 3], 3).tolist()}"
        self.get_logger().info(f"tool error {error * 100:.1f} cm, plan cost {plan.cost:.3f}, latency {latency * 1e3:.0f} ms, "
                               f"joints {np.round(mpc_reaching.joint_positions(self.mjm, self.data.qpos), 2).tolist()}{cube}",
                               throttle_duration_sec=1.0)
        if self.command_pub is not None and points:
            with_gripper = self.cycler.target.gripper is not None
            msg = JointTrajectory(joint_names=list(mpc_reaching.ARM_JOINTS) + [mpc_reaching.GRIPPER_JOINT] * with_gripper)
            for t, positions in points:
                if with_gripper:
                    positions = positions + [self.gripper.at(sent_at + t)]
                # Positions only: without velocities the controller interpolates linearly between points (20 ms
                # apart), where a spline through finite-difference velocities can overshoot them.
                msg.points.append(JointTrajectoryPoint(positions=positions, time_from_start=_duration(t)))
            self.command_pub.publish(msg)

    def publish_target(self):
        target = self.cycler.target
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, target.xmat)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, target.position)
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = map(float, quat)
        self.target_pub.publish(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    active_model.add_model_argument(ap, MODELS)
    ap.add_argument("--task", choices=("grasp", "square"), default="grasp",
                    help="grasp: pick up the cube and put it back (default); square: cycle through a square's corners")
    ap.add_argument("--cube", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="the cube's center, m, base frame (default: where the model has its body 'cube')")
    ap.add_argument("--approach", type=float, default=mpc_reaching.GRASP_APPROACH,
                    help=f"m above the cube to come down from and lift to (default {mpc_reaching.GRASP_APPROACH})")
    ap.add_argument("--controller", default="joint_trajectory_controller", help="controller to send trajectories to")
    ap.add_argument("--rate", type=float, default=2.0,
                    help="plans per second (default 2: a plan over the 1.28 s horizon takes ~0.4 s)")
    ap.add_argument("--samples", type=int, default=1024, help="MPC samples per iteration (default 1024)")
    ap.add_argument("--horizon", type=int, default=mpc_reaching.HORIZON_STEPS,
                    help=f"MPC horizon, control steps of 40 ms (default {mpc_reaching.HORIZON_STEPS})")
    ap.add_argument("--nodes", type=int, default=mpc_reaching.HORIZON_NODES,
                    help=f"control nodes - 1 (default {mpc_reaching.HORIZON_NODES})")
    ap.add_argument("--noise", type=float, default=mpc_reaching.DEFAULT_SIGMA_SCALE,
                    help=f"MPC noise scale on the last control node, in half joint ranges (default {mpc_reaching.DEFAULT_SIGMA_SCALE})")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="seconds over which a new trajectory merges into the setpoint underway (default 0.3)")
    ap.add_argument("--extra-latency", type=float, default=0.0,
                    help="seconds, besides planning, between the measured state and the controller using the trajectory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=0.3, help="seconds at a waypoint before the next (default 0.3)")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds before giving up on a waypoint (default 10)")
    ap.add_argument("--no-command", action="store_true", help="plan and publish the target, but send nothing")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    rclpy.init(args=ros_args)
    node = MpcReach(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
