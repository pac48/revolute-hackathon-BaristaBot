"""Runs a LeRobot policy on the arm (simulated or real) through the trajectory controller, until Ctrl-C.

Usage: bazel run //src/lerobot_policy:run_policy -- CHECKPOINT [--rate 20] [--device cuda] [--task "..."]
           [--max-joint-step 0.1] [--lookahead S] [--max-age 0.25] [--duration 0] [--model rebot_arm_gemini_305]
           [--ros-args -p use_sim_time:=true]

CHECKPOINT: a pretrained_model directory (lerobot-train's outputs/train/<run>/checkpoints/<step>/pretrained_model).

Every 1/--rate s (the datasets' fps, 20): the latest /joint_states and images of the policy's cameras
(lerobot_export.conventions.CAMERAS) -> PolicyRunner.act -> a one-point JointTrajectory of joint1..7, reached in
--lookahead s (default 1/--rate), on joint_trajectory_controller/joint_trajectory: the controller the datasets
were recorded through (their action is its reference), streamed to as teleop does. It holds (sends nothing:
the controller keeps its last target) while an input is missing or older than --max-age s, and when act refuses
an action (IK, a joint jumping more than --max-joint-step rad); it says why, at most once a second.

Time is the node's clock: with use_sim_time, the rate and the ages are the simulation's.
"""

import argparse
import sys

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from src.lerobot_export.conventions import CAMERAS, CONTROLLER, JOINT_STATES, JOINTS, decode_color
from src.lerobot_policy.policy_runner import DEFAULT_TASK, PolicyError, PolicyRunner


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


class PolicyNode:
    def __init__(self, node, runner: PolicyRunner, args):
        self.node, self.runner, self.args = node, runner, args
        self.log = node.get_logger()
        self.joints = None  # (received at, positions in JOINTS order)
        self.images = {}  # camera -> (received at, CompressedImage data)
        node.create_subscription(JointState, JOINT_STATES, self._on_joint_states, 10)
        for camera in runner.cameras:
            node.create_subscription(CompressedImage, CAMERAS[camera][0],
                                     lambda msg, camera=camera: self._on_image(camera, msg), qos_profile_sensor_data)
        self.publisher = node.create_publisher(JointTrajectory, f"/{CONTROLLER}/joint_trajectory", 10)
        self.lookahead = args.lookahead if args.lookahead is not None else 1.0 / args.rate
        self.steps = 0
        self.started = None
        node.create_timer(1.0 / args.rate, self._step)

    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg: JointState):
        index = {n: i for i, n in enumerate(msg.name)}
        if all(j in index for j in JOINTS):
            self.joints = (self._now(), np.array([msg.position[index[j]] for j in JOINTS]))

    def _on_image(self, camera: str, msg: CompressedImage):
        self.images[camera] = (self._now(), bytes(msg.data))

    def _hold(self, why: str):
        self.log.warn(f"holding: {why}", throttle_duration_sec=1.0)

    def _step(self):
        now = self._now()
        if self.started is None:
            self.started = now
        if self.args.duration > 0 and now - self.started > self.args.duration:
            raise SystemExit(0)
        inputs = {"joint states": self.joints} | {f"camera {c}": self.images.get(c) for c in self.runner.cameras}
        for what, latest in inputs.items():
            if latest is None:
                return self._hold(f"no {what} yet")
            if now - latest[0] > self.args.max_age:
                return self._hold(f"{what} {now - latest[0]:.2f} s old (max {self.args.max_age})")
        joints = self.joints[1]
        try:
            images = {c: decode_color(self.images[c][1]) for c in self.runner.cameras}
            targets = self.runner.act(joints, images)
        except (PolicyError, ValueError) as e:
            return self._hold(str(e))
        self.publisher.publish(JointTrajectory(joint_names=JOINTS, points=[
            JointTrajectoryPoint(positions=[float(v) for v in targets], time_from_start=_duration(self.lookahead))]))
        self.steps += 1
        if self.steps == 1:
            self.log.info(f"commanding (first action after {now - self.started:.1f} s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the policy's pretrained_model directory")
    ap.add_argument("--rate", type=float, default=20.0, help="actions per second: the datasets' fps (default 20)")
    ap.add_argument("--device", default="cuda", help="torch device (default cuda; cpu is ~35x slower for ACT on 3 cameras)")
    ap.add_argument("--task", default=DEFAULT_TASK, help=f"the task, for language-conditioned policies (default {DEFAULT_TASK!r})")
    ap.add_argument("--model", default="rebot_arm_gemini_305", help="the robot model (its kinematics, for ee-space policies)")
    ap.add_argument("--max-joint-step", type=float, default=0.1,
                    help="rad an arm joint may move in one action; more is refused (default 0.1: 2 rad/s at 20 Hz)")
    ap.add_argument("--lookahead", type=float, help="s to reach each target (default 1/--rate)")
    ap.add_argument("--max-age", type=float, default=0.25, help="s: older joint states or images hold the arm (default 0.25)")
    ap.add_argument("--duration", type=float, default=0.0, help="s to run, then exit (default 0: until Ctrl-C)")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])

    print(f"loading {args.checkpoint}", flush=True)
    with PolicyRunner(args.checkpoint, device=args.device, task=args.task, model=args.model,
                      max_joint_step=args.max_joint_step) as runner:
        print(f"{runner.config.type} policy, {runner.space} space, cameras {', '.join(runner.cameras) or '(none)'}", flush=True)
        rclpy.init(args=sys.argv)
        node = rclpy.create_node("lerobot_policy")
        PolicyNode(node, runner, args)
        runner.reset()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
