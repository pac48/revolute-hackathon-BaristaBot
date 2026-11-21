"""How still the simulated arm is at a stationary target: records /joint_states and reports,
per joint, the variation of the position (standard deviation, peak to peak), the velocity (RMS),
the effort (standard deviation) and the dominant frequency of the position's oscillation.

  bazel run //src/ros2_control_sim:stationary_diagnostic [-- --model rebot_arm_gemini_305]
      starts the model's simulation headless, lets it settle, and records it
  bazel run //src/ros2_control_sim:stationary_diagnostic -- --attach
      records a simulation that is already running (it doesn't move it); use the same
      ROS_DOMAIN_ID as that simulation

The arm holds the pose it starts in (the controller holds its position from activation).
"""

import argparse
import time

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import numpy as np
import rclpy
from sensor_msgs.msg import JointState

from src.robot_model.active_model import active_model
from src.ros2_control_sim import models
from src.ros2_control_sim.sim import Simulation


def move_to(pose, seconds):
    """Send one trajectory to `pose` ({joint: position}) and wait for it to end."""
    rclpy.init()
    try:
        node = rclpy.create_node("stationary_diagnostic_move")
        pub = node.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
        while pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.1)
        point = JointTrajectoryPoint(positions=list(pose.values()), velocities=[0.0] * len(pose))
        point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
        pub.publish(JointTrajectory(joint_names=list(pose), points=[point]))
        end = time.monotonic() + seconds + 1.0
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        rclpy.shutdown()


def record(seconds):
    rclpy.init()
    try:
        node = rclpy.create_node("stationary_diagnostic")
        rows = []  # (time, {joint: (position, velocity, effort)})
        node.create_subscription(
            JointState,
            "/joint_states",
            lambda m: rows.append(
                (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, {n: (p, v, e) for n, p, v, e in zip(m.name, m.position, m.velocity, m.effort)})
            ),
            100,
        )
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
        return rows
    finally:
        rclpy.shutdown()


def report(rows):
    if len(rows) < 100:
        raise SystemExit(f"only {len(rows)} /joint_states received; is the simulation running (same ROS_DOMAIN_ID)?")
    t = np.array([r[0] for r in rows])
    rate = (len(t) - 1) / (t[-1] - t[0])
    print(f"{len(rows)} samples over {t[-1] - t[0]:.1f} s of simulation time ({rate:.0f} Hz)")
    print(f"{'joint':12s} {'mean':>9s} {'pos std':>10s} {'pos p-p':>10s} {'vel rms':>10s} {'effort std':>11s} {'dominant':>9s}")
    for joint in rows[0][1]:
        data = np.array([r[1][joint] for r in rows])
        position, velocity, effort = data[:, 0], data[:, 1], data[:, 2]
        centered = position - position.mean()
        spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
        frequency = np.fft.rfftfreq(len(centered), 1.0 / rate)
        dominant = frequency[1 + int(np.argmax(spectrum[1:]))] if centered.std() > 0 else 0.0
        print(
            f"{joint:12s} {position.mean():9.5f} {position.std():10.2e} {np.ptp(position):10.2e} "
            f"{np.sqrt(np.mean(velocity**2)):10.2e} {effort.std():11.2e} {dominant:7.1f}Hz"
        )


def main():
    ap = argparse.ArgumentParser()
    active_model.add_model_argument(ap, models.MODELS)
    ap.add_argument("--attach", action="store_true", help="record a running simulation instead of starting one")
    ap.add_argument("--settle", type=float, default=3.0, help="s to wait before recording a simulation it starts")
    ap.add_argument("--pose", nargs="*", default=[], metavar="JOINT=VALUE", help="move here first (2 s), then hold")
    ap.add_argument("--seconds", type=float, default=10.0, help="s to record")
    args = ap.parse_args()
    if args.attach:
        report(record(args.seconds))
        return
    with Simulation(args.model, headless=True):
        if args.pose:
            move_to({k: float(v) for k, v in (p.split("=") for p in args.pose)}, 2.0)
        time.sleep(args.settle)
        report(record(args.seconds))


if __name__ == "__main__":
    main()
