"""Does the real driver keep its rate while teleop's command stream arrives? Nothing moves.

Usage: bazel run //src/ros2_control_sim:hold_load_diagnostic[.remote] -- [--rate 200] [--duration 15] [--joints N]

Needs real_arm (or the sim) running. Like the teleop node, publishes a one-point JointTrajectory every 1/--rate s
(20 ms lookahead) for the robot's joints, but each point is the joint's *measured* position, so the arms hold where
they are. Reports how that stream and the driver behave: the publish rate achieved, the /joint_states and
controller_state intervals (a controller_manager that lags shows up as a longer or jittery /joint_states
interval), and how far the measured positions drift (they should not).
"""

import argparse
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def describe(name, times):
    dt = np.diff(np.asarray(times)) * 1e3
    if len(dt) < 2:
        print(f"{name:28s} no data")
        return
    print(
        f"{name:28s} {1e3 / dt.mean():7.1f} Hz  interval mean {dt.mean():6.2f} ms  p50 {np.percentile(dt, 50):6.2f}  "
        f"p99 {np.percentile(dt, 99):6.2f}  max {dt.max():6.2f} ms  (n={len(dt) + 1})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=200.0, help="command rate, Hz (teleop's --rate)")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--lookahead", type=float, default=0.02, help="time_from_start of each point, s")
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--no-load", action="store_true", help="only listen (the baseline)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("hold_load_diagnostic")
    js_times, st_times = [], []
    latest = {}
    first = {}

    def on_js(msg):
        js_times.append(time.monotonic())
        for n, p in zip(msg.name, msg.position):
            latest[n] = p
            first.setdefault(n, p)

    node.create_subscription(JointState, "/joint_states", on_js, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_subscription(
        JointTrajectoryControllerState,
        f"/{args.controller}/controller_state",
        lambda m: st_times.append(time.monotonic()),
        10,
    )
    pub = node.create_publisher(JointTrajectory, f"/{args.controller}/joint_trajectory", 10)

    t0 = time.monotonic()
    while not latest and time.monotonic() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not latest:
        raise SystemExit("no /joint_states: is the driver running?")
    names = [n for n in latest if n.endswith(tuple(f"joint{i}" for i in range(1, 7)))]
    print(f"holding {len(names)} joints; {'no load' if args.no_load else f'streaming at {args.rate:g} Hz'} for {args.duration:g} s")

    pub_times, late = [], []
    period = 1.0 / args.rate
    start = time.monotonic()
    next_t = start
    while time.monotonic() - start < args.duration:
        rclpy.spin_once(node, timeout_sec=max(0.0, next_t - time.monotonic()))
        now = time.monotonic()
        if now < next_t:
            continue
        late.append(now - next_t)
        next_t += period
        if now - next_t > period:
            next_t = now
        if args.no_load:
            continue
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [latest[n] for n in names]
        pt.time_from_start = Duration(sec=0, nanosec=int(args.lookahead * 1e9))
        msg.points = [pt]
        pub.publish(msg)
        pub_times.append(now)

    if pub_times:
        describe("published trajectories", pub_times)
    print(f"{'publisher lateness':28s} mean {np.mean(late) * 1e3:.2f} ms  max {np.max(late) * 1e3:.2f} ms")
    describe("/joint_states", js_times)
    describe("controller_state", st_times)
    drift = max(abs(latest[n] - first[n]) for n in names)
    print(f"largest joint drift: {drift:.4f} rad")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
