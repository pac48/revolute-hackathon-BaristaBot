"""Move a running sim (or robot) the way the teleop node streams to it, until Ctrl-C.

Usage: bazel run //src/trajectory_generator:teleop_stream -- [--joints joint1 joint2 joint3] [--wave cosine|square]
       [--amplitude 0.15] [--frequency 0.5] [--rate 200] [--lookahead 0.02] [--duration 0]
       [--controller joint_trajectory_controller]

Like teleop: every 1/--rate s (200 Hz) a one-point JointTrajectory with every controller joint, reached in
--lookahead (20 ms), on <controller>/joint_trajectory. The point follows --wave on --joints (the others hold),
from the controller's setpoint when it starts (q0): a raised cosine q0 + A (1 - cos 2 pi f t), or a square
wave between q0 and q0 + 2 A (half a period each; its steps show how the controller's fit smooths a jump in
the targets). Both stay on the positive side of the start pose and start without a jump. On Ctrl-C (or after
--duration s, if > 0) it sends the arm back to q0 in --return-time s and exits.

Watch <controller>/controller_state meanwhile: reference.positions (the targets) against output.positions
(the controller's fitted curve) against feedback.positions (the measured joints).
"""

import argparse
import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--joints", nargs="+", default=["joint1", "joint2", "joint3"], help="joints that move")
    ap.add_argument("--wave", choices=["cosine", "square"], default="cosine",
                    help="cosine: q0 + A (1 - cos 2 pi f t); square: q0, then q0 + 2 A, each for half a period")
    ap.add_argument("--amplitude", type=float, default=0.15, help="rad (the motion spans 2 A)")
    ap.add_argument("--frequency", type=float, default=0.5, help="Hz")
    ap.add_argument("--rate", type=float, default=200.0, help="Hz")
    ap.add_argument("--lookahead", type=float, default=0.02, help="s")
    ap.add_argument("--duration", type=float, default=0.0, help="s; 0: until Ctrl-C")
    ap.add_argument("--return-time", type=float, default=2.0, help="s to go back to the start pose at the end")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("teleop_stream")
    setpoint = {}
    node.create_subscription(
        JointTrajectoryControllerState, f"/{args.controller}/controller_state",
        lambda m: setpoint.update(zip(m.joint_names, m.reference.positions)), 10)
    pub = node.create_publisher(JointTrajectory, f"/{args.controller}/joint_trajectory", 10)

    print(f"waiting for /{args.controller}/controller_state and a subscriber ...", flush=True)
    while rclpy.ok() and (not setpoint or pub.get_subscription_count() == 0):
        rclpy.spin_once(node, timeout_sec=0.1)
    joints = list(setpoint)
    unknown = [j for j in args.joints if j not in setpoint]
    if unknown:
        raise SystemExit(f"not joints of {args.controller}: {unknown} (it has {joints})")
    q0 = dict(setpoint)

    def send(positions, seconds):
        msg = JointTrajectory(joint_names=joints)
        msg.points = [JointTrajectoryPoint(positions=[positions[j] for j in joints], time_from_start=_duration(seconds))]
        pub.publish(msg)

    def target(t):
        phase = (args.frequency * t) % 1.0
        if args.wave == "square":  # q0 for the first half period, q0 + 2 A for the second
            offset = 0.0 if phase < 0.5 else 2 * args.amplitude
        else:
            offset = args.amplitude * (1 - math.cos(2 * math.pi * phase))
        return {j: q0[j] + (offset if j in args.joints else 0.0) for j in joints}

    print(f"streaming a {args.wave} wave on {args.joints} at {args.rate:.0f} Hz (A {args.amplitude} rad, "
          f"{args.frequency} Hz); Ctrl-C to stop", flush=True)
    start = time.monotonic()
    next_send = start
    try:
        while rclpy.ok() and (args.duration <= 0 or time.monotonic() - start < args.duration):
            now = time.monotonic()
            if now >= next_send:
                send(target(now - start + args.lookahead), args.lookahead)
                next_send += 1.0 / args.rate
                if next_send < now:  # fell behind (e.g. suspended): don't burst to catch up
                    next_send = now + 1.0 / args.rate
            rclpy.spin_once(node, timeout_sec=max(0.0, min(0.0005, next_send - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            print(f"returning to the start pose in {args.return_time} s", flush=True)
            send(q0, args.return_time)
            rclpy.spin_once(node, timeout_sec=0.1)  # let it go out
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
