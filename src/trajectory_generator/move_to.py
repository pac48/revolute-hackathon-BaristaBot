"""Move joints to a goal with a point-to-point trajectory (joint_trajectory_controller).

Usage: bazel run //src/trajectory_generator:move_to -- joint1=0.5 joint2=-0.3 [--duration 2]

Joints not listed keep their position only if the controller allows partial
goals; by default list every joint the controller commands.
"""

import argparse
import sys

import rclpy

from src.trajectory_generator.cli_args import parse_goal
from src.trajectory_generator.client import TrajectoryClient, TrajectoryError


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("goal", nargs="+", metavar="JOINT=POSITION")
    ap.add_argument("--duration", type=float, help="seconds; default: shortest within the limits")
    ap.add_argument("--max-velocity", type=float, default=1.0, help="rad/s or m/s per joint (default 1)")
    ap.add_argument("--max-acceleration", type=float, help="rad/s^2 or m/s^2 per joint (default: none)")
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument(
        "--tolerance", type=float, default=0.01, help="allowed final error per joint, rad or m (default 0.01)"
    )
    ap.add_argument("--settle-timeout", type=float, default=2.0, help="seconds to wait for the joints to settle")
    args = ap.parse_args()
    goal = parse_goal(args.goal)

    rclpy.init()
    try:
        node = rclpy.create_node("move_to")
        client = TrajectoryClient(node, controller=args.controller, timeout=args.timeout)
        client.move_to(
            goal, duration=args.duration, max_velocity=args.max_velocity, max_acceleration=args.max_acceleration
        )
        # The controller reports success when the trajectory ends; check where
        # the joints actually are.
        measured = client.wait_until_at(goal, args.tolerance, args.settle_timeout)
        errors = {j: measured[j] - goal[j] for j in goal}
        worst = max(errors, key=lambda j: abs(errors[j]))
        print("measured: " + ", ".join(f"{j}={measured[j]:.4f}" for j in goal), file=sys.stderr)
        if abs(errors[worst]) > args.tolerance:
            print(f"move_to: {worst} is {errors[worst]:+.4f} from the goal (tolerance {args.tolerance})", file=sys.stderr)
            return 1
        print(f"reached the goal (max error {abs(errors[worst]):.4f} at {worst})", file=sys.stderr)
        return 0
    except TrajectoryError as e:
        print(f"move_to: {e}", file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
