"""Plan a collision-free motion with a robot model's planner and execute it.

Usage: bazel run //src/trajectory_generator:plan_to -- [--model rebot_arm] joint1=0.5 joint2=1.0
           [--box CX,CY,CZ,HX,HY,HZ[,ROLL,PITCH,YAW]] [--sphere CX,CY,CZ,R] [--max-velocity 1]

Starts from the current /joint_states; planning joints not given keep their
positions. Obstacles (virtual, in the robot's base frame) are only for the
planner: --box takes a center and half-extents, --sphere a center and radius.
"""

import argparse
import sys

import rclpy

from src.robot_model.active_model import active_model
from src.trajectory_generator.cli_args import attach_values, numbers, parse_goal
from src.trajectory_generator.client import TrajectoryClient, TrajectoryError
from src.trajectory_generator.planner_client import PLANNERS, PlanningError


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("goal", nargs="+", metavar="JOINT=POSITION")
    active_model.add_model_argument(ap, PLANNERS)
    ap.add_argument("--box", action="append", default=[], type=lambda t: numbers(t, (6, 9)))
    ap.add_argument("--sphere", action="append", default=[], type=lambda t: numbers(t, (4,)))
    ap.add_argument("--max-velocity", type=float, default=1.0, help="rad/s or m/s per joint (default 1)")
    ap.add_argument("--max-acceleration", type=float, help="rad/s^2 or m/s^2 per joint (default: none)")
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument(
        "--tolerance", type=float, default=0.01, help="allowed final error per joint, rad or m (default 0.01)"
    )
    ap.add_argument("--settle-timeout", type=float, default=2.0, help="seconds to wait for the joints to settle")
    args = ap.parse_args(attach_values(sys.argv[1:] if argv is None else argv, ("--box", "--sphere")))
    goal = parse_goal(args.goal)
    obstacles = {
        "boxes": [{"center": b[:3], "half_extents": b[3:6], "rpy": b[6:9] or [0, 0, 0]} for b in args.box],
        "spheres": [{"center": s[:3], "radius": s[3]} for s in args.sphere],
    }

    rclpy.init()
    try:
        node = rclpy.create_node("plan_to")
        client = TrajectoryClient(node, controller=args.controller, timeout=args.timeout)
        path, _ = client.plan_to(args.model, goal, obstacles, args.max_velocity, args.max_acceleration)
        print(f"plan_to: {len(path.waypoints)} waypoints, planned in {path.planning_time_ms:.2f} ms "
              f"(+{path.simplify_time_ms:.2f} ms simplify)", file=sys.stderr)
        measured = client.wait_until_at(goal, args.tolerance, args.settle_timeout)
        errors = {j: measured[j] - goal[j] for j in goal}
        worst = max(errors, key=lambda j: abs(errors[j]))
        print("measured: " + ", ".join(f"{j}={measured[j]:.4f}" for j in goal), file=sys.stderr)
        if abs(errors[worst]) > args.tolerance:
            print(f"plan_to: {worst} is {errors[worst]:+.4f} from the goal (tolerance {args.tolerance})", file=sys.stderr)
            return 1
        print(f"reached the goal (max error {abs(errors[worst]):.4f} at {worst})", file=sys.stderr)
        return 0
    except (TrajectoryError, PlanningError) as e:
        print(f"plan_to: {e}", file=sys.stderr)
        return 1
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
