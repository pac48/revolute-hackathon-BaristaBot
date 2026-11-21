"""Move a robot's tool in straight lines (Cartesian space) and execute it.

Usage: bazel run //src/trajectory_generator:move_cartesian -- [--model rebot_arm]
           [--by DX,DY,DZ[,ROLL,PITCH,YAW]] [--to X,Y,Z[,ROLL,PITCH,YAW]] ... [--speed 0.1]

Waypoints, in order, starting from the current pose (tool = the model's tip link,
e.g. the reBot's gripper_end, in its base frame; meters and radians):
  --by DX,DY,DZ[,R,P,Y]  move by DX,DY,DZ in the base frame and turn by R,P,Y about
                         the tool's own axes
  --to X,Y,Z[,R,P,Y]     go to X,Y,Z; with R,P,Y also to that orientation (base frame,
                         URDF rpy), otherwise keep the current one
  --print                only print the current pose
The path is sampled densely and solved with IK (see TrajectoryClient.move_cartesian).
"""

import argparse
import math
import sys

import rclpy

from src.robot_model.active_model import active_model
from src.trajectory_generator.cartesian_path import Pose
from src.trajectory_generator.client import TrajectoryClient, TrajectoryError
from src.trajectory_generator.kinematics_client import KINEMATICS, KinematicsError
from src.trajectory_generator.cli_args import attach_values, numbers


def describe(pose: Pose) -> str:
    x, y, z = pose.position
    r, p, w = pose.rpy
    return f"xyz=({x:.4f}, {y:.4f}, {z:.4f}) rpy=({r:.3f}, {p:.3f}, {w:.3f})"


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    active_model.add_model_argument(ap, KINEMATICS)
    ap.add_argument("--by", dest="waypoints", action="append", default=[],
                    type=lambda t: ("by", numbers(t, (3, 6))), help="relative move (repeatable)")
    ap.add_argument("--to", dest="waypoints", action="append",
                    type=lambda t: ("to", numbers(t, (3, 6))), help="absolute move (repeatable)")
    ap.add_argument("--print", action="store_true", help="print the current pose and exit")
    ap.add_argument("--speed", type=float, default=0.1, help="max tool speed, m/s (default 0.1)")
    ap.add_argument("--angular-speed", type=float, default=0.5, help="max tool turn rate, rad/s (default 0.5)")
    ap.add_argument("--max-joint-velocity", type=float, default=1.0, help="rad/s (default 1)")
    ap.add_argument("--dt", type=float, default=0.01, help="IK sample period, s (default 0.01)")
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--tolerance", type=float, default=0.005, help="allowed final position error, m (default 0.005)")
    args = ap.parse_args(attach_values(sys.argv[1:] if argv is None else argv, ("--by", "--to")))
    if not args.print and not args.waypoints:
        ap.error("give --by/--to waypoints or --print")

    rclpy.init()
    try:
        client = TrajectoryClient(rclpy.create_node("move_cartesian"), controller=args.controller,
                                  timeout=args.timeout)
        pose = client.current_pose(args.model)
        print(f"current: {describe(pose)}", file=sys.stderr)
        if args.print:
            return 0
        waypoints = []
        for kind, v in args.waypoints:
            if kind == "by":
                pose = pose.translated(*v[:3])
                if len(v) == 6:
                    pose = pose.rotated(*v[3:])
            else:
                pose = Pose.from_xyz_rpy(*v) if len(v) == 6 else Pose(v[:3], pose.quaternion)
            waypoints.append(pose)
        move, _ = client.move_cartesian(args.model, waypoints, max_linear_velocity=args.speed,
                                        max_angular_velocity=args.angular_speed,
                                        max_joint_velocity=args.max_joint_velocity, dt=args.dt)
        print(f"move_cartesian: {len(move.ik.positions)} IK samples in {move.ik.time_ms:.1f} ms "
              f"(max error {move.ik.max_position_error * 1e3:.3f} mm, "
              f"{math.degrees(move.ik.max_orientation_error):.3f} deg), {move.trajectory.duration:.2f} s"
              + (f" (slowed {move.slowdown:.2f}x for joint speed)" if move.slowdown > 1 else ""), file=sys.stderr)
        joints = move.ik.joint_names
        client.wait_until_at(dict(zip(joints, move.ik.positions[-1])), 0.002, 2.0)
        reached = client.current_pose(args.model)
        error = reached.distance(waypoints[-1])
        print(f"reached: {describe(reached)}; {error * 1e3:.1f} mm, "
              f"{math.degrees(reached.angle(waypoints[-1])):.2f} deg from the goal", file=sys.stderr)
        if error > args.tolerance:
            print(f"move_cartesian: {error * 1e3:.1f} mm from the goal (tolerance {args.tolerance * 1e3:.1f} mm)",
                  file=sys.stderr)
            return 1
        return 0
    except (TrajectoryError, KinematicsError) as e:
        print(f"move_cartesian: {e}", file=sys.stderr)
        return 1
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
