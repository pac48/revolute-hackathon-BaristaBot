"""Demo of Cartesian moves on the reBot: a square, a wide sweep, a reach and wrist twists.

Usage: bazel run //src/trajectory_generator:cartesian_demo -- [--speed 0.3] [--shape square|sweep|reach|twist ...]

Moves to a start pose with the wrist bent (joint2=1.2, joint3=1.4, joint4=-0.7;
tool at (0.40, 0, 0.34) m), then runs each shape as one move_cartesian (straight
lines between the corners, dense IK, stopping at each corner), with the start
pose's tool orientation turned about the tool's own axes where given. Every shape
starts and ends at that center pose.

The waypoints were checked with dense IK and MuJoCo collision checking: they're
reachable within the joint limits and free of contacts. With the wrist straight
(joint4=0), moving sideways while keeping the tool's orientation swings joint5
into contact between link3 and link5 (their convex collision hulls); see
kinematics_cli's check_collisions.
"""

import argparse
import math
import sys

import rclpy

from src.robot_model.active_model import active_model
from src.trajectory_generator.cartesian_path import Pose
from src.trajectory_generator.client import TrajectoryClient, TrajectoryError
from src.trajectory_generator.kinematics_client import KINEMATICS, KinematicsError

START = {"joint1": 0.0, "joint2": 1.2, "joint3": 1.4, "joint4": -0.7, "joint5": 0.0, "joint6": 0.0}
CENTER = (0.40, 0.0, 0.34)  # the tool at START

# (x, y, z[, (roll, pitch, yaw) about the tool's axes]) in the base frame.
SHAPES = {
    # 30 x 18 cm in the vertical plane in front of the robot.
    "square": [(0.40, -0.15, 0.24), (0.40, 0.15, 0.24), (0.40, 0.15, 0.42), (0.40, -0.15, 0.42),
               (0.40, -0.15, 0.24), CENTER],
    # 60 cm left to right.
    "sweep": [(0.40, -0.30, 0.32), (0.40, 0.30, 0.32), CENTER],
    # Out and down, then in and up (38 cm along the line).
    "reach": [(0.52, 0.0, 0.20), (0.30, 0.0, 0.45), CENTER],
    # In place: yaw +-0.6 rad, then roll +-0.5 rad about the tool's axes.
    "twist": [(*CENTER, (0, 0, 0.6)), (*CENTER, (0, 0, -0.6)), CENTER,
              (*CENTER, (0.5, 0, 0)), (*CENTER, (-0.5, 0, 0)), CENTER],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    active_model.add_model_argument(ap, KINEMATICS)
    ap.add_argument("--shape", action="append", choices=sorted(SHAPES), help="default: all, in order")
    ap.add_argument("--speed", type=float, default=0.3, help="max tool speed, m/s (default 0.3)")
    ap.add_argument("--angular-speed", type=float, default=1.5, help="rad/s (default 1.5)")
    ap.add_argument("--max-joint-velocity", type=float, default=3.0, help="rad/s (default 3)")
    args = ap.parse_args()

    rclpy.init()
    try:
        client = TrajectoryClient(rclpy.create_node("cartesian_demo"))
        client.move_to(START, max_velocity=1.5)
        client.wait_until_at(START, 0.02, 2.0)
        orientation = client.current_pose(args.model, commanded=True).quaternion
        for name in args.shape or list(SHAPES):
            waypoints = []
            for x, y, z, *rot in SHAPES[name]:
                pose = Pose((x, y, z), orientation)
                waypoints.append(pose.rotated(*rot[0]) if rot else pose)
            move, _ = client.move_cartesian(args.model, waypoints, max_linear_velocity=args.speed,
                                            max_angular_velocity=args.angular_speed,
                                            max_joint_velocity=args.max_joint_velocity)
            client.wait_until_at(dict(zip(move.ik.joint_names, move.ik.positions[-1])), 0.005, 1.0)
            reached = client.current_pose(args.model)
            print(f"{name}: {len(waypoints)} waypoints, {len(move.ik.positions)} IK samples in "
                  f"{move.ik.time_ms:.0f} ms, {move.trajectory.duration:.1f} s"
                  + (f" (slowed {move.slowdown:.2f}x for joint speed)" if move.slowdown > 1.001 else "")
                  + f"; ended {reached.distance(waypoints[-1]) * 1e3:.1f} mm, "
                  f"{math.degrees(reached.angle(waypoints[-1])):.2f} deg from the goal", file=sys.stderr, flush=True)
        return 0
    except (TrajectoryError, KinematicsError) as e:
        print(f"cartesian_demo: {e}", file=sys.stderr)
        return 1
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
