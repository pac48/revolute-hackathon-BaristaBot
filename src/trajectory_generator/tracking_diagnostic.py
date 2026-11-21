"""Record and break down how the reBot follows a Cartesian move (debugging tool).

Usage: bazel run //src/trajectory_generator:tracking_diagnostic -- [--shape square] [--speed 0.3] [--csv out.csv]

Runs one cartesian_demo shape while recording joint_trajectory_controller's
controller_state (per joint: reference = the controller's interpolated setpoint,
feedback = measured). Then, with FK, reports where the tool's deviation comes from:
  planned    the IK samples sent (vs the straight Cartesian segments)
  reference  FK(reference) vs the segments: our trajectory as the controller interpolates it
  feedback   FK(feedback) vs the segments: what the robot actually did
  tracking   FK(feedback) vs FK(reference), and per-joint feedback - reference
with the times of the worst values, so spikes can be lined up with the motion.
--csv writes every recorded state (time, reference, feedback, Cartesian errors) for plotting.
"""

import argparse
import csv
import math
import sys

import rclpy
from control_msgs.msg import JointTrajectoryControllerState

from src.robot_model.active_model import active_model
from src.trajectory_generator import kinematics_client
from src.trajectory_generator.cartesian_demo import SHAPES, START
from src.trajectory_generator.cartesian_path import Pose
from src.trajectory_generator.client import TrajectoryClient


def distance_to_polyline(p, points):
    best = math.inf
    for a, b in zip(points, points[1:]):
        ab = [y - x for x, y in zip(a, b)]
        ap = [y - x for x, y in zip(a, p)]
        n = sum(v * v for v in ab)
        s = 0.0 if n == 0 else max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / n))
        best = min(best, math.dist(p, [x + s * v for x, v in zip(a, ab)]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    active_model.add_model_argument(ap, kinematics_client.KINEMATICS)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="square")
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--max-joint-velocity", type=float, default=3.0)
    ap.add_argument("--csv", help="write every recorded state here")
    args = ap.parse_args()

    rclpy.init()
    try:
        node = rclpy.create_node("tracking_diagnostic")
        client = TrajectoryClient(node)
        client.move_to(START, max_velocity=1.5)
        client.wait_until_at(START, 0.02, 2.0)
        orientation = client.current_pose(args.model, commanded=True).quaternion
        waypoints = []
        for x, y, z, *rot in SHAPES[args.shape]:
            pose = Pose((x, y, z), orientation)
            waypoints.append(pose.rotated(*rot[0]) if rot else pose)

        states = []

        def record(msg: JointTrajectoryControllerState):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            states.append((t, list(msg.joint_names), list(msg.reference.positions), list(msg.feedback.positions),
                           list(msg.reference.velocities), list(msg.feedback.velocities)))

        node.create_subscription(JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state",
                                 record, 1000)
        start_pose = client.current_pose(args.model, commanded=True)
        states.clear()
        move, _ = client.move_cartesian(args.model, waypoints, max_linear_velocity=args.speed,
                                        max_angular_velocity=1.5, max_joint_velocity=args.max_joint_velocity)
        client.wait_until_at(dict(zip(move.ik.joint_names, move.ik.positions[-1])), 0.002, 1.0)
    finally:
        rclpy.try_shutdown()

    joints = move.ik.joint_names
    corners = [p.position for p in [start_pose, *waypoints]]
    # Keep only the recording while the trajectory ran (reference moving) plus 0.5 s after.
    t0 = states[0][0]
    rows = [(t - t0, [dict(zip(n, r))[j] for j in joints], [dict(zip(n, f))[j] for j in joints],
             [dict(zip(n, rv))[j] for j in joints] if rv else [], [dict(zip(n, fv))[j] for j in joints] if fv else [])
            for t, n, r, f, rv, fv in states]
    print(f"{args.shape} at {args.speed} m/s: {len(rows)} controller states over {rows[-1][0]:.1f} s "
          f"({len(rows) / max(rows[-1][0], 1e-9):.0f} Hz), trajectory {move.trajectory.duration:.2f} s, "
          f"{len(move.ik.positions)} IK samples", file=sys.stderr)

    planned = kinematics_client.forward_many(args.model, [dict(zip(joints, q)) for q in move.ik.positions])
    reference = kinematics_client.forward_many(args.model, [dict(zip(joints, r)) for _, r, _, _, _ in rows])
    feedback = kinematics_client.forward_many(args.model, [dict(zip(joints, f)) for _, _, f, _, _ in rows])

    def worst(values, times):
        i = max(range(len(values)), key=lambda k: values[k])
        return values[i], times[i]

    times = [t for t, *_ in rows]
    planned_dev = max(distance_to_polyline(p.position, corners) for p in planned)
    ref_dev = [distance_to_polyline(p.position, corners) for p in reference]
    fb_dev = [distance_to_polyline(p.position, corners) for p in feedback]
    track = [a.distance(b) for a, b in zip(reference, feedback)]
    print(f"  planned   max {planned_dev * 1e3:7.3f} mm from the segments (IK samples)", file=sys.stderr)
    for name, values in (("reference", ref_dev), ("feedback", fb_dev), ("tracking", track)):
        v, t = worst(values, times)
        mean = sum(values) / len(values)
        print(f"  {name:9s} max {v * 1e3:7.2f} mm at t={t:5.2f} s, mean {mean * 1e3:6.2f} mm"
              + ("  (FK(feedback) - FK(reference))" if name == "tracking" else "  (from the segments)"),
              file=sys.stderr)
    print("  per joint: max |feedback - reference| (rad) at t, max |reference velocity| (rad/s)", file=sys.stderr)
    for j, name in enumerate(joints):
        errors = [abs(f[j] - r[j]) for _, r, f, _, _ in rows]
        e, t = worst(errors, times)
        v = max((abs(rv[j]) for *_, rv, _ in rows if rv), default=float("nan"))
        print(f"    {name}: {e:.4f} at t={t:5.2f} s, ref speed {v:.2f}", file=sys.stderr)
    # Big steps in the reference between consecutive states (interpolation / goal-start glitches).
    steps = [max(abs(b[j] - a[j]) for j in range(len(joints))) / max(tb - ta, 1e-9)
             for (ta, a, *_), (tb, b, *_) in zip(rows, rows[1:])]
    s, t = worst(steps, times[1:])
    print(f"  reference max joint speed between states {s:.2f} rad/s at t={t:5.2f} s", file=sys.stderr)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"ref_{j}" for j in joints] + [f"fb_{j}" for j in joints]
                       + ["ref_dev_mm", "fb_dev_mm", "tracking_mm"])
            for (t, r, fb, _, _), rd, fd, tr in zip(rows, ref_dev, fb_dev, track):
                w.writerow([f"{t:.4f}"] + [f"{v:.5f}" for v in r + fb] + [f"{rd * 1e3:.3f}", f"{fd * 1e3:.3f}",
                                                                        f"{tr * 1e3:.3f}"])
        print(f"  wrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
