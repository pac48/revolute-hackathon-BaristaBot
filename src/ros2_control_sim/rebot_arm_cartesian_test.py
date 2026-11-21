"""End-to-end: the simulated reBot moves its gripper along straight lines.

Starts the rebot_arm simulation headless, moves to cartesian_demo's start pose,
then runs TrajectoryClient.move_cartesian through its "reach" waypoints (out and
down, in and up, back; checked free of collisions). While it moves, /joint_states
is recorded; FK of those states gives the gripper's actual path, which must stay
close to the straight segments and end at the goal. The remaining deviation is the
simulated joints' tracking (position PIDs, gravity), not the commanded path.
"""

import math
import unittest

import rclpy
from sensor_msgs.msg import JointState

from src.ros2_control_sim.sim import Simulation
from src.trajectory_generator import kinematics_client
from src.trajectory_generator.cartesian_demo import SHAPES, START
from src.trajectory_generator.cartesian_path import Pose
from src.trajectory_generator.client import TrajectoryClient

# m, measured tool path vs the straight segments. Mostly static sag under gravity
# (the position PIDs in rebot_arm/pids.yaml leave ~0.02-0.03 rad on joint2),
# measured ~11.5 mm; tighter gains would lower it.
LINE_TOLERANCE = 0.015
GOAL_TOLERANCE = 0.004  # m


def distance_to_segment(p, a, b):
    ab = [y - x for x, y in zip(a, b)]
    ap = [y - x for x, y in zip(a, p)]
    length2 = sum(v * v for v in ab)
    s = 0.0 if length2 == 0 else max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / length2))
    return math.dist(p, [x + s * v for x, v in zip(a, ab)])


class RebotArmCartesianTest(unittest.TestCase):
    def test_follows_straight_lines(self):
        with Simulation("rebot_arm", headless=True):
            rclpy.init()
            try:
                node = rclpy.create_node("rebot_arm_cartesian_test")
                client = TrajectoryClient(node)
                client.move_to(START, duration=2.0)
                client.wait_until_at(START, 0.005, 3.0)

                start = client.current_pose("rebot_arm", commanded=True)
                waypoints = [Pose(p, start.quaternion) for p in SHAPES["reach"]]
                recorded = []
                joints = [f"joint{i}" for i in range(1, 7)]

                def record(msg: JointState):
                    positions = dict(zip(msg.name, msg.position))
                    if all(j in positions for j in joints):
                        recorded.append({j: positions[j] for j in joints})

                node.create_subscription(JointState, "/joint_states", record, 100)
                move, _ = client.move_cartesian("rebot_arm", waypoints, max_linear_velocity=0.1)
                client.wait_until_at(dict(zip(move.ik.joint_names, move.ik.positions[-1])), 0.002, 2.0)
                print(f"{len(move.ik.positions)} IK samples in {move.ik.time_ms:.1f} ms, "
                      f"trajectory {move.trajectory.duration:.2f} s, slowdown {move.slowdown:.2f}", flush=True)

                goal = waypoints[-1]
                reached = client.current_pose("rebot_arm")
                print(f"goal error {reached.distance(goal) * 1e3:.2f} mm, "
                      f"{math.degrees(reached.angle(goal)):.2f} deg", flush=True)
                self.assertLess(reached.distance(goal), GOAL_TOLERANCE)
                self.assertLess(reached.angle(goal), 0.02)

                # The tool's actual path (every 10th joint state) vs the segments.
                corners = [start.position] + [w.position for w in waypoints]
                path = kinematics_client.forward_many("rebot_arm", recorded[::10])
                deviation = max(min(distance_to_segment(p.position, a, b) for a, b in zip(corners, corners[1:]))
                                for p in path)
                print(f"max deviation from the straight lines over {len(path)} states: "
                      f"{deviation * 1e3:.2f} mm", flush=True)
                self.assertGreater(len(path), 20)
                self.assertLess(deviation, LINE_TOLERANCE)
            finally:
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
