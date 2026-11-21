"""End-to-end: the simulated reBot follows a motion-planned path around a virtual obstacle.

Starts the rebot_arm simulation headless, moves to a raised pose, then plans
(trajectory_generator.planner_client, VAMP) around a pole that blocks the
direct motion and executes the path with joint_trajectory_controller.
"""

import unittest

import rclpy

from src.ros2_control_sim.sim import Simulation
from src.trajectory_generator.client import TrajectoryClient

LEFT = {"joint1": -1.2, "joint2": 1.2, "joint3": 1.0, "joint4": 0.3, "joint5": -0.3, "joint6": 0.5}
RIGHT = {"joint1": 1.2, "joint2": 1.2, "joint3": 1.0, "joint4": 0.3, "joint5": -0.3, "joint6": -0.5}
# Virtual pole (base frame) that blocks the direct joint-space motion LEFT -> RIGHT.
POLE = {"boxes": [{"center": [-0.15, 0.0, 0.4], "half_extents": [0.03, 0.03, 0.4]}]}
TOLERANCE = 0.02


class RebotArmPlanTest(unittest.TestCase):
    def test_follows_planned_path_around_obstacle(self):
        with Simulation("rebot_arm", headless=True):
            rclpy.init()
            try:
                client = TrajectoryClient(rclpy.create_node("rebot_arm_plan_test"))
                client.move_to(LEFT, duration=2.0)
                path, _ = client.plan_to("rebot_arm", RIGHT, POLE, max_velocity=1.5)
                print(f"{len(path.waypoints)} waypoints, planned in {path.planning_time_ms:.2f} ms", flush=True)
                self.assertGreater(len(path.waypoints), 2, "the pole should force a detour")
                reached = client.wait_until_at(RIGHT, TOLERANCE, 3.0)
                for joint, value in RIGHT.items():
                    self.assertAlmostEqual(reached[joint], value, delta=TOLERANCE, msg=joint)
            finally:
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
