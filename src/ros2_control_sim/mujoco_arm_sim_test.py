"""End-to-end check: the simulated robot follows a point-to-point trajectory.

Starts the simulation of $SIM_MODEL (headless unless $SIM_VIEWER=1), sends a
quintic point-to-point move to joint_trajectory_controller with the
trajectory_generator client, and checks the controller reports success and
/joint_states settles at the goal. Takes a few seconds; every wait is bounded so
a broken setup fails fast instead of hanging.
"""

import os
import time
import unittest

import rclpy

from src.ros2_control_sim import controller_config, models
from src.ros2_control_sim.sim import Simulation
from src.trajectory_generator.client import TrajectoryClient

# Per model: joint -> goal position (rad or m).
GOALS = {
    "two_link_arm": {"joint1": 0.5, "joint2": -0.3},
    "rebot_arm": {
        "joint1": 0.3,
        "joint2": 0.5,
        "joint3": 0.5,
        "joint4": 0.2,
        "joint5": -0.2,
        "joint6": 0.3,
        "joint7": 0.02,  # gripper opening (m)
    },
}
GOALS["rebot_arm_gemini_305"] = GOALS["rebot_arm"]  # the same arm, with a camera
# Two of them, side by side: both arms to the same pose (parallel, 25 cm apart).
GOALS["rebot_dual_arm_gemini_305"] = {f"{arm}{joint}": q for arm in ("left_", "right_") for joint, q in GOALS["rebot_arm"].items()}
DURATION = 1.0  # s
TOLERANCE = {"joint7": 0.003, "left_joint7": 0.003, "right_joint7": 0.003}  # prismatic gripper; everything else uses DEFAULT_TOLERANCE
DEFAULT_TOLERANCE = 0.02
SETTLE_TIMEOUT = 3.0  # s after the trajectory ends


class MujocoArmSimTest(unittest.TestCase):
    def test_arm_follows_point_to_point_trajectory(self):
        model = models.get(os.environ["SIM_MODEL"])
        goal = GOALS[model.name]
        config = model.controller_config()
        # One trajectory controller per arm (the dual arm has two); each is sent its own joints' goal.
        controllers = {
            name: config[name]["ros__parameters"]["joints"]
            for name in controller_config.controller_names(config)
            if name != "joint_state_broadcaster"
        }
        commanded = [j for joints in controllers.values() for j in joints]
        self.assertEqual(sorted(commanded), sorted(goal), "GOALS must cover the commanded joints")

        with Simulation(model, headless=os.environ.get("SIM_VIEWER") != "1"):
            rclpy.init()
            try:
                node = rclpy.create_node("mujoco_arm_sim_test")
                for name, joints in controllers.items():
                    client = TrajectoryClient(node, controller=name)
                    part = {j: goal[j] for j in joints}
                    start = time.monotonic()
                    client.move_to(part, duration=DURATION)
                    print(f"{model.name} {name}: trajectory done after {time.monotonic() - start:.2f}s", flush=True)

                    # The controller reports success at the end of the trajectory;
                    # give the joints a moment to settle on the goal.
                    tolerance = {j: TOLERANCE.get(j, DEFAULT_TOLERANCE) for j in part}
                    positions = client.wait_until_at(part, tolerance, SETTLE_TIMEOUT)
                    print(f"{model.name} {name}: {positions}", flush=True)
                    for joint, position in positions.items():
                        self.assertAlmostEqual(position, goal[joint], delta=tolerance[joint], msg=joint)
            finally:
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
