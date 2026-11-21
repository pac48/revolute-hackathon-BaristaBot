"""End-to-end on mock hardware: the real arm's controllers (arm in MIT mode, gripper by position) run.

mock_components/GenericSystem has the real arm's joint interfaces but no motors: a state follows its
position command. So this checks the controller claims and drives them (the 5 MIT interfaces of joint1-6,
position and speed limit of the gripper joint7) and follows a trajectory; not the motors or the CAN bus.
"""

import time
import unittest

import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from src.ros2_control_sim.real import RealRobot

JOINTS = [f"joint{i}" for i in range(1, 8)]
GOAL = [0.3, 0.5, 0.5, 0.2, -0.2, 0.3, 0.04]  # the gripper in m (0 closed .. 0.05 open)
ARMS = ("left_", "right_")


class RealMockTest(unittest.TestCase):
    def test_trajectory_with_the_gripper(self):
        with RealRobot("rebot_arm", gripper=True, mock=True):
            rclpy.init()
            try:
                node = rclpy.create_node("real_mock_test")
                state = {}
                node.create_subscription(JointState, "/joint_states", lambda m: state.update(zip(m.name, m.position)), 10)
                publisher = node.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
                end = time.monotonic() + 20
                while not all(j in state for j in JOINTS) and time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.1)
                self.assertTrue(all(j in state for j in JOINTS), f"states of {sorted(state)}")
                while publisher.get_subscription_count() == 0:
                    rclpy.spin_once(node, timeout_sec=0.1)

                point = JointTrajectoryPoint()
                point.positions = GOAL
                point.velocities = [0.0] * 7
                point.time_from_start = Duration(sec=2)
                msg = JointTrajectory(joint_names=JOINTS, points=[point])
                publisher.publish(msg)
                end = time.monotonic() + 6
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.05)
                for joint, goal in zip(JOINTS, GOAL):
                    self.assertAlmostEqual(state[joint], goal, delta=0.01, msg=joint)
            finally:
                node.destroy_node()
                rclpy.shutdown()


    def test_dual_arm_trajectory_with_the_grippers(self):
        """Both arms on mock hardware, one controller: one trajectory moves each arm and gripper to its own goal."""
        joints = [arm + j for arm in ARMS for j in JOINTS]
        # Different goals, so an arm following the other's would show: the right arm's mirrored, except
        # joints 2 and 3, whose ranges ([-0.2, 3.14]) don't reach the mirror (the controller clamps into them).
        goals = {"left_": GOAL, "right_": [-GOAL[0], 0.8, 0.3, -GOAL[3], -GOAL[4], -GOAL[5], 0.01]}
        goal = [q for arm in ARMS for q in goals[arm]]
        with RealRobot("rebot_dual_arm_gemini_305", gripper=True, mock=True):
            rclpy.init()
            try:
                node = rclpy.create_node("real_dual_mock_test")
                state = {}
                node.create_subscription(JointState, "/joint_states", lambda m: state.update(zip(m.name, m.position)), 10)
                publisher = node.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
                end = time.monotonic() + 20
                while not all(j in state for j in joints) and time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.1)
                self.assertTrue(all(j in state for j in joints), f"states of {sorted(state)}")
                while publisher.get_subscription_count() == 0:
                    rclpy.spin_once(node, timeout_sec=0.1)

                point = JointTrajectoryPoint()
                point.positions = goal
                point.velocities = [0.0] * len(goal)
                point.time_from_start = Duration(sec=2)
                publisher.publish(JointTrajectory(joint_names=joints, points=[point]))
                end = time.monotonic() + 6
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.05)
                for joint, q in zip(joints, goal):
                    self.assertAlmostEqual(state[joint], q, delta=0.01, msg=joint)
            finally:
                node.destroy_node()
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
