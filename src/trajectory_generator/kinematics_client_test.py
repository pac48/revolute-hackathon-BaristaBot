"""kinematics_client against the real rebot_arm kinematics CLI."""

import unittest

from src.trajectory_generator import cartesian_path, kinematics_client
from src.trajectory_generator.kinematics_client import KinematicsError

JOINTS = [f"joint{i}" for i in range(1, 7)]
START = {"joint2": 1.2, "joint3": 1.0}


class KinematicsClientTest(unittest.TestCase):
    def test_chain(self):
        chain = kinematics_client.chain("rebot_arm")
        self.assertEqual((chain.joint_names, chain.root_link, chain.tip_link), (JOINTS, "base_link", "gripper_end"))

    def test_forward(self):
        pose = kinematics_client.forward("rebot_arm", START)
        many = kinematics_client.forward_many("rebot_arm", [START, {}])
        self.assertAlmostEqual(pose.distance(many[0]), 0)
        self.assertGreater(pose.distance(many[1]), 0.01)

    def test_straight_line(self):
        start_pose = kinematics_client.forward("rebot_arm", START)
        goal = start_pose.translated(0, 0.1, -0.05).rotated(0, 0, 0.3)
        samples = cartesian_path.plan([start_pose, goal]).samples(0.02)
        path = kinematics_client.ik_path("rebot_arm", START, [p for _, p in samples])
        self.assertEqual(path.joint_names, JOINTS)
        self.assertEqual(len(path.positions), len(samples))
        self.assertLess(path.max_position_error, 1e-3)
        # Every solution reproduces its pose (checked with FK), and the joints move continuously.
        reached = kinematics_client.forward_many("rebot_arm", [dict(zip(JOINTS, q)) for q in path.positions])
        for (_, target), pose in zip(samples, reached):
            self.assertLess(pose.distance(target), 1e-3)
            self.assertLess(pose.angle(target), 1e-2)
        steps = [max(abs(b - a) for a, b in zip(p, q)) for p, q in zip(path.positions, path.positions[1:])]
        self.assertLess(max(steps), 0.2)

    def test_unreachable(self):
        start_pose = kinematics_client.forward("rebot_arm", START)
        far = start_pose.translated(1.0, 0, 0)
        poses = [p for _, p in cartesian_path.plan([start_pose, far]).samples(0.05)]
        with self.assertRaises(KinematicsError) as e:
            kinematics_client.ik_path("rebot_arm", START, poses)
        self.assertGreater(e.exception.index, 0)
        self.assertEqual(len(e.exception.positions), e.exception.index)

    def test_jump_rejected(self):
        # A reachable pose far from the start in one step: solvable, but only with a big joint move.
        far = kinematics_client.forward("rebot_arm", {"joint1": 1.0, "joint2": 1.2, "joint3": 1.0})
        with self.assertRaisesRegex(KinematicsError, "jumped at pose 0: joint1"):
            kinematics_client.ik_path("rebot_arm", START, [far])

    def test_errors(self):
        with self.assertRaises(ValueError):
            kinematics_client.chain("no_such_robot")
        with self.assertRaisesRegex(KinematicsError, "not in the MuJoCo model|not found|nope"):
            kinematics_client.forward("rebot_arm", {"nope": 1})

    def test_session(self):
        """One process answers many requests, the same as one each; an error does not end it."""
        start_pose = kinematics_client.forward("rebot_arm", START)
        goal = start_pose.translated(0, 0.05, 0)
        with kinematics_client.Kinematics("rebot_arm") as k:
            for _ in range(3):
                self.assertAlmostEqual(k.forward(START).distance(start_pose), 0)
            self.assertEqual(len(k.forward_many([START, {}])), 2)
            with self.assertRaisesRegex(KinematicsError, "not in the MuJoCo model|not found|nope"):
                k.forward({"nope": 1})
            path = k.ik_path(START, [goal])
            self.assertLess(k.forward(dict(zip(path.joint_names, path.positions[0]))).distance(goal), 1e-3)
            with self.assertRaises(KinematicsError) as e:
                k.ik_path(START, [start_pose.translated(1.0, 0, 0)])  # unreachable
            self.assertEqual(e.exception.index, 0)
            self.assertAlmostEqual(k.forward(START).distance(start_pose), 0)  # still serving
        with self.assertRaisesRegex(KinematicsError, "exited"):
            k.forward(START)  # closed


if __name__ == "__main__":
    unittest.main()
