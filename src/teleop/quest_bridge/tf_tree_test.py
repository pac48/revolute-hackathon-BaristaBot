import math
import unittest

from src.teleop.quest_bridge.protocol import Pose
from src.teleop.quest_bridge.tf_tree import TfTree, compose, inverse, rotate


def yaw(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


class TfTreeTest(unittest.TestCase):
    def assertPose(self, a: Pose, p, q, places=6):
        for x, y in zip(a.p, p):
            self.assertAlmostEqual(x, y, places=places)
        sign = 1 if sum(x * y for x, y in zip(a.q, q)) >= 0 else -1
        for x, y in zip(a.q, q):
            self.assertAlmostEqual(x, sign * y, places=places)

    def test_rotate_and_inverse(self):
        v = rotate(yaw(math.pi / 2), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 1.0)
        t = Pose((1.0, 2.0, 3.0), yaw(0.7))
        self.assertPose(compose(t, inverse(t)), (0, 0, 0), (0, 0, 0, 1))

    def test_chain_lookup(self):
        tf = TfTree()
        # world -> base_link (1 m in x, turned 90 deg) -> link1 (0.5 m up) -> tool (0.2 m along link1 x)
        tf.set("world", "base_link", Pose((1.0, 0.0, 0.0), yaw(math.pi / 2)))
        tf.set("base_link", "link1", Pose((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)))
        tf.set("link1", "tool", Pose((0.2, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        self.assertPose(tf.lookup("base_link", "tool"), (0.2, 0.0, 0.5), (0, 0, 0, 1))
        # In world, base_link's x is world y.
        self.assertPose(tf.lookup("world", "tool"), (1.0, 0.2, 0.5), yaw(math.pi / 2))
        # Reverse direction and sibling lookups.
        self.assertPose(tf.lookup("tool", "base_link"), (-0.2, 0.0, -0.5), (0, 0, 0, 1))
        tf.set("base_link", "camera", Pose((0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        self.assertPose(tf.lookup("camera", "tool"), (0.2, -1.0, 0.5), (0, 0, 0, 1))

    def test_updates_replace(self):
        tf = TfTree()
        tf.set("base_link", "tool", Pose((0.1, 0.0, 0.0), (0, 0, 0, 1)))
        tf.set("/base_link", "tool", Pose((0.3, 0.0, 0.0), (0, 0, 0, 1)))  # leading slash ignored
        self.assertPose(tf.lookup("base_link", "tool"), (0.3, 0, 0), (0, 0, 0, 1))

    def test_unconnected(self):
        tf = TfTree()
        tf.set("a", "b", Pose())
        with self.assertRaises(LookupError):
            tf.lookup("a", "missing")
        tf.set("b", "a", Pose())
        with self.assertRaises(LookupError):
            tf.lookup("x", "a")  # cycle


if __name__ == "__main__":
    unittest.main()
