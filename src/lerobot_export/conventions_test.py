"""conventions: the 6D rotation and its inverse."""

import unittest

import numpy as np

from src.lerobot_export import conventions as c


def random_quaternions(n: int, rng) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class RotationTest(unittest.TestCase):
    def test_rotation_6d(self):
        s = np.sqrt(0.5)
        q = np.array([[0, 0, 0, 1], [0, 0, s, s], [s, 0, 0, s]])  # identity, 90 deg about z, 90 deg about x
        expected = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, -1, 0, 0],  # x -> y, y -> -x
            [1, 0, 0, 0, 0, 1],  # x -> x, y -> z
        ])
        np.testing.assert_allclose(c.rotation_6d(q), expected, atol=1e-6)
        np.testing.assert_allclose(c.rotation_6d(-q), expected, atol=1e-6)  # q and -q: the same rotation, the same 6D

    def test_6d_round_trip(self):
        rng = np.random.default_rng(0)
        s = np.sqrt(0.5)
        # Random rotations and the ones Shepperd's branches are for: identity, and half turns about each axis.
        qs = np.concatenate([random_quaternions(200, rng), [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [s, s, 0, 0]]])
        for q in qs:
            back = c.quaternion_from_6d(c.rotation_6d(q[None])[0])
            self.assertAlmostEqual(abs(float(back @ q)), 1.0, places=5, msg=f"{q} -> {back}")  # q or -q
            self.assertGreaterEqual(back[3], 0)

    def test_6d_of_a_policy_need_not_be_a_rotation(self):
        r = np.array([2.0, 0.0, 0.0, 0.3, 0.5, 0.0])  # not unit, not orthogonal: x, then y once orthogonalized
        np.testing.assert_allclose(c.quaternion_from_6d(r), [0, 0, 0, 1], atol=1e-9)

    def test_ee_vector(self):
        pose = np.array([[0.4, 0.0, 0.3, 0, 0, 0, 1]])
        joints = np.array([[0.1, 1.2, 1.4, -0.7, 0, 0, 0.05]])
        v = c.ee_vector(pose, joints)
        self.assertEqual(v.shape, (1, len(c.EE_NAMES)))
        np.testing.assert_allclose(v[0], [0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 0.05], atol=1e-6)
        self.assertEqual(c.SPACES, {7: "joint", 10: "ee"})


class ResizeTest(unittest.TestCase):
    def test_color_and_depth(self):
        rng = np.random.default_rng(0)
        color = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.assertEqual(c.resize(color, c.IMAGE_SIZE).shape, (360, 640, 3))
        self.assertIs(c.resize(color, (720, 1280)), color)  # already its size
        flat = np.full((720, 1280, 3), 77, np.uint8)
        np.testing.assert_array_equal(c.resize(flat, (360, 640)), 77)
        depth = rng.choice(np.array([400, 900], np.uint16), (720, 1280, 1))
        small = c.resize(depth, c.IMAGE_SIZE, depth=True)
        self.assertEqual((small.shape, small.dtype), ((360, 640, 1), np.uint16))
        self.assertEqual(set(np.unique(small)), {400, 900})  # the nearest pixel's: no blended depths


if __name__ == "__main__":
    unittest.main()
