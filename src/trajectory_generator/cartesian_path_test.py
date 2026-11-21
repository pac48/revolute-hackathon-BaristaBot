"""cartesian_path: pose helpers, straight segments, slerp, timing within limits."""

import math
import unittest

from src.trajectory_generator import cartesian_path as cp
from src.trajectory_generator.cartesian_path import Pose


def rotate(q, v):
    """v rotated by unit quaternion q."""
    x, y, z, w = cp.quaternion_multiply(cp.quaternion_multiply(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return (x, y, z)


class PoseTest(unittest.TestCase):
    def test_rpy_round_trip(self):
        pose = Pose.from_xyz_rpy(1, 2, 3, 0.1, -0.4, 2.0)
        for a, b in zip(pose.rpy, (0.1, -0.4, 2.0)):
            self.assertAlmostEqual(a, b)

    def test_rpy_is_zyx(self):
        # yaw 90 deg then roll 90 deg about fixed x: tool x -> base y, tool y -> base z.
        q = cp.quaternion_from_rpy(math.pi / 2, 0, math.pi / 2)
        for a, b in zip(rotate(q, (1, 0, 0)), (0, 1, 0)):
            self.assertAlmostEqual(a, b)
        for a, b in zip(rotate(q, (0, 1, 0)), (0, 0, 1)):
            self.assertAlmostEqual(a, b)

    def test_translated_and_rotated(self):
        pose = Pose.from_xyz_rpy(0, 0, 0, 0, 0, math.pi / 2)
        moved = pose.translated(0.1, 0, 0)
        self.assertEqual(moved.position, (0.1, 0.0, 0.0))
        self.assertEqual(moved.quaternion, pose.quaternion)
        turned = pose.rotated(0, 0, math.pi / 2)  # about the tool's z: total yaw 180 deg
        self.assertAlmostEqual(turned.angle(Pose.from_xyz_rpy(0, 0, 0, 0, 0, math.pi)), 0, places=6)
        self.assertAlmostEqual(pose.angle(turned), math.pi / 2)

    def test_json(self):
        pose = Pose.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        self.assertEqual(Pose.from_json(pose.to_json()), pose)


class PathTest(unittest.TestCase):
    def setUp(self):
        self.a = Pose.from_xyz_rpy(0.4, 0.0, 0.3, 0, 0.5, 0)
        self.b = Pose.from_xyz_rpy(0.4, 0.2, 0.2, 0, 0.5, 1.0)
        self.path = cp.plan([self.a, self.b], max_linear_velocity=0.1, max_angular_velocity=0.5)

    def test_ends_and_straight_line(self):
        self.assertEqual(self.path.sample(0), self.a)
        self.assertAlmostEqual(self.path.sample(self.path.duration).distance(self.b), 0)
        direction = [y - x for x, y in zip(self.a.position, self.b.position)]
        for t, pose in self.path.samples(0.05):
            offset = [p - x for p, x in zip(pose.position, self.a.position)]
            cross = [offset[1] * direction[2] - offset[2] * direction[1],
                     offset[2] * direction[0] - offset[0] * direction[2],
                     offset[0] * direction[1] - offset[1] * direction[0]]
            self.assertAlmostEqual(math.hypot(*cross), 0, places=12)

    def test_slerp_halfway(self):
        middle = self.path.sample(self.path.duration / 2)
        self.assertAlmostEqual(middle.angle(self.a), middle.angle(self.b), places=9)
        self.assertAlmostEqual(middle.angle(self.a), self.a.angle(self.b) / 2, places=9)

    def test_speed_limits(self):
        # The slower of distance / 0.1 m/s and angle / 0.5 rad/s, times the quintic's 15/8 peak factor.
        self.assertAlmostEqual(self.path.duration, 15 / 8 * max(self.a.distance(self.b) / 0.1,
                                                                 self.a.angle(self.b) / 0.5))
        samples = self.path.samples(0.001)
        linear = max(p.distance(q) / (t1 - t0) for (t0, p), (t1, q) in zip(samples, samples[1:]))
        angular = max(p.angle(q) / (t1 - t0) for (t0, p), (t1, q) in zip(samples, samples[1:]))
        self.assertLessEqual(linear, 0.1 + 1e-6)
        self.assertLessEqual(angular, 0.5 + 1e-6)
        self.assertGreater(linear, 0.099)  # the binding limit is reached

    def test_waypoints_sampled_once_and_at_rest(self):
        c = self.b.translated(0, 0, 0.1)
        path = cp.plan([self.a, self.b, c])
        samples = path.samples(0.1)
        times = [t for t, _ in samples]
        self.assertEqual(times, sorted(set(times)))
        self.assertAlmostEqual(times[-1], path.duration)
        boundary = path.segments[0].duration
        self.assertTrue(any(abs(t - boundary) < 1e-12 for t in times))
        # Stopped at the middle waypoint: no motion just around it.
        self.assertLess(path.sample(boundary - 1e-3).distance(path.sample(boundary + 1e-3)), 1e-8)

    def test_errors(self):
        with self.assertRaises(ValueError):
            cp.plan([self.a])
        with self.assertRaises(ValueError):
            cp.plan([self.a, self.b], max_linear_velocity=0)
        with self.assertRaises(ValueError):
            Pose((0, 0, 0), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
