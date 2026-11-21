"""point_to_point: boundary conditions, limits, sampling."""

import math
import unittest

from src.trajectory_generator import point_to_point as p2p


class PointToPointTest(unittest.TestCase):
    def setUp(self):
        self.traj = p2p.PointToPoint(start=(0.0, 1.0), goal=(1.0, -1.0), duration=2.0)

    def test_starts_and_ends_at_rest(self):
        for t, expected in ((0.0, self.traj.start), (2.0, self.traj.goal)):
            s = self.traj.sample(t)
            self.assertEqual(s.positions, expected)
            for v in (*s.velocities, *s.accelerations):
                self.assertAlmostEqual(v, 0.0)

    def test_midpoint_and_peak_velocity(self):
        s = self.traj.sample(1.0)
        self.assertEqual(s.positions, (0.5, 0.0))
        # 15/8 * distance / duration
        self.assertAlmostEqual(s.velocities[0], 15 / 8 * 1.0 / 2.0)
        self.assertAlmostEqual(s.velocities[1], 15 / 8 * -2.0 / 2.0)

    def test_velocity_matches_position_derivative(self):
        h = 1e-6
        for t in (0.3, 1.1, 1.7):
            numeric = (self.traj.sample(t + h).positions[1] - self.traj.sample(t - h).positions[1]) / (2 * h)
            self.assertAlmostEqual(self.traj.sample(t).velocities[1], numeric, places=5)

    def test_samples_cover_the_whole_move(self):
        samples = self.traj.samples(0.3)
        self.assertEqual(samples[0].time, 0.0)
        self.assertEqual(samples[-1].time, 2.0)
        self.assertEqual(samples[-1].positions, self.traj.goal)
        self.assertTrue(all(b.time > a.time for a, b in zip(samples, samples[1:])))
        self.assertLessEqual(max(b.time - a.time for a, b in zip(samples, samples[1:])), 0.3)

    def test_minimum_duration_respects_limits(self):
        start, goal = (0.0, 0.0), (1.0, 0.5)
        duration = p2p.minimum_duration(start, goal, max_velocity=[0.5, 2.0], max_acceleration=1.0)
        traj = p2p.PointToPoint(start, goal, duration)
        samples = traj.samples(duration / 1000)
        self.assertLessEqual(max(abs(s.velocities[0]) for s in samples), 0.5 + 1e-9)
        self.assertLessEqual(max(abs(s.accelerations[i]) for s in samples for i in (0, 1)), 1.0 + 1e-9)
        # The binding limit is reached, so the duration is not longer than needed.
        self.assertAlmostEqual(duration, max(15 / 8 * 1.0 / 0.5, math.sqrt(10 / math.sqrt(3) * 1.0 / 1.0)))

    def test_plan(self):
        self.assertEqual(p2p.plan([0], [1], duration=3).duration, 3.0)
        self.assertAlmostEqual(p2p.plan([0], [1], max_velocity=1.0).duration, 15 / 8)
        self.assertEqual(p2p.plan([1], [1]).duration, 0.1)  # no motion: min_duration

    def test_invalid(self):
        with self.assertRaises(ValueError):
            p2p.PointToPoint((0.0,), (1.0, 2.0), 1.0)
        with self.assertRaises(ValueError):
            p2p.PointToPoint((0.0,), (1.0,), 0.0)
        with self.assertRaises(ValueError):
            p2p.minimum_duration([0, 0], [1, 1], max_velocity=[1.0])


if __name__ == "__main__":
    unittest.main()
