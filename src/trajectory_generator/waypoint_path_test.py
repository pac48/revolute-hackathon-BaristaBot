"""waypoint_path: segments through the waypoints, rest at each, monotonic samples."""

import unittest

from src.trajectory_generator import waypoint_path


class WaypointPathTest(unittest.TestCase):
    def setUp(self):
        self.path = waypoint_path.plan([[0, 0], [1, 0], [1, 2]], max_velocity=1.0)

    def test_segments_and_duration(self):
        self.assertEqual(len(self.path.segments), 2)
        self.assertEqual(self.path.dof, 2)
        # Quintic peak velocity 15/8 * distance / duration = max_velocity.
        self.assertAlmostEqual(self.path.duration, 15 / 8 * 1 + 15 / 8 * 2)
        self.assertEqual(self.path.waypoints, [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)])

    def test_passes_through_waypoints_at_rest(self):
        middle = self.path.sample(self.path.segments[0].duration)
        self.assertEqual(middle.positions, (1.0, 0.0))
        self.assertTrue(all(abs(v) < 1e-12 for v in middle.velocities))
        self.assertEqual(self.path.sample(self.path.duration).positions, (1.0, 2.0))

    def test_stays_on_the_straight_segments(self):
        for t in (0.3, 1.0, 1.5):  # first segment: joint 1 stays 0
            self.assertEqual(self.path.sample(t).positions[1], 0.0)

    def test_samples(self):
        samples = self.path.samples(0.1)
        times = [s.time for s in samples]
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], self.path.duration)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertEqual(sum(1 for s in samples if s.positions == (1.0, 0.0)), 1)  # waypoint once

    def test_needs_two_waypoints(self):
        with self.assertRaises(ValueError):
            waypoint_path.plan([[0, 0]])


if __name__ == "__main__":
    unittest.main()
