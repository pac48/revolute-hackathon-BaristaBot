"""sampled_trajectory: finite-difference velocities, velocity limits, samples."""

import unittest

from src.trajectory_generator.sampled_trajectory import SampledTrajectory


class SampledTrajectoryTest(unittest.TestCase):
    def test_velocities(self):
        # q = t^2 on uneven times: weighted central differences are exact for quadratics.
        times = [0.0, 0.1, 0.25, 0.3, 0.5]
        traj = SampledTrajectory(times, [(t * t, -t) for t in times])
        v = traj.velocities()
        self.assertEqual(v[0], (0.0, 0.0))
        self.assertEqual(v[-1], (0.0, 0.0))
        for t, (v0, v1) in zip(times[1:-1], v[1:-1]):
            self.assertAlmostEqual(v0, 2 * t)
            self.assertAlmostEqual(v1, -1)

    def test_limit_velocity(self):
        traj = SampledTrajectory([0, 1, 2], [(0, 0), (2, 0.5), (3, 1)])
        self.assertEqual(traj.peak_velocities(), [2.0, 0.5])
        slow = traj.limit_velocity(1.0)
        self.assertEqual(slow.times, (0.0, 2.0, 4.0))
        self.assertEqual(slow.positions, traj.positions)
        self.assertIs(traj.limit_velocity([2.0, 1.0]), traj)  # already within

    def test_samples(self):
        traj = SampledTrajectory([1.0, 1.5, 2.0], [(0,), (1,), (2,)])
        samples = traj.samples(0.05)
        self.assertEqual([s.time for s in samples], [0.0, 0.5, 1.0])
        self.assertEqual(samples[1].velocities, (2.0,))
        self.assertEqual(samples[1].accelerations, ())
        self.assertEqual((traj.dof, traj.duration), (1, 1.0))

    def test_errors(self):
        with self.assertRaises(ValueError):
            SampledTrajectory([0], [(0,)])
        with self.assertRaises(ValueError):
            SampledTrajectory([0, 0], [(0,), (1,)])
        with self.assertRaises(ValueError):
            SampledTrajectory([0, 1], [(0,), (1, 2)])


if __name__ == "__main__":
    unittest.main()
