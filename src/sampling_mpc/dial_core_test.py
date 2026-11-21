import unittest

import numpy as np

from src.sampling_mpc.dial_mpc import DialConfig, noise_schedule, softmax_weights


class NoiseScheduleTest(unittest.TestCase):
    def test_shape_and_annealing(self):
        config = DialConfig(Hnode=4, horizon_diffuse_factor=0.9, traj_diffuse_factor=0.5, sigma_scale=2.0)
        sigma = noise_schedule(config, 3)
        self.assertEqual(sigma.shape, (3, 5))
        # The last node is perturbed most, the first least; each iteration halves the noise.
        np.testing.assert_allclose(sigma[0, -1], 2.0)
        self.assertTrue(np.all(np.diff(sigma[0]) > 0))
        np.testing.assert_allclose(sigma[1], sigma[0] * 0.5)
        np.testing.assert_allclose(sigma[2], sigma[0] * 0.25)


class SoftmaxWeightsTest(unittest.TestCase):
    def test_normalized_and_prefers_lower_cost(self):
        weights = softmax_weights(np.array([3.0, 1.0, 2.0, 5.0]), reference_cost=3.0, temperature=0.5)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertEqual(int(np.argmax(weights)), 1)
        self.assertEqual(int(np.argmin(weights)), 3)

    def test_scale_invariant(self):
        costs = np.array([3.0, 1.0, 2.0, 5.0])
        np.testing.assert_allclose(
            softmax_weights(costs, 3.0, 0.1), softmax_weights(100 * costs, 300.0, 0.1), atol=1e-12
        )

    def test_equal_costs_are_uniform(self):
        np.testing.assert_allclose(softmax_weights(np.ones(8), 1.0, 0.06), np.full(8, 1 / 8))

    def test_low_temperature_does_not_overflow(self):
        weights = softmax_weights(np.array([0.0, 1e6, 2e6]), 0.0, 1e-6)
        self.assertTrue(np.all(np.isfinite(weights)))


if __name__ == "__main__":
    unittest.main()
