import unittest

import numpy as np

from src.trajectory_generator.shake_diagnostic import roughness


class RoughnessTest(unittest.TestCase):
    def test_a_slow_sine_has_no_high_frequency_share(self):
        t = np.arange(0, 20, 0.002)
        r = roughness(t, 0.2 * np.sin(2 * np.pi * 0.4 * t), cutoff=5.0, smooth=0.02)
        self.assertLess(r["high_share"], 0.01)
        self.assertAlmostEqual(r["vel_rms"], 0.2 * 2 * np.pi * 0.4 / np.sqrt(2), delta=0.01)
        self.assertAlmostEqual(r["acc_rms"], 0.2 * (2 * np.pi * 0.4) ** 2 / np.sqrt(2), delta=0.05)

    def test_a_shake_is_counted_as_high_frequency(self):
        t = np.arange(0, 20, 0.002)
        x = 0.2 * np.sin(2 * np.pi * 0.4 * t) + 0.003 * np.sin(2 * np.pi * 15 * t)
        r = roughness(t, x, cutoff=5.0, smooth=0.0)
        self.assertGreater(r["high_share"], 0.1)


if __name__ == "__main__":
    unittest.main()
