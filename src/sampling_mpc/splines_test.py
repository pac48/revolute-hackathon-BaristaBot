import unittest

import numpy as np

from src.sampling_mpc.splines import NodeSpline


class NodeSplineTest(unittest.TestCase):
    def test_constant_nodes_give_constant_controls(self):
        spline = NodeSpline(steps=16, nodes=4)
        controls = spline.controls(np.full((5, 2), 0.3))
        np.testing.assert_allclose(controls, 0.3, atol=1e-12)

    def test_controls_pass_through_the_nodes_at_their_times(self):
        spline = NodeSpline(steps=13, nodes=4)  # node times 0, 3, 6, 9, 12 are step times
        nodes = np.random.default_rng(0).normal(size=(5, 3))
        controls = spline.controls(nodes)
        np.testing.assert_allclose(controls[[0, 3, 6, 9, 12]], nodes, atol=1e-10)

    def test_fit_recovers_the_nodes_of_a_smooth_sequence(self):
        spline = NodeSpline(steps=16, nodes=4)
        t = np.linspace(0, 1, 16)
        sequence = np.stack([np.sin(t), t**2], axis=1)
        nodes = spline.fit(sequence)
        np.testing.assert_allclose(spline.controls(nodes), sequence, atol=5e-3)

    def test_batched(self):
        spline = NodeSpline(steps=16, nodes=4)
        nodes = np.random.default_rng(1).normal(size=(7, 5, 2))
        self.assertEqual(spline.controls(nodes).shape, (7, 16, 2))

    def test_rejects_too_few_steps(self):
        with self.assertRaises(ValueError):
            NodeSpline(steps=2, nodes=1)


if __name__ == "__main__":
    unittest.main()
