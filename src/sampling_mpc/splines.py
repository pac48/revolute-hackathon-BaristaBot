"""Control parametrization of DIAL-MPC: a few control *nodes* over the horizon, interpolated to one
control per step by a quadratic spline.

The interpolation is linear in the nodes, so it is a fixed matrix: `nodes_to_controls` (H x K) and
its counterpart `controls_to_nodes` (K x H), which fits the nodes to a control sequence (used to
shift a plan forward in time).
"""

import numpy as np
from scipy.interpolate import make_interp_spline


def _interpolation_matrix(x_from: np.ndarray, x_to: np.ndarray, order: int) -> np.ndarray:
    """Matrix M (len(x_to) x len(x_from)) so that M @ y evaluates the spline through (x_from, y) at x_to."""
    identity = np.eye(len(x_from))
    return make_interp_spline(x_from, identity, k=order)(x_to)


class NodeSpline:
    """Interpolation between `nodes + 1` control nodes and `steps` controls spanning the same time."""

    def __init__(self, steps: int, nodes: int, order: int = 2):
        if nodes < 1:
            raise ValueError("need at least one interval between control nodes (nodes >= 1)")
        if steps < order + 1 or nodes + 1 < order + 1:
            raise ValueError(f"a spline of order {order} needs at least {order + 1} steps and nodes + 1")
        self.steps = steps
        self.nodes = nodes
        # Unit time between steps; the nodes are evenly spread over the same span.
        step_times = np.arange(steps, dtype=float)
        node_times = np.linspace(0.0, steps - 1.0, nodes + 1)
        self.nodes_to_controls = _interpolation_matrix(node_times, step_times, order)  # (steps, nodes + 1)
        self.controls_to_nodes = _interpolation_matrix(step_times, node_times, order)  # (nodes + 1, steps)

    def controls(self, nodes: np.ndarray) -> np.ndarray:
        """Controls (..., steps, nu) from nodes (..., nodes + 1, nu)."""
        return np.einsum("sk,...ku->...su", self.nodes_to_controls, nodes)

    def fit(self, controls: np.ndarray) -> np.ndarray:
        """Nodes (..., nodes + 1, nu) of a control sequence (..., steps, nu)."""
        return np.einsum("ks,...su->...ku", self.controls_to_nodes, controls)
