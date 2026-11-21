"""Cost kernels on Warp's CPU device (no GPU): a stand-in for MuJoCo Warp's Data with just the arrays they read."""

import types
import unittest

import mujoco
import numpy as np
import warp as wp

from src.sampling_mpc.costs import EffortCost, GroundCost, VelocityLimitCost

MODEL = """
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".1"/>
    </body>
    <body>
      <joint name="a" type="hinge"/>
      <geom size=".1"/>
      <body>
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
A, B = 6, 7  # the hinge's and the slide's qvel addresses (after the free joint's 6)


def evaluate(cost, a, b, terminal=False):
    qvel = np.zeros((len(a), 8), dtype=np.float32)
    qvel[:, A], qvel[:, B] = a, b
    qvel[:, :6] = 50.0  # the free joint: never limited
    d = types.SimpleNamespace(qvel=wp.array(qvel, dtype=wp.float32, device="cpu"))
    out = wp.zeros(len(a), dtype=wp.float32, device="cpu")
    (cost.terminal if terminal else cost.running)(None, d, out)
    return out.numpy()


class VelocityLimitCostTest(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.mjm = mujoco.MjModel.from_xml_string(MODEL)

    def test_nothing_within_the_limit_quadratic_past_it_either_way(self):
        cost = VelocityLimitCost(self.mjm, joints=["a"], limit=1.0, weight=1000.0, device="cpu")
        values = evaluate(cost, [0.0, 0.99, -1.0, 1.1, -1.3], [9.0] * 5)  # b is not in `joints`
        np.testing.assert_allclose(values, [0.0, 0.0, 0.0, 1000 * 0.1**2, 1000 * 0.3**2], rtol=1e-4, atol=1e-6)

    def test_every_hinge_and_slide_by_default_with_per_joint_limits(self):
        cost = VelocityLimitCost(self.mjm, limit={"a": 1.0, "b": 0.1}, weight=1.0, terminal_scale=5.0, device="cpu")
        np.testing.assert_allclose(evaluate(cost, [1.5], [0.3]), [0.5**2 + 0.2**2], rtol=1e-4)
        np.testing.assert_allclose(evaluate(cost, [1.5], [0.0], terminal=True), [5 * 0.5**2], rtol=1e-4)


class EffortCostTest(unittest.TestCase):
    def test_begin_allocates_its_per_world_state(self):
        wp.init()
        mjm = mujoco.MjModel.from_xml_string(MODEL.replace("</worldbody>", '</worldbody><actuator><motor joint="a"/></actuator>'))
        cost = EffortCost(mjm, force_weight=1.0, rate_weight=1.0, device="cpu")
        d = types.SimpleNamespace(nworld=3, qpos=wp.zeros((3, mjm.nq), dtype=wp.float32, device="cpu"))
        with wp.ScopedDevice("cpu"):
            cost.begin(None, d)
        self.assertEqual(cost._prev_ctrl.shape, (3, 1))


ARM = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="base" pos="0 0 .05">
      <geom name="base" type="box" size=".1 .1 .05"/>
      <body name="link" pos="0 0 .05">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom name="link" type="box" size=".2 .02 .01" pos=".2 0 0"/>
        <geom name="visual" type="box" size=".5 .5 .5" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="cube" pos=".5 0 .02">
      <freejoint/>
      <geom type="box" size=".02 .02 .02"/>
    </body>
  </worldbody>
</mujoco>
"""


class GroundCostTest(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.mjm = mujoco.MjModel.from_xml_string(ARM)

    def evaluate(self, cost, angles):
        """The cost with the link turned by each of `angles` (rad, about y: positive turns it down)."""
        xpos, xmat = [], []
        for a in angles:
            data = mujoco.MjData(self.mjm)
            data.qpos[self.mjm.joint("j").qposadr[0]] = a
            mujoco.mj_kinematics(self.mjm, data)
            xpos.append(data.geom_xpos.copy())
            xmat.append(data.geom_xmat.reshape(-1, 3, 3).copy())
        d = types.SimpleNamespace(geom_xpos=wp.array(np.array(xpos, dtype=np.float32), dtype=wp.vec3f, device="cpu"),
                                  geom_xmat=wp.array(np.array(xmat, dtype=np.float32), dtype=wp.mat33f, device="cpu"))
        out = wp.zeros(len(angles), dtype=wp.float32, device="cpu")
        cost.running(None, d, out)
        return out.numpy()

    def test_only_the_moving_collision_geoms(self):
        cost = GroundCost(self.mjm, "base", device="cpu")
        # The base rests on the floor (welded: not counted), the visual geom has no collisions, the cube is not the robot's.
        self.assertEqual([mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in cost.geoms], ["link"])

    def test_nothing_above_the_floor_quadratic_below(self):
        cost = GroundCost(self.mjm, "base", weight=1.0, device="cpu")
        # Level, the link's bottom is at 9 cm. Turned down by a, its far bottom corner is at
        # 0.1 - 0.4 sin(a) - 0.01 cos(a) (pivot at 10 cm, 40 cm long, 1 cm half-thick).
        a = np.arcsin(0.3)
        depth = 0.4 * np.sin(a + 0.1) + 0.01 * np.cos(a + 0.1) - 0.1
        values = self.evaluate(cost, [0.0, 0.1, a + 0.1])
        np.testing.assert_allclose(values, [0.0, 0.0, depth**2], rtol=1e-3, atol=1e-9)
        self.assertGreater(depth, 0.03)


if __name__ == "__main__":
    unittest.main()
