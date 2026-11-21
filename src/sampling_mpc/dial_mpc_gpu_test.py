"""DIAL-MPC on the GPU: reaching with the cost functions, and the plumbing (shift, rollouts)."""

import unittest

import mujoco
import numpy as np
import warp as wp

from src.sampling_mpc.costs import EffortCost, SiteCost, StateCost, SumCost
from src.sampling_mpc.rollout import BatchedRollout
from src.sampling_mpc.dial_mpc import DialConfig, DialMpc

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" mass="1"/>
      <body pos="0.4 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="0.5"/>
        <site name="tip" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100" kv="10" ctrlrange="-3 3"/>
    <position name="a2" joint="j2" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""


def run_closed_loop(mjm, mpc, steps):
    data = mujoco.MjData(mjm)
    for _ in range(steps):
        plan = mpc.plan(data.qpos, data.qvel)
        data.ctrl[:] = plan.controls[0]
        for _ in range(mpc.substeps):
            mujoco.mj_step(mjm, data)
        mpc.shift()
    mujoco.mj_forward(mjm, data)
    return data


class DialMpcGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.quiet = True
        cls.mjm = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(cls.mjm)
        data.qpos[:] = [0.8, -0.6]
        mujoco.mj_forward(cls.mjm, data)
        cls.target = data.site_xpos[0].copy()
        cls.config = DialConfig(Nsample=512, Hsample=16, Hnode=4, seed=1)

    def test_site_cost_reaches_the_target(self):
        cost = SumCost(
            SiteCost(self.mjm, "tip", self.target, weight=100.0, terminal_scale=5.0),
            StateCost(self.mjm, qvel_weight=0.01, ctrl_weight=0.001),
        )
        mpc = DialMpc(self.mjm, cost, self.config)
        data = run_closed_loop(self.mjm, mpc, 40)
        self.assertLess(np.linalg.norm(data.site_xpos[0] - self.target), 0.08)

    def test_state_cost_reaches_joint_targets(self):
        cost = StateCost(self.mjm, qpos_target={"j1": -0.7, "j2": 0.5}, qpos_weight=10.0, qvel_weight=0.05,
                         terminal_scale=5.0)
        mpc = DialMpc(self.mjm, cost, self.config)
        data = run_closed_loop(self.mjm, mpc, 40)
        np.testing.assert_allclose(data.qpos, [-0.7, 0.5], atol=0.12)

    def test_different_costs_give_different_plans(self):
        a = DialMpc(self.mjm, StateCost(self.mjm, qpos_target={"j1": 1.0}, qpos_weight={"j1": 10.0}), self.config)
        b = DialMpc(self.mjm, StateCost(self.mjm, qpos_target={"j1": -1.0}, qpos_weight={"j1": 10.0}), self.config)
        qpos, qvel = np.zeros(2), np.zeros(2)
        self.assertGreater(a.plan(qpos, qvel).controls[:, 0].mean(), 0.5)
        self.assertLess(b.plan(qpos, qvel).controls[:, 0].mean(), -0.5)

    def test_plan_shapes_and_control_limits(self):
        mpc = DialMpc(self.mjm, StateCost(self.mjm, qpos_target={"j1": 3.0}, qpos_weight=1.0), self.config)
        plan = mpc.plan(np.zeros(2), np.zeros(2), iterations=2)
        self.assertEqual(plan.controls.shape, (16, 2))
        self.assertEqual(plan.nodes.shape, (5, 2))
        self.assertEqual(plan.iterations, 2)
        self.assertTrue(np.all(np.abs(plan.nodes) <= 1.0 + 1e-9))

    def test_shift_moves_the_plan_forward_in_time(self):
        mpc = DialMpc(self.mjm, StateCost(self.mjm, qpos_target={"j1": 1.0}, qpos_weight=1.0), self.config)
        mpc.plan(np.zeros(2), np.zeros(2), iterations=3)
        before = mpc.spline.controls(mpc.nodes)
        mpc.shift()
        after = mpc.spline.controls(mpc.nodes)
        # The nodes are refit, so it is the shifted plan up to the fit's error, and closer to it than to the old one.
        moved = np.abs(after[:-2] - before[1:-1]).max()
        self.assertLess(moved, 0.4 * np.abs(before).max())
        self.assertLess(moved, np.abs(after[:-2] - before[:-2]).max())

    def test_predict_is_the_motion_under_the_plan(self):
        mpc = DialMpc(self.mjm, StateCost(self.mjm), self.config)
        qpos = np.array([0.1, -0.2])
        controls = np.tile([1.0, 0.5], (16, 1))
        trace = mpc.predict(qpos, np.zeros(2), controls)
        self.assertEqual(trace.shape, (17, 2))
        np.testing.assert_allclose(trace[0], qpos, atol=1e-6)
        # The same motion as stepping MuJoCo itself.
        data = mujoco.MjData(self.mjm)
        data.qpos[:] = qpos
        for u in controls:
            data.ctrl[:] = u
            for _ in range(mpc.substeps):
                mujoco.mj_step(self.mjm, data)
        np.testing.assert_allclose(trace[-1], data.qpos, atol=2e-3)
        self.assertGreater(trace[-1, 0], 0.3)  # moving toward the position target

    def test_hold_sets_the_plan_to_the_positions(self):
        mpc = DialMpc(self.mjm, StateCost(self.mjm), self.config)
        mpc.hold([1.5, -0.75])
        np.testing.assert_allclose(mpc.plan(np.zeros(2), np.zeros(2), iterations=0).controls, np.tile([1.5, -0.75], (16, 1)), atol=1e-9)

    def test_effort_cost_penalizes_jittery_controls_and_forces(self):
        q0, qd = np.zeros(2), np.zeros(2)
        smooth = np.tile([0.4, 0.2], (16, 1))
        jitter = smooth + 0.3 * np.array([[1.0, -1.0] if k % 2 else [-1.0, 1.0] for k in range(16)])
        controls = np.stack([smooth, jitter])
        rate = BatchedRollout(self.mjm, EffortCost(self.mjm, rate_weight=100.0), 2, 16, 10)
        costs = rate.run(q0, qd, controls)
        # Rate only: constant controls have none; the jitter flips by 0.6 each step over 15 steps, two actuators.
        self.assertAlmostEqual(costs[0], 0.0, places=5)
        self.assertAlmostEqual(costs[1], 100.0 * 15 * 2 * 0.6**2, delta=1.0)
        force = BatchedRollout(self.mjm, EffortCost(self.mjm, force_weight=1.0), 2, 16, 10)
        costs = force.run(q0, qd, np.stack([np.zeros((16, 2)), np.full((16, 2), 2.0)]))
        self.assertLess(costs[0], costs[1])  # a far position target needs a large servo force

    def test_effort_cost_keeps_the_plan_smoother(self):
        goal = StateCost(self.mjm, qpos_target={"j1": 1.0, "j2": -0.5}, qpos_weight=10.0, terminal_scale=5.0)
        rates = []
        for rate_weight in (0.0, 200.0):
            cost = SumCost(goal, EffortCost(self.mjm, rate_weight=rate_weight))
            mpc = DialMpc(self.mjm, cost, self.config)
            plan = mpc.plan(np.zeros(2), np.zeros(2), iterations=6)
            rates.append(np.abs(np.diff(plan.controls, axis=0)).mean())
        self.assertLess(rates[1], rates[0])

    def test_update_gain_takes_a_longer_step_the_same_way(self):
        steps = {}
        for gain in (1.0, 3.0):
            config = DialConfig(Nsample=512, Hsample=16, Hnode=4, seed=1, sigma_scale=0.05, update_gain=gain)
            mpc = DialMpc(self.mjm, StateCost(self.mjm, qpos_target={"j1": 1.0}, qpos_weight=10.0), config)
            mpc.hold([0.0, 0.0])
            mpc.plan(np.zeros(2), np.zeros(2), iterations=1)
            steps[gain] = mpc.nodes[-1, 0]
        self.assertGreater(steps[1.0], 0.0)  # toward the target
        self.assertAlmostEqual(steps[3.0] / steps[1.0], 3.0, delta=0.3)  # the same direction, three times as far

    def test_bad_ctrl_dt_is_rejected(self):
        with self.assertRaises(ValueError):
            DialMpc(self.mjm, StateCost(self.mjm), DialConfig(ctrl_dt=0.0031))


if __name__ == "__main__":
    unittest.main()
