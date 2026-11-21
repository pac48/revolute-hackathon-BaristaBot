"""The reaching MPC on the GPU, in closed loop with a CPU MuJoCo simulation of the servo model."""

import unittest

import mujoco
import numpy as np
import warp as wp

from src.robot_model.rebot_arm_data import rs_mjcf
from src.ros2_control_sim import controller_config, rebot_arm_controllers
from src.trajectory_generator import mpc_reaching as mr


def time_to_first_corner(seconds: float = 8.0, tolerance: float = 0.03, **config):
    """Sim time (s) until the tool is within `tolerance` of the first waypoint, from the start pose, or None.
    `config` overrides mpc_reaching.reaching_config()."""
    wp.config.quiet = True
    mjm = mujoco.MjModel.from_xml_path(rs_mjcf.path())
    params = controller_config.load(rebot_arm_controllers.path())["joint_trajectory_controller"]["ros__parameters"]
    waypoint = mr.square_waypoints(mjm)[0]
    mpc, pose_cost = mr.make_reaching_mpc(mjm, mr.servo_gains(params), waypoint, mr.reaching_config(seed=0, **config))
    # The plant is the same servo model (mjm was converted in place), simulated on the CPU, from the start pose.
    data = mujoco.MjData(mjm)
    for name, value in mr.START.items():
        data.qpos[mjm.joint(name).qposadr[0]] = value
    data.ctrl[:] = data.qpos[:mjm.nu]
    mujoco.mj_forward(mjm, data)
    site = mjm.site(mr.TOOL_SITE).id
    mpc.hold(mr.actuator_positions(mjm, data))
    t = 0.0
    while t < seconds:
        mujoco.mj_kinematics(mjm, data)
        if np.linalg.norm(data.site_xpos[site] - waypoint.position) < tolerance:
            return t
        plan = mpc.plan(data.qpos, data.qvel)
        for k in range(3):  # the first 3 control steps (0.12 s) of the plan
            data.ctrl[:] = plan.controls[k]
            for _ in range(mpc.substeps):
                mujoco.mj_step(mjm, data)
        mpc.shift(3)
        t += 3 * mpc.config.ctrl_dt
    return None


class MpcReachingGpuTest(unittest.TestCase):
    def test_tool_reaches_the_first_corner_quickly(self):
        reached = time_to_first_corner()
        self.assertIsNotNone(reached)
        self.assertLess(reached, 3.0)


if __name__ == "__main__":
    unittest.main()
