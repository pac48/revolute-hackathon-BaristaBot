import unittest

import mujoco
import numpy as np

from src.robot_model.rebot_arm_data import rs_mjcf
from src.robot_model.rebot_arm_gemini_305_data import mjcf as gemini_mjcf
from src.ros2_control_sim import controller_config, rebot_arm_controllers
from src.sampling_mpc.servo import position_servos
from src.trajectory_generator import mpc_reaching as mr


def load():
    return mujoco.MjModel.from_xml_path(rs_mjcf.path())


class WaypointTest(unittest.TestCase):
    def test_square_waypoints_keep_the_start_orientation(self):
        mjm = load()
        waypoints = mr.square_waypoints(mjm)
        self.assertEqual(len(waypoints), 4)
        start = mr.tool_pose(mjm, mr.START)
        np.testing.assert_allclose(start.position, [0.40, 0.0, 0.34], atol=0.01)  # the documented start pose
        for w in waypoints:
            np.testing.assert_allclose(w.xmat, start.xmat)

    def test_cycler_advances_after_the_dwell_and_wraps(self):
        mjm = load()
        waypoints = mr.square_waypoints(mjm)
        cycler = mr.WaypointCycler(waypoints, dwell=0.3, timeout=100.0)
        a = waypoints[0]
        self.assertFalse(cycler.update(0.0, a.position + 0.5, a.xmat))  # far
        self.assertFalse(cycler.update(1.0, a.position, a.xmat))  # there, dwelling
        self.assertFalse(cycler.update(1.2, a.position, a.xmat))
        self.assertTrue(cycler.update(1.31, a.position, a.xmat))
        self.assertEqual(cycler.index, 1)
        for i in range(3):  # the others, to the wrap
            w = cycler.target
            cycler.update(10.0 + i, w.position, w.xmat)
            cycler.update(10.4 + i, w.position, w.xmat)
        self.assertEqual(cycler.index, 0)
        self.assertEqual(cycler.laps, 1)

    def test_leaving_the_waypoint_resets_the_dwell(self):
        waypoints = mr.square_waypoints(load())
        cycler = mr.WaypointCycler(waypoints, dwell=0.3, timeout=100.0)
        a = waypoints[0]
        cycler.update(0.0, a.position, a.xmat)
        cycler.update(0.2, a.position + 0.5, a.xmat)  # left
        self.assertFalse(cycler.update(0.4, a.position, a.xmat))  # back: dwell restarts
        self.assertTrue(cycler.update(0.71, a.position, a.xmat))

    def test_cycler_gives_up_after_the_timeout(self):
        waypoints = mr.square_waypoints(load())
        cycler = mr.WaypointCycler(waypoints, timeout=5.0)
        far = waypoints[0].position + 1.0
        self.assertFalse(cycler.update(0.0, far, waypoints[0].xmat))
        self.assertTrue(cycler.update(5.0, far, waypoints[0].xmat))
        self.assertEqual(cycler.index, 1)


class GraspTest(unittest.TestCase):
    def test_the_cube_is_where_the_scene_has_it(self):
        mjm = mujoco.MjModel.from_xml_path(gemini_mjcf.path())
        np.testing.assert_allclose(mr.body_position(mjm, "cube"), [0.3, 0.0, 0.02])

    def test_top_down_points_the_fingers_down_and_closes_them_across_the_reach(self):
        R = mr.top_down_xmat([0.0, 0.3, 0.02]).reshape(3, 3)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        np.testing.assert_allclose(R[:, 0], [0, 0, -1])  # tcp x (along the fingers) down
        np.testing.assert_allclose(R[:, 1], [-1, 0, 0], atol=1e-12)  # fingers close across the base -> cube line

    def test_grasp_lap_opens_descends_closes_lifts_puts_back_and_releases(self):
        cube = np.array([0.3, 0.0, 0.02])
        steps = mr.grasp_waypoints(cube, approach=0.08)
        above = cube + [0, 0, 0.08]
        expected = [(above, mr.GRIPPER_OPEN), (cube, mr.GRIPPER_OPEN), (cube, mr.GRIPPER_CLOSED),
                    (above, mr.GRIPPER_CLOSED), (cube, mr.GRIPPER_CLOSED), (cube, mr.GRIPPER_OPEN)]
        self.assertEqual(len(steps), len(expected))
        for step, (position, gripper) in zip(steps, expected):
            np.testing.assert_allclose(step.position, position)
            self.assertEqual(step.gripper, gripper)
            np.testing.assert_allclose(step.xmat, mr.top_down_xmat(cube))
        self.assertGreaterEqual(steps[2].dwell, (mr.GRIPPER_OPEN - mr.GRIPPER_CLOSED) / mr.GripperRamp(0.0).speed)

    def test_fingers_close_on_the_faces_nearest_the_default_direction(self):
        at = [0.3, 0.0, 0.02]  # default: fingers closing along y (90 degrees)
        def yawed(degrees):
            return [np.cos(np.radians(degrees) / 2), 0.0, 0.0, np.sin(np.radians(degrees) / 2)]
        self.assertAlmostEqual(mr.cube_finger_angle(at, yawed(0)), np.pi / 2)
        self.assertAlmostEqual(mr.cube_finger_angle(at, yawed(30)), np.radians(120))
        self.assertAlmostEqual(mr.cube_finger_angle(at, yawed(80)), np.radians(80))
        self.assertAlmostEqual(mr.cube_finger_angle(at, yawed(-80)), np.radians(100))
        # On its side (rolled 90 degrees about x): its x axis is still horizontal.
        rolled = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
        self.assertAlmostEqual(mr.cube_finger_angle(at, rolled), np.pi / 2)

    def test_grasp_task_follows_the_cube_until_it_closes(self):
        task = mr.GraspTask([0.3, 0.0, 0.02], dwell=0.3, timeout=100.0)
        moved = np.array([0.32, 0.05, 0.02])
        task.track(moved, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(task.target.position, moved + [0, 0, mr.GRASP_APPROACH])
        for t in (0.0, 0.31):  # above it, then down to it
            task.update(t, task.target.position, task.target.xmat)
        self.assertEqual(task.index, 1)
        np.testing.assert_allclose(task.target.position, moved)
        task.update(1.0, task.target.position, task.target.xmat)
        task.update(1.31, task.target.position, task.target.xmat)
        self.assertEqual(task.index, 2)  # closing: the targets stay where it was grasped
        task.track(moved + [0, 0, 0.05], [1.0, 0.0, 0.0, 0.0])  # lifted with the gripper
        np.testing.assert_allclose(task.target.position, moved)
        np.testing.assert_allclose(task.cycler.waypoints[3].position, moved + [0, 0, mr.GRASP_APPROACH])

    def test_a_waypoints_own_dwell_overrides_the_cyclers(self):
        steps = mr.grasp_waypoints([0.3, 0.0, 0.02], close_dwell=1.0)
        cycler = mr.WaypointCycler(steps, dwell=0.3, timeout=100.0)
        cycler.index = 2
        w = cycler.target
        cycler.update(0.0, w.position, w.xmat)
        self.assertFalse(cycler.update(0.5, w.position, w.xmat))  # past the cycler's 0.3 s
        self.assertTrue(cycler.update(1.01, w.position, w.xmat))

    def test_gripper_ramps_to_its_target_at_its_speed_and_retargets_from_where_it_is(self):
        ramp = mr.GripperRamp(0.05, speed=0.1)
        ramp.set_target(1.0, 0.0)
        self.assertAlmostEqual(ramp.at(1.0), 0.05)
        self.assertAlmostEqual(ramp.at(1.2), 0.03)
        self.assertAlmostEqual(ramp.at(2.0), 0.0)  # there, and stays
        ramp.set_target(1.2, 0.05)  # back from 0.03
        self.assertAlmostEqual(ramp.at(1.3), 0.04)
        self.assertAlmostEqual(ramp.at(5.0), 0.05)


class ServoTest(unittest.TestCase):
    def test_gains_come_from_the_controller_parameters(self):
        params = controller_config.load(rebot_arm_controllers.path())["joint_trajectory_controller"]["ros__parameters"]
        gains = mr.servo_gains(params)
        for joint in mr.ARM_JOINTS:  # whatever config/rebot_arm_shared.yaml sets: not pinned here
            self.assertEqual(gains[joint], (params["gains"][joint]["stiffness"], params["gains"][joint]["damping"]))
        self.assertEqual(set(gains), set(mr.ARM_JOINTS))

    def test_position_servos_hold_a_target_position(self):
        mjm = load()
        params = controller_config.load(rebot_arm_controllers.path())["joint_trajectory_controller"]["ros__parameters"]
        position_servos(mjm, mr.servo_gains(params))
        data = mujoco.MjData(mjm)
        a = mjm.actuator("joint1_motor").id
        self.assertEqual(mjm.actuator_ctrlrange[a].tolist(), mjm.jnt_range[mjm.joint("joint1").id].tolist())
        self.assertTrue(mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        # The servo force is kp (target - q), within the motor's +-36.
        data.ctrl[a] = 0.1
        mujoco.mj_forward(mjm, data)
        self.assertAlmostEqual(data.actuator_force[a], params["gains"]["joint1"]["stiffness"] * 0.1, places=4)
        # Limited to the motor's old range.
        data.ctrl[a] = 2.0
        mujoco.mj_forward(mjm, data)
        self.assertAlmostEqual(data.actuator_force[a], 36.0, places=4)


class JointStateTest(unittest.TestCase):
    def test_writes_positions_and_velocities_and_reports_missing_required_joints(self):
        mjm = load()
        data = mujoco.MjData(mjm)
        states = {"joint1": (0.3, 0.1), "joint2": (0.7, float("nan")), "joint5": (float("nan"), 0.0), "other": (9.0, 9.0)}
        missing = mr.set_joint_state(mjm, data, states, required=["joint1", "joint5", "joint6"])
        self.assertEqual(missing, ["joint5", "joint6"])
        self.assertEqual(data.qpos[mjm.joint("joint1").qposadr[0]], 0.3)
        self.assertEqual(data.qvel[mjm.joint("joint1").dofadr[0]], 0.1)
        self.assertEqual(data.qpos[mjm.joint("joint2").qposadr[0]], 0.7)
        self.assertEqual(data.qvel[mjm.joint("joint2").dofadr[0]], 0.0)  # non-finite velocity: zero
        self.assertEqual(data.qpos[mjm.joint("joint5").qposadr[0]], 0.0)  # untouched


class QuinticSetpointTest(unittest.TestCase):
    def test_reaches_the_target_with_its_velocity_and_zero_acceleration(self):
        setpoint = mr.QuinticSetpoint([0.0, 1.0])
        setpoint.retarget(10.0, [1.0, 0.0], [0.5, -0.5], 2.0)
        p, v, a = setpoint.state(12.0)
        np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(v, [0.5, -0.5], atol=1e-9)
        np.testing.assert_allclose(a, [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(setpoint.state(11.0)[0].shape, (2,))  # in between
        np.testing.assert_allclose(setpoint.state(20.0)[0], [1.0, 0.0], atol=1e-9)  # holds the end

    def test_a_new_segment_continues_position_velocity_and_acceleration(self):
        setpoint = mr.QuinticSetpoint([0.0])
        setpoint.retarget(0.0, [1.0], [0.0], 1.0)
        p, v, a = setpoint.state(0.4)
        setpoint.retarget(0.4, [-3.0], [2.0], 0.8)  # a very different plan
        p2, v2, a2 = setpoint.state(0.4)
        np.testing.assert_allclose([p2, v2, a2], [p, v, a], atol=1e-9)
        # Neither a jump in position nor in velocity a moment later.
        eps = 1e-4
        np.testing.assert_allclose(setpoint.state(0.4 + eps)[0], p + v * eps, atol=1e-6)
        np.testing.assert_allclose(setpoint.state(0.4 + eps)[1], v + a * eps, atol=1e-5)

    def test_points_are_every_dt_to_the_end_of_the_segment(self):
        setpoint = mr.QuinticSetpoint([0.0])
        setpoint.retarget(5.0, [1.0], [0.0], 0.3)
        points = setpoint.points(5.0, 0.02)
        self.assertEqual(len(points), 15)
        self.assertAlmostEqual(points[0][0], 0.02)
        self.assertAlmostEqual(points[-1][0], 0.3)
        self.assertAlmostEqual(points[-1][1][0], 1.0)
        # Mid-segment: what is left.
        self.assertEqual(len(setpoint.points(5.1, 0.02)), 10)

    def test_the_setpoint_stays_smooth_under_jumpy_replans(self):
        rng = np.random.default_rng(0)
        setpoint = mr.QuinticSetpoint([0.0])
        times = np.arange(0.0, 6.0, 0.002)
        values, next_replan = [], 0.0
        for t in times:
            if t >= next_replan:  # a replan every 0.25 s, to a random target
                setpoint.retarget(t, rng.uniform(-1, 1, 1), rng.uniform(-1, 1, 1), 0.4)
                next_replan += 0.25
            values.append(setpoint.state(t)[0][0])
        x = np.array(values)
        v = np.diff(x) / 0.002
        a = np.diff(v) / 0.002
        # Replans to targets up to 2 rad away within 0.4 s are fast, but continuous: a jump of 1 rad would be 500 rad/s,
        # of 1 rad/s in velocity 500 rad/s^2.
        self.assertLess(np.abs(v).max(), 15.0)
        self.assertLess(np.abs(a).max(), 400.0)


class PlanEndTest(unittest.TestCase):
    def test_a_line_fit_recovers_a_straight_plan_exactly(self):
        t = 0.04 * np.arange(16)
        targets = np.stack([0.5 * t + 0.1, np.full(16, 1.0)], axis=1)
        end, velocity = mr.plan_end(targets, 0.04)
        np.testing.assert_allclose(end, [0.5 * t[-1] + 0.1, 1.0], atol=1e-9)
        np.testing.assert_allclose(velocity, [0.5, 0.0], atol=1e-9)

    def test_a_noisy_last_target_moves_the_end_less_than_it_moves(self):
        t = 0.04 * np.arange(16)
        targets = np.stack([0.5 * t], axis=1)
        targets[-1, 0] += 0.3
        end, _ = mr.plan_end(targets, 0.04)
        self.assertLess(abs(end[0] - 0.5 * t[-1]), 0.2)

    def test_the_end_velocity_is_limited(self):
        targets = np.stack([10.0 * 0.04 * np.arange(16)], axis=1)
        self.assertEqual(mr.plan_end(targets, 0.04)[1][0], mr.VELOCITY_LIMIT)


class JointPositionsTest(unittest.TestCase):
    def test_joint_positions_picks_the_joints_from_qpos(self):
        mjm = load()
        qpos = np.arange(2 * mjm.nq, dtype=float).reshape(2, mjm.nq)
        picked = mr.joint_positions(mjm, qpos, ("joint1", "joint3"))
        np.testing.assert_array_equal(picked, qpos[:, [mjm.joint("joint1").qposadr[0], mjm.joint("joint3").qposadr[0]]])


if __name__ == "__main__":
    unittest.main()
