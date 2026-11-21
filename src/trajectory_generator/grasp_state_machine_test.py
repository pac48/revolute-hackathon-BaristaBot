"""grasp_state_machine with a fake arm: the episode's states in order, the poses, randomizing and the retries."""

import math
import unittest

import numpy as np

from src.trajectory_generator import grasp_state_machine as gsm


def rotation(q):
    """Rotation matrix of a Pose quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class FakeArm:
    """Records what is asked. The cube is where set_cube put it, and follows the tool while the gripper is
    closed on it (if `grips`); `reach` decides which poses are reachable, `unreachable_moves` how many moves fail."""

    def __init__(self, grips=True, closes_on_cube=True, reach=None, unreachable_moves=0):
        self.grips = grips
        self.closes_on_cube = closes_on_cube
        self.reach = reach or (lambda poses: True)
        self.unreachable_moves = unreachable_moves
        self.cube = None
        self.quat = None
        self.calls = []
        self.closed = False
        self.tool = None

    def go_home(self):
        self.calls.append(("home",))

    def cube_pose(self):
        return self.cube.copy(), self.quat

    def set_cube(self, position, quat):
        self.calls.append(("set_cube", np.round(position, 4).tolist()))
        self.cube, self.quat = np.asarray(position, dtype=float), np.asarray(quat)
        self.closed = False

    def reachable(self, above, at):
        return self.reach([above, at])

    def move_tool(self, poses, speed, near_cube=False):
        if self.unreachable_moves:
            self.unreachable_moves -= 1
            raise gsm.Unreachable("no IK")
        self.calls.append(("move", [np.round(p.position, 4).tolist() for p in poses], speed))
        if self.closed and self.grips and self.closes_on_cube:
            self.cube = np.asarray(poses[-1].position, dtype=float)
        self.tool = poses[-1]

    def begin_episode(self):
        self.calls.append(("begin",))

    def end_episode(self):
        self.calls.append(("end",))

    def gripper(self, position):
        self.calls.append(("gripper", position))
        self.closed = position < 0.01
        if self.closed:
            return 0.024 if self.closes_on_cube else 0.0
        return position


def machine(arm, **config):
    return gsm.GraspStateMachine(arm, gsm.GraspConfig(**config), rng=np.random.default_rng(0), log=lambda _: None)


class GraspStateMachineTest(unittest.TestCase):
    def test_an_episode_in_order(self):
        arm = FakeArm()
        m = machine(arm, episodes=1)
        self.assertIs(m.run(), gsm.State.DONE)
        self.assertEqual((m.successes, m.failures), (1, 0))
        c = gsm.GraspConfig()
        cube = np.array(arm.calls[1][1])
        r = lambda p: np.round(p, 4).tolist()
        self.assertEqual(arm.calls, [
            ("home",),
            ("set_cube", r(cube)),
            ("begin",),
            ("gripper", gsm.GRIPPER_OPEN),
            ("move", [r(cube + [0, 0, c.approach])], c.approach_speed),
            ("move", [r(cube)], c.contact_speed),
            ("gripper", gsm.GRIPPER_CLOSED),
            ("move", [r(cube + [0, 0, c.lift])], c.contact_speed),
            ("end",),
        ])

    def test_only_successful_episodes_are_ended(self):
        arm = FakeArm(unreachable_moves=1)  # the first episode fails, the second succeeds
        machine(arm, episodes=2).run()
        marks = [c[0] for c in arm.calls if c[0] in ("home", "begin", "end")]
        self.assertEqual(marks, ["home", "begin", "home", "begin", "end"])

    def test_every_episode_starts_at_home(self):
        arm = FakeArm(unreachable_moves=1)  # the first episode fails, the others succeed
        m = machine(arm, episodes=3)
        m.run()
        firsts = [arm.calls[i - 1][0] for i, c in enumerate(arm.calls) if c[0] == "set_cube"]
        self.assertEqual(firsts, ["home"] * 3)

    def test_the_cube_is_put_in_the_region_on_the_floor(self):
        arm = FakeArm()
        machine(arm, episodes=20).run()
        cubes = np.array([c[1] for c in arm.calls if c[0] == "set_cube"])
        self.assertEqual(len(cubes), 20)
        c = gsm.GraspConfig()
        self.assertTrue(np.all((cubes[:, 0] >= c.x_range[0]) & (cubes[:, 0] <= c.x_range[1])))
        self.assertTrue(np.all((cubes[:, 1] >= c.y_range[0]) & (cubes[:, 1] <= c.y_range[1])))
        np.testing.assert_allclose(cubes[:, 2], c.cube_z)
        self.assertGreater(len({tuple(p) for p in cubes}), 15)  # different every time

    def test_unreachable_poses_are_sampled_again(self):
        asked = []

        def reach(poses):  # only cubes on the left (y > 0)
            asked.append(poses)
            return poses[-1].position[1] > 0

        arm = FakeArm(reach=reach)
        self.assertIs(machine(arm, episodes=3).run(), gsm.State.DONE)
        self.assertTrue(all(c[1][1] > 0 for c in arm.calls if c[0] == "set_cube"))
        self.assertGreater(len(asked), 3)
        above, at = asked[0]  # the poses checked: above the cube and at it, pointing down
        self.assertAlmostEqual(above.position[2] - at.position[2], gsm.GraspConfig().approach)
        np.testing.assert_allclose(rotation(at.quaternion)[:, 0], [0, 0, -1], atol=1e-9)

    def test_no_reachable_pose_fails(self):
        self.assertIs(machine(FakeArm(reach=lambda poses: False), max_samples=5).run(), gsm.State.FAILED)

    def test_the_fingers_close_on_the_cubes_faces(self):
        arm = FakeArm()
        m = machine(arm, episodes=1)
        for _ in range(5):  # home, randomize, open, approach, descend
            m.step()
        R = rotation(arm.tool.quaternion)
        np.testing.assert_allclose(R[:, 0], [0, 0, -1], atol=1e-9)
        w, x, y, z = arm.quat
        yaw = 2 * math.atan2(z, w)
        finger = math.atan2(R[1, 1], R[0, 1])
        self.assertAlmostEqual(math.sin(2 * (finger - yaw)), 0.0, places=6)  # along one of its horizontal axes

    def test_quaternion_round_trip(self):
        for angle in (0.0, 0.7, math.pi / 2, 2.5, -2.0):
            R = rotation(gsm.top_down_pose([0, 0, 0], angle).quaternion)
            np.testing.assert_allclose(R[:, 0], [0, 0, -1], atol=1e-9)
            np.testing.assert_allclose(R[:, 1], [math.cos(angle), math.sin(angle), 0], atol=1e-9)

    def test_finger_angle_nearest_the_default(self):
        cube = [0.3, 0.0, 0.02]
        self.assertAlmostEqual(gsm.finger_angle(cube, None), math.pi / 2)
        self.assertAlmostEqual(gsm.finger_angle(cube, gsm.yaw_quat(math.radians(30))), math.radians(120))
        self.assertAlmostEqual(gsm.finger_angle(cube, gsm.yaw_quat(math.radians(80))), math.radians(80))

    def test_a_moved_cube_is_approached_again(self):
        arm = FakeArm()
        m = machine(arm, episodes=1)
        for _ in range(4):  # home, randomize, open, approach
            m.step()
        moved = arm.cube + [0.05, 0.02, 0.0]
        arm.cube = moved
        self.assertIs(m.step(), gsm.State.APPROACH)
        self.assertIs(m.step(), gsm.State.DESCEND)
        self.assertEqual(arm.calls[-1][1], [np.round(moved + [0, 0, gsm.GraspConfig().approach], 4).tolist()])

    def test_closing_on_nothing_retries_then_the_episode_fails(self):
        arm = FakeArm(closes_on_cube=False)
        m = machine(arm, episodes=2, max_attempts=2)
        self.assertIs(m.run(), gsm.State.DONE)
        self.assertEqual((m.successes, m.failures), (0, 2))
        self.assertEqual(sum(1 for c in arm.calls if c == ("gripper", gsm.GRIPPER_CLOSED)), 4)
        self.assertEqual(sum(1 for c in arm.calls if c[0] == "set_cube"), 2)

    def test_a_cube_that_does_not_come_up_is_tried_again(self):
        arm = FakeArm(grips=False)  # the gripper stops on it, but it slips out
        m = machine(arm, episodes=1, max_attempts=3)
        m.run()
        self.assertEqual((m.successes, m.failures), (0, 1))
        self.assertEqual(sum(1 for c in arm.calls if c == ("gripper", gsm.GRIPPER_CLOSED)), 3)

    def test_an_unreachable_move_ends_the_episode_with_a_new_cube(self):
        arm = FakeArm(unreachable_moves=1)  # the first approach fails
        m = machine(arm, episodes=2)
        self.assertIs(m.run(), gsm.State.DONE)
        self.assertEqual((m.successes, m.failures), (1, 1))
        self.assertEqual(sum(1 for c in arm.calls if c[0] == "set_cube"), 2)


if __name__ == "__main__":
    unittest.main()
