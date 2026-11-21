"""PolicyRunner on untrained checkpoints, made as lerobot-train saves them (a small ACT in each space): loading,
the observation it is given, actions to joint targets (IK for the ee space) and the refusals."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.lerobot_export import conventions as c
from src.lerobot_policy.policy_runner import PolicyError, PolicyRunner
from src.lerobot_policy.testing import IMAGE, save_checkpoint
from src.trajectory_generator import kinematics_client

CPU = "cpu"  # the GPU's test is policy_runner_gpu_test
HOME = np.array([0.0, 1.2, 1.4, -0.7, 0.0, 0.0, 0.04])  # grasp_cube's home pose, the gripper open


class PolicyRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.joint = save_checkpoint(cls.dir / "joint", 7)
        cls.ee = save_checkpoint(cls.dir / "ee", 10, cameras=("scene", "center"))

    def images(self, runner):
        return {cam: np.zeros(IMAGE, np.uint8) for cam in runner.cameras}

    def test_joint_space(self):
        with PolicyRunner(self.joint, device=CPU, max_joint_step=10.0) as runner:
            self.assertEqual((runner.space, runner.cameras), ("joint", ["scene"]))
            runner.reset()
            for _ in range(7):  # past the first chunk (5 actions)
                targets = runner.act(HOME, self.images(runner))
                self.assertEqual(targets.shape, (7,))
                self.assertTrue(np.all(np.isfinite(targets)))

    def test_observation_is_the_datasets_frame(self):
        with PolicyRunner(self.joint, device=CPU) as runner:
            frame = runner.observation(HOME, self.images(runner))
            self.assertEqual(set(frame), {"observation.state", "observation.images.scene"})
            np.testing.assert_allclose(frame["observation.state"], HOME, atol=1e-6)
            # A camera's full-size image: resized to the policy's size, as the dataset's were.
            full = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
            image = runner.observation(HOME, {"scene": full})["observation.images.scene"]
            np.testing.assert_array_equal(image, c.resize(full, IMAGE[:2]))
            with self.assertRaisesRegex(ValueError, "3-channel uint8"):
                runner.observation(HOME, {"scene": np.zeros((48, 64), np.uint8)})
            with self.assertRaisesRegex(ValueError, "3-channel uint8"):
                runner.observation(HOME, {"scene": np.zeros(IMAGE, np.float32)})
            with self.assertRaisesRegex(ValueError, "no image of camera scene"):
                runner.observation(HOME, {})
            with self.assertRaisesRegex(ValueError, "expected 7 values"):
                runner.observation(HOME[:6], self.images(runner))

    def test_jump_refused(self):
        # Untrained, with the action's std 0.01 around 0: an action near 0, far from HOME's joint2 (1.2 rad).
        with PolicyRunner(self.joint, device=CPU, max_joint_step=0.1) as runner:
            with self.assertRaisesRegex(PolicyError, "moves joint\\d by"):
                runner.act(HOME, self.images(runner))

    def test_ee_space(self):
        with PolicyRunner(self.ee, device=CPU, model="rebot_arm_gemini_305") as runner:
            self.assertEqual((runner.space, sorted(runner.cameras)), ("ee", ["center", "scene"]))
            # The observation: the tool's pose at HOME by FK (grasp_cube's home: the tool at (0.40, 0, 0.34)) and the gripper.
            state = runner.observation(HOME, self.images(runner))["observation.state"]
            np.testing.assert_allclose(state[:3], [0.40, 0.0, 0.34], atol=0.01)
            self.assertAlmostEqual(float(state[9]), HOME[6], places=6)
            # An action: a pose near HOME's -> IK -> joints that reach it, the gripper as commanded.
            moved = HOME + [0.05, -0.03, 0.02, 0.02, 0.03, -0.04, -0.02]
            pose = kinematics_client.forward("rebot_arm_gemini_305", dict(zip(c.ARM_JOINTS, moved[:6])))
            action = c.ee_vector(np.array([[*pose.position, *pose.quaternion]]), moved[None])[0]
            targets = runner.joints_from_ee(action, HOME)
            reached = kinematics_client.forward("rebot_arm_gemini_305", dict(zip(c.ARM_JOINTS, targets[:6])))
            self.assertLess(reached.distance(pose), 1e-3)
            self.assertLess(reached.angle(pose), 1e-2)
            self.assertAlmostEqual(targets[6], moved[6])
            # Unreachable: refused.
            far = action.copy()
            far[:3] += [1.0, 0, 0]
            with self.assertRaisesRegex(PolicyError, "IK"):
                runner.joints_from_ee(far, HOME)

    def test_checkpoint_of_another_shape(self):
        with self.assertRaisesRegex(ValueError, "expected both"):
            PolicyRunner(save_checkpoint(self.dir / "five", 5), device=CPU)
        with self.assertRaisesRegex(ValueError, "camera one of"):
            PolicyRunner(save_checkpoint(self.dir / "cam", 7, cameras=("nope",)), device=CPU)


if __name__ == "__main__":
    unittest.main()
