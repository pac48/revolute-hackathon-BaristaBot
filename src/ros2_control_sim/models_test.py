"""Each model's controller config only commands joints/interfaces its description offers."""

import unittest

from src.ros2_control_sim import controller_config, models


class ModelsTest(unittest.TestCase):
    def test_controllers_match_description(self):
        for model in models.MODELS.values():
            with self.subTest(model.name):
                commanded = controller_config.command_interfaces(model.controller_config())
                offered = model.description.COMMAND_INTERFACES
                for joint, interfaces in commanded.items():
                    self.assertIn(joint, offered)
                    self.assertLessEqual(set(interfaces), set(offered[joint]), joint)

    def test_rebot_arm_joints_have_the_real_motors_interfaces(self):
        # rebot_robstride_protocol's RobStrideProtocol: MIT-mode commands, and position/velocity/effort states.
        mit = ["position", "velocity", "kp", "kd", "t_ff"]
        for name in ("rebot_arm", "rebot_arm_gemini_305"):
            with self.subTest(name):
                model = models.get(name)
                commanded = controller_config.command_interfaces(model.controller_config())
                for joint in [f"joint{i}" for i in range(1, 7)]:
                    self.assertEqual(commanded[joint], mit, joint)
                    self.assertEqual(model.description.COMMAND_INTERFACES[joint], mit, joint)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            models.get("nope")


if __name__ == "__main__":
    unittest.main()
