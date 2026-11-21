"""The real arm's description and controllers offer the interfaces the simulation's do (arm joints)."""

import unittest
import xml.etree.ElementTree as ET

from src.ros2_control_sim import controller_config, models, real


class RealTest(unittest.TestCase):
    def blocks(self, model, gripper=False):
        """The <ros2_control> block of each arm (one CAN bus each: can7, can8)."""
        cans = [f"can{7 + i}" for i in range(len(model.can_interfaces))]
        return ET.fromstring(model.description.real_robot_description(*cans, gripper)).findall("ros2_control")

    def interfaces(self, model, gripper=False):
        """The first block, and the command and state interfaces of the joints of all blocks."""
        blocks = self.blocks(model, gripper)
        commands = {j.get("name"): [c.get("name") for c in j.iter("command_interface")] for b in blocks for j in b.iter("joint")}
        states = {j.get("name"): [c.get("name") for c in j.iter("state_interface")] for b in blocks for j in b.iter("joint")}
        return blocks[0], commands, states

    def test_hardware_is_can_system_on_the_interface(self):
        for model in real.MODELS.values():
            with self.subTest(model.name):
                buses = []
                for block in self.blocks(model):
                    params = {p.get("name"): p.text for p in block.find("hardware").iter("param")}
                    self.assertEqual(block.find("hardware/plugin").text, "shared_can_ros2/CanSystem")
                    self.assertEqual(params["transport_plugin"], "shared_can_ros2/SocketCanTransport")
                    buses.append(params["can_interface"])
                self.assertEqual(buses, [f"can{7 + i}" for i in range(len(model.can_interfaces))])

    def test_arm_joints_match_the_simulation(self):
        for name, model in real.MODELS.items():
            with self.subTest(name):
                _, commands, states = self.interfaces(model)
                sim = models.get(name)
                prefixes = ("left_", "right_") if "dual" in name else ("",)
                for joint in [f"{p}joint{i}" for p in prefixes for i in range(1, 7)]:
                    self.assertEqual(commands[joint], sim.description.COMMAND_INTERFACES[joint], joint)
                    self.assertEqual(states[joint], ["position", "velocity", "effort"], joint)

    def test_controller_values_match_the_simulation(self):
        # One source (config/rebot_arm_shared.yaml): the real arm runs the simulation's arm gains (only the
        # update rate differs: the sim evaluates the motors' MIT law per update, so it runs faster).
        for name, model in real.MODELS.items():
            with self.subTest(name):
                real_config, sim_config = model.controller_config(), models.get(name).controller_config()
                params = lambda c: c["joint_trajectory_controller"]["ros__parameters"]  # noqa: E731
                self.assertEqual(params(real_config)["mit_joints"], params(sim_config)["mit_joints"])
                for joint in params(sim_config)["mit_joints"]:
                    self.assertEqual(params(real_config)["gains"][joint], params(sim_config)["gains"][joint], joint)

    def test_motors_are_distinct_and_have_a_protocol(self):
        block, _, _ = self.interfaces(real.MODELS["rebot_arm"])
        ids = []
        for joint in block.iter("joint"):
            params = {p.get("name"): p.text for p in joint.findall("param")}
            self.assertEqual(params["protocol_plugin"], "rebot_robstride_protocol/RobStrideProtocol")
            ids.append(params["motor_id"])
        self.assertEqual(len(set(ids)), 6)

    def test_controllers_only_command_what_the_real_arm_offers(self):
        for name, model in real.MODELS.items():
            with self.subTest(name):
                for gripper in (False, True):
                    _, offered, _ = self.interfaces(model, gripper)
                    commanded = controller_config.command_interfaces(model.controller_config(gripper))
                    self.assertEqual(set(commanded), set(offered), gripper)
                    for joint, interfaces in commanded.items():
                        self.assertLessEqual(set(interfaces), set(offered[joint]), joint)

    def test_gripper_is_in_meters_on_the_gripper_protocol(self):
        block, commands, states = self.interfaces(real.MODELS["rebot_arm"], gripper=True)
        joint7 = next(j for j in block.iter("joint") if j.get("name") == "joint7")
        params = {p.get("name"): p.text for p in joint7.findall("param")}
        self.assertEqual(params["protocol_plugin"], "rebot_robstride_protocol/RebotGripperProtocol")
        self.assertEqual(params["motor_id"], "0x07")
        self.assertEqual(float(params["stroke"]), 0.05)
        self.assertEqual(commands["joint7"], ["position", "velocity"])
        self.assertEqual(states["joint7"], ["position", "velocity", "effort", "temperature"])
        # The same units and sign as the simulation's slide.
        self.assertEqual(models.get("rebot_arm").description.COMMAND_INTERFACES["joint7"], ["position", "effort"])

    def test_dual_arm_has_a_can_system_and_protocols_per_arm(self):
        model = real.MODELS["rebot_dual_arm_gemini_305"]
        left, right = self.blocks(model, gripper=True)
        self.assertNotEqual(left.get("name"), right.get("name"))
        for block, prefix in ((left, "left_"), (right, "right_")):
            joints = [j.get("name") for j in block.iter("joint")]
            self.assertEqual(joints, [f"{prefix}joint{i}" for i in range(1, 8)])
            ids = [j.find("param[@name='motor_id']").text for j in block.iter("joint")]
            self.assertEqual(sorted(ids), [f"0x0{i}" for i in range(1, 8)])  # motors 1-7 on each bus
            instances = {j.find("param[@name='protocol_instance']").text for j in block.iter("joint")}
            self.assertTrue(all(i.startswith(prefix) for i in instances), instances)
        # A different protocol instance per arm: the same motor ids on two buses must not share one.
        all_instances = {p.text for b in (left, right) for p in b.iter("param") if p.get("name") == "protocol_instance"}
        self.assertEqual(len(all_instances), 4)

    def test_dual_arm_is_one_controller_over_both_arms_and_the_grippers_are_by_position(self):
        model = real.MODELS["rebot_dual_arm_gemini_305"]
        config = model.controller_config(gripper=True)
        self.assertEqual(controller_config.controller_names(config), ["joint_state_broadcaster", "joint_trajectory_controller"])
        params = config["joint_trajectory_controller"]["ros__parameters"]
        arms = [f"{arm}_joint{i}" for arm in ("left", "right") for i in range(1, 8)]
        self.assertEqual(params["joints"], arms)
        self.assertEqual(params["mit_joints"], [j for j in arms if not j.endswith("joint7")])
        self.assertEqual(params["position_joints"], ["left_joint7", "right_joint7"])
        arms_only = model.controller_config(gripper=False)["joint_trajectory_controller"]["ros__parameters"]
        self.assertEqual(len(arms_only["joints"]), 12)
        self.assertNotIn("position_joints", arms_only)

    def test_gripper_gains_are_low_and_adjustable(self):
        model = real.MODELS["rebot_arm"]

        def params(options=None):
            urdf = ET.fromstring(model.description.real_robot_description("can7", True, False, options))
            joint7 = next(j for j in urdf.find("ros2_control").iter("joint") if j.get("name") == "joint7")
            return {p.get("name"): p.text for p in joint7.findall("param")}

        default = params()
        # Not the arm's 50: the squeeze is kp * overshoot (~100 N per N*m through the ~95 rad/m transmission).
        self.assertEqual((float(default["mit_kp"]), float(default["mit_kd"])), (0.5, 0.25))
        self.assertNotIn("max_torque", default)  # no outright limit unless asked for
        tuned = params({"kp": 0.3, "kd": 0.1, "force": 50.0})
        self.assertEqual((float(tuned["mit_kp"]), float(tuned["mit_kd"])), (0.3, 0.1))
        # F = tau * (open - closed) / stroke: 50 N over 4.73 rad (the measured travel) per 0.05 m
        self.assertAlmostEqual(float(tuned["max_torque"]), 50.0 * 0.05 / 4.73, places=5)

    def test_no_gripper_by_default(self):
        _, commands, _ = self.interfaces(real.MODELS["rebot_arm"])
        self.assertNotIn("joint7", commands)


if __name__ == "__main__":
    unittest.main()
