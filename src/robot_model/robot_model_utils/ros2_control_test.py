"""ros2_control.render()/robot_description() on small inline URDFs."""

import unittest
import xml.etree.ElementTree as ET

from src.robot_model.robot_model_utils import ros2_control

URDF = """<robot name="r">
  <link name="a"/><link name="b"/><link name="c"/><link name="d"/>
  <joint name="fixed" type="fixed"><parent link="a"/><child link="b"/></joint>
  <joint name="j1" type="revolute"><parent link="b"/><child link="c"/>
    <limit lower="0.5" upper="1.0" effort="1" velocity="1"/></joint>
  <joint name="j2" type="prismatic"><parent link="c"/><child link="d"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""


def joints(block):
    return {
        j.get("name"): [c.get("name") for c in j.iter("command_interface")] for j in block.iter("joint")
    }


class Ros2ControlTest(unittest.TestCase):
    def test_movable_joints_with_their_command_interfaces(self):
        block = ET.fromstring(ros2_control.render(URDF, "/m.xml", {"j1": ["position"]}))
        self.assertEqual(joints(block), {"j1": ["position"], "j2": []})  # j2 passive, fixed skipped

    def test_hardware_params(self):
        block = ET.fromstring(
            ros2_control.render(URDF, "/m.xml", {}, headless=True, pids_config_file="/pids.yaml")
        )
        params = {p.get("name"): p.text for p in block.find("hardware").iter("param")}
        self.assertEqual(
            params, {"mujoco_model": "/m.xml", "headless": "true", "pids_config_file": "/pids.yaml"}
        )

    def test_no_pids_param_by_default(self):
        block = ET.fromstring(ros2_control.render(URDF, "/m.xml", {}))
        self.assertNotIn("pids_config_file", [p.get("name") for p in block.iter("param")])

    def test_initial_position_clamped_to_limits(self):
        self.assertEqual(ros2_control.joints(URDF)[0]["initial_position"], 0.5)

    def test_unknown_commanded_joint_is_an_error(self):
        with self.assertRaises(ValueError):
            ros2_control.render(URDF, "/m.xml", {"nope": ["position"]})

    def test_robot_description_is_valid_urdf(self):
        robot = ET.fromstring(ros2_control.robot_description(URDF, "/m.xml", {"j1": ["position"]}))
        self.assertEqual(robot.tag, "robot")
        self.assertIsNotNone(robot.find("ros2_control"))


if __name__ == "__main__":
    unittest.main()
