"""The two_link_arm robot description is a valid URDF whose <ros2_control> block matches the model."""

import unittest
import xml.etree.ElementTree as ET

from src.robot_model.two_link_arm import description


class DescriptionTest(unittest.TestCase):
    def test_commanded_joints(self):
        block = ET.fromstring(description.robot_description(headless=True)).find("ros2_control")
        commanded = {
            j.get("name"): [c.get("name") for c in j.iter("command_interface")]
            for j in block.iter("joint")
            if j.find("command_interface") is not None
        }
        self.assertEqual(commanded, description.COMMAND_INTERFACES)


if __name__ == "__main__":
    unittest.main()
