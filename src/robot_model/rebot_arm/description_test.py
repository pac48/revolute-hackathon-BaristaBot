"""The rebot_arm robot description is a valid URDF whose <ros2_control> block matches the model."""

import os
import unittest
import xml.etree.ElementTree as ET

from python.runfiles import Runfiles

from src.robot_model.rebot_arm import description


class DescriptionTest(unittest.TestCase):
    def test_commanded_joints(self):
        block = ET.fromstring(description.robot_description(headless=True)).find("ros2_control")
        commanded = {
            j.get("name"): [c.get("name") for c in j.iter("command_interface")]
            for j in block.iter("joint")
            if j.find("command_interface") is not None
        }
        self.assertEqual(commanded, description.COMMAND_INTERFACES)

    def test_gripper_fingers_are_passive_and_pids_are_set(self):
        block = ET.fromstring(description.robot_description()).find("ros2_control")
        joints = {j.get("name"): j for j in block.iter("joint")}
        for finger in ("joint_left", "joint_right"):
            self.assertIsNone(joints[finger].find("command_interface"), finger)
        params = {p.get("name"): p.text for p in block.find("hardware").iter("param")}
        self.assertTrue(params["pids_config_file"].endswith("pids.yaml"))

    def test_visual_meshes_are_package_uris(self):
        # Not file:// paths: RViz may be on another machine than the simulation.
        meshes = [m.get("filename") for m in ET.fromstring(description.robot_description()).iter("mesh")]
        self.assertGreater(len(meshes), 20)
        for uri in meshes:
            self.assertTrue(uri.startswith("package://rebot_arm_description/meshes/"), uri)

    def test_package_uris_are_in_the_ament_index(self):
        # What an RViz with these indexes in its data resolves package:// URIs to.
        runfiles = Runfiles.Create()
        for m in ET.fromstring(description.robot_description()).iter("mesh"):
            package, _, path = m.get("filename").removeprefix("package://").partition("/")
            found = runfiles.Rlocation(f"_main/share/{package}/{path}")
            self.assertTrue(found and os.path.isfile(found), m.get("filename"))


if __name__ == "__main__":
    unittest.main()
