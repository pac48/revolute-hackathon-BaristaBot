"""The rebot_arm_gemini_305 robot description: a valid URDF with the camera, and the reBot's <ros2_control> block."""

import os
import unittest
import xml.etree.ElementTree as ET

from python.runfiles import Runfiles

from src.robot_model.rebot_arm_gemini_305 import description


class DescriptionTest(unittest.TestCase):
    def test_commanded_joints_are_the_rebots(self):
        block = ET.fromstring(description.robot_description(headless=True)).find("ros2_control")
        commanded = {
            j.get("name"): [c.get("name") for c in j.iter("command_interface")]
            for j in block.iter("joint")
            if j.find("command_interface") is not None
        }
        self.assertEqual(commanded, description.COMMAND_INTERFACES)

    def test_camera_links_are_in_the_urdf(self):
        links = {l.get("name") for l in ET.fromstring(description.robot_description()).iter("link")}
        self.assertTrue({"gripper_end", "camera_bottom_screw", "camera_link", "camera_left_color_optical_frame"} <= links)

    def test_visual_meshes_are_package_uris(self):
        # Not file:// paths: RViz may be on another machine than the simulation.
        meshes = [m.get("filename") for m in ET.fromstring(description.robot_description()).iter("mesh")]
        prefixes = ("package://rebot_arm_description/meshes/", "package://orbbec_description/meshes/gemini305/")
        self.assertTrue(any(uri.startswith(prefixes[1]) for uri in meshes))
        for uri in meshes:
            self.assertTrue(uri.startswith(prefixes), uri)

    def test_package_uris_are_in_the_ament_index(self):
        # What an RViz with these indexes in its data resolves package:// URIs to.
        runfiles = Runfiles.Create()
        for m in ET.fromstring(description.robot_description()).iter("mesh"):
            package, _, path = m.get("filename").removeprefix("package://").partition("/")
            found = runfiles.Rlocation(f"_main/share/{package}/{path}")
            self.assertTrue(found and os.path.isfile(found), m.get("filename"))


if __name__ == "__main__":
    unittest.main()
