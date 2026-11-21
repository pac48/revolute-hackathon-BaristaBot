"""The dual-arm robot description: a valid URDF of two prefixed arms 34.5 cm apart, with their <ros2_control> blocks."""

import os
import unittest
import xml.etree.ElementTree as ET

from python.runfiles import Runfiles

from src.robot_model.rebot_dual_arm_gemini_305 import description

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def commanded(block):
    return {
        j.get("name"): [c.get("name") for c in j.iter("command_interface")]
        for j in block.iter("joint")
        if j.find("command_interface") is not None
    }


class DescriptionTest(unittest.TestCase):
    def test_commanded_joints(self):
        block = ET.fromstring(description.robot_description(headless=True)).find("ros2_control")
        self.assertEqual(commanded(block), description.COMMAND_INTERFACES)
        self.assertEqual(len(description.COMMAND_INTERFACES), 14)

    def test_both_arms_and_their_cameras_are_in_the_urdf(self):
        urdf = ET.fromstring(description.robot_description())
        links = {l.get("name") for l in urdf.iter("link")}
        joints = {j.get("name") for j in urdf.findall("joint")}
        for arm in ("left_", "right_"):
            self.assertTrue({arm + "base_link", arm + "gripper_end", arm + "camera_link"} <= links, arm)
            self.assertTrue({arm + j for j in ARM_JOINTS} <= joints, arm)

    def test_bases_are_34_5_cm_apart_in_y_and_aligned_in_x(self):
        urdf = ET.fromstring(description.robot_description())
        origin = {}
        for joint in urdf.findall("joint"):
            if joint.find("child").get("link") in ("left_base_link", "right_base_link"):
                origin[joint.find("child").get("link")] = [float(v) for v in joint.find("origin").get("xyz").split()]
        left, right = origin["left_base_link"], origin["right_base_link"]
        self.assertAlmostEqual(left[0], right[0])
        self.assertAlmostEqual(left[2], right[2])
        self.assertAlmostEqual(left[1] - right[1], 0.345)

    def test_real_description_has_a_can_system_per_arm(self):
        urdf = ET.fromstring(description.real_robot_description("can3", "can4", gripper=True))
        blocks = {b.get("name"): b for b in urdf.findall("ros2_control")}
        self.assertEqual(len(blocks), 2)
        buses = {}
        for name, block in blocks.items():
            params = {p.get("name"): p.text for p in block.find("hardware").iter("param")}
            self.assertEqual(block.find("hardware/plugin").text, "shared_can_ros2/CanSystem")
            buses[name] = params["can_interface"]
            joints = [j.get("name") for j in block.iter("joint")]
            prefix = "left_" if "left" in name else "right_"
            self.assertEqual(joints, [prefix + j for j in ARM_JOINTS])
            self.assertEqual(len({j.find("param[@name='motor_id']").text for j in block.iter("joint")}), 7)
        self.assertEqual(sorted(buses.values()), ["can3", "can4"])
        # Protocol instances are per arm, so the arms' motor ids (1-7 on each bus) don't clash.
        instances = {p.text for p in urdf.iter("param") if p.get("name") == "protocol_instance"}
        self.assertEqual(len(instances), 4)

    def test_visual_meshes_are_package_uris(self):
        meshes = [m.get("filename") for m in ET.fromstring(description.robot_description()).iter("mesh")]
        prefixes = ("package://rebot_arm_description/meshes/", "package://orbbec_description/meshes/gemini305/")
        for uri in meshes:
            self.assertTrue(uri.startswith(prefixes), uri)

    def test_package_uris_are_in_the_ament_index(self):
        runfiles = Runfiles.Create()
        for m in ET.fromstring(description.robot_description()).iter("mesh"):
            package, _, path = m.get("filename").removeprefix("package://").partition("/")
            found = runfiles.Rlocation(f"_main/share/{package}/{path}")
            self.assertTrue(found and os.path.isfile(found), m.get("filename"))


if __name__ == "__main__":
    unittest.main()
