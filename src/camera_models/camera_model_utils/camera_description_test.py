"""camera_description: which URDF joints are the driver's, and the rpy quaternion."""

import math
import unittest

from src.camera_models.camera_model_utils import camera_description as cd

URDF = """<robot name="cam">
  <link name="bottom"/><link name="camera_link"/><link name="screw"/><link name="color"/><link name="optical"/>
  <joint name="a" type="fixed"><parent link="bottom"/><child link="camera_link"/><origin xyz="0 0 0.02"/></joint>
  <joint name="b" type="fixed"><parent link="bottom"/><child link="screw"/><origin xyz="0.01 0 0"/></joint>
  <joint name="c" type="fixed"><parent link="camera_link"/><child link="color"/></joint>
  <joint name="d" type="fixed"><parent link="color"/><child link="optical"/>
    <origin rpy="-1.5707963 0 -1.5707963"/></joint>
</robot>"""


class CameraDescriptionTest(unittest.TestCase):
    def test_driver_subtree_left_out(self):
        joints = cd.model_only_joints(cd.fixed_joints(URDF), "camera_link")
        self.assertEqual(sorted((j.parent, j.child) for j in joints), [("bottom", "camera_link"), ("bottom", "screw")])
        self.assertEqual(joints[0].xyz, (0.0, 0.0, 0.02))

    def test_non_fixed_joint_rejected(self):
        with self.assertRaises(ValueError):
            cd.fixed_joints(URDF.replace('name="b" type="fixed"', 'name="b" type="revolute"'))

    def test_quaternion(self):
        for value, expected in [((0, 0, 0), (0, 0, 0, 1)), ((0, 0, math.pi / 2), (0, 0, math.sqrt(0.5), math.sqrt(0.5))),
                                ((-math.pi / 2, 0, -math.pi / 2), (-0.5, 0.5, -0.5, 0.5))]:
            for a, b in zip(cd.quaternion(value), expected):
                self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
