"""fast_kinematics_gen: segments + tip transform reproduce URDF forward kinematics."""

import math
import unittest

import numpy as np

from src.robot_model.robot_model_utils import fast_kinematics_gen as gen

# Fixed joints before/between/after the joints, flipped and tilted axes, multi-angle rpy.
URDF = """<robot name="r">
  <joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.1" rpy="0 0 0.3"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0.01 0.02 0.075"/><axis xyz="0 0 -1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.02 0.03 0.07" rpy="-1.5708 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="bracket" type="fixed"><parent link="l2"/><child link="l2b"/><origin xyz="0.1 0 0" rpy="0.2 0.4 -0.1"/></joint>
  <joint name="j3" type="continuous"><parent link="l2b"/><child link="l3"/>
    <origin xyz="0.0365 0 0.048" rpy="3.14159265 1.57079265 3.14159265"/><axis xyz="1 0 0"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0.1 0.2 0.3" rpy="0.5 -0.2 0.9"/><axis xyz="0.3 -0.5 0.8"/></joint>
  <joint name="tool" type="fixed"><parent link="l4"/><child link="tip"/>
    <origin xyz="0 0 0.16621" rpy="3.67e-06 -1.57079265 -3.14158898"/></joint>
  <joint name="finger" type="prismatic"><parent link="tip"/><child link="f"/><axis xyz="0 1 0"/></joint>
</robot>"""


def axis_rotation(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k


def urdf_fk(joints, q):
    t, i = np.eye(4), 0
    for j in joints:
        t = t @ j["origin"]
        if j["type"] != "fixed":
            t = t @ gen.homogeneous(axis_rotation(j["axis"], q[i]))
            i += 1
    return t


def segment_fk(segs, tip, q):
    t = np.eye(4)
    for s, qi in zip(segs, q):
        t = t @ gen.homogeneous(s["rotation"], s["offset"]) @ gen.homogeneous(axis_rotation([0, 0, 1], qi))
    return t @ tip


class FastKinematicsGenTest(unittest.TestCase):
    def test_matches_urdf_fk(self):
        joints = gen.chain(URDF, "world", "tip")
        segs, tip = gen.segments(joints)
        self.assertEqual([s["name"] for s in segs], ["j1", "j2", "j3", "j4"])
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform(-math.pi, math.pi, 4)
            np.testing.assert_allclose(segment_fk(segs, tip, q), urdf_fk(joints, q), atol=1e-12)

    def test_rpy_is_zyx(self):
        r = gen.rpy_matrix(0.1, 0.2, 0.3)
        np.testing.assert_allclose(r, axis_rotation([0, 0, 1], 0.3) @ axis_rotation([0, 1, 0], 0.2)
                                   @ axis_rotation([1, 0, 0], 0.1), atol=1e-15)

    def test_align_z(self):
        for axis in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0.3, -0.5, 0.8]):
            a = gen.align_z(axis)
            np.testing.assert_allclose(a @ [0, 0, 1], np.asarray(axis) / np.linalg.norm(axis), atol=1e-12)
            np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(a), 1.0)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "only revolute"):
            gen.segments(gen.chain(URDF, "world", "f"))
        with self.assertRaisesRegex(ValueError, "not below"):
            gen.chain(URDF, "l3", "l1")

    def test_template_reads_columns(self):
        # The template reads written values as R11 R21 R31 R12 ...; check the
        # first segment's numbers come out in that order.
        segs, _ = gen.segments(gen.chain(URDF, "world", "tip"))
        code = gen.render("{% for r in rotations %}{% for row in r %}{% for v in row %}{{v}} {% endfor %}"
                          "{% endfor %};{% endfor %}", segs)
        first = [float(v) for v in code.split(";")[0].split()]
        np.testing.assert_allclose(first, segs[0]["rotation"].T.flatten(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
