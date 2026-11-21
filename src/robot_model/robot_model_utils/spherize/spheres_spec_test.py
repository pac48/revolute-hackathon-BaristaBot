"""spheres_spec: spec round trip, joint freezing, SRDF pairs, sphere FK and overlap check."""

import math
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from src.robot_model.robot_model_utils.spherize import spheres_spec as ss

# base -(fixed)- mount -(revolute z)- arm -(prismatic x)- slider
URDF = """<robot name="r">
  <link name="base"/><link name="mount"/><link name="arm"/><link name="slider"/>
  <joint name="fix" type="fixed"><parent link="base"/><child link="mount"/>
    <origin xyz="0 0 0.1"/></joint>
  <joint name="j1" type="revolute"><parent link="mount"/><child link="arm"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
  <joint name="j2" type="prismatic"><parent link="arm"/><child link="slider"/>
    <origin xyz="0.5 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
</robot>"""


class SpheresSpecTest(unittest.TestCase):
    def robot(self):
        return ET.fromstring(URDF)

    def test_round_trip(self):
        spec = ss.Spec({"arm": [(0.1, 0, 0, 0.05)]}, {"j2": 0.2}, [("base", "slider")])
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(ss.dump_spec(spec, "header"))
        try:
            loaded = ss.load_spec(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual((loaded.fixed_joints, loaded.disable_collisions), (spec.fixed_joints, spec.disable_collisions))
        self.assertEqual(loaded.spheres["arm"][0][:3], (0.1, 0.0, 0.0))
        self.assertAlmostEqual(loaded.spheres["arm"][0][3], 0.050001, places=9)  # rounded up by 1 um

    def test_written_spheres_still_cover(self):
        # A point on the sphere's surface stays inside after writing (centers and radii rounded).
        center, r = np.array([0.1234567, -0.0000004, 0.3333336]), 0.0123454
        point = center + r * np.array([1, 1, 1]) / math.sqrt(3)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(ss.dump_spec(ss.Spec({"l": [(*center, r)]}, {}, [])))
        try:
            x, y, z, written_r = ss.load_spec(f.name).spheres["l"][0]
        finally:
            os.unlink(f.name)
        self.assertLessEqual(np.linalg.norm(point - [x, y, z]), written_r)

    def test_freeze_prismatic_moves_origin(self):
        robot = self.robot()
        ss.freeze_joints(robot, {"j2": 0.25})
        j2 = next(j for j in robot.iter("joint") if j.get("name") == "j2")
        self.assertEqual(j2.get("type"), "fixed")
        self.assertIsNone(j2.find("limit"))
        np.testing.assert_allclose([float(v) for v in j2.find("origin").get("xyz").split()], [0.75, 0, 0])

    def test_freeze_revolute_rotates_origin(self):
        robot = self.robot()
        ss.freeze_joints(robot, {"j1": math.pi / 2})
        j1 = next(j for j in robot.iter("joint") if j.get("name") == "j1")
        np.testing.assert_allclose([float(v) for v in j1.find("origin").get("rpy").split()], [0, 0, math.pi / 2],
                                   atol=1e-9)

    def test_sphere_collisions_replace_existing(self):
        robot = self.robot()
        ss.set_sphere_collisions(robot, {"arm": [(0.1, 0, 0, 0.05), (0.2, 0, 0, 0.04)]})
        arm = next(l for l in robot.iter("link") if l.get("name") == "arm")
        self.assertEqual([c.find("geometry/sphere").get("radius") for c in arm.findall("collision")],
                         ["0.05", "0.04"])
        with self.assertRaises(ValueError):
            ss.set_sphere_collisions(robot, {"nope": [(0, 0, 0, 1)]})

    def test_merge_specs(self):
        a = ss.Spec({"base": [(0, 0, 0, 1)]}, {"j": 0.0}, [("x", "y")])
        b = ss.Spec({"cam": [(0, 0, 0, 2)]}, {"j": 0.0, "k": 1.0}, [("x", "y"), ("cam", "base")])
        merged = ss.merge_specs([a, b])
        self.assertEqual(sorted(merged.spheres), ["base", "cam"])
        self.assertEqual(merged.fixed_joints, {"j": 0.0, "k": 1.0})
        self.assertEqual(merged.disable_collisions, [("x", "y"), ("cam", "base")])
        with self.assertRaises(ValueError):
            ss.merge_specs([a, a])
        with self.assertRaises(ValueError):
            ss.merge_specs([a, ss.Spec({}, {"j": 1.0}, [])])

    def test_disabled_pairs(self):
        pairs = {frozenset(p[:2]): p[2] for p in ss.automatic_disabled_pairs(self.robot())}
        self.assertEqual(pairs[frozenset(("base", "mount"))], "Rigid")
        self.assertEqual(pairs[frozenset(("base", "arm"))], "Adjacent")  # base+mount is one body
        self.assertEqual(pairs[frozenset(("arm", "slider"))], "Adjacent")
        self.assertNotIn(frozenset(("base", "slider")), pairs)

    def test_srdf_adds_user_pairs_once(self):
        root = ET.fromstring(ss.srdf(self.robot(), [("base", "slider"), ("arm", "slider")]))
        pairs = [(e.get("link1"), e.get("link2"), e.get("reason")) for e in root]
        self.assertIn(("base", "slider", "User"), pairs)
        self.assertEqual(sum(1 for a, b, _ in pairs if {a, b} == {"arm", "slider"}), 1)

    def test_sphere_positions_follow_joints(self):
        robot = self.robot()
        placed = ss.sphere_positions(robot, {"slider": [(0.1, 0, 0, 0.05)]}, {"j1": math.pi / 2, "j2": 0.2})
        np.testing.assert_allclose(placed[0][1], [0, 0.8, 0.2], atol=1e-9)

    def test_colliding_pairs(self):
        robot = self.robot()
        spheres = {"base": [(0, 0, 0.2, 0.1)], "slider": [(-0.5, 0, 0, 0.05)]}
        hits = ss.colliding_pairs(robot, spheres, {}, ss.automatic_disabled_pairs(robot))
        self.assertAlmostEqual(hits[frozenset(("base", "slider"))], 0.15)
        self.assertEqual(ss.colliding_pairs(robot, spheres, {"j2": 1.0}, ss.automatic_disabled_pairs(robot)), {})


if __name__ == "__main__":
    unittest.main()
