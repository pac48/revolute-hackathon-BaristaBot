import unittest
import xml.etree.ElementTree as ET

from src.robot_model.robot_model_utils.spherize import spheres_mjcf

XML = """<mujoco><worldbody><geom name="floor" type="plane" size="1 1 1"/>
<body name="a"><geom type="mesh" mesh="m" contype="0" conaffinity="0"/><geom type="mesh" mesh="m" group="3"/>
<body name="b"><geom type="box" size="1 1 1"/></body></body></worldbody></mujoco>"""


class SpheresMjcf(unittest.TestCase):
    def test_replaces_collision_geoms_only_of_spherized_bodies(self):
        root = ET.fromstring(XML)
        done = spheres_mjcf.spherize_tree(root, {"a": [(0.0, 0.0, 0.1, 0.05), (0.1, 0.0, 0.0, 0.02)]})
        self.assertEqual(done, {"a"})
        geoms = {b.get("name"): b.findall("geom") for b in root.iter("body")}
        self.assertEqual([g.get("type") for g in geoms["a"]], ["mesh", "sphere", "sphere"])  # visual kept
        self.assertEqual(geoms["a"][0].get("contype"), "0")
        self.assertEqual(len(geoms["b"]), 1)  # not in the spec: untouched
        self.assertEqual(len(root.find("worldbody").findall("geom")), 1)


if __name__ == "__main__":
    unittest.main()
