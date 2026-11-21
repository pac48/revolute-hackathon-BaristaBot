import io
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from src.robot_model.robot_model_utils.spherize import export_spheres_obj as ex
from src.robot_model.robot_model_utils.spherize import import_spheres_obj as im
from src.robot_model.robot_model_utils.spherize import spheres_spec

URDF = """<robot name="r"><link name="base"/><link name="arm_1"/>
<joint name="j" type="revolute"><parent link="base"/><child link="arm_1"/>
<origin xyz="0.1 0.2 0.3" rpy="0 0 1.0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint></robot>"""


class RoundTrip(unittest.TestCase):
    def test_export_then_import(self):
        robot = ET.fromstring(URDF)
        q = {"j": 0.4}
        poses = spheres_spec.link_poses(robot, q)
        spheres = {"base": [(0.0, 0.0, 0.05, 0.02)], "arm_1": [(0.01, 0.02, 0.03, 0.04), (0.1, 0, 0, 0.01)]}
        uv, uf = ex.icosphere(2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.obj")
            with open(path, "w") as f:
                w = ex.ObjWriter(f)
                for link, items in spheres.items():
                    for i, (x, y, z, r) in enumerate(items):
                        w.add(f"sphere_{link}_{i}.001", ex.transform(uv * r + [x, y, z], poses[link]), uf)
                w.add("mesh_arm_1", np.eye(3), np.array([[0, 1, 2]]))
            got = im.spheres_from_objects(im.read_obj_objects(path), poses)
        self.assertEqual(set(got), set(spheres))
        for link in spheres:
            np.testing.assert_allclose(got[link], spheres[link], atol=1e-6)

    def test_copies_share_the_spec(self):
        spec = spheres_spec.Spec({"a": [(0.0, 0.0, 0.0, 0.01)]}, {}, [])
        moved = [(0.1, 0.0, 0.0, 0.01)]
        same = list(spec.spheres["a"])
        # only the left copy was edited
        self.assertEqual(im.resolve([spec], ["l_", "r_"], {"l_a": moved, "r_a": same}), {"a": moved})
        # nothing edited
        self.assertEqual(im.resolve([spec], ["l_", "r_"], {"l_a": same, "r_a": same}), {})
        # both, differently
        with self.assertRaises(ValueError):
            im.resolve([spec], ["l_", "r_"], {"l_a": moved, "r_a": [(0.2, 0.0, 0.0, 0.01)]})
        # deleted in one copy
        self.assertEqual(im.resolve([spec], ["l_", "r_"], {"r_a": same}), {"a": []})

    def test_fit_rigid_and_exploded_layout(self):
        rng = np.random.default_rng(0)
        local = rng.normal(size=(30, 3))
        pose = spheres_spec.link_poses(ET.fromstring(URDF), {"j": 0.7})["arm_1"]
        np.testing.assert_allclose(im.fit_rigid(local, ex.transform(local, pose)), pose, atol=1e-9)
        self.assertIsNone(im.fit_rigid(local, ex.transform(local, pose) + rng.normal(size=local.shape) * 0.01))
        poses = spheres_spec.link_poses(ET.fromstring(URDF), {})
        geometry = {"base": rng.normal(size=(20, 3)) * 0.1, "arm_1": rng.normal(size=(20, 3)) * 0.1}
        laid = ex.exploded_poses(poses, geometry)
        centers = [ex.transform(g, laid[k]).mean(axis=0) for k, g in geometry.items()]
        self.assertGreater(np.linalg.norm(centers[0] - centers[1]), 0.2)
        np.testing.assert_allclose(laid["arm_1"][:3, :3], poses["arm_1"][:3, :3])

    def test_link_pose_from_moved_mesh(self):
        local = np.random.default_rng(1).normal(size=(12, 3))
        moved = spheres_spec.link_poses(ET.fromstring(URDF), {"j": 1.2})["arm_1"]
        got = im.link_poses_in_obj({"mesh_arm_1": ex.transform(local, moved)}, {"arm_1": np.eye(4)},
                                   {"base": np.eye(4)}, {"arm_1": local})
        np.testing.assert_allclose(got["arm_1"], moved, atol=1e-9)

    def test_spheres_go_to_the_link_they_are_near(self):
        robot = ET.fromstring(URDF)
        poses = ex.exploded_poses(spheres_spec.link_poses(robot, {}), {})
        poses["base"] = np.eye(4)
        poses["arm_1"] = np.eye(4)
        poses["arm_1"][:3, 3] = [1.0, 0, 0]
        balls = {"base": (np.zeros(3), 0.3), "arm_1": (np.zeros(3), 0.3)}
        uv, _ = ex.icosphere(1)
        objects = {"Sphere.003": uv * 0.05 + [1.1, 0.1, 0], "sphere_wrongname_0": uv * 0.05 + [0.1, 0, 0],
                   "mesh_base": uv}
        got = im.spheres_from_objects(objects, poses, balls)
        np.testing.assert_allclose(got["arm_1"], [(0.1, 0.1, 0.0, 0.05)], atol=1e-6)
        np.testing.assert_allclose(got["base"], [(0.1, 0.0, 0.0, 0.05)], atol=1e-6)

    def test_unknown_link_without_balls(self):
        with self.assertRaises(ValueError):
            im.spheres_from_objects({"sphere_nolink_0": np.eye(3) + 1}, {"base": np.eye(4)})

    def test_longest_link_wins(self):
        self.assertEqual(im.parse_name("sphere_arm_1_3", {"arm", "arm_1"}), "arm_1")


if __name__ == "__main__":
    unittest.main()
