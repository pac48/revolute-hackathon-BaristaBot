"""prefix_spec renames links and joints and keeps the spheres."""

import unittest

from src.robot_model.robot_model_utils.spherize import prefix_spec, spheres_spec


class PrefixSpecTest(unittest.TestCase):
    def test_names_are_prefixed_and_spheres_kept(self):
        spec = spheres_spec.Spec(
            spheres={"link1": [(0.0, 0.1, 0.2, 0.03)], "link2": [(0.0, 0.0, 0.0, 0.01)]},
            fixed_joints={"joint7": 0.0},
            disable_collisions=[("link1", "link2")],
        )
        out = prefix_spec.prefixed(spec, "left_")
        self.assertEqual(set(out.spheres), {"left_link1", "left_link2"})
        self.assertEqual(out.spheres["left_link1"], spec.spheres["link1"])
        self.assertEqual(out.fixed_joints, {"left_joint7": 0.0})
        self.assertEqual(out.disable_collisions, [("left_link1", "left_link2")])


if __name__ == "__main__":
    unittest.main()
