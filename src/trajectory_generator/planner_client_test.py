"""planner_client against the real rebot_arm planner CLI."""

import unittest

from src.trajectory_generator import planner_client

LEFT = {"joint1": -1.2, "joint2": 1.2, "joint3": 1.0, "joint4": 0.3, "joint5": -0.3, "joint6": 0.5}


class PlannerClientTest(unittest.TestCase):
    def test_joint_names(self):
        self.assertEqual(planner_client.joint_names("rebot_arm"), [f"joint{i}" for i in range(1, 7)])

    def test_plans_from_start_to_goal(self):
        path = planner_client.plan("rebot_arm", start={}, goal=LEFT)
        self.assertEqual(path.joint_names, [f"joint{i}" for i in range(1, 7)])
        self.assertEqual(path.waypoints[0], [0.0] * 6)
        for value, expected in zip(path.waypoints[-1], LEFT.values()):
            self.assertAlmostEqual(value, expected, places=5)

    def test_reports_failures(self):
        everything = {"boxes": [{"center": [0, 0, 0], "half_extents": [2, 2, 2]}]}
        with self.assertRaisesRegex(planner_client.PlanningError, "start is in collision"):
            planner_client.plan("rebot_arm", start={}, goal=LEFT, obstacles=everything)
        with self.assertRaisesRegex(planner_client.PlanningError, "not in the MuJoCo model"):
            planner_client.plan("rebot_arm", start={}, goal={"nope": 1})
        with self.assertRaises(ValueError):
            planner_client.plan("no_such_robot", start={}, goal={})


if __name__ == "__main__":
    unittest.main()
