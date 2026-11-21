"""End-to-end: the simulation publishes its whole MuJoCo state and /set_mujoco_state sets it.

Starts the rebot_arm_gemini_305 simulation (headless), whose scene has a free cube: /mujoco_state carries
every qpos and qvel (the arm's and the cube's), and moving the cube with /set_mujoco_state shows up in it.
Every wait is bounded.
"""

import threading
import time
import unittest

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy

from mujoco_state_msgs.msg import MujocoState
from mujoco_state_msgs.srv import SetMujocoState
from src.ros2_control_sim import models
from src.ros2_control_sim.sim import Simulation

TIMEOUT = 10.0  # s
# scene.xml's state: the arm's 9 hinge/slide joints, then the cube's free joint (x y z qw qx qy qz; 6 velocities).
NQ, NV = 16, 15
CUBE = 9  # the cube's qpos address


class MujocoStateTest(unittest.TestCase):
    def test_state_is_published_and_set(self):
        cube = CUBE
        with Simulation(models.get("rebot_arm_gemini_305"), headless=True):
            rclpy.init()
            try:
                node = rclpy.create_node("mujoco_state_test")
                lock = threading.Lock()
                latest = {}

                def on_state(msg):
                    with lock:
                        latest["msg"] = msg
                        latest["count"] = latest.get("count", 0) + 1

                node.create_subscription(MujocoState, "/mujoco_state", on_state,
                                         QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
                executor = rclpy.executors.SingleThreadedExecutor()
                executor.add_node(node)

                def spin_until(done):
                    end = time.monotonic() + TIMEOUT
                    while time.monotonic() < end:
                        executor.spin_once(timeout_sec=0.05)
                        with lock:
                            if done():
                                return True
                    return False

                self.assertTrue(spin_until(lambda: latest.get("count", 0) >= 50), "no /mujoco_state")
                with lock:
                    msg = latest["msg"]
                self.assertEqual(len(msg.qpos), NQ)
                self.assertEqual(len(msg.qvel), NV)
                np.testing.assert_allclose(msg.qpos[cube:cube + 3], [0.3, 0.0, 0.02], atol=0.005)  # resting

                client = node.create_client(SetMujocoState, "/set_mujoco_state")
                self.assertTrue(client.wait_for_service(timeout_sec=TIMEOUT))

                def call(request):
                    future = client.call_async(request)
                    executor.spin_until_future_complete(future, timeout_sec=TIMEOUT)
                    self.assertTrue(future.done())
                    return future.result()

                bad = call(SetMujocoState.Request(qpos=[0.0] * 3))
                self.assertFalse(bad.success)
                self.assertIn(str(NQ), bad.message)

                qpos = np.array(msg.qpos)
                qpos[cube:cube + 3] = [0.35, 0.1, 0.02]
                self.assertTrue(call(SetMujocoState.Request(qpos=qpos.tolist())).success)
                with lock:
                    latest.clear()

                def moved():
                    m = latest.get("msg")
                    return m is not None and abs(m.qpos[cube] - 0.35) < 0.005 and abs(m.qpos[cube + 1] - 0.1) < 0.005

                self.assertTrue(spin_until(moved), f"cube not moved: {latest.get('msg') and latest['msg'].qpos[cube:cube + 3]}")
            finally:
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
