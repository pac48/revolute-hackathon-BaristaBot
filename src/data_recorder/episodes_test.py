"""The recorder's --episodic mode: ~/start and ~/stop make one finalized bag per episode, a second ~/start throws
the episode away, and so does exiting before ~/stop.

Runs the recorder with a JointState publisher of its own (200 Hz on /joint_states); no simulation.
"""

import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import unittest

import rclpy
from python.runfiles import Runfiles
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

RATE = 200.0  # Hz


def message_count(bag: str) -> int:
    with open(os.path.join(bag, "metadata.yaml")) as f:
        return int(re.search(r"message_count:\s*(\d+)", f.read()).group(1))  # the bag's total: the first


class EpisodesTest(unittest.TestCase):
    def test_start_stop_discard(self):
        recorder = Runfiles.Create().Rlocation(os.environ["RECORDER"])
        out = os.path.join(tempfile.mkdtemp(), "episodes")
        proc = subprocess.Popen([recorder, "--episodic", "--no-voice", "-o", out, "--topics", "/joint_states", "--stats-period", "100"])
        rclpy.init()
        node = rclpy.create_node("episodes_test")
        publisher = node.create_publisher(JointState, "/joint_states", 10)
        node.create_timer(1.0 / RATE, lambda: publisher.publish(JointState(name=["j"], position=[0.0])))
        start = node.create_client(Trigger, "/data_recorder/start")
        stop = node.create_client(Trigger, "/data_recorder/stop")
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        try:
            self.assertTrue(start.wait_for_service(timeout_sec=20.0) and stop.wait_for_service(timeout_sec=5.0))
            time.sleep(2.0)  # the recorder finds /joint_states (it looks every second)
            call = lambda client: client.call(Trigger.Request(), timeout_sec=10.0)
            episode1, episode2, episode3 = (os.path.join(out, f"episode_000{i}") for i in (1, 2, 3))

            r = call(start)
            self.assertTrue(r.success)
            self.assertEqual(r.message, episode1)
            time.sleep(0.5)
            r = call(stop)
            self.assertTrue(r.success)
            self.assertEqual(r.message, episode1)
            self.assertFalse(call(stop).success)  # not recording

            self.assertTrue(call(start).success)
            time.sleep(1.0)  # thrown away by the next start
            r = call(start)
            self.assertTrue(r.success)
            self.assertEqual(r.message, episode2)  # the same episode, again
            time.sleep(0.5)
            self.assertEqual(call(stop).message, episode2)
            time.sleep(0.5)  # not recorded
            self.assertEqual(call(start).message, episode3)  # never stopped: discarded at exit
            time.sleep(0.3)
        finally:
            proc.send_signal(signal.SIGINT)
            code = proc.wait(timeout=20.0)
            executor.shutdown()
            rclpy.try_shutdown()
        self.assertIn(code, (0, 2))  # 2: messages lost (not expected; still a finished recording)
        n1, n2 = message_count(episode1), message_count(episode2)
        print(f"episode 1: {n1} messages, episode 2: {n2}", flush=True)
        self.assertGreater(n1, 0.3 * RATE)
        self.assertGreater(n2, 0.3 * RATE)
        self.assertLess(n2, 0.9 * RATE)  # only the 0.5 s after the second start, not the discarded 1 s
        self.assertFalse(os.path.exists(episode3))  # started but never stopped: discarded
        self.assertEqual(sorted(os.listdir(out)), ["episode_0001", "episode_0002"])


if __name__ == "__main__":
    unittest.main()
