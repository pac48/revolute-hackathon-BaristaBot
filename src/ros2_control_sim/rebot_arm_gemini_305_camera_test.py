"""End-to-end: the simulated Gemini 305's color cameras publish at 30 FPS or better.

Starts the rebot_arm_gemini_305 simulation (headless) and counts the 1280x720 images
on /camera/left_color/image_raw and /camera/right_color/image_raw over a few seconds
of wall-clock time. The subscriptions are reliable, as the plugin's publishers are:
best-effort ones lose the 2.7 MB samples over UDP and see ~1 FPS. They take the
serialized messages (raw), since deserializing 2.7 MB in Python would cap the rate.
"""

import time
import unittest

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image  # noqa: F401

from src.ros2_control_sim.sim import Simulation

TOPICS = ["/camera/left_color/image_raw", "/camera/right_color/image_raw"]
MIN_FPS = 30.0
WINDOW = 5.0  # s


class CameraRateTest(unittest.TestCase):
    def test_images_at_30_fps(self):
        with Simulation("rebot_arm_gemini_305", headless=True):
            rclpy.init()
            try:
                node = rclpy.create_node("camera_rate_test")
                counts = {t: 0 for t in TOPICS}
                shape = {}

                def counter(topic):
                    def on_image(msg):  # raw serialized bytes: deserializing 2.7 MB in Python would cap the rate
                        counts[topic] += 1
                        shape[topic] = len(msg)

                    return on_image

                qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
                for topic in TOPICS:
                    node.create_subscription(Image, topic, counter(topic), qos, raw=True)
                info = []
                node.create_subscription(CameraInfo, "/camera/left_color/camera_info", info.append, qos)

                # Wait for the first image (the renderer starts after the model loads).
                deadline = time.monotonic() + 20.0
                while counts[TOPICS[0]] == 0 and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                self.assertGreater(counts[TOPICS[0]], 0, "no image within 20 s")

                clock = []
                node.create_subscription(Clock, "/clock", lambda m: clock.append(m.clock.sec + m.clock.nanosec * 1e-9), 10)
                before = dict(counts)
                start = time.monotonic()
                while time.monotonic() - start < WINDOW:
                    rclpy.spin_once(node, timeout_sec=0.01)
                elapsed = time.monotonic() - start
                if len(clock) > 1:
                    print(f"sim time advanced {clock[-1] - clock[0]:.2f} s in {elapsed:.2f} s wall", flush=True)
                for topic in TOPICS:
                    fps = (counts[topic] - before[topic]) / elapsed
                    print(f"{topic}: {fps:.1f} FPS, {shape[topic]} bytes", flush=True)
                    self.assertGreaterEqual(fps, MIN_FPS, topic)
                self.assertTrue(info, "no camera_info")
            finally:
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
