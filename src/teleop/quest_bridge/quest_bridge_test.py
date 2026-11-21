"""End-to-end: a WebSocket client in place of the headset, against the real bridge and ROS.

Sends hello + controller messages and checks the ROS topics, publishes a static TF for the
gripper and checks the client receives it. Every wait is bounded.
"""

import asyncio
import json
import struct
import threading
import time
import unittest

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Joy
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
from websockets.sync.client import connect

from src.teleop.quest_bridge import bridge
from src.teleop.quest_bridge.bridge import QuestBridge, QuestBridgeNode

TIMEOUT = 5.0


def controllers(seq: int) -> str:
    return json.dumps(
        {
            "type": "controllers",
            "seq": seq,
            "t": seq / 72,
            "head": {"p": [-0.3, 0.0, 0.6], "q": [0, 0, 0, 1]},
            "left": {"active": False, "trigger": 0, "grip": 0, "stick": [0, 0],
                     "buttons": {"x": 1, "y": 0, "stick": 0, "menu": 0}},
            "right": {"active": True, "p": [0.4, -0.1, 0.3], "q": [0, 0, 0, 1],
                      "trigger": 0.5, "grip": 1.0, "stick": [0.2, -0.3], "buttons": {"a": 0, "b": 1, "stick": 0}},
        }
    )


class Recorder(Node):
    def __init__(self):
        super().__init__("quest_bridge_test")
        self.msgs: dict[str, list] = {"right": [], "left_pose": [], "left_joy": [], "right_joy": [], "active": [],
                                      "view": []}
        self.create_subscription(CompressedImage, "/quest/view/image_raw/compressed", self.msgs["view"].append, 10)
        self.create_subscription(PoseStamped, "/quest/right/pose", self.msgs["right"].append, 10)
        self.create_subscription(PoseStamped, "/quest/left/pose", self.msgs["left_pose"].append, 10)
        self.create_subscription(Joy, "/quest/left/joy", self.msgs["left_joy"].append, 10)
        self.create_subscription(Joy, "/quest/right/joy", self.msgs["right_joy"].append, 10)
        self.create_subscription(Bool, "/quest/active", self.msgs["active"].append, 10)
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.static_pub = self.create_publisher(TFMessage, "/tf_static", qos)

    def publish_gripper(self):
        tf = TransformStamped()
        tf.header.frame_id = "world"
        tf.child_frame_id = "gripper_end"
        tf.transform.translation.x = 0.25
        tf.transform.translation.z = 0.15
        tf.transform.rotation.w = 1.0
        self.static_pub.publish(TFMessage(transforms=[tf]))


def wait_for(cond, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


class QuestBridgeTest(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = QuestBridgeNode("world", "gripper_end")
        self.recorder = Recorder()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.recorder)  # spun by the bridge's event loop, with its node

        self.bridge = QuestBridge(self.node, self.executor, "127.0.0.1", 0, gripper_rate=50.0, timeout=0.25)
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        self.server = threading.Thread(
            target=lambda: self.loop.run_until_complete(self.bridge.run(ready)), daemon=True
        )
        self.server.start()
        self.assertTrue(ready.wait(TIMEOUT), "bridge did not start")

    def tearDown(self):
        self.bridge.stop()
        self.server.join(TIMEOUT)
        self.assertFalse(self.server.is_alive(), "bridge did not stop")
        self.loop.close()
        self.executor.shutdown()
        self.node.destroy_node()
        self.recorder.destroy_node()
        rclpy.try_shutdown()

    def test_controllers_to_ros_and_gripper_back(self):
        self.recorder.publish_gripper()
        with connect(f"ws://127.0.0.1:{self.bridge.bound_port}", open_timeout=TIMEOUT) as ws:
            ws.send(json.dumps({"type": "hello", "version": 1, "device": "test"}))
            seq = 0

            def send_some():
                nonlocal seq
                seq += 1
                ws.send(controllers(seq))
                return False

            self.assertTrue(
                wait_for(lambda: send_some() or (self.recorder.msgs["right"] and self.recorder.msgs["right_joy"])),
                "no controller topics",
            )
            pose = self.recorder.msgs["right"][-1]
            self.assertEqual(pose.header.frame_id, "world")
            self.assertAlmostEqual(pose.pose.position.x, 0.4)
            self.assertAlmostEqual(pose.pose.position.y, -0.1)
            joy = self.recorder.msgs["right_joy"][-1]
            for got, want in zip(joy.axes, [0.5, 1.0, 0.2, -0.3], strict=True):
                self.assertAlmostEqual(got, want, places=5)
            self.assertEqual(list(joy.buttons), [0, 1, 0])
            self.assertEqual(list(self.recorder.msgs["left_joy"][-1].buttons), [1, 0, 0, 0])
            self.assertEqual(self.recorder.msgs["left_pose"], [], "untracked hand must not publish a pose")
            self.assertTrue(wait_for(lambda: send_some() or any(m.data for m in self.recorder.msgs["active"])))

            # Gripper pose from /tf_static comes back over the socket.
            ws.socket.settimeout(TIMEOUT)
            deadline = time.monotonic() + TIMEOUT
            gripper = None
            while gripper is None and time.monotonic() < deadline:
                msg = json.loads(ws.recv(timeout=TIMEOUT))
                if msg["type"] == "gripper_pose":
                    gripper = msg
            self.assertIsNotNone(gripper, "no gripper_pose received")
            self.assertEqual(gripper["p"], [0.25, 0.0, 0.15])
            self.assertEqual(gripper["q"], [0.0, 0.0, 0.0, 1.0])

        # Watchdog: after the headset goes away, /quest/active turns false.
        self.recorder.msgs["active"].clear()
        self.assertTrue(wait_for(lambda: any(not m.data for m in self.recorder.msgs["active"])))

    def test_view_to_compressed_image(self):
        jpeg = b"\xff\xd8\xff\xe0fake jpeg body\xff\xd9"
        with connect(f"ws://127.0.0.1:{self.bridge.bound_port}", open_timeout=TIMEOUT) as ws:
            ws.send(b"VIEW" + struct.pack("<HHd", 64, 48, 1.0) + jpeg)
            ws.send(b"JUNK" + bytes(20))  # unknown binary: logged, connection stays
            ws.send(b"VIEW" + struct.pack("<HHd", 64, 48, 1.1) + jpeg)
            self.assertTrue(wait_for(lambda: len(self.recorder.msgs["view"]) == 2), "no view images")
        msg = self.recorder.msgs["view"][-1]
        self.assertEqual(msg.format, "jpeg")
        self.assertEqual(bytes(msg.data), jpeg)
        self.assertEqual(msg.header.frame_id, "quest_head")

    def test_bad_messages_do_not_drop_connection(self):
        with connect(f"ws://127.0.0.1:{self.bridge.bound_port}", open_timeout=TIMEOUT) as ws:
            ws.send("garbage")
            ws.send('{"type":"controllers","head":{"p":[0],"q":[0,0,0,1]}}')
            ws.send('{"type":"future_message"}')
            ws.send(controllers(1))
            self.assertTrue(wait_for(lambda: self.bridge.messages == 1))



class CommandLineTest(unittest.TestCase):
    def test_poses_are_in_the_world_frame_by_default(self):
        args = bridge.build_parser().parse_args([])
        self.assertEqual((args.world_frame, args.tip_frame, args.port), ("world", "gripper_end", 9090))

    def test_options(self):
        args = bridge.build_parser().parse_args(["--world-frame", "map", "--tip-frame", "left_gripper_end", "--port", "1"])
        self.assertEqual((args.world_frame, args.tip_frame, args.port), ("map", "left_gripper_end", 1))


if __name__ == "__main__":
    unittest.main()
