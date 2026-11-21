"""WebSocket <-> ROS 2 bridge for the Quest Teleop headset app (//src/teleop/quest_app).

    bazel run //src/teleop/quest_bridge -- [--port 9090] [--tip-frame gripper_end]

The headset connects as a WebSocket client and streams controller state every frame; the
bridge publishes it (all in the world frame, `world`, the root of every robot's TF tree; ROS convention):

    /quest/{head,left,right}/pose  geometry_msgs/PoseStamped  (hands only while tracked)
    /quest/{left,right}/joy        sensor_msgs/Joy
        axes:    [trigger, grip, stick_x, stick_y]   (0..1, 0..1, -1..1, -1..1)
        buttons: left [x, y, stick, menu], right [a, b, stick]
    /tf                            base -> quest_head, quest_left, quest_right
    /quest/active                  std_msgs/Bool: a headset is connected and its last message
                                   is younger than --timeout (use as a teleop watchdog)

    /quest/view/image_raw/compressed  sensor_msgs/CompressedImage: the headset's left-eye view
                                   (JPEG, from the app's view stream), frame quest_head

and sends back the gripper pose (--tip-frame in the world frame, from /tf and /tf_static) at
--gripper-rate Hz.

Over USB: `adb reverse tcp:9090 tcp:9090` and the app's default host (127.0.0.1:9090).
Over Wi-Fi: `adb shell setprop debug.quest_teleop.host <this PC's IP>:9090`, restart the app.
"""

import argparse
import asyncio
import logging
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import Executor, ShutdownException, SingleThreadedExecutor, TimeoutException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Joy
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from src.teleop.quest_bridge import protocol
from src.teleop.quest_bridge.protocol import LEFT_BUTTONS, RIGHT_BUTTONS, Pose
from src.teleop.quest_bridge.tf_tree import TfTree

log = logging.getLogger("quest_bridge")


def joy_from(hand: protocol.Hand, names: tuple[str, ...]) -> tuple[list[float], list[int]]:
    return (
        [hand.trigger, hand.grip, hand.stick[0], hand.stick[1]],
        [int(hand.buttons.get(n, False)) for n in names],
    )


class QuestBridgeNode(Node):
    def __init__(self, world_frame: str, tip_frame: str, frame_prefix: str = "quest_"):
        super().__init__("quest_bridge")
        self.world_frame = world_frame
        self.tip_frame = tip_frame
        self.frame_prefix = frame_prefix
        self.tf = TfTree()

        self.pose_pubs = {
            name: self.create_publisher(PoseStamped, f"/quest/{name}/pose", 10) for name in ("head", "left", "right")
        }
        self.joy_pubs = {name: self.create_publisher(Joy, f"/quest/{name}/joy", 10) for name in ("left", "right")}
        self.tf_pub = self.create_publisher(TFMessage, "/tf", 10)
        self.active_pub = self.create_publisher(Bool, "/quest/active", 10)
        self.view_pub = self.create_publisher(CompressedImage, "/quest/view/image_raw/compressed", 2)

        self.create_subscription(TFMessage, "/tf", self.tf.add_message, 100)
        static_qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TFMessage, "/tf_static", self.tf.add_message, static_qos)

    def publish_controllers(self, c: protocol.Controllers):
        stamp = self.get_clock().now().to_msg()
        tfs = []

        def pose_out(name: str, pose: Pose):
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.world_frame
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose.p
            o = msg.pose.orientation
            o.x, o.y, o.z, o.w = pose.q
            self.pose_pubs[name].publish(msg)
            tf = TransformStamped()
            tf.header = msg.header
            tf.child_frame_id = self.frame_prefix + name
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pose.p
            tf.transform.rotation = o
            tfs.append(tf)

        if c.head:
            pose_out("head", c.head)
        for name, hand, names in (("left", c.left, LEFT_BUTTONS), ("right", c.right, RIGHT_BUTTONS)):
            if hand.active and hand.pose:
                pose_out(name, hand.pose)
            joy = Joy()
            joy.header.stamp = stamp
            joy.header.frame_id = self.frame_prefix + name
            joy.axes, joy.buttons = joy_from(hand, names)
            self.joy_pubs[name].publish(joy)
        if tfs:
            self.tf_pub.publish(TFMessage(transforms=tfs))

    def publish_view(self, view: protocol.View):
        msg = CompressedImage(format="jpeg", data=view.jpeg)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_prefix + "head"
        self.view_pub.publish(msg)

    def publish_active(self, active: bool):
        self.active_pub.publish(Bool(data=active))

    def gripper_pose(self) -> Pose | None:
        try:
            return self.tf.lookup(self.world_frame, self.tip_frame)
        except LookupError:
            return None


class QuestBridge:
    """The WebSocket server. Accepts any number of headsets; all get the gripper pose and all
    publish to the same topics (meant for one at a time)."""

    def __init__(
        self,
        node: QuestBridgeNode,
        executor: Executor,
        host: str,
        port: int,
        gripper_rate: float,
        timeout: float,
    ):
        self.node = node
        self.executor = executor
        self.host = host
        self.port = port
        self.gripper_period = 1.0 / gripper_rate
        self.timeout = timeout
        self.clients: set[ServerConnection] = set()
        self.last_message = 0.0  # time.monotonic() of the newest controller message
        self.messages = 0
        self.views = 0
        self.bound_port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None

    async def handle(self, ws: ServerConnection):
        peer = ws.remote_address
        log.info("headset connected: %s", peer)
        self.clients.add(ws)
        last_seq = None
        try:
            async for raw in ws:
                try:
                    if isinstance(raw, bytes):  # binary: the headset view
                        self.node.publish_view(protocol.parse_view(raw))
                        self.views += 1
                        continue
                    msg = protocol.parse(raw)
                    if msg["type"] == "hello":
                        log.info("hello from %s: %s (protocol v%s)", peer, msg.get("device"), msg.get("version"))
                    elif msg["type"] == "controllers":
                        c = protocol.controllers_from(msg)
                        if last_seq is not None and c.seq > last_seq + 1:
                            log.debug("dropped %d headset frame(s)", c.seq - last_seq - 1)
                        last_seq = c.seq
                        self.last_message = time.monotonic()
                        self.messages += 1
                        self.node.publish_controllers(c)
                except protocol.ProtocolError as e:
                    log.warning("bad message from %s: %s", peer, e)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("headset disconnected: %s", peer)

    def active(self) -> bool:
        return bool(self.clients) and time.monotonic() - self.last_message < self.timeout

    async def gripper_loop(self):
        was_active = None
        next_active_pub = 0.0
        while True:
            await asyncio.sleep(self.gripper_period)
            active = self.active()
            now = time.monotonic()
            if active != was_active or now >= next_active_pub:  # on change, and at 2 Hz
                self.node.publish_active(active)
                was_active, next_active_pub = active, now + 0.5
            if not self.clients:
                continue
            pose = self.node.gripper_pose()
            if pose is None:
                continue
            text = protocol.encode_gripper_pose(pose, time.time())
            for ws in list(self.clients):
                try:
                    await ws.send(text)
                except ConnectionClosed:
                    pass

    async def spin_ros(self):
        """Runs ROS callbacks (TF subscriptions) on this event loop, so nothing touches rclpy
        from a second thread."""
        while True:
            for _ in range(100):  # drain what's ready, bounded so the socket stays served
                try:
                    handler, _, _ = self.executor.wait_for_ready_callbacks(timeout_sec=0)
                except (TimeoutException, ShutdownException):
                    break
                handler()
                handler.result()  # re-raise a callback's exception
            await asyncio.sleep(0.002)

    async def run(self, ready: threading.Event | None = None):
        # Small messages at frame rate: no compression, disable Nagle (websockets sets
        # TCP_NODELAY on accepted sockets).
        async with serve(self.handle, self.host, self.port, compression=None, ping_interval=5, ping_timeout=5) as server:
            self.bound_port = next(iter(server.sockets)).getsockname()[1]
            log.info("listening on ws://%s:%d (base %s, tip %s)", self.host, self.bound_port,
                     self.node.world_frame, self.node.tip_frame)
            self._loop = asyncio.get_running_loop()
            self._stop = asyncio.Event()
            if ready:
                ready.set()
            tasks = [asyncio.create_task(self.gripper_loop()), asyncio.create_task(self.spin_ros())]
            await self._stop.wait()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        """Makes run() return; callable from any thread."""
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="listen address (default: all interfaces)")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--world-frame", default="world", help="frame of every pose: the robot's world frame")
    ap.add_argument("--tip-frame", default="gripper_end", help="TF frame sent to the headset as the gripper")
    ap.add_argument("--gripper-rate", type=float, default=30.0, help="Hz")
    ap.add_argument("--timeout", type=float, default=0.25, help="s without a message before /quest/active goes false")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[quest_bridge] %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)  # connects are logged by handle()

    rclpy.init()
    node = QuestBridgeNode(args.world_frame, args.tip_frame)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    bridge = QuestBridge(node, executor, args.host, args.port, args.gripper_rate, args.timeout)
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
