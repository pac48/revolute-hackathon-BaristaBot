"""End-to-end: the clutched VR teleop node drives the simulated reBot.

Starts the rebot_arm_gemini_305 simulation headless and moves it to cartesian_demo's START, then
runs //src/teleop/controller:teleop while this test plays the headset
(/quest/active, /quest/right/pose and /quest/right/joy as //src/teleop/quest_bridge publishes
them, at 72 Hz). Checks, on the controller's setpoint (FK of its reference), that:
  - moving the controller with the trigger released moves nothing,
  - with the trigger held, the tool follows the controller's displacement,
  - after releasing, moving the hand back, and engaging again, it continues from where
    it stopped (ratcheting), and follows the controller's rotation,
  - squeezing the grip closes the gripper,
  - the target is persistent: when the arm lags it (the hand jumped, then released), engaging
    again keeps closing the gap to the same target,
  - pressing A snaps the target to the end effector, so engaging again doesn't move the arm.
test_wrist_roll_stops_short_of_joint_limit rolls the wrist ~4.5 rad: joint6 never goes past its
+-3.14 limit (IK's box constraints, or IK moves to another branch; the measured joint may overshoot
slightly) and
joint_trajectory_controller stays active (ros2_control's limiter used to deactivate it, dropping
the arm).
"""

import math
import signal
import threading
import time
import unittest

import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Bool

from src.ros2_control_sim import teleop_node
from src.ros2_control_sim.sim import Simulation
from src.trajectory_generator.cartesian_demo import START
from src.trajectory_generator.cartesian_path import Pose, quaternion_multiply
from src.trajectory_generator import kinematics_client
from src.trajectory_generator.client import TrajectoryClient

MODEL = "rebot_arm_gemini_305"  # the sim and the teleop node run this model

POSITION_TOLERANCE = 0.002  # m, setpoint vs expected
ANGLE_TOLERANCE = 0.01  # rad
HZ = 72.0
# The operator's hand, somewhere in the base frame, held at an arbitrary orientation.
HAND = Pose((0.1, -0.4, 0.9), (0.2, -0.1, 0.3, 0.9))
DELTA = (0.04, 0.03, -0.03)


def yaw(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


class Headset:
    """Publishes the current pose/trigger/grip at HZ from its own thread and node, as
    quest_bridge does, so the stream never pauses while the test waits or measures."""

    def __init__(self):
        self.node = rclpy.create_node("fake_headset")
        self.pose_pub = self.node.create_publisher(PoseStamped, "/quest/right/pose", 10)
        self.joy_pub = self.node.create_publisher(Joy, "/quest/right/joy", 10)
        self.active_pub = self.node.create_publisher(Bool, "/quest/active", 10)
        self.lock = threading.Lock()
        self.pose, self.trigger, self.grip, self.a = HAND, 0.0, 0.0, 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            with self.lock:
                p, trigger, grip, a = self.pose, self.trigger, self.grip, self.a
            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p.position
            o = msg.pose.orientation
            o.x, o.y, o.z, o.w = p.quaternion
            self.pose_pub.publish(msg)
            self.joy_pub.publish(Joy(axes=[trigger, grip, 0.0, 0.0], buttons=[a, 0, 0]))
            self.active_pub.publish(Bool(data=True))
            time.sleep(1 / HZ)

    def set(self, pose=None, trigger=None, grip=None, a=None):
        with self.lock:
            self.pose = pose if pose is not None else self.pose
            self.trigger = trigger if trigger is not None else self.trigger
            self.grip = grip if grip is not None else self.grip
            self.a = a if a is not None else self.a

    def hold(self, seconds, pose=None, trigger=None, grip=None, a=None):
        self.set(pose, trigger, grip, a)
        time.sleep(seconds)

    def play(self, duration, pose_at, trigger=None, grip=None):
        """Moves through pose_at(s), s from 0 to 1, over `duration`."""
        self.set(pose_at(0.0), trigger, grip)
        t0 = time.monotonic()
        while (s := (time.monotonic() - t0) / duration) < 1.0:
            self.set(pose_at(s))
            time.sleep(1 / (2 * HZ))
        self.set(pose_at(1.0))

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.node.destroy_node()


def commanded(client):
    """The tool at the controller's current setpoint (fresh, not a cached message)."""
    client._references.clear()
    return client.current_pose(MODEL, commanded=True)


class TargetWatcher:
    """The latest /teleop/target_pose, fresh: the first message stamped after the call (the
    subscription's queue may still hold older ones)."""

    def __init__(self, node):
        self.node = node
        self.msgs = []
        node.create_subscription(PoseStamped, "/teleop/target_pose", self.msgs.append, 10)

    def latest(self, timeout=2.0) -> Pose:
        since = self.node.get_clock().now().nanoseconds
        self.msgs.clear()

        def fresh():
            return [m for m in self.msgs if m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec >= since]

        deadline = time.monotonic() + timeout
        while not fresh() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        msgs = fresh()
        assert msgs, "no fresh /teleop/target_pose"
        p, q = msgs[0].pose.position, msgs[0].pose.orientation
        return Pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def axis_angle(axis, angle):
    s = math.sin(angle / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def rotate(q, v):
    x, y, z, w = q
    t = (2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0]))
    return (v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0]))


def moved(pose, d, scale=1.0, rotation=(0.0, 0.0, 0.0, 1.0)):
    return Pose(tuple(p + scale * v for p, v in zip(pose.position, d)), quaternion_multiply(rotation, pose.quaternion))


class RebotArmTeleopTest(unittest.TestCase):
    def assertPose(self, actual, expected, what):
        print(f"{what}: {actual.distance(expected) * 1e3:.2f} mm, {math.degrees(actual.angle(expected)):.2f} deg",
              flush=True)
        self.assertLess(actual.distance(expected), POSITION_TOLERANCE, what)
        self.assertLess(actual.angle(expected), ANGLE_TOLERANCE, what)

    def test_clutch_teleop(self):
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_arm_teleop_test")
                client = TrajectoryClient(node)
                client.move_to(START, duration=2.0)
                client.wait_until_at(START, 0.005, 3.0)
                start = commanded(client)

                teleop = teleop_node.popen("--model", MODEL)
                headset = Headset()
                headset.hold(2.0, HAND)  # let the node start and see the inputs

                # Released: moving the hand does nothing.
                headset.play(0.5, lambda s: moved(HAND, DELTA, s))
                headset.hold(0.5, moved(HAND, DELTA))
                self.assertPose(commanded(client), start, "released, hand moved")

                # Engage with the hand back at HAND; move by DELTA.
                headset.hold(0.3, HAND)
                headset.hold(0.3, HAND, trigger=1.0)
                headset.play(1.0, lambda s: moved(HAND, DELTA, s), trigger=1.0)
                headset.hold(0.5, moved(HAND, DELTA), trigger=1.0)
                first = moved(start, DELTA)
                self.assertPose(commanded(client), first, "engaged, followed DELTA")

                # Release, bring the hand back and squeeze the grip: the arm stays, the gripper closes.
                headset.hold(0.3, moved(HAND, DELTA), trigger=0.0)
                headset.play(0.5, lambda s: moved(HAND, DELTA, 1 - s), grip=1.0)
                headset.hold(1.0, HAND, grip=1.0)
                self.assertPose(commanded(client), first, "released, hand back")
                client._references.clear()
                self.assertAlmostEqual(client.commanded_positions(["joint7"])[0], 0.0, delta=0.002)

                # Engage again: DELTA once more and a 0.3 rad turn about base z.
                turn = yaw(0.3)
                headset.hold(0.3, HAND, trigger=1.0)
                headset.play(1.0, lambda s: moved(HAND, DELTA, s, yaw(0.3 * s)), trigger=1.0)
                headset.hold(0.5, moved(HAND, DELTA, 1.0, turn), trigger=1.0)
                second = Pose(moved(first, DELTA).position, quaternion_multiply(turn, start.quaternion))
                self.assertPose(commanded(client), second, "ratcheted + turned")

                # The measured arm got there too (within the sim's tracking/sag).
                self.assertLess(client.current_pose(MODEL).distance(second), 0.015)

                # Persistent target: jump the hand 10 cm and release within 50 ms. The target
                # jumps; the arm, speed-limited to 1 m/s, gets only part of the way.
                watcher = TargetWatcher(node)
                jump = (-0.10, 0.0, 0.0)
                headset.hold(0.3, HAND, trigger=0.0)
                headset.hold(0.3, HAND, trigger=1.0)
                headset.hold(0.05, moved(HAND, jump), trigger=1.0)
                headset.hold(0.3, moved(HAND, jump), trigger=0.0)
                third = moved(second, jump)
                self.assertPose(watcher.latest(), third, "target after the jump")
                lag = commanded(client).distance(third)
                print(f"arm behind the target after release: {lag * 1e3:.1f} mm", flush=True)
                self.assertGreater(lag, 0.02)
                # Hand back, engage again holding still: the arm finishes reaching the same target.
                headset.hold(0.3, HAND, trigger=0.0)
                headset.hold(1.0, HAND, trigger=1.0)
                self.assertPose(commanded(client), third, "engaged again, caught up with the target")

                # Reset: make the target run ahead again, then press A while released.
                headset.hold(0.3, HAND, trigger=0.0)
                headset.hold(0.3, HAND, trigger=1.0)
                headset.hold(0.05, moved(HAND, (0.0, 0.0, 0.08)), trigger=1.0)
                headset.hold(0.3, moved(HAND, (0.0, 0.0, 0.08)), trigger=0.0)
                here = commanded(client)
                self.assertGreater(watcher.latest().distance(here), 0.02, "target should be ahead of the arm")
                headset.hold(0.2, HAND, a=1)
                headset.hold(0.2, HAND, a=0)
                self.assertPose(watcher.latest(), here, "target reset to the end effector")
                headset.hold(1.0, HAND, trigger=1.0)  # engaging doesn't move the arm
                self.assertPose(commanded(client), here, "engaged after reset, arm stayed")
            finally:
                if headset:
                    headset.close()
                if teleop:
                    teleop.send_signal(signal.SIGINT)
                    try:
                        code = teleop.wait(timeout=5)
                    except Exception:
                        teleop.kill()
                        raise
                    self.assertEqual(code, 0, "teleop exit code")
                rclpy.shutdown()

    def test_wrist_roll_stops_short_of_joint_limit(self):
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_arm_teleop_limit_test")
                client = TrajectoryClient(node)
                client.move_to(START, duration=2.0)
                client.wait_until_at(START, 0.005, 3.0)
                start = commanded(client)
                joint6 = []

                def record(msg: JointState):
                    if "joint6" in msg.name:
                        joint6.append(msg.position[msg.name.index("joint6")])

                node.create_subscription(JointState, "/joint_states", record, 100)

                teleop = teleop_node.popen("--model", MODEL)
                headset = Headset()
                headset.hold(2.0, HAND)
                # Roll about the tool's approach axis (its x, in the base frame): joint6.
                axis = rotate(start.quaternion, (1.0, 0.0, 0.0))
                for _ in range(5):  # 5 x 0.9 rad, past the +-3.14 limit
                    headset.hold(0.3, HAND, trigger=0.0)
                    headset.hold(0.3, HAND, trigger=1.0)
                    headset.play(0.6, lambda s: Pose(HAND.position,
                                                     quaternion_multiply(axis_angle(axis, 0.9 * s), HAND.quaternion)),
                                 trigger=1.0)
                    headset.hold(0.4, trigger=1.0)
                    end = time.monotonic() + 0.1
                    while time.monotonic() < end:
                        rclpy.spin_once(node, timeout_sec=0.01)
                headset.hold(0.5, HAND, trigger=0.0)
                end = time.monotonic() + 0.5
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.01)

                peak = max(abs(q) for q in joint6)
                print(f"joint6: peak |measured| {peak:.4f} rad over {len(joint6)} states (limit 3.14)", flush=True)
                # Near the limit IK may continue on another branch (other joints taking over the roll).
                self.assertGreater(peak, 2.0, "the roll should drive joint6 far")
                self.assertLess(peak, 3.16, "joint6 went well past its limit")
                # The controller is still active: a full move back must succeed.
                client.move_to(START, duration=3.0)
                client.wait_until_at(START, 0.01, 3.0)
            finally:
                if headset:
                    headset.close()
                if teleop:
                    teleop.send_signal(signal.SIGINT)
                    try:
                        code = teleop.wait(timeout=5)
                    except Exception:
                        teleop.kill()
                        raise
                    self.assertEqual(code, 0, "teleop exit code")
                rclpy.shutdown()

    def test_stops_before_self_collision(self):
        """Driving the gripper back into the arm: the command stops at the last collision-free
        point of the joint-space path, short of the target."""
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_arm_teleop_collision_test")
                client = TrajectoryClient(node)
                client.move_to(START, duration=2.0)
                client.wait_until_at(START, 0.005, 3.0)
                start = commanded(client)
                teleop = teleop_node.popen("--model", MODEL)
                headset = Headset()
                headset.hold(2.0, HAND)
                # Back and up: reachable all the way, but link3 hits link5 ~83% of the way along.
                into_arm = (-0.2, 0.0, 0.1)
                headset.hold(0.3, HAND, trigger=1.0)
                headset.play(2.5, lambda s: moved(HAND, into_arm, s), trigger=1.0)
                headset.hold(1.5, moved(HAND, into_arm), trigger=1.0)
                headset.hold(0.3, trigger=0.0)

                target = moved(start, into_arm)
                reached = commanded(client)
                print(f"commanded {reached.distance(target) * 1e3:.0f} mm short of the target", flush=True)
                self.assertGreater(reached.distance(target), 0.01, "should have stopped before the collision")
                self.assertGreater(reached.distance(start), 0.1, "should have moved most of the way first")
                # The commanded configuration is collision-free.
                joints = [f"joint{i}" for i in range(1, 7)]
                client._references.clear()
                q = dict(zip(joints, client.commanded_positions(joints)))
                path = kinematics_client.ik_path(MODEL, q, [kinematics_client.forward(MODEL, q)])
                self.assertEqual(len(path.positions), 1)
            finally:
                if headset:
                    headset.close()
                if teleop:
                    teleop.send_signal(signal.SIGINT)
                    try:
                        code = teleop.wait(timeout=5)
                    except Exception:
                        teleop.kill()
                        raise
                    self.assertEqual(code, 0, "teleop exit code")
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
