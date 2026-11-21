"""End-to-end: one teleop node drives the two simulated reBot arms as one robot, one hand each.

Starts the rebot_dual_arm_gemini_305 simulation headless and moves both arms to cartesian_demo's START,
then runs //src/teleop/controller:teleop --model rebot_dual_arm_gemini_305 (one node, one trajectory
controller, one planner over both arms) while this test plays the headset (/quest/active,
/quest/{left,right}/pose in the world frame and /quest/{left,right}/joy, as //src/teleop/quest_bridge publishes
them). Checks, on the tool pose each arm is being commanded to (TF teleop_command_<hand> in the world frame), that:
  - the left hand with its trigger held moves the left arm's tool by the hand's displacement and leaves the
    right arm alone, and the right hand likewise the right arm,
  - both hands engaged at once, moving apart, move both arms (each by its own hand's displacement),
  - a hand that moves its arm's tool through the other arm stops short of it (the arms don't collide),
and that Ctrl-C sends both arms home (all joints at zero) in one collision-free trip before the node exits.
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
from tf2_msgs.msg import TFMessage

from src.ros2_control_sim import teleop_node
from src.ros2_control_sim.sim import Simulation
from src.trajectory_generator.cartesian_demo import START
from src.trajectory_generator.cartesian_path import Pose
from src.trajectory_generator.client import TrajectoryClient

MODEL = "rebot_dual_arm_gemini_305"
HANDS = ("left", "right")
POSITION_TOLERANCE = 0.003  # m, commanded tool vs expected
HZ = 72.0
# The operator's hands in the world frame, apart by the arms' spacing.
HAND = {"left": Pose((0.1, 0.2, 0.9), (0.0, 0.0, 0.0, 1.0)), "right": Pose((0.1, -0.2, 0.9), (0.0, 0.0, 0.0, 1.0))}
DELTA = {"left": (0.04, 0.03, -0.03), "right": (0.03, -0.04, 0.02)}


def moved(pose, d, s=1.0):
    return Pose(tuple(p + s * v for p, v in zip(pose.position, d)), pose.quaternion)


class Headset:
    """Publishes both hands' pose/trigger at HZ from its own thread and node, as quest_bridge does."""

    def __init__(self):
        self.node = rclpy.create_node("fake_headset")
        self.pose_pub = {h: self.node.create_publisher(PoseStamped, f"/quest/{h}/pose", 10) for h in HANDS}
        self.joy_pub = {h: self.node.create_publisher(Joy, f"/quest/{h}/joy", 10) for h in HANDS}
        self.active_pub = self.node.create_publisher(Bool, "/quest/active", 10)
        self.lock = threading.Lock()
        self.pose = dict(HAND)
        self.trigger = {h: 0.0 for h in HANDS}
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            with self.lock:
                poses, triggers = dict(self.pose), dict(self.trigger)
            for hand in HANDS:
                msg = PoseStamped()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.header.frame_id = "world"
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = poses[hand].position
                o = msg.pose.orientation
                o.x, o.y, o.z, o.w = poses[hand].quaternion
                self.pose_pub[hand].publish(msg)
                self.joy_pub[hand].publish(Joy(axes=[triggers[hand], 0.0, 0.0, 0.0], buttons=[0, 0, 0]))
            self.active_pub.publish(Bool(data=True))
            time.sleep(1 / HZ)

    def set(self, hand, pose=None, trigger=None):
        with self.lock:
            if pose is not None:
                self.pose[hand] = pose
            if trigger is not None:
                self.trigger[hand] = trigger

    def play(self, duration, deltas, triggers):
        """Moves each hand of `deltas` ({hand: delta}) from HAND by delta over `duration`, triggers held."""
        for hand, trigger in triggers.items():
            self.set(hand, trigger=trigger)
        time.sleep(0.4)  # the node engages (the clutch's reference is the hand pose then) before the hand moves
        t0 = time.monotonic()
        while (s := (time.monotonic() - t0) / duration) < 1.0:
            for hand, delta in deltas.items():
                self.set(hand, moved(HAND[hand], delta, s))
            time.sleep(1 / (2 * HZ))
        for hand, delta in deltas.items():
            self.set(hand, moved(HAND[hand], delta))

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.node.destroy_node()


class Commanded:
    """The tool poses the teleop node is commanding, from its TF frames teleop_command_<hand> (in the world frame).

    /tf also carries robot_state_publisher's frames at the joint states' rate, so the subscription's queue
    holds a few hundred ms of older messages: a position is the first one *stamped* after the call."""

    def __init__(self, node):
        self.node = node
        self.latest = {}  # hand -> (stamp ns, position)
        node.create_subscription(TFMessage, "/tf", self._on_tf, 2000)

    def _on_tf(self, msg):
        for t in msg.transforms:
            for hand in HANDS:
                if t.child_frame_id == f"teleop_command_{hand}":
                    p = t.transform.translation
                    self.latest[hand] = (t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec, (p.x, p.y, p.z))

    def position(self, hand, timeout=3.0):
        since = self.node.get_clock().now().nanoseconds
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if hand in self.latest and self.latest[hand][0] > since:
                return self.latest[hand][1]
        raise AssertionError(f"no teleop_command_{hand} TF")


def distance(a, b):
    return math.dist(a, b)


class RebotDualArmTeleopTest(unittest.TestCase):
    def assertNear(self, actual, expected, what):
        print(f"{what}: {distance(actual, expected) * 1e3:.2f} mm", flush=True)
        self.assertLess(distance(actual, expected), POSITION_TOLERANCE, what)

    def start_both_arms(self, node):
        client = TrajectoryClient(node)  # one controller over both arms
        goal = {f"{hand}_{joint}": q for hand in HANDS for joint, q in START.items()}
        client.move_to(goal, duration=2.0)
        client.wait_until_at(goal, 0.005, 3.0)

    def test_swap_hands_drives_each_arm_with_the_other_controller(self):
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_dual_arm_swap_test")
                self.start_both_arms(node)
                teleop = teleop_node.popen("--model", MODEL, "--swap-hands")
                headset = Headset()
                commanded = Commanded(node)
                time.sleep(5.0)
                start = {h: commanded.position(h) for h in HANDS}

                # The left controller now drives the right arm (facing the robot, it is on your left).
                headset.play(1.0, {"left": DELTA["left"]}, {"left": 1.0, "right": 0.0})
                time.sleep(0.5)
                self.assertNear(commanded.position("right"), moved(Pose(start["right"], (0, 0, 0, 1)), DELTA["left"]).position,
                                "left controller: right arm followed")
                self.assertNear(commanded.position("left"), start["left"], "left controller: left arm stayed")
                headset.set("left", trigger=0.0)

                # And the right controller the left arm.
                headset.play(1.0, {"right": DELTA["right"]}, {"right": 1.0})
                time.sleep(0.5)
                self.assertNear(commanded.position("left"), moved(Pose(start["left"], (0, 0, 0, 1)), DELTA["right"]).position,
                                "right controller: left arm followed")
            finally:
                if headset:
                    headset.close()
                if teleop and teleop.poll() is None:
                    teleop.kill()
                rclpy.shutdown()

    def test_arms_move_from_the_home_pose(self):
        """From all joints at zero (the arms folded, each with a self-contact of its own), the other arm's
        contact must not block this arm's motion."""
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_dual_arm_home_test")
                teleop = teleop_node.popen("--model", MODEL)
                headset = Headset()
                commanded = Commanded(node)
                time.sleep(5.0)
                start = {h: commanded.position(h) for h in HANDS}
                for hand, delta in (("left", (0.0, 0.0, 0.04)), ("right", (0.0, 0.0, 0.04))):
                    headset.play(1.0, {hand: delta}, {hand: 1.0})
                    time.sleep(0.5)
                    moved_to = commanded.position(hand)
                    print(f"{hand} tool moved {math.dist(moved_to, start[hand]) * 1e3:.1f} mm of {math.hypot(*delta) * 1e3:.0f}", flush=True)
                    self.assertGreater(moved_to[2] - start[hand][2], 0.02, f"the {hand} arm should rise from the home pose")
                    headset.set(hand, trigger=0.0, pose=HAND[hand])
                    time.sleep(0.5)
            finally:
                if headset:
                    headset.close()
                if teleop and teleop.poll() is None:
                    teleop.kill()
                rclpy.shutdown()

    def test_an_arm_stops_before_hitting_the_other(self):
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_dual_arm_collision_test")
                self.start_both_arms(node)
                teleop = teleop_node.popen("--model", MODEL)
                headset = Headset()
                commanded = Commanded(node)
                time.sleep(5.0)
                start = {h: commanded.position(h) for h in HANDS}
                gap = start["left"][1] - start["right"][1]  # the tools' spacing in y (the arms' 25 cm)
                self.assertAlmostEqual(gap, 0.25, delta=0.01)

                # The left hand pushes the left tool 0.30 m toward -y: through the right arm's tool.
                headset.play(3.0, {"left": (0.0, -0.30, 0.0)}, {"left": 1.0, "right": 0.0})
                time.sleep(1.0)
                left, right = commanded.position("left"), commanded.position("right")
                print(f"left tool y {left[1]:.3f} (target {start['left'][1] - 0.30:.3f}), right tool y {right[1]:.3f}", flush=True)
                self.assertNear(right, start["right"], "the right arm stayed")
                self.assertLess(left[1], start["left"][1] - 0.02, "the left arm moved toward the right one")
                # It stopped short of the other arm's tool (the tools are ~ a gripper wide), not through it.
                self.assertGreater(left[1], right[1] + 0.03, "the left tool stopped before the right arm")
            finally:
                if headset:
                    headset.close()
                if teleop and teleop.poll() is None:
                    teleop.kill()
                rclpy.shutdown()

    def test_each_hand_drives_its_own_arm_and_ctrl_c_sends_both_home(self):
        with Simulation(MODEL, headless=True):
            rclpy.init()
            teleop = headset = None
            try:
                node = rclpy.create_node("rebot_dual_arm_teleop_test")
                self.start_both_arms(node)

                teleop = teleop_node.popen("--model", MODEL)
                headset = Headset()
                commanded = Commanded(node)
                time.sleep(5.0)  # the nodes (a model and a planner each) start and see the inputs
                start = {h: commanded.position(h) for h in HANDS}

                # Only the left trigger: the left arm follows the left hand; the right arm stays.
                headset.play(1.0, {"left": DELTA["left"]}, {"left": 1.0, "right": 0.0})
                time.sleep(0.5)
                left_first = moved(Pose(start["left"], (0, 0, 0, 1)), DELTA["left"]).position
                self.assertNear(commanded.position("left"), left_first, "left hand engaged: left arm followed")
                self.assertNear(commanded.position("right"), start["right"], "left hand engaged: right arm stayed")

                # Release the left hand; only the right trigger: the right arm follows, the left stays.
                headset.set("left", trigger=0.0)
                headset.play(1.0, {"right": DELTA["right"]}, {"right": 1.0})
                time.sleep(0.5)
                right_first = moved(Pose(start["right"], (0, 0, 0, 1)), DELTA["right"]).position
                self.assertNear(commanded.position("right"), right_first, "right hand engaged: right arm followed")
                self.assertNear(commanded.position("left"), left_first, "right hand engaged: left arm stayed")

                # Both at once, moving apart: each by its own hand's displacement (the left hand moves again
                # from where it stopped, the right hand back and then on).
                headset.set("right", trigger=0.0)
                headset.set("left", HAND["left"])
                headset.set("right", HAND["right"])
                time.sleep(0.3)
                apart = {"left": (0.0, 0.05, 0.0), "right": (0.0, -0.05, 0.0)}
                headset.play(1.0, apart, {"left": 1.0, "right": 1.0})
                time.sleep(0.5)
                self.assertNear(commanded.position("left"), moved(Pose(left_first, (0, 0, 0, 1)), apart["left"]).position, "both engaged: left")
                self.assertNear(commanded.position("right"), moved(Pose(right_first, (0, 0, 0, 1)), apart["right"]).position, "both engaged: right")
                headset.set("left", trigger=0.0)
                headset.set("right", trigger=0.0)

                # Ctrl-C: the node plans both arms home and exits; the joints end at zero.
                positions = {}
                node.create_subscription(JointState, "/joint_states", lambda m: positions.update(zip(m.name, m.position)), 10)
                teleop.send_signal(signal.SIGINT)
                deadline = time.monotonic() + 60
                while teleop.poll() is None and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                self.assertEqual(teleop.poll(), 0, "the teleop node should exit by itself after the trip home")
                end = time.monotonic() + 2.0
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.05)
                for hand in HANDS:
                    for i in range(1, 7):
                        joint = f"{hand}_joint{i}"
                        self.assertAlmostEqual(positions[joint], 0.0, delta=0.03, msg=joint)
            finally:
                if headset:
                    headset.close()
                if teleop and teleop.poll() is None:
                    teleop.kill()
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
