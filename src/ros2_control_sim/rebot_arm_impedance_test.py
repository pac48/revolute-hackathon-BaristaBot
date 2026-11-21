"""End-to-end: the reBot's arm under joint impedance control tracks a joint trajectory.

Starts the rebot_arm simulation (//src/controllers/joint_impedance_controller on joint1-7: the
arm in MIT mode like the real motors, the gripper on effort), and checks that
  0. the arm joints have the real motors' hardware interfaces (rebot_robstride_protocol):
     the MIT commands position, velocity, kp, kd, t_ff, and the states position, velocity, effort;
  1. with no trajectory the arm holds its pose: the gravity feedforward carries the arm, so it
     does not sag (the position-PID model sags ~0.02-0.03 rad on joint2);
  2. a trajectory published as a JointTrajectory message is followed (cubic splines through
     positions + velocities) and ends at the goal;
  3. the tracking error during the move stays small.
"""

import time
import unittest

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import ListHardwareInterfaces
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from src.ros2_control_sim.sim import Simulation

JOINTS = [f"joint{i}" for i in range(1, 7)]
GOAL = [0.3, 0.5, 0.5, 0.2, -0.2, 0.3]
HOLD_TOLERANCE = 0.01  # rad, from the start pose with no trajectory
GOAL_TOLERANCE = 0.01  # rad, after settling
TRACKING_TOLERANCE = 0.06  # rad, to the straight-line-in-time reference during the move
MOVE_TIME = 2.0  # s


MIT_COMMANDS = {"position", "velocity", "kp", "kd", "t_ff"}
STATES = {"position", "velocity", "effort"}


class RebotArmImpedanceTest(unittest.TestCase):
    def test_arm_joints_have_the_real_motors_interfaces(self):
        with Simulation("rebot_arm", headless=True):
            rclpy.init()
            try:
                node = rclpy.create_node("rebot_arm_interfaces_test")
                client = node.create_client(ListHardwareInterfaces, "/controller_manager/list_hardware_interfaces")
                self.assertTrue(client.wait_for_service(timeout_sec=30.0))
                future = client.call_async(ListHardwareInterfaces.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
                response = future.result()
                commands, states = {}, {}
                for interfaces, out in ((response.command_interfaces, commands), (response.state_interfaces, states)):
                    for interface in interfaces:
                        joint, name = interface.name.split("/")
                        out.setdefault(joint, set()).add(name)
                for joint in JOINTS:
                    self.assertEqual(commands[joint], MIT_COMMANDS, joint)
                    self.assertEqual(states[joint], STATES, joint)
                # The gripper is a sim-only slide: effort (and PID position), no MIT gains.
                self.assertEqual(commands["joint7"], {"position", "effort"})
            finally:
                node.destroy_node()
                rclpy.shutdown()

    def spin_for(self, node, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    def test_tracks_trajectory(self):
        with Simulation("rebot_arm", headless=True):
            rclpy.init()
            try:
                node = rclpy.create_node("rebot_arm_impedance_test")
                state = {}
                history = []

                def on_state(msg):
                    state.update(zip(msg.name, msg.position))
                    history.append((time.monotonic(), [state[j] for j in JOINTS]) if all(j in state for j in JOINTS) else None)

                node.create_subscription(JointState, "/joint_states", on_state, 10)
                states = []  # (stamp, message) of ~/controller_state
                node.create_subscription(
                    JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state",
                    lambda m: states.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m)), 100)
                publisher = node.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
                while not all(j in state for j in JOINTS):
                    rclpy.spin_once(node, timeout_sec=0.1)
                start = [state[j] for j in JOINTS]

                # 1. Hold.
                self.spin_for(node, 2.0)
                held = [state[j] for j in JOINTS]
                for j, a, b in zip(JOINTS, start, held):
                    self.assertAlmostEqual(a, b, delta=HOLD_TOLERANCE, msg=f"{j} drifted while holding")

                # 2. Move: rest-to-rest through a midpoint, positions and velocities (cubic).
                def point(fraction, seconds, velocity):
                    p = JointTrajectoryPoint()
                    p.positions = [a + fraction * (g - a) for a, g in zip(start, GOAL)]
                    p.velocities = [velocity * (g - a) for a, g in zip(start, GOAL)]
                    p.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
                    return p

                msg = JointTrajectory()
                msg.joint_names = list(reversed(JOINTS))  # any order is accepted
                points = [point(0.5, MOVE_TIME / 2, 1.0 / MOVE_TIME * 1.5), point(1.0, MOVE_TIME, 0.0)]
                for p in points:
                    for field in ("positions", "velocities"):
                        setattr(p, field, list(reversed(getattr(p, field))))
                msg.points = points
                while publisher.get_subscription_count() == 0:
                    rclpy.spin_once(node, timeout_sec=0.1)
                history.clear()
                states.clear()
                t0 = time.monotonic()
                publisher.publish(msg)
                self.spin_for(node, MOVE_TIME + 2.0)

                # 3. End at the goal; the arm moved through the middle, close to the reference.
                final = [state[j] for j in JOINTS]
                for j, q, g in zip(JOINTS, final, GOAL):
                    self.assertAlmostEqual(q, g, delta=GOAL_TOLERANCE, msg=f"{j} did not reach the goal")
                mid = min(
                    (h for h in history if h and h[0] - t0 > MOVE_TIME / 2),
                    key=lambda h: h[0],
                )
                worst = max(abs(q - (a + 0.5 * (g - a))) for q, a, g in zip(mid[1], start, GOAL))
                print(f"error at the midpoint (wall clock, so approximate): {worst:.3f} rad", flush=True)
                self.assertLess(worst, TRACKING_TOLERANCE)

                # 4. The output comes from the fitted curve: its velocity is its position's rate of
                # change, it accelerates, and its effort is the control law's (gravity at least). The
                # reference is the targets alone: positions, ending at the goal.
                outputs = [(t, m.output) for t, m in states]
                moving = [(t, r) for t, r in outputs if max(abs(v) for v in r.velocities) > 0.05]
                self.assertGreater(len(moving), 10, "the output velocity should be nonzero while moving")
                for (t1, r1), (t2, r2) in zip(moving, moving[1:]):
                    if not 0 < t2 - t1 < 0.05:
                        continue
                    for i in range(len(JOINTS)):
                        rate = (r2.positions[i] - r1.positions[i]) / (t2 - t1)
                        mean = 0.5 * (r1.velocities[i] + r2.velocities[i])
                        self.assertAlmostEqual(rate, mean, delta=0.02 + 0.05 * abs(mean), msg=f"output velocity of {JOINTS[i]}")
                self.assertGreater(max(abs(a) for _, r in moving for a in r.accelerations), 0.1)
                self.assertGreater(max(abs(e) for _, r in outputs for e in r.effort), 0.1)
                self.assertTrue(all(v == 0.0 for _, m in states for v in m.reference.velocities))
                final_reference = dict(zip(states[-1][1].joint_names, states[-1][1].reference.positions))
                for j, g in zip(JOINTS, GOAL):
                    self.assertAlmostEqual(final_reference[j], g, delta=1e-6, msg=f"reference of {j} at the end")
                peak = max(abs(v) for _, r in moving for v in r.velocities)
                print(f"output: {len(moving)} moving samples, peak velocity {peak:.2f} rad/s, "
                      f"peak effort {max(abs(e) for _, r in outputs for e in r.effort):.2f}", flush=True)
            finally:
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
