"""How closely the simulated reBot follows streamed setpoints, as the teleop node commands it.

The teleop node streams a one-point JointTrajectory every 5 ms (200 Hz) whose point is the next
IK solution, reached in --lookahead (20 ms) from the controller's setpoint. This does the same
along a raised cosine q0 + A (1 - cos 2 pi f t) on joints 1-3 (it stays on the positive side of
the start pose: joints 2 and 3 can't go below it) and reports, per simulation variant, how far the measured joints
lag the reference (the delay that best aligns them), their amplitude, the RMS tracking error, and
the simulation's speed relative to wall time.

  bazel run //src/ros2_control_sim:teleop_tracking_diagnostic -- [--models a b ...] [--jtc-config PATH]

--jtc-config: a controllers.yaml with a joint_trajectory_controller (position), added as the
variant "rebot_arm_jtc" (the reBot's controllers before joint_impedance_controller:
`git show 9144937^:src/ros2_control_sim/config/rebot_arm/controllers.yaml > /tmp/jtc.yaml`).
"""

import argparse
import math
import os
import time
import types

import rclpy
from builtin_interfaces.msg import Duration
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from src.robot_model.rebot_arm import description as rebot_arm_description
from src.ros2_control_sim import models
from src.ros2_control_sim.sim import Simulation

JOINTS = [f"joint{i}" for i in range(1, 8)]
MOVING = ["joint1", "joint2", "joint3"]


def thread_cpu(pid):
    """CPU seconds used so far by each thread of a process (utime + stime from /proc)."""
    out = {}
    tick = os.sysconf("SC_CLK_TCK")
    for task in os.listdir(f"/proc/{pid}/task"):
        with open(f"/proc/{pid}/task/{task}/stat") as f:
            stat = f.read()
        fields = stat[stat.rindex(")") + 2 :].split()
        out[task] = (int(fields[11]) + int(fields[12])) / tick
    return out


def run(model, args):
    with Simulation(model, headless=True) as sim:
        rclpy.init()
        try:
            node = rclpy.create_node("teleop_tracking_diagnostic")
            states = []  # (wall, {joint: position})
            clock = []
            node.create_subscription(
                JointState, "/joint_states", lambda m: states.append((time.monotonic(), dict(zip(m.name, m.position)))), 50
            )
            node.create_subscription(Clock, "/clock", lambda m: clock.append((time.monotonic(), m.clock.sec + m.clock.nanosec * 1e-9)), 50)
            pub = node.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
            while not states or pub.get_subscription_count() == 0:
                rclpy.spin_once(node, timeout_sec=0.1)
            deadline = time.monotonic() + 1.0  # settle
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.01)
            q0 = {j: states[-1][1][j] for j in JOINTS}

            def reference(t):
                return {j: q0[j] + (args.amplitude * (1 - math.cos(2 * math.pi * args.frequency * t)) if j in MOVING else 0.0) for j in JOINTS}

            states.clear()
            clock.clear()
            cpu_before = thread_cpu(sim.control_node.pid)
            start = time.monotonic()
            next_send = start
            while time.monotonic() - start < args.duration:
                now = time.monotonic()
                if now >= next_send:
                    target = reference(now - start + args.lookahead)
                    msg = JointTrajectory(joint_names=JOINTS)
                    point = JointTrajectoryPoint(positions=[target[j] for j in JOINTS])
                    point.time_from_start = Duration(sec=0, nanosec=int(args.lookahead * 1e9))
                    msg.points = [point]
                    pub.publish(msg)
                    next_send += 1.0 / args.rate
                rclpy.spin_once(node, timeout_sec=0.0005)

            elapsed = time.monotonic() - start
            cpu_after = thread_cpu(sim.control_node.pid)
            busiest = sorted(((cpu_after[t] - cpu_before.get(t, 0.0)) / elapsed for t in cpu_after), reverse=True)[:3]

            # Delay: the shift of the reference that best explains the measured joints.
            samples = [(t - start, s) for t, s in states if t - start > 1.0]  # after the first second
            best = None
            for delay_ms in range(0, 400, 2):
                delay = delay_ms / 1000.0
                err = 0.0
                for t, s in samples:
                    ref = reference(t - delay)
                    err += sum((s[j] - ref[j]) ** 2 for j in MOVING)
                err = (err / len(samples) / len(MOVING)) ** 0.5
                if best is None or err < best[1]:
                    best = (delay, err)
            rms_now = (sum(sum((s[j] - reference(t)[j]) ** 2 for j in MOVING) for t, s in samples) / len(samples) / len(MOVING)) ** 0.5
            amplitude = max(abs(s["joint1"] - q0["joint1"]) for _, s in samples)
            sim_speed = (clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0]) if len(clock) > 1 else float("nan")
            print(
                f"{model.name:24s} lag {best[0] * 1000:5.0f} ms (rms {best[1]:.4f} rad at that lag; "
                f"{rms_now:.4f} rad vs the reference now), joint1 travelled {amplitude:.3f} of {2 * args.amplitude} rad, "
                f"sim speed {sim_speed:.2f}x, busiest threads {', '.join(f'{100 * c:.0f}%' for c in busiest)} of a core",
                flush=True,
            )
        finally:
            rclpy.shutdown()


def probe(seconds):
    """Rates of a running simulation (this does not move it): its speed against wall time, and
    the rates of /joint_states and the controller's state."""
    from control_msgs.msg import JointTrajectoryControllerState

    rclpy.init()
    try:
        node = rclpy.create_node("sim_probe")
        clock, joint_states, controller_state = [], [], []
        node.create_subscription(Clock, "/clock", lambda m: clock.append((time.monotonic(), m.clock.sec + m.clock.nanosec * 1e-9)), 100)
        node.create_subscription(JointState, "/joint_states", lambda m: joint_states.append(time.monotonic()), 100)
        node.create_subscription(
            JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state", lambda m: controller_state.append(time.monotonic()), 100
        )
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
        if len(clock) > 1:
            print(f"sim speed {(clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0]):.2f}x wall time")
        print(f"/clock {len(clock) / seconds:.0f} Hz, /joint_states {len(joint_states) / seconds:.0f} Hz, controller_state {len(controller_state) / seconds:.0f} Hz")
    finally:
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=float, metavar="SECONDS", help="only measure a running simulation for this long")
    ap.add_argument("--models", nargs="+", default=["rebot_arm_gemini_305", "rebot_arm"])
    ap.add_argument("--jtc-config", help="controllers.yaml of the position-controlled reBot, for the variant rebot_arm_jtc")
    ap.add_argument("--amplitude", type=float, default=0.15, help="rad (the motion spans 2 A)")
    ap.add_argument("--frequency", type=float, default=0.5, help="Hz")
    ap.add_argument("--duration", type=float, default=8.0, help="s")
    ap.add_argument("--rate", type=float, default=200.0, help="Hz")
    ap.add_argument("--lookahead", type=float, default=0.02, help="s")
    args = ap.parse_args()
    if args.probe:
        return probe(args.probe)
    for name in args.models:
        if name == "rebot_arm_jtc":
            controllers = types.SimpleNamespace(path=lambda: args.jtc_config)
            model = models.SimModel("rebot_arm_jtc", rebot_arm_description, controllers)
        else:
            model = models.get(name)
        run(model, args)


if __name__ == "__main__":
    main()
