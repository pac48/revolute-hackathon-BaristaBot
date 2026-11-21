"""Is the arm's shake from the controller or from the trajectories sent to it?

Usage: bazel run //src/trajectory_generator:shake_diagnostic -- --mode record|sine|sine-long [--duration 15]

Needs the sim (or robot) running. Records /joint_trajectory_controller/controller_state (the controller's reference,
50 Hz) and /joint_states (the measured motion, 500 Hz) for --duration seconds and prints, per joint, how rough each is:
RMS velocity, RMS acceleration, and the share of the velocity's variance above --cutoff Hz (smooth motion has none).

  record     only records: run it while mpc_reach (or teleop) drives the arm.
  sine       drives joint1 and joint3 with a smooth sine (--amplitude, --frequency) as mpc_reach delivers its
             trajectories: every --period s a new 0.32 s trajectory of 20 ms positions-only points, trimmed to start
             from now. The reference is smooth by construction, so any roughness in the measured motion is the
             controller's (or the delivery pattern's).
  sine-long  the same sine as one long trajectory, sent once.

If sine's measured motion is as rough as mpc_reach's while its reference is smooth, the shake is the controller;
if mpc_reach's reference is rough, it is the optimization's output.
"""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def roughness(times: np.ndarray, values: np.ndarray, cutoff: float, smooth: float = 0.0) -> dict:
    """Roughness of a signal sampled at `times` (s, roughly uniform): RMS of its derivative (`values` are
    positions), RMS of the second derivative, and the share of the derivative's variance at frequencies above
    `cutoff` Hz. `smooth` > 0 averages the derivative over that many seconds before differentiating again, so
    the second derivative of a 500 Hz signal is not dominated by quantization."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    dt = float(np.median(np.diff(times)))
    # Uniform resample (messages arrive jittered).
    grid = np.arange(times[0], times[-1], dt)
    x = np.interp(grid, times, values)
    v = np.diff(x) / dt
    if smooth > 0:
        n = max(1, int(round(smooth / dt)))
        v_s = np.convolve(v, np.ones(n) / n, mode="valid")
    else:
        v_s = v
    a = np.diff(v_s) / dt
    spectrum = np.abs(np.fft.rfft(v - v.mean())) ** 2
    freqs = np.fft.rfftfreq(len(v), dt)
    total = spectrum[1:].sum()
    high = spectrum[freqs > cutoff].sum()
    return dict(vel_rms=float(np.sqrt(np.mean(v**2))), acc_rms=float(np.sqrt(np.mean(a**2))),
                high_share=float(high / total) if total > 0 else 0.0)


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


class Diagnostic(Node):
    def __init__(self, args):
        super().__init__("shake_diagnostic")
        self.args = args
        self.lock = threading.Lock()
        self.reference = []  # (t, positions)
        self.measured = []
        self.start = None
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, qos)
        self.create_subscription(JointTrajectoryControllerState, f"/{args.controller}/controller_state",
                                 self.on_state, QoSProfile(depth=50))
        self.pub = self.create_publisher(JointTrajectory, f"/{args.controller}/joint_trajectory", 10)
        self.t0 = time.monotonic()

    def on_joint_states(self, msg):
        t = time.monotonic()
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        if not all(j in pos for j in ARM_JOINTS):
            return
        row = [pos[j] for j in ARM_JOINTS]
        with self.lock:
            if self.start is None:
                self.start = list(row)
            self.measured.append((t, row))

    def on_state(self, msg):
        t = time.monotonic()
        pos = dict(zip(msg.joint_names, msg.reference.positions))
        if all(j in pos for j in ARM_JOINTS):
            with self.lock:
                self.reference.append((t, [pos[j] for j in ARM_JOINTS]))

    def sine(self, t_rel: np.ndarray) -> np.ndarray:
        """Positions (len(t_rel), 6): the starting pose, with joint1 and joint3 on a sine (starting at zero)."""
        out = np.tile(np.array(self.start), (len(t_rel), 1))
        s = self.args.amplitude * np.sin(2 * np.pi * self.args.frequency * t_rel)
        out[:, 0] += s
        out[:, 2] += s
        return out

    def send(self, t_from: float, points: int, dt: float = 0.02):
        """A trajectory of `points` positions-only points at dt, 2 dt, ... from now (sine time t_from)."""
        times = t_from + dt * np.arange(1, points + 1)
        positions = self.sine(times)
        msg = JointTrajectory(joint_names=list(ARM_JOINTS))
        for k in range(points):
            msg.points.append(JointTrajectoryPoint(positions=positions[k].tolist(),
                                                   time_from_start=_duration(dt * (k + 1))))
        self.pub.publish(msg)

    def drive(self):
        """Runs the mode for --duration; returns when done."""
        while self.start is None:
            time.sleep(0.05)
        a = self.args
        t_start = time.monotonic()
        if a.mode == "sine-long":
            self.send(0.0, int(a.duration / 0.02))
        while time.monotonic() - t_start < a.duration:
            if a.mode == "sine":
                self.send(time.monotonic() - t_start, 16)
                time.sleep(a.period)
            else:
                time.sleep(0.1)

    def report(self, t_from: float):
        print(f"\n{'':10s}{'vel RMS':>9s}{'acc RMS':>10s}{'>%g Hz' % self.args.cutoff:>9s}")
        for name, rows, smooth in (("reference", self.reference, 0.0), ("measured", self.measured, 0.02)):
            with self.lock:
                data = [(t, r) for t, r in rows if t >= t_from]
            if len(data) < 20:
                print(f"{name}: too few samples ({len(data)})")
                continue
            times = np.array([t for t, _ in data])
            values = np.array([r for _, r in data])
            print(name)
            for j, joint in enumerate(ARM_JOINTS):
                r = roughness(times, values[:, j], self.args.cutoff, smooth)
                print(f"  {joint:8s}{r['vel_rms']:9.3f}{r['acc_rms']:10.2f}{r['high_share']:9.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["record", "sine", "sine-long"], default="record")
    ap.add_argument("--controller", default="joint_trajectory_controller")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--amplitude", type=float, default=0.15, help="rad (sine modes)")
    ap.add_argument("--frequency", type=float, default=0.4, help="Hz (sine modes)")
    ap.add_argument("--period", type=float, default=0.2, help="s between trajectories (sine mode)")
    ap.add_argument("--cutoff", type=float, default=5.0, help="Hz above which motion counts as shake")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    rclpy.init(args=ros_args)
    node = Diagnostic(args)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    t0 = time.monotonic() + 1.0  # skip the first second
    if args.mode == "record":
        time.sleep(args.duration)
    else:
        node.drive()
    node.report(t0)
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
