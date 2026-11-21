"""Latency diagnostic: point-to-point trajectories against a local sim, then a remote one.

Usage:
  bazel run //src/ros2_control_sim:latency_diagnostic -- [--model rebot_arm] [--trials 10]
      [--only local|remote] [--mode fresh|reuse|both] [--json results.json]

Runs the same trials twice, from this PC:
  local   the sim runs here (headless, in-process launcher)
  remote  the sim runs on the remote build host (tools/remote, the `.remote` setup)
Each phase uses its own ROS_DOMAIN_ID, so they can't see each other or anything
else on the network. Comparing the two shows what the remote sim and the network
add. Per trial it measures, with this PC's clock:

  discovery  new client until the action server and /joint_states are seen
  accept     goal sent until the controller accepts it
  overhead   accepted until the result, minus the planned duration
  error      largest joint error --settle seconds after the result
  lost       goals never accepted within --accept-timeout

Client modes: "fresh" makes a new DDS participant per trial (like the move_to
CLI, so discovery is part of every trial); "reuse" keeps one client (discovery
once). If goals are lost only in "fresh", the problem is discovery.
"""

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import rclpy

from src.robot_model.active_model import active_model
from src.ros2_control_sim import models
from src.ros2_control_sim.sim import Simulation, SimulationError
from src.trajectory_generator.client import TrajectoryClient, TrajectoryError

# Two poses per model to alternate between; joints are the controller's.
POSES = {
    "two_link_arm": [{"joint1": 0.5, "joint2": -0.3}, {"joint1": 0.0, "joint2": 0.0}],
    "rebot_arm": [
        {"joint1": 0.3, "joint2": 0.5, "joint3": 0.5, "joint4": 0.2, "joint5": -0.2, "joint6": 0.3, "joint7": 0.02},
        {"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0, "joint7": 0.0},
    ],
}
POSES["rebot_arm_gemini_305"] = POSES["rebot_arm"]  # the same arm and controllers


@dataclasses.dataclass
class Trial:
    phase: str
    mode: str
    index: int
    discovery: float | None = None
    accept: float | None = None
    overhead: float | None = None
    error: float | None = None
    failure: str = ""  # "" if the move succeeded, else the failing stage


def run_trials(phase, model, mode, args) -> list[Trial]:
    """`args.trials` moves alternating between the model's poses."""
    poses = POSES[model]
    trials = []
    client = node = None

    def new_client():
        rclpy.init()
        n = rclpy.create_node(f"latency_diagnostic_{os.getpid()}")
        return n, TrajectoryClient(n, timeout=args.accept_timeout)

    try:
        for i in range(args.trials):
            trial = Trial(phase, mode, i)
            trials.append(trial)
            goal = poses[i % len(poses)]
            try:
                if client is None:
                    start = time.monotonic()
                    node, client = new_client()
                    client.timeout = args.discovery_timeout
                    client.wait_for_server()
                    client.current_positions(list(goal))
                    trial.discovery = time.monotonic() - start
                    client.timeout = args.accept_timeout
                client.move_to(goal, duration=args.duration)
                timing = client.last_timing
                trial.accept, trial.overhead = timing.accept, timing.overhead
                # Tolerance 0: always wait the full settle time, then measure.
                settled = client.wait_until_at(goal, 0.0, args.settle)
                trial.error = max(abs(settled[j] - goal[j]) for j in goal)
            except TrajectoryError as e:
                trial.failure = e.stage or "error"
                print(f"  {phase}/{mode} trial {i}: {e}", file=sys.stderr)
                if e.stage == "result" and client.last_timing is None:
                    # Unknown state on the controller; let the arm come to rest.
                    time.sleep(args.duration)
            if mode == "fresh" or trial.failure in ("server", "joint_states"):
                if node is not None:
                    node.destroy_node()
                    rclpy.shutdown()
                client = node = None
            print(f"  {phase}/{mode} trial {i}: {_fmt_trial(trial)}", file=sys.stderr, flush=True)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
    return trials


def _fmt_trial(t: Trial) -> str:
    parts = [f"{k}={getattr(t, k) * 1000:.0f}ms" for k in ("discovery", "accept", "overhead") if getattr(t, k) is not None]
    if t.error is not None:
        parts.append(f"error={t.error:.4f}")
    return ", ".join(parts) + (f"  FAILED ({t.failure})" if t.failure else "")


def _parse_env(items: list[str]) -> dict[str, str]:
    env = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"latency_diagnostic: expected KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def run_phase(phase, model, args) -> list[Trial]:
    os.environ["ROS_DOMAIN_ID"] = str(args.local_domain if phase == "local" else args.remote_domain)
    if phase == "remote":
        # Read by rclpy/rmw at each rclpy.init(), i.e. for every new client.
        os.environ.update(_parse_env(args.client_env))
    modes = ["fresh", "reuse"] if args.mode == "both" else [args.mode]
    print(f"== {phase}: {model}, ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}", file=sys.stderr, flush=True)
    if phase == "local":
        with Simulation(model, headless=not args.viewer, timeout=args.start_timeout):
            return [t for mode in modes for t in run_trials(phase, model, mode, args)]
    with RemoteSim(model, args):
        return [t for mode in modes for t in run_trials(phase, model, mode, args)]


class RemoteSim:
    """The sim on the remote build host, via `tools/remote bazel run`.

    The remote launcher runs with --exit-on-stdin-eof; closing the ssh session's
    stdin on exit stops it cleanly instead of orphaning it on the host.
    """

    def __init__(self, model, args):
        self.args = args
        self.cmd = [
            os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], "tools", "remote"),
            "bazel",
            "run",
            f"--run_env=ROS_DOMAIN_ID={args.remote_domain}",
            *(f"--run_env={kv}" for kv in args.remote_env),
            "//src/ros2_control_sim:mujoco_arm_sim",
            "--",
            "--model",
            model,
            "--exit-on-stdin-eof",
            *([] if args.viewer else ["--headless"]),
        ]
        self.log_path = args.remote_log
        self.proc = None

    def __enter__(self):
        print(f"  starting: {' '.join(self.cmd[1:])} (log: {self.log_path})", file=sys.stderr, flush=True)
        log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.environ["BUILD_WORKSPACE_DIRECTORY"],
        )
        ready = threading.Event()

        def pump():
            for line in self.proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith("controllers active"):
                    ready.set()
            log.close()

        threading.Thread(target=pump, daemon=True).start()
        deadline = time.monotonic() + self.args.remote_start_timeout
        while not ready.wait(0.2):
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.__exit__()
                raise SimulationError(f"remote sim did not start (see {self.log_path})")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()  # remote launcher sees EOF and stops its nodes
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(5)


def _stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
    return {"n": len(values), "median": statistics.median(values), "p95": p95, "max": values[-1]}


def summarize(trials: list[Trial], tolerance: float) -> dict:
    groups = {}
    for t in trials:
        groups.setdefault((t.phase, t.mode), []).append(t)
    summary = {}
    for (phase, mode), ts in groups.items():
        summary[f"{phase}/{mode}"] = {
            "trials": len(ts),
            "failures": {f: sum(t.failure == f for t in ts) for f in sorted({t.failure for t in ts} - {""})},
            "misses": sum(t.error is not None and t.error > tolerance for t in ts),
            **{k: _stats(getattr(t, k) for t in ts) for k in ("discovery", "accept", "overhead", "error")},
        }
    return summary


def print_report(summary: dict, tolerance: float):
    def ms(s, key):
        return "-" if s is None else f"{s[key] * 1000:.0f}"

    print("\n{:<14} {:>6} {:>6}  {:>17}  {:>17}  {:>17}  {:>14}".format(
        "", "trials", "failed", "discovery ms", "accept ms", "overhead ms", "error max/miss"))
    print("{:<14} {:>6} {:>6}  {:>17}  {:>17}  {:>17}  {:>14}".format(
        "", "", "", "med / p95 / max", "med / p95 / max", "med / p95 / max", f"tol {tolerance}"))
    for name, s in summary.items():
        failed = sum(s["failures"].values())
        cells = [
            "-" if s[k] is None else f"{ms(s[k], 'median')} / {ms(s[k], 'p95')} / {ms(s[k], 'max')}"
            for k in ("discovery", "accept", "overhead")
        ]
        err = "-" if s["error"] is None else f"{s['error']['max']:.4f} / {s['misses']}"
        print(f"{name:<14} {s['trials']:>6} {failed:>6}  {cells[0]:>17}  {cells[1]:>17}  {cells[2]:>17}  {err:>14}")
        if s["failures"]:
            print(f"{'':<14} failures by stage: {s['failures']}")

    print("\nremote - local (median ms):")
    for mode in ("fresh", "reuse"):
        local, remote = summary.get(f"local/{mode}"), summary.get(f"remote/{mode}")
        if not (local and remote):
            continue
        diffs = []
        for k in ("discovery", "accept", "overhead"):
            if local[k] and remote[k]:
                diffs.append(f"{k} {(remote[k]['median'] - local[k]['median']) * 1000:+.0f}")
        lost = sum(remote["failures"].values()) - sum(local["failures"].values())
        print(f"  {mode}: " + ", ".join(diffs) + f", failures {lost:+d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    active_model.add_model_argument(ap, POSES)
    ap.add_argument("--trials", type=int, default=10, help="per phase and mode (default 10)")
    ap.add_argument("--only", choices=["local", "remote"], help="run one phase")
    ap.add_argument("--mode", choices=["fresh", "reuse", "both"], default="both")
    ap.add_argument("--duration", type=float, default=0.5, help="planned seconds per move (default 0.5)")
    ap.add_argument("--tolerance", type=float, default=0.02, help="final error to count as a miss (default 0.02)")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds after the result to measure the error")
    ap.add_argument("--accept-timeout", type=float, default=3.0, help="seconds before a goal counts as lost")
    ap.add_argument("--discovery-timeout", type=float, default=10.0)
    ap.add_argument("--start-timeout", type=float, default=30.0, help="local sim startup")
    ap.add_argument("--remote-start-timeout", type=float, default=900.0, help="remote sync, build and startup")
    ap.add_argument("--local-domain", type=int, default=91)
    ap.add_argument("--remote-domain", type=int, default=92)
    ap.add_argument("--remote-log", default="/tmp/latency_diagnostic_remote.log")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer in both sims")
    ap.add_argument(
        "--remote-env", action="append", default=[], metavar="KEY=VALUE", help="env for the remote sim (repeatable)"
    )
    ap.add_argument(
        "--client-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="env for this PC's client in the remote phase, e.g. ROS_STATIC_PEERS=<host> (repeatable)",
    )
    ap.add_argument("--json", help="also write the trials and summary here")
    args = ap.parse_args()

    trials = []
    for phase in ["local", "remote"] if args.only is None else [args.only]:
        try:
            trials += run_phase(phase, args.model, args)
        except SimulationError as e:
            print(f"latency_diagnostic: {phase} sim failed: {e}", file=sys.stderr)
            return 1
    summary = summarize(trials, args.tolerance)
    print_report(summary, args.tolerance)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"trials": [dataclasses.asdict(t) for t in trials], "summary": summary}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
