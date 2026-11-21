"""Prototype: run ros2_control (controller_manager) against a MuJoCo-simulated robot.

Usage: bazel run //src/ros2_control_sim:mujoco_arm_sim -- [--model rebot_arm] [--headless]
Then move it with joint_trajectory_controller, e.g. for two_link_arm:
  bazel run //src/trajectory_generator:move_to -- joint1=0.5 joint2=-0.3
"""

import argparse
import os
import signal
import sys
import threading

from src.robot_model.active_model import active_model
from src.ros2_control_sim import models
from src.ros2_control_sim.sim import Simulation, SimulationError


def _exit_on_sigterm(signum, frame):
    # Unwind like Ctrl-C so Simulation stops its nodes instead of orphaning them.
    raise KeyboardInterrupt


def _stop_on_stdin_eof():
    sys.stdin.read()
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    ap = argparse.ArgumentParser()
    active_model.add_model_argument(ap, models.MODELS, help="robot to simulate")
    ap.add_argument("--headless", action="store_true", help="run without the MuJoCo viewer")
    ap.add_argument(
        "--exit-on-stdin-eof",
        action="store_true",
        help="stop when stdin closes (lets a caller over ssh stop a remote sim by closing it)",
    )
    ap.add_argument(
        "--exit-when-ready",
        action="store_true",
        help="stop as soon as the controllers are active (smoke check of startup)",
    )
    args = ap.parse_args()
    if args.exit_on_stdin_eof:
        threading.Thread(target=_stop_on_stdin_eof, daemon=True).start()

    try:
        with Simulation(args.model, headless=args.headless) as sim:
            if args.exit_when_ready:
                print("controllers active", file=sys.stderr, flush=True)
                return 0
            print("controllers active; Ctrl-C to stop", file=sys.stderr, flush=True)
            sim.control_node.wait()
    except SimulationError as e:
        print(f"mujoco_arm_sim: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
