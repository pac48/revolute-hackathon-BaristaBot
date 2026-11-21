"""Run ros2_control against the real reBot Arm B601-RS (RobStride motors on a SocketCAN bus).

Usage: bazel run //src/ros2_control_sim:real_arm[.remote] -- [--model rebot_arm_gemini_305] [--can-interface can0]
       bazel run //src/ros2_control_sim:real_arm[.remote] -- --model rebot_dual_arm_gemini_305 --can-interface can0 can1

The real version of `bazel run //src/ros2_control_sim:mujoco_arm_sim -- --model rebot_arm_gemini_305`: the same
controllers and topics (/joint_trajectory_controller/..., /joint_states), so
`bazel run //src/trajectory_generator:move_to -- joint1=0.5 ...` moves the real arm. Run it on the machine with the
CAN adapter (`.remote` for the build host); bring the interface up first:
  sudo ip link set can0 up type can bitrate 1000000

WARNING: starting it ENABLES the six arm motors and the gripper's (--no-gripper if it has none); the arm holds its position, then follows trajectories with the
simulation's untuned gains. Keep the workspace clear and an emergency stop at hand. Ctrl-C stops the driver; see
the controller's on_deactivate for what the arm does then. With --model rebot_dual_arm_gemini_305 both arms are driven (left_ on the first CAN interface, right_ on the second, motors 1-7 on each),
as one robot: one controller over both arms' joints. The gripper is joint7, in m (0 closed .. 0.05 open), part of the same trajectories. The camera is separate:
`bazel run //src/camera_drivers:orbbec_camera[.remote]`.
"""

import argparse
import os
import signal
import sys
import threading

from src.robot_model.active_model import active_model
from src.ros2_control_sim import real


def _exit_on_sigterm(signum, frame):
    # Unwind like Ctrl-C so RealRobot stops its nodes instead of orphaning them.
    raise KeyboardInterrupt


def _stop_on_stdin_eof():
    sys.stdin.read()
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    ap = argparse.ArgumentParser()
    active_model.add_model_argument(ap, real.MODELS, help="robot description")
    ap.add_argument(
        "--can-interface",
        nargs="+",
        default=os.environ["CAN_INTERFACE"].split() if "CAN_INTERFACE" in os.environ else None,
        help="SocketCAN interface (default can0); one per arm for the dual arm (default can0 can1: left right)",
    )
    ap.add_argument("--gripper-kp", type=float, help="the gripper's position gain, N*m/rad (default 0.5; squeeze F ~ 100 * kp * overshoot[rad] N)")
    ap.add_argument("--gripper-kd", type=float, help="the gripper's damping, N*m*s/rad (default 0.25)")
    ap.add_argument("--gripper-force", type=float, help="also limit the squeeze to this many N (off by default)")
    ap.add_argument("--no-gripper", action="store_true", help="the arm has no official gripper (motor 7)")
    ap.add_argument(
        "--exit-on-stdin-eof",
        action="store_true",
        help="stop when stdin closes (lets a caller over ssh stop the driver by closing it)",
    )
    ap.add_argument(
        "--exit-when-ready",
        action="store_true",
        help="stop as soon as the controllers are active (smoke check of startup)",
    )
    args = ap.parse_args()
    if args.exit_on_stdin_eof:
        threading.Thread(target=_stop_on_stdin_eof, daemon=True).start()

    options = {k: v for k, v in (
        ("kp", args.gripper_kp), ("kd", args.gripper_kd), ("force", args.gripper_force)) if v is not None}
    try:
        with real.RealRobot(args.model, can_interface=args.can_interface, gripper=not args.no_gripper, gripper_options=options) as robot:
            if args.exit_when_ready:
                print("controllers active", file=sys.stderr, flush=True)
                return 0
            print(f"controllers active on {robot.can_interfaces}; Ctrl-C to stop", file=sys.stderr, flush=True)
            robot.control_node.wait()
    except real.RealRobotError as e:
        print(f"real_arm: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
