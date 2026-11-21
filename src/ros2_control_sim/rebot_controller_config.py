"""Writes a reBot controller parameter file from config/rebot_arm_shared.yaml (the values; see there).

    rebot_controller_config SHARED OUT --backend sim|real [--prefixes left_ right_] [--gripper]

--backend sim: controller_manager + joint impedance controller over the arm joints (MIT mode) and the sim's
gripper (effort). --backend real: the same without the gripper; with --gripper, the parameter file that adds
the real gripper (position joints) on top of it instead. --prefixes: one arm per prefix (default: one, none).
"""

import argparse

import yaml

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER = "joint7"
CONTROLLER = "joint_trajectory_controller"


def controllers(shared: dict, backend: str, prefixes: list[str]) -> dict:
    sim = backend == "sim"
    joints, gains = [], {}
    for p in prefixes:
        for j in ARM_JOINTS:
            joints.append(p + j)
            gains[p + j] = dict(shared["arm_gains"][j])
        if sim:
            joints.append(p + GRIPPER)
            gains[p + GRIPPER] = dict(shared["gripper"]["sim"])
    return {
        "controller_manager": {
            "ros__parameters": {
                "update_rate": shared["update_rate"][backend],
                "use_sim_time": sim,
                "enforce_command_limits": shared["enforce_command_limits"],
                "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
                CONTROLLER: {"type": "joint_impedance_controller/JointImpedanceController"},
            }
        },
        CONTROLLER: {
            "ros__parameters": {
                "joints": joints,
                "mit_joints": [p + j for p in prefixes for j in ARM_JOINTS],
                "allow_partial_joints_goal": True,
                "gains": gains,
            }
        },
    }


def real_gripper(shared: dict, prefixes: list[str]) -> dict:
    """Merged over the real controllers.yaml (a later --params-file): the grippers as position joints."""
    return {
        CONTROLLER: {
            "ros__parameters": {
                "joints": [p + j for p in prefixes for j in ARM_JOINTS + [GRIPPER]],
                "position_joints": [p + GRIPPER for p in prefixes],
                "gains": {p + GRIPPER: dict(shared["gripper"]["real"]) for p in prefixes},
            }
        }
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shared")
    ap.add_argument("out")
    ap.add_argument("--backend", choices=["sim", "real"], required=True)
    ap.add_argument("--prefixes", nargs="+", default=[""])
    ap.add_argument("--gripper", action="store_true", help="the real gripper's parameter file")
    args = ap.parse_args(argv)
    if args.gripper and args.backend != "real":
        ap.error("--gripper is the real gripper's file: needs --backend real")
    with open(args.shared) as f:
        shared = yaml.safe_load(f)
    config = real_gripper(shared, args.prefixes) if args.gripper else controllers(shared, args.backend, args.prefixes)
    with open(args.out, "w") as f:
        f.write("# Generated from src/ros2_control_sim/config/rebot_arm_shared.yaml by rebot_controller_config.py: "
                "edit that file, not this one.\n")
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None, width=1000)


if __name__ == "__main__":
    main()
