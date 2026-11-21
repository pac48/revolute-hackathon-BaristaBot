"""Helpers for a controller_manager parameter file (config/<model>/controllers.yaml)."""

import yaml

# The reBot Arm RS's motor commands (MIT mode); see //src/controllers/joint_impedance_controller.
MIT_INTERFACES = ["position", "velocity", "kp", "kd", "t_ff"]


def load(path: str) -> dict:
    """The parsed parameter file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge(*configs: dict) -> dict:
    """The configs deep-merged, later ones overriding (lists are replaced), like ROS's --params-file order."""
    out: dict = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
    return out


def controller_names(config: dict) -> list[str]:
    """Controllers declared under controller_manager, in file order."""
    params = config["controller_manager"]["ros__parameters"]
    return [k for k, v in params.items() if isinstance(v, dict) and "type" in v]


def command_interfaces(config: dict) -> dict[str, list[str]]:
    """Joint -> command interfaces the configured controllers write to.

    Reads each controller's `joints` with `interface_name` (forward command
    controllers) or `command_interfaces` (e.g. joint_trajectory_controller); a
    joint_impedance_controller writes effort, or position, velocity, kp, kd and t_ff for its `mit_joints`,
    or position (and velocity, with a velocity_limit) for its `position_joints`.
    Joints no controller commands are left out, so they get state interfaces only.
    """
    out: dict[str, list[str]] = {}
    for name in controller_names(config):
        params = config.get(name, {}).get("ros__parameters", {})
        interfaces = params.get("command_interfaces") or (
            [params["interface_name"]] if "interface_name" in params else []
        )
        impedance = config["controller_manager"]["ros__parameters"][name]["type"].startswith(
            "joint_impedance_controller/"
        )
        for joint in params.get("joints", []):
            if impedance:
                if joint in params.get("mit_joints", []):
                    interfaces = MIT_INTERFACES
                elif joint in params.get("position_joints", []):
                    limit = params.get("gains", {}).get(joint, {}).get("velocity_limit", 0.0)
                    interfaces = ["position", "velocity"] if limit > 0 else ["position"]
                else:
                    interfaces = ["effort"]
            for interface in interfaces:
                if interface not in out.setdefault(joint, []):
                    out[joint].append(interface)
    return out
