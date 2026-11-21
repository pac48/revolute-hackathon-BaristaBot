"""The real reBot Arm B601-RS's ros2_control block: shared_can_ros2/CanSystem and RobStrideProtocol.

    from src.robot_model.rebot_arm import real_description
    real_description.robot_description(visual_urdf, "can0")   # URDF + <ros2_control> block

The counterpart of description.py (the MuJoCo simulation): each arm joint has the same command and state
interfaces (the RobStride MIT-mode commands position, velocity, kp, kd, t_ff; the states position,
velocity, effort), but is a RobStride motor on a SocketCAN bus instead of a torque actuator. The motor table
is upstream's (xiayip/rebot_arm_rs_ros2, rebot_arm_rs.ros2_control.xacro). The gripper (joint7) is the
official RS00 gripper (RebotGripperProtocol) when `gripper` is set: its position, velocity (a speed limit)
and, in SI units, a 0.05 m stroke, 0 closed, like the simulation's slide; the simulation drives it by effort.

The joint limits are the URDF's (from the MJCF), tighter than upstream's xacro for joint2-4.
"""

import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

from src.robot_model.rebot_arm.description import MIT_INTERFACES

# name, motor id, model, mit_kp, mit_kd (the motor's gains until a controller writes its own)
JOINTS = [
    ("joint1", "0x01", "rs-06", 50.0, 3.0),
    ("joint2", "0x02", "rs-06", 150.0, 10.0),
    ("joint3", "0x03", "rs-06", 150.0, 10.0),
    ("joint4", "0x04", "rs-00", 50.0, 5.0),
    ("joint5", "0x05", "rs-00", 50.0, 4.0),
    ("joint6", "0x06", "rs-00", 50.0, 4.0),
]
# Command ranges of the model's MIT frames (codec limits_for_model): velocity (rad/s), t_ff (N*m), kp, kd.
LIMITS = {
    "rs-06": {"velocity": 50.0, "t_ff": 36.0, "kp": 5000.0, "kd": 100.0},
    "rs-00": {"velocity": 33.0, "t_ff": 14.0, "kp": 500.0, "kd": 5.0},
}
GRIPPER_STROKE = 0.05  # m, closed to open: the URDF's joint7 range
GRIPPER_OPEN, GRIPPER_CLOSED = 4.73, 0.0  # rad: the motor's endpoints (RebotGripperProtocol's open_position and closed_position)
# The motor's zero is set at the closed stop (//src/rebot_robstride_protocol:gripper_zero, zero range 0..2pi); the open stop
# measured 4.734 and 4.759 rad on the two grippers, so 4.73 keeps a full-open command off the stop on both.
# The gripper's MIT gains. The squeeze on an object is the law's torque, tau = kp * (target - position): through the
# transmission (100 rad/m: F = tau * (open - closed) / stroke) F [N] ~ 100 * kp * overshoot [rad], the overshoot being how far
# past the object the target is (the closed end, with a full squeeze: 1 cm of finger travel is 1 rad). At kp 50 that
# saturates the motor's 14 N*m, ~1400 N. Raise them for a firmer, more precise gripper, lower them for a gentler one
# (real_arm --gripper-kp / --gripper-kd). kd damps the deviation from the slew's velocity (feedforward).
GRIPPER_KP = 0.5  # was 1.0: the squeeze, ~100 * kp * overshoot N, was still too high
GRIPPER_KD = 0.25  # kd / kp = 0.5 s, as before
# GRIPPER_FORCE (N), if set, also limits the squeeze outright (RebotGripperProtocol's max_torque): a motor stalled
# behind its target is held that far ahead. Off by default: a limit under the gripper's friction can stall it.
GRIPPER_FORCE = None
GRIPPER_MAX_SPEED = 0.2  # m/s: the largest speed limit a controller may write (the motor's would be 0.33)
GRIPPER_SPEED = GRIPPER_STROKE / 4  # m/s: the protocol's default, a quarter of the travel per second
HARDWARE_PLUGIN = "shared_can_ros2/CanSystem"
TRANSPORT_PLUGIN = "shared_can_ros2/SocketCanTransport"


def _joint_limits(urdf_xml: str) -> dict[str, tuple[float, float]]:
    out = {}
    for joint in ET.fromstring(urdf_xml).iter("joint"):
        limit = joint.find("limit")
        if limit is not None:
            out[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return out


def gripper_max_torque(force: float) -> float:
    """The motor torque (N*m) that squeezes with `force` (N), through the ideal transmission."""
    return force * GRIPPER_STROKE / (GRIPPER_OPEN - GRIPPER_CLOSED)


def _gripper_lines(upper: float, prefix: str = "", options: dict | None = None) -> list[str]:
    options = options or {}
    kp, kd, force = options.get("kp", GRIPPER_KP), options.get("kd", GRIPPER_KD), options.get("force", GRIPPER_FORCE)
    return [
        f'  <joint name="{prefix}joint7">',
        '    <param name="protocol_plugin">rebot_robstride_protocol/RebotGripperProtocol</param>',
        f'    <param name="protocol_instance">{prefix}official_gripper_robstride</param>',
        '    <param name="motor_id">0x07</param>',
        '    <param name="model">rs-00</param>',
        '    <param name="direction">1.0</param>',
        '    <param name="offset">0.0</param>',
        f'    <param name="mit_kp">{kp}</param>',
        f'    <param name="mit_kd">{kd}</param>',
        # The motor's endpoints (rad), and the finger travel they span (m).
        f'    <param name="open_position">{GRIPPER_OPEN}</param>',
        f'    <param name="closed_position">{GRIPPER_CLOSED}</param>',
        *([f'    <param name="max_torque">{gripper_max_torque(force):g}</param>'] if force else []),
        f'    <param name="stroke">{GRIPPER_STROKE}</param>',
        f'    <param name="default_velocity">{GRIPPER_SPEED}</param>',
        '    <param name="min_position">0.0</param>',
        f'    <param name="max_position">{upper}</param>',
        '    <command_interface name="position">',
        '      <param name="min">0.0</param>',
        f'      <param name="max">{upper}</param>',
        "    </command_interface>",
        '    <command_interface name="velocity">',
        '      <param name="min">0.0</param>',
        f'      <param name="max">{GRIPPER_MAX_SPEED}</param>',
        f'      <param name="initial_value">{GRIPPER_SPEED}</param>',
        "    </command_interface>",
        *[
            f'    <state_interface name="{i}"><param name="initial_value">0.0</param></state_interface>'
            for i in ("position", "velocity", "effort", "temperature")
        ],
        "  </joint>",
    ]


def ros2_control_block(
    urdf_xml: str,
    can_interface: str = "can0",
    gripper: bool = False,
    mock: bool = False,
    name: str = "RebotArmRs",
    prefix: str = "",
    gripper_options: dict | None = None,
) -> str:
    """The <ros2_control> block of the arm joints (and the gripper) of `urdf_xml` on `can_interface`.

    `prefix` is the joints' name prefix in a model with several arms (left_joint1, ...); each arm is its
    own system (`name`) on its own CAN bus.

    `mock` swaps the CAN hardware for ros2_control's mock_components/GenericSystem, which has the same joint
    interfaces but no motors (the state follows the position command): to test the controllers without the arm.
    """
    limits = _joint_limits(urdf_xml)
    hardware = (
        [
            "    <plugin>mock_components/GenericSystem</plugin>",
            '    <param name="calculate_dynamics">false</param>',
        ]
        if mock
        else [
            f"    <plugin>{HARDWARE_PLUGIN}</plugin>",
            f"    <param name=\"transport_plugin\">{TRANSPORT_PLUGIN}</param>",
            f"    <param name=\"can_interface\">{quoteattr(can_interface)[1:-1]}</param>",
            '    <param name="host_id">0xFD</param>',
            '    <param name="require_all_devices">true</param>',
            '    <param name="initial_read_timeout_ms">1200</param>',
            '    <param name="poll_divider">1</param>',
            '    <param name="max_command_step">0.02</param>',
        ]
    )
    lines = [f'<ros2_control name="{name}" type="system">', "  <hardware>", *hardware, "  </hardware>"]
    for joint, motor_id, model, kp, kd in JOINTS:
        lower, upper = limits[prefix + joint]
        limit = LIMITS[model]
        initial = {"velocity": 0.0, "kp": kp, "kd": kd, "t_ff": 0.0}
        ranges = {
            "position": ("min", lower, "max", upper),
            "velocity": ("min", -limit["velocity"], "max", limit["velocity"]),
            "kp": ("minimum", 0.0, "maximum", limit["kp"]),
            "kd": ("minimum", 0.0, "maximum", limit["kd"]),
            "t_ff": ("minimum", -limit["t_ff"], "maximum", limit["t_ff"]),
        }
        lines += [
            f'  <joint name="{prefix}{joint}">',
            '    <param name="protocol_plugin">rebot_robstride_protocol/RobStrideProtocol</param>',
            f'    <param name="protocol_instance">{prefix}arm_robstride</param>',
            f'    <param name="motor_id">{motor_id}</param>',
            f'    <param name="model">{model}</param>',
            '    <param name="direction">1.0</param>',
            '    <param name="offset">0.0</param>',
            f'    <param name="mit_kp">{kp}</param>',
            f'    <param name="mit_kd">{kd}</param>',
            f'    <param name="min_position">{lower}</param>',
            f'    <param name="max_position">{upper}</param>',
        ]
        for interface in MIT_INTERFACES:
            low_name, low, high_name, high = ranges[interface]
            lines.append(f'    <command_interface name="{interface}">')
            lines.append(f'      <param name="{low_name}">{low}</param>')
            lines.append(f'      <param name="{high_name}">{high}</param>')
            if interface in initial:
                lines.append(f'      <param name="initial_value">{initial[interface]}</param>')
            lines.append("    </command_interface>")
        for interface in ("position", "velocity", "effort"):
            lines.append(f'    <state_interface name="{interface}"><param name="initial_value">0.0</param></state_interface>')
        lines.append("  </joint>")
    if gripper:
        lines += _gripper_lines(limits[prefix + "joint7"][1], prefix, gripper_options)
    lines.append("</ros2_control>\n")
    return "\n".join(lines)


def robot_description(
    urdf_xml: str,
    can_interface: str = "can0",
    gripper: bool = False,
    mock: bool = False,
    gripper_options: dict | None = None,
) -> str:
    """`urdf_xml` with the real arm's <ros2_control> block inserted. `gripper_options` ({"kp", "kd", "force"}) override the gripper's gains and squeeze limit."""
    end = urdf_xml.rindex("</robot>")
    block = ros2_control_block(urdf_xml, can_interface, gripper, mock, gripper_options=gripper_options)
    return urdf_xml[:end] + block + urdf_xml[end:]


def multi_arm_robot_description(
    urdf_xml: str, arms: dict[str, str], gripper: bool = False, mock: bool = False, gripper_options: dict | None = None
) -> str:
    """`urdf_xml` with one <ros2_control> block per arm. `arms` maps a joint-name prefix to the CAN
    interface of that arm's bus, e.g. {"left_": "can0", "right_": "can1"} (the motors are 1-7 on each)."""
    end = urdf_xml.rindex("</robot>")
    blocks = "".join(
        ros2_control_block(
            urdf_xml, can, gripper, mock, name=f"RebotArmRs_{prefix.rstrip('_')}", prefix=prefix, gripper_options=gripper_options
        )
        for prefix, can in arms.items()
    )
    return urdf_xml[:end] + blocks + urdf_xml[end:]
