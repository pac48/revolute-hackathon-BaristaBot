"""ros2_control.xml.j2, plus helpers to add a mujoco_ros2_control <ros2_control> block to a URDF.

Generic: each robot model package (two_link_arm, rebot_arm, ...) calls
robot_description() with its own URDF, MJCF, command interfaces and PID gains.
"""

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

import jinja2

from ._ros2_control_data import *  # noqa: F401,F403  path(), read_text(), read_bytes()
from ._ros2_control_data import read_text

# URDF joint types that ros2_control can command (fixed/floating are skipped).
_MOVABLE = ("revolute", "continuous", "prismatic")


def joints(urdf_xml: str, command_interfaces: Mapping[str, Sequence[str]] = {}) -> list[dict]:
    """Movable joints of a URDF, in file order, with a start position inside their limits.

    `command_interfaces` maps joint name -> command interfaces (e.g. ["position"]);
    other joints are state-only (passive).
    """
    urdf_joints = [j for j in ET.fromstring(urdf_xml).iter("joint") if j.get("type") in _MOVABLE]
    unknown = set(command_interfaces) - {j.get("name") for j in urdf_joints}
    if unknown:
        raise ValueError(f"commanded joints not in the URDF: {sorted(unknown)}")

    out = []
    for joint in urdf_joints:
        initial = 0.0
        limit = joint.find("limit")
        if limit is not None and joint.get("type") != "continuous":
            lower = float(limit.get("lower", 0.0))
            upper = float(limit.get("upper", 0.0))
            initial = min(max(initial, lower), upper)
        out.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "initial_position": initial,
                "command_interfaces": list(command_interfaces.get(joint.get("name"), [])),
            }
        )
    return out


def render(
    urdf_xml: str,
    mujoco_model: str,
    command_interfaces: Mapping[str, Sequence[str]],
    headless: bool = False,
    pids_config_file: str | None = None,
    name: str = "MujocoSystem",
    plugin: str = "mujoco_ros2_control/MujocoSystemInterface",
) -> str:
    """The <ros2_control> block for every movable joint of `urdf_xml`."""
    template = jinja2.Environment(
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    ).from_string(read_text())
    return template.render(
        name=name,
        plugin=plugin,
        mujoco_model=mujoco_model,
        headless=headless,
        pids_config_file=pids_config_file,
        joints=joints(urdf_xml, command_interfaces),
    )


def robot_description(urdf_xml: str, mujoco_model: str, command_interfaces, **kwargs) -> str:
    """`urdf_xml` with its generated <ros2_control> block inserted (kwargs as for render())."""
    end = urdf_xml.rindex("</robot>")
    return urdf_xml[:end] + render(urdf_xml, mujoco_model, command_interfaces, **kwargs) + urdf_xml[end:]
