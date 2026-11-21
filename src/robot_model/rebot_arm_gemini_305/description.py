"""reBot Arm B601-RS with a Gemini 305: robot description for ros2_control with mujoco_ros2_control.

    from src.robot_model.rebot_arm_gemini_305 import description
    description.robot_description(headless=True)  # URDF + <ros2_control> block

The arm's actuators, command interfaces and PIDs are the reBot's (rebot_arm.description);
the URDF also has the camera's links, with their meshes as package:// URIs.
"""

from src.robot_model.rebot_arm import pids, real_description
from src.robot_model.rebot_arm.description import COMMAND_INTERFACES, HARDWARE_PLUGIN  # noqa: F401 (COMMAND_INTERFACES re-exported)
from src.robot_model.rebot_arm_gemini_305_data import mjcf, visual_urdf
from src.robot_model.robot_model_utils import ros2_control


def visual_urdf_text() -> str:
    """The URDF generated from scene.xml with its visual meshes as package:// URIs: the arm's
    (package://rebot_arm_description/meshes/...) and the camera's
    (package://orbbec_description/meshes/gemini305/...). They resolve through
    @rebot_arm_description//:ament_index and @orbbec_description//:ament_index, which an RViz
    that shows the robot needs in its data; the simulation may run on another machine than RViz."""
    return visual_urdf.read_text()


def robot_description(headless: bool = False) -> str:
    """The URDF generated from scene.xml (with visual meshes) plus a mujoco_ros2_control
    <ros2_control> block."""
    return ros2_control.robot_description(
        visual_urdf_text(),
        mjcf.path(),
        COMMAND_INTERFACES,
        headless=headless,
        pids_config_file=pids.path(),
        plugin=HARDWARE_PLUGIN,
    )


def real_robot_description(can_interface: str = "can0", gripper: bool = False, mock: bool = False, gripper_options: dict | None = None) -> str:
    """The same URDF with the real arm's <ros2_control> block (rebot_arm.real_description) instead of the simulation's."""
    return real_description.robot_description(visual_urdf_text(), can_interface, gripper, mock, gripper_options)
