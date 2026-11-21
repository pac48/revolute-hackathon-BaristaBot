"""Two reBot Arm B601-RS with a Gemini 305 each: robot description for ros2_control.

    from src.robot_model.rebot_dual_arm_gemini_305 import description
    description.robot_description(headless=True)   # URDF + the simulation's <ros2_control> block
    description.real_robot_description("can0", "can1")  # the same URDF + the real arms' blocks

The arms are the reBot's (rebot_arm.description) with every name prefixed: left_joint1 ...
left_joint7, left_gripper_end, left_camera_link (bases at y = +0.1725), and right_* (y = -0.1725). Each
arm's joints have its interfaces: joint1-6 the MIT-mode commands and joint7 (the gripper) an
effort or a position through a PID in the simulation.
"""

from src.robot_model.rebot_arm import real_description
from src.robot_model.rebot_arm.description import HARDWARE_PLUGIN, MIT_INTERFACES
from src.robot_model.rebot_dual_arm_gemini_305 import pids
from src.robot_model.rebot_dual_arm_gemini_305_data import mjcf, visual_urdf
from src.robot_model.robot_model_utils import ros2_control

ARMS = ("left_", "right_")
COMMAND_INTERFACES = {
    f"{arm}joint{i}": (MIT_INTERFACES if i <= 6 else ["position", "effort"]) for arm in ARMS for i in range(1, 8)
}


def visual_urdf_text() -> str:
    """The URDF generated from scene.xml with its visual meshes as package:// URIs (the arms' and the
    cameras': see //src/robot_model/rebot_dual_arm_gemini_305_data:visual_urdf_gen)."""
    return visual_urdf.read_text()


def robot_description(headless: bool = False) -> str:
    """The URDF (with visual meshes) plus a mujoco_ros2_control <ros2_control> block."""
    return ros2_control.robot_description(
        visual_urdf_text(),
        mjcf.path(),
        COMMAND_INTERFACES,
        headless=headless,
        pids_config_file=pids.path(),
        plugin=HARDWARE_PLUGIN,
    )


def real_robot_description(
    left_can_interface: str = "can0", right_can_interface: str = "can1", gripper: bool = False, mock: bool = False, gripper_options: dict | None = None
) -> str:
    """The same URDF with a real-arm <ros2_control> block per arm (rebot_arm.real_description): the left
    arm's motors on `left_can_interface`, the right's on `right_can_interface` (motors 1-7 on each)."""
    return real_description.multi_arm_robot_description(
        visual_urdf_text(), {"left_": left_can_interface, "right_": right_can_interface}, gripper, mock, gripper_options
    )
