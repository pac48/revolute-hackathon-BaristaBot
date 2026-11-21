"""two_link_arm robot description for ros2_control with mujoco_ros2_control.

    from src.robot_model.two_link_arm import description
    description.robot_description(headless=True)  # URDF + <ros2_control> block
"""

from src.robot_model.robot_model_utils import ros2_control
from src.robot_model.two_link_arm_data import mjcf, urdf

# scene.xml has a MuJoCo <position> actuator per joint, so position commands
# are native (no PIDs needed).
COMMAND_INTERFACES = {
    "joint1": ["position"],
    "joint2": ["position"],
}


def robot_description(headless: bool = False) -> str:
    """The generated URDF with a mujoco_ros2_control <ros2_control> block for scene.xml."""
    return ros2_control.robot_description(
        urdf.read_text(), mjcf.path(), COMMAND_INTERFACES, headless=headless
    )
