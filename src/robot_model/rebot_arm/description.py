"""reBot Arm B601-RS robot description for ros2_control with mujoco_ros2_control.

    from src.robot_model.rebot_arm import description
    description.robot_description(headless=True)  # URDF + <ros2_control> block
"""

from src.robot_model.rebot_arm import pids
from src.robot_model.rebot_arm_data import rs_mjcf, rs_visual_urdf
from src.robot_model.robot_model_utils import ros2_control

# rs_arm.xml drives joint1-7 with torque <motor> actuators. joint1-6 have the interfaces of the real
# arm's RobStride motors (rebot_robstride_protocol's RobStrideProtocol): the MIT-mode command
#   tau = kp (position - q) + kd (velocity - dq) + t_ff
# and the states position, velocity and effort, which MitMujocoSystemInterface provides on the torque
# actuators (the MJCF's ctrlrange is the motor's torque limit). joint7, the gripper, is a slide (m, N) with the real
# gripper's units and sign (rebot_robstride_protocol's RebotGripperProtocol with a stroke: position in
# m, 0 closed .. 0.05 open) but not its interfaces: it takes "effort" directly or "position" through
# the PIDs in pids.yaml, one kind at a time, while the real gripper takes a position and a speed
# limit. joint_left/joint_right have no actuator: the MJCF couples
# them to joint7 with equality constraints, so they are state-only.
MIT_INTERFACES = ["position", "velocity", "kp", "kd", "t_ff"]
COMMAND_INTERFACES = {
    **{f"joint{i}": MIT_INTERFACES for i in range(1, 7)},
    "joint7": ["position", "effort"],
}
HARDWARE_PLUGIN = "mit_system_interface/MitMujocoSystemInterface"


def visual_urdf() -> str:
    """The URDF generated from rs_arm.xml with its visual meshes as package://rebot_arm_description/meshes/...
    URIs (see //src/robot_model/rebot_arm_data:rs_visual_urdf_gen). They resolve through
    @rebot_arm_description//:ament_index, which an RViz that shows the arm needs in its data; the
    simulation may run on another machine than RViz."""
    return rs_visual_urdf.read_text()


def robot_description(headless: bool = False) -> str:
    """The URDF generated from rs_arm.xml (with visual meshes) plus a mujoco_ros2_control
    <ros2_control> block."""
    return ros2_control.robot_description(
        visual_urdf(),
        rs_mjcf.path(),
        COMMAND_INTERFACES,
        headless=headless,
        pids_config_file=pids.path(),
        plugin=HARDWARE_PLUGIN,
    )


def real_robot_description(can_interface: str = "can0", gripper: bool = False, mock: bool = False, gripper_options: dict | None = None) -> str:
    """The same URDF with the real arm's <ros2_control> block (real_description) instead of the simulation's."""
    from src.robot_model.rebot_arm import real_description

    return real_description.robot_description(visual_urdf(), can_interface, gripper, mock, gripper_options)
