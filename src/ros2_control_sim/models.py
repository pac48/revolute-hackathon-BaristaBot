"""Robot models the ros2_control simulation can run.

Each SimModel pairs a robot model's description module (from //src/robot_model:
the URDF + mujoco_ros2_control block, which the model package renders because
it knows its own actuators) with this package's controller config for it
(config/<name>/controllers.yaml). Add a model by adding an entry to MODELS and
its controllers.yaml.
"""

import dataclasses
from types import ModuleType

from src.robot_model.rebot_arm import description as rebot_arm_description
from src.robot_model.rebot_arm_gemini_305 import description as rebot_arm_gemini_305_description
from src.robot_model.rebot_dual_arm_gemini_305 import description as rebot_dual_arm_gemini_305_description
from src.robot_model.two_link_arm import description as two_link_arm_description
from src.ros2_control_sim import (
    controller_config,
    rebot_arm_controllers,
    rebot_arm_gemini_305_cameras,
    rebot_dual_arm_gemini_305_cameras,
    rebot_dual_arm_gemini_305_controllers,
    two_link_arm_controllers,
)


@dataclasses.dataclass(frozen=True)
class SimModel:
    name: str
    description: ModuleType  # robot_description(headless) and COMMAND_INTERFACES
    controllers: ModuleType  # controller_manager parameters (path())
    extra_params: tuple[ModuleType, ...] = ()  # more parameter files for the control node (path()), e.g. cameras

    def controller_config(self) -> dict:
        return controller_config.load(self.controllers.path())


MODELS = {
    m.name: m
    for m in [
        SimModel("two_link_arm", two_link_arm_description, two_link_arm_controllers),
        SimModel("rebot_arm", rebot_arm_description, rebot_arm_controllers),
        # The same arm: the reBot's controllers, and the simulated camera.
        SimModel(
            "rebot_arm_gemini_305",
            rebot_arm_gemini_305_description,
            rebot_arm_controllers,
            (rebot_arm_gemini_305_cameras,),
        ),
        # Two of them, 25 cm apart in y (left_ and right_ prefixes), one controller over both.
        SimModel(
            "rebot_dual_arm_gemini_305",
            rebot_dual_arm_gemini_305_description,
            rebot_dual_arm_gemini_305_controllers,
            (rebot_dual_arm_gemini_305_cameras,),
        ),
    ]
}


def get(name: str) -> SimModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None
