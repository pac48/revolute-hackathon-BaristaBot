"""Start and stop the real robot's ros2_control driver, as sim.py does the simulation.

    with RealRobot("rebot_arm", can_interface="can0") as robot:
        ...  # controllers are active here (the motors are enabled)

robot_state_publisher, the controller manager with shared_can_ros2/CanSystem (RobStrideProtocol over SocketCAN),
then every controller in the model's real controllers.yaml (with the gripper's gripper.yaml). CanSystem reads the
motors' state and ENABLES them when the hardware activates, so the arm holds its position from the first moment the controllers
are active. Leaving the context stops the nodes (SIGINT).
"""

import dataclasses
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Sequence
from types import ModuleType

from src.robot_model.rebot_arm import description as rebot_arm_description
from src.robot_model.rebot_arm_gemini_305 import description as rebot_arm_gemini_305_description
from src.robot_model.rebot_dual_arm_gemini_305 import description as rebot_dual_arm_gemini_305_description
from src.ros2_control_sim import (
    controller_config,
    rebot_arm_real_controllers,
    rebot_arm_real_gripper,
    rebot_dual_arm_real_controllers,
    rebot_dual_arm_real_gripper,
    robot_state_publisher,
    spawner,
)
from src.ros2_control_sim import real_ros2_control_node


class RealRobotError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class RealModel:
    name: str
    description: ModuleType  # real_robot_description(*can_interfaces, gripper, mock)
    controllers: ModuleType  # controller_manager parameters (path()), for the arm
    gripper: ModuleType = rebot_arm_real_gripper  # parameters added by the gripper (path())
    can_interfaces: tuple[str, ...] = ("can0",)  # the default bus of each arm (one CanSystem each)

    def param_files(self, gripper: bool) -> list[str]:
        return [self.controllers.path(), *([self.gripper.path()] if gripper else [])]

    def controller_config(self, gripper: bool = True) -> dict:
        return controller_config.merge(*map(controller_config.load, self.param_files(gripper)))


MODELS = {
    m.name: m
    for m in [
        RealModel("rebot_arm", rebot_arm_description, rebot_arm_real_controllers),
        # The arm with the Gemini 305's links in its URDF (the camera driver is //src/camera_drivers:orbbec_camera).
        RealModel("rebot_arm_gemini_305", rebot_arm_gemini_305_description, rebot_arm_real_controllers),
        # Two arms, each on its own CAN bus (left_ on can0, right_ on can1), one controller over both.
        RealModel(
            "rebot_dual_arm_gemini_305",
            rebot_dual_arm_gemini_305_description,
            rebot_dual_arm_real_controllers,
            rebot_dual_arm_real_gripper,
            ("can0", "can1"),
        ),
    ]
}


class RealRobot:
    def __init__(
        self,
        model: str = "rebot_arm",
        can_interface: "str | Sequence[str] | None" = None,
        gripper: bool = True,
        mock: bool = False,
        gripper_options: dict | None = None,
        timeout: float = 20.0,
    ):
        try:
            self.model = MODELS[model]
        except KeyError:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}") from None
        # One SocketCAN interface per arm (a bare string is the one arm's); None: the model's defaults.
        buses = [can_interface] if isinstance(can_interface, str) else list(can_interface or self.model.can_interfaces)
        if len(buses) != len(self.model.can_interfaces):
            raise ValueError(f"{model} has {len(self.model.can_interfaces)} arm(s) and needs that many CAN interfaces, got {buses}")
        self.can_interfaces = buses
        self.gripper = gripper
        self.gripper_options = gripper_options  # {"kp", "kd", "force"}: the gripper's gains and squeeze limit (real_description's defaults)
        self.mock = mock  # ros2_control's mock hardware instead of the CAN motors: for testing the controllers
        self.timeout = timeout
        self.procs: list[subprocess.Popen] = []

    @property
    def control_node(self) -> subprocess.Popen:
        return self.procs[1]

    def __enter__(self) -> "RealRobot":
        model = self.model
        config = model.controller_config(self.gripper)
        robot_description = model.description.real_robot_description(*self.can_interfaces, self.gripper, self.mock, self.gripper_options)
        param_files = model.param_files(self.gripper)
        params = [a for path in param_files for a in ("--params-file", path)]
        try:
            self.procs.append(
                robot_state_publisher.popen(
                    "--ros-args", "-p", "robot_description:=" + robot_description, "-p", "use_sim_time:=false"
                )
            )
            prefix = shlex.split(os.environ.get("ROS2_CONTROL_NODE_PREFIX", ""))
            self.procs.append(
                subprocess.Popen(
                    [*prefix, real_ros2_control_node.path(), "--ros-args", *params]
                )
            )
            names = controller_config.controller_names(config)
            spawn = spawner.popen(
                *names, "--controller-manager-timeout", self.timeout, *[a for path in param_files for a in ("--param-file", path)]
            )
            self.procs.append(spawn)
            deadline = time.monotonic() + self.timeout + 5
            while spawn.poll() is None:
                self._check_nodes()
                if time.monotonic() > deadline:
                    raise RealRobotError(f"spawning {names} took longer than {self.timeout}s")
                time.sleep(0.05)
            self.procs.remove(spawn)
            if spawn.returncode != 0:
                raise RealRobotError(f"spawner failed for {names} (exit code {spawn.returncode})")
            self._check_nodes()
        except BaseException:
            self.stop()
            raise
        return self

    def _check_nodes(self):
        for p in self.procs[:2]:
            if p.poll() is not None:
                raise RealRobotError(f"{' '.join(map(os.path.basename, p.args[:1]))} exited with code {p.returncode}")

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        # The controller manager first, so the controllers deactivate while the others are still up.
        for p in reversed(self.procs):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
