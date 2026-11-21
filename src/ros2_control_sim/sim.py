"""Start and stop the MuJoCo + ros2_control simulation of a robot model.

    with Simulation("rebot_arm", headless=True) as sim:
        ...  # controllers are active here

`model` is a name from models.MODELS (or a models.SimModel). The robot
description comes from the model's own package; entering the context starts
robot_state_publisher and the mujoco_ros2_control controller
manager, then spawns every controller in the model's controllers.yaml and
returns once they are all active. Leaving it stops the nodes. Anything that
goes wrong raises SimulationError within `timeout` seconds instead of hanging.
"""

import ctypes.util
import os
import shlex
import signal
import subprocess
import sys
import time

from src.ros2_control_sim import (
    controller_config,
    models,
    robot_state_publisher,
    ros2_control_node,
    spawner,
)


class SimulationError(RuntimeError):
    pass


def viewer_env() -> dict[str, str]:
    """Environment for a node that opens the MuJoCo viewer.

    The RCR's GLFW links a hermetic libX11 with libxcb built in (qtbase's
    x11_shared), but the window's GL context comes from the system GLX driver,
    which uses the system libxcb on the same X connection. Two xcb copies on one
    connection crash intermittently in xcb_get_extension_data. Preloading the
    system libX11/libxcb makes GLFW use the same ones as the driver (Xlib's ABI
    is stable). Headless runs don't open a window and stay hermetic.
    """
    env = dict(os.environ)
    system_libs = [ctypes.util.find_library(name) for name in ("X11", "xcb")]
    if all(system_libs):
        env["LD_PRELOAD"] = ":".join(filter(None, [*system_libs, env.get("LD_PRELOAD")]))
    else:
        print("mujoco_arm_sim: system libX11/libxcb not found; the viewer may crash", file=sys.stderr)
    return env


class Simulation:
    def __init__(self, model: "str | models.SimModel" = "two_link_arm", headless: bool = False, timeout: float = 20.0):
        self.model = models.get(model) if isinstance(model, str) else model
        self.headless = headless
        self.timeout = timeout
        self.procs: list[subprocess.Popen] = []

    @property
    def control_node(self) -> subprocess.Popen:
        return self.procs[1]

    def __enter__(self) -> "Simulation":
        model = self.model
        config = model.controller_config()
        robot_description = model.description.robot_description(headless=self.headless)
        controllers = model.controllers.path()
        try:
            self.procs.append(
                robot_state_publisher.popen(
                    "--ros-args", "-p", "robot_description:=" + robot_description, "-p", "use_sim_time:=true"
                )
            )
            # $ROS2_CONTROL_NODE_PREFIX wraps the node like a ROS launch prefix,
            # e.g. "gdb -batch -ex run -ex bt --args" to debug a crash.
            prefix = shlex.split(os.environ.get("ROS2_CONTROL_NODE_PREFIX", ""))
            self.procs.append(
                subprocess.Popen(
                    [
                        *prefix,
                        ros2_control_node.path(),
                        "--ros-args",
                        "--params-file",
                        controllers,
                        *[a for module in model.extra_params for a in ("--params-file", module.path())],
                    ],
                    env=None if self.headless else viewer_env(),
                )
            )
            # One spawner call loads, configures and activates all controllers.
            # Watch the nodes meanwhile, so a crashed node fails now rather than
            # after the spawner's timeout.
            names = controller_config.controller_names(config)
            spawn = spawner.popen(*names, "--controller-manager-timeout", self.timeout, "--param-file", controllers)
            self.procs.append(spawn)
            deadline = time.monotonic() + self.timeout + 5
            while spawn.poll() is None:
                self._check_nodes()
                if time.monotonic() > deadline:
                    raise SimulationError(f"spawning {names} took longer than {self.timeout}s")
                time.sleep(0.05)
            self.procs.remove(spawn)
            if spawn.returncode != 0:
                raise SimulationError(f"spawner failed for {names} (exit code {spawn.returncode})")
            self._check_nodes()
        except BaseException:
            self.stop()
            raise
        return self

    def _check_nodes(self):
        for p in self.procs[:2]:
            if p.poll() is not None:
                raise SimulationError(f"{' '.join(map(os.path.basename, p.args[:1]))} exited with code {p.returncode}")

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in self.procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
