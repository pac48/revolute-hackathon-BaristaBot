"""Forward and inverse kinematics from a robot model's kinematics CLI.

    kinematics_client.forward("rebot_arm", {"joint1": 0.1, ...})   -> Pose
    path = kinematics_client.ik_path("rebot_arm", start={...}, poses=[Pose, ...])
    path.joint_names, path.positions

Runs the model's kinematics CLI (e.g. //src/robot_model/rebot_arm:kinematics;
JSON protocol in //src/robot_model/robot_model_utils:kinematics_cli.hpp). Poses
are of the model's tip link in its root link's frame (cartesian_path.Pose).
"""

import dataclasses
import json
import subprocess
from collections.abc import Mapping, Sequence

from src.trajectory_generator import rebot_arm_gemini_305_kinematics, rebot_arm_kinematics
from src.trajectory_generator.cartesian_path import Pose

# Model name -> kinematics CLI (py_data_library). Add an entry per model with kinematics.
KINEMATICS = {
    "rebot_arm": rebot_arm_kinematics,
    "rebot_arm_gemini_305": rebot_arm_gemini_305_kinematics,
}


class KinematicsError(RuntimeError):
    """`index` is the pose that failed (ik_path), `positions` the solutions before it."""

    def __init__(self, message: str, index: int = -1, positions: list | None = None):
        super().__init__(message)
        self.index = index
        self.positions = positions or []


@dataclasses.dataclass(frozen=True)
class Chain:
    joint_names: list[str]
    root_link: str
    tip_link: str


@dataclasses.dataclass(frozen=True)
class IkPath:
    joint_names: list[str]
    positions: list[list[float]]  # one per pose, in joint_names order
    time_ms: float
    max_position_error: float  # m
    max_orientation_error: float  # rad


def _run(model: str, request: dict, timeout: float) -> dict:
    if model not in KINEMATICS:
        raise ValueError(f"no kinematics for model {model!r}; known: {sorted(KINEMATICS)}")
    proc = KINEMATICS[model].run(input=json.dumps(request), capture_output=True, text=True, timeout=timeout)
    try:
        response = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise KinematicsError(f"{model} kinematics failed (exit {proc.returncode}): {proc.stderr.strip()}") from None
    if proc.returncode != 0:
        raise KinematicsError(f"{model} kinematics: {response.get('error', proc.stderr.strip())}")
    return response


def chain(model: str, timeout: float = 30.0) -> Chain:
    """The model's kinematic chain: joints (root to tip), root and tip link."""
    r = _run(model, {"describe": True}, timeout)
    return Chain(r["joint_names"], r["root_link"], r["tip_link"])


def forward(model: str, joints: Mapping[str, float], timeout: float = 30.0) -> Pose:
    """Tip pose for `joints` (chain joints not given are at the model's defaults)."""
    return Pose.from_json(_run(model, {"fk": dict(joints)}, timeout)["pose"])


def forward_many(model: str, states: Sequence[Mapping[str, float]], timeout: float = 30.0) -> list[Pose]:
    return [Pose.from_json(p) for p in _run(model, {"fk": [dict(s) for s in states]}, timeout)["poses"]]


def ik_path(
    model: str,
    start: Mapping[str, float],
    poses: Sequence[Pose],
    max_joint_step: float = 0.2,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1e-2,
    check_collisions: bool = True,
    timeout: float = 60.0,
) -> IkPath:
    """Joint positions for each of `poses` in turn, each solved from the previous (the first from `start`).

    Raises KinematicsError at the first pose without a solution, or whose
    solution moves a joint more than `max_joint_step` rad (a jump to another IK
    branch: sample the poses more densely, or the path crosses a singularity or
    a joint limit), or, with `check_collisions`, whose solution puts bodies of the
    model in contact (e.g. self-collision).
    """
    r = _run(model, _ik_request(start, poses, max_joint_step, position_tolerance, orientation_tolerance, check_collisions), timeout)
    return _ik_result(model, r)


def _ik_request(start, poses, max_joint_step, position_tolerance, orientation_tolerance, check_collisions) -> dict:
    return {"ik_path": {
        "start": dict(start),
        "poses": [p.to_json() for p in poses],
        "max_joint_step": max_joint_step,
        "check_collisions": check_collisions,
        "settings": {"position_tolerance": position_tolerance, "orientation_tolerance": orientation_tolerance},
    }}


def _ik_result(model: str, r: dict) -> IkPath:
    if not r["solved"]:
        raise KinematicsError(f"{model}: {r['error']}", r["failed_index"], r["path"])
    return IkPath(r["joint_names"], r["path"], r["time_ms"], r["max_position_error"], r["max_orientation_error"])


class Kinematics:
    """The model's kinematics in one process kept for many requests (the module's functions start one per
    request, which loads the model: ~0.3 s; here a request costs its solve, ~1 ms). Not thread-safe.

        with kinematics_client.Kinematics("rebot_arm_gemini_305") as k:
            k.forward({...}); k.ik_path(start={...}, poses=[...])
    """

    def __init__(self, model: str):
        if model not in KINEMATICS:
            raise ValueError(f"no kinematics for model {model!r}; known: {sorted(KINEMATICS)}")
        self.model = model
        self._proc = KINEMATICS[model].popen(stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def request(self, request: dict) -> dict:
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):  # ValueError: closed
            raise KinematicsError(f"{self.model} kinematics exited (code {self._proc.poll()})") from None
        line = self._proc.stdout.readline()
        if not line:
            raise KinematicsError(f"{self.model} kinematics exited (code {self._proc.wait()})")
        response = json.loads(line)
        if "error" in response and "solved" not in response:
            raise KinematicsError(f"{self.model} kinematics: {response['error']}")
        return response

    def forward(self, joints: Mapping[str, float]) -> Pose:
        return Pose.from_json(self.request({"fk": dict(joints)})["pose"])

    def forward_many(self, states: Sequence[Mapping[str, float]]) -> list[Pose]:
        return [Pose.from_json(p) for p in self.request({"fk": [dict(s) for s in states]})["poses"]]

    def ik_path(self, start: Mapping[str, float], poses: Sequence[Pose], max_joint_step: float = 0.2,
                position_tolerance: float = 1e-3, orientation_tolerance: float = 1e-2, check_collisions: bool = True) -> IkPath:
        """As the module's ik_path."""
        return _ik_result(self.model, self.request(
            _ik_request(start, poses, max_joint_step, position_tolerance, orientation_tolerance, check_collisions)))

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()  # the CLI exits at the end of its input
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def __enter__(self) -> "Kinematics":
        return self

    def __exit__(self, *exc):
        self.close()
