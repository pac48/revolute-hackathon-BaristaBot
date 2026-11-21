"""Collision-free joint paths from a robot model's motion planner.

    path = planner_client.plan("rebot_arm", start={"joint1": 0, ...}, goal={...},
                               obstacles={"boxes": [{"center": [x, y, z], "half_extents": [hx, hy, hz]}]})
    path.joint_names, path.waypoints

Runs the model's planner CLI (e.g. //src/robot_model/rebot_arm:plan; JSON
protocol in //src/robot_model/robot_model_utils:planner_cli.hpp). Obstacles are
spheres ({"center", "radius"}) and boxes ({"center", "half_extents", "rpy"}) in
the robot's base frame. Planning joints missing from start/goal stay at the
model's defaults.
"""

import dataclasses
import json
from collections.abc import Mapping

from src.trajectory_generator import rebot_arm_gemini_305_planner, rebot_arm_planner

# Model name -> planner CLI (py_data_library). Add an entry per model with a planner.
PLANNERS = {
    "rebot_arm": rebot_arm_planner,
    "rebot_arm_gemini_305": rebot_arm_gemini_305_planner,
}


class PlanningError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class PlannedPath:
    joint_names: list[str]
    waypoints: list[list[float]]  # start ... goal, in joint_names order
    planning_time_ms: float
    simplify_time_ms: float
    iterations: int


def _run(model: str, request: dict, timeout: float) -> dict:
    if model not in PLANNERS:
        raise ValueError(f"no planner for model {model!r}; known: {sorted(PLANNERS)}")
    proc = PLANNERS[model].run(input=json.dumps(request), capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise PlanningError(f"{model} planner failed (exit {proc.returncode}): {proc.stderr.strip()}") from None


def joint_names(model: str, timeout: float = 30.0) -> list[str]:
    """The planning joints of `model`'s planner."""
    return _run(model, {"describe": True}, timeout)["joint_names"]


def plan(
    model: str,
    start: Mapping[str, float],
    goal: Mapping[str, float],
    obstacles: Mapping | None = None,
    max_iterations: int = 100000,
    simplify: bool = True,
    timeout: float = 30.0,
) -> PlannedPath:
    """Plan with `model`'s planner; raises PlanningError if there is no path."""
    request = {
        "start": dict(start),
        "goal": dict(goal),
        "obstacles": dict(obstacles or {}),
        "settings": {"max_iterations": max_iterations, "simplify": simplify},
    }
    response = _run(model, request, timeout)
    if not response.get("solved"):
        raise PlanningError(f"{model} planner: {response.get('error') or 'no path'}")
    return PlannedPath(
        joint_names=response["joint_names"],
        waypoints=response["path"],
        planning_time_ms=response["planning_time_ms"],
        simplify_time_ms=response["simplify_time_ms"],
        iterations=response["iterations"],
    )
