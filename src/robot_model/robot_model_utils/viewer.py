"""Open an MJCF in MuJoCo's viewer (`simulate`), to inspect a model.

    bazel run //src/robot_model/<model>:view [-- EXTRA_SIMULATE_ARGS]

Each model package has a `view` target made by viewer.bzl's model_viewer(), which passes the
MJCF's runfiles path as the first argument. From Python: viewer.view(path).

The viewer is MuJoCo's own `simulate` app: orbit with the left mouse button, pan with the right
(or ctrl+left), zoom with the wheel; Space pauses; the Rendering panel (Tab) toggles
groups (the models' visual meshes are group 1, collision meshes group 3), cameras and frames; the
Physics/Control panels show joints and actuators. A model's cameras are in the Camera drop-down.
Needs a display.
"""

import os
import sys

from python.runfiles import Runfiles

from src.robot_model.robot_model_utils import viewer_simulate


def resolve(path: str) -> str:
    """`path` itself if it is a file, else a runfiles (rlocation) path, made absolute."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    found = Runfiles.Create().Rlocation(path)
    if not found or not os.path.isfile(found):
        raise FileNotFoundError(f"MJCF not found: {path}")
    return found


def view(mjcf: str, *extra_args: str) -> int:
    """Run the viewer on `mjcf` (a path or rlocation path); returns when it is closed."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("viewer: no DISPLAY; the viewer needs a window", file=sys.stderr)
        return 1
    return viewer_simulate.run(resolve(mjcf), *extra_args).returncode


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: viewer MJCF [SIMULATE_ARGS...]", file=sys.stderr)
        return 2
    return view(*sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
