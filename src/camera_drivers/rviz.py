"""Launch RViz with a camera view.

Usage: [ROS_DOMAIN_ID=<camera's domain>] bazel run //src/camera_drivers:rviz [-- RVIZ_ARGS...]
       [ROS_DOMAIN_ID=<camera's domain>] bazel run //src/camera_drivers:orbbec_rviz [-- RVIZ_ARGS...]

With no arguments, `rviz` opens config/camera.rviz (usb_cam's /image_raw) and
`orbbec_rviz` opens config/orbbec.rviz (Orbbec Gemini 305: color, depth, point
cloud, TF); any arguments are passed to rviz2 instead (e.g. `-d other.rviz`).
RViz must be on the same ROS_DOMAIN_ID as the camera.

Started without a display (e.g. `bazel run //src/camera_drivers:orbbec_rviz.remote`,
over ssh), RViz opens on that machine's logged-in desktop: X display :0 (Xwayland
under a Wayland session), authorized with the session's X authority file.
"""

import glob
import os
import sys

from src.camera_drivers import rviz2


def desktop_display_env() -> dict[str, str]:
    """DISPLAY/XAUTHORITY for the local desktop session when none is set (started over ssh)."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return {}
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    candidates = [
        os.environ.get("XAUTHORITY", ""),
        *sorted(glob.glob(f"{runtime_dir}/.mutter-Xwaylandauth.*")),  # GNOME (Wayland)
        f"{runtime_dir}/gdm/Xauthority",  # GNOME (Xorg)
        os.path.expanduser("~/.Xauthority"),
    ]
    env = {"DISPLAY": ":0"}
    xauthority = next((c for c in candidates if c and os.path.exists(c)), None)
    if xauthority:
        env["XAUTHORITY"] = xauthority
    print(f"rviz: no display set, using the desktop's ({env})", file=sys.stderr)
    return env


def main(config=None):
    if config is None:
        from src.camera_drivers import camera_rviz_config as config
    args = sys.argv[1:] or ["-d", config.path()]
    os.environ.update(desktop_display_env())
    os.execv(rviz2.path(), [rviz2.path(), *args])


if __name__ == "__main__":
    main()
