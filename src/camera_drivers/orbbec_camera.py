"""Run the Orbbec camera driver (e.g. a Gemini 305) on this machine.

Usage: bazel run //src/camera_drivers:orbbec_camera[.remote] [-- [--dual-color] [--jpeg-quality Q] [--ros-args -p NAME:=VALUE ...]]

Starts orbbec_camera_node as /camera/camera (topics /camera/color/image_raw,
/camera/depth/image_raw, /camera/depth/points, ... as upstream's launch files
name them), so `bazel run //src/camera_drivers:orbbec_rviz` shows it.

--dual-color uses the Gemini 305's "Dual Color Streams" preset (upstream's
config/gemini305_dual_color.yaml): both color cameras, as
/camera/left_color/image_raw and /camera/right_color/image_raw.

Color topics are also published JPEG-compressed on <topic>/compressed
(image_transport's compressed plugin, encoding on this machine; the Gemini 305
has no MJPEG mode for its color streams). That is a fraction of the raw
bandwidth, so view those from another machine (orbbec_rviz does). The node only
converts/encodes what has subscribers. --jpeg-quality (1-100, default 80) trades
size for quality.

Arguments after --ros-args are passed to the node, e.g. `--ros-args -p enable_point_cloud:=false`.

The camera's URDF model (orbbec_description's gemini_305) is published on
/camera/robot_description (latched), with static TF for the model frames the
driver doesn't publish (camera_bottom_screw -> camera_link, the screws); the
driver publishes camera_link and below from the device calibration. The model
is //src/camera_models/gemini_305 (MJCF in gemini_305_data); see
camera_models/camera_model_utils/camera_description.py.

USB access needs Orbbec's udev rules on the machine with the camera:
  sudo cp $(bazel info output_base)/external/orbbec_camera+/scripts/99-obsensor-libusb.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
"""

import argparse
import signal
import subprocess
import sys

import rclpy

from src.camera_drivers import orbbec_camera_node
from src.camera_models.camera_model_utils.camera_description import CameraDescription
from src.camera_models.gemini_305 import description

CAMERA_NAME = "camera"
DEFAULTS = {
    "camera_name": CAMERA_NAME,
    "enable_color": "true",
    "enable_depth": "true",
    "enable_point_cloud": "true",
}

# Upstream's config/gemini305_dual_color.yaml (other parameters keep their defaults).
DUAL_COLOR = {
    "device_preset": "Dual Color Streams",
    "enable_left_color": "true",
    "enable_right_color": "true",
}


def main():
    argv = sys.argv[1:]
    ros_args = argv[argv.index("--ros-args") + 1:] if "--ros-args" in argv else []
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dual-color", action="store_true", help="both color cameras (Gemini 305 preset)")
    ap.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality of <topic>/compressed (default 80)")
    args = ap.parse_args(argv[:len(argv) - len(ros_args) - (1 if "--ros-args" in argv else 0)])
    params = {**DEFAULTS, **(DUAL_COLOR if args.dual_color else {})}
    for stream in ("color", "left_color", "right_color"):
        params[f"{stream}.image_raw.compressed.jpeg_quality"] = str(args.jpeg_quality)

    cmd = [orbbec_camera_node.path(), "--ros-args", "-r", f"__ns:=/{CAMERA_NAME}", "-r", f"__node:={CAMERA_NAME}"]
    for name, value in params.items():
        cmd += ["-p", f"{name}:={value}"]
    driver = subprocess.Popen(cmd + ros_args)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, _: driver.send_signal(signum))

    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    try:
        node = rclpy.create_node("description", namespace=f"/{CAMERA_NAME}")
        published = CameraDescription(node, description.urdf_text(), driver_root=description.DRIVER_ROOT)
        print(f"orbbec_camera: /{CAMERA_NAME}/robot_description published; static TF for {published.frames}",
              file=sys.stderr, flush=True)
        while driver.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        rclpy.try_shutdown()
        if driver.poll() is None:
            driver.terminate()
    return driver.wait()


if __name__ == "__main__":
    sys.exit(main())
