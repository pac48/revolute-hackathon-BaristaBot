"""Run a USB camera (usb_cam) plus a clock echo for latency measurements.

Usage (on the machine with the camera, e.g. the remote build host):
  bazel run //src/camera_drivers:camera.remote -- [--device D] [--width W --height H --fps F]
      [--format raw | --pixel-format P] [--namespace NS]

Several cameras at once: one run per camera, each with its own --device and --namespace, e.g.
  bazel run //src/camera_drivers:camera -- --namespace /scene_camera --device /dev/v4l/by-id/...
  bazel run //src/camera_drivers:camera -- --namespace /center_camera --device /dev/v4l/by-id/...
publishes /scene_camera/image_raw and /center_camera/image_raw (the names lerobot_export.conventions.CAMERAS
expects, after the recorder compresses them to <topic>/compressed).

The camera settings come from config/usb_cam.yaml; each flag given overrides
the matching parameter for this run only:
  --device        video_device; /dev/v4l/by-id/...-video-index0 paths are stable across
                  replugging (resolved here: usb_cam can't open symlinks itself)
  --width/--height/--fps   image_width / image_height / framerate
  --namespace     ROS namespace of this camera's topics and nodes (default: none, /image_raw)

image_width / image_height of 0 (the default) pick the mode automatically from
what the camera reports (v4l2_modes.select_mode): its native aspect ratio, the
largest size up to --max-width that reaches framerate. Set only one of them to
choose a size in the native aspect ratio; set both to force a supported size.
  --format raw    pixel_format yuyv2rgb: camera YUYV converted to rgb8 on /image_raw
  --pixel-format  any usb_cam pixel_format

JPEG passthrough (/image_raw/compressed) doesn't work in usb_cam 0.8.1: the
format is registered as "raw_mjpeg" but the node only sets up the compressed
publisher for "mjpeg", and it publishes a fixed-size buffer rather than each
JPEG's length.

raw uses the camera's uncompressed YUYV mode, so this machine doesn't decode
JPEG; --pixel-format mjpeg2rgb decodes camera MJPEG instead (needs ffmpeg's
MJPEG decoder, enabled in .bazelrc).

Header stamps are the V4L2 capture time, on this machine's wall clock.

The clock echo (clock_probe.py) lets a receiver on another machine measure the
offset between the two clocks; see image_latency.py.
"""

import argparse
import os
import signal
import sys
import threading
import time

import rclpy
from std_msgs.msg import Header

import yaml

from src.camera_drivers import clock_probe, usb_cam_node, usb_cam_params, v4l2_modes
from src.camera_models.camera_model_utils.camera_description import CameraDescription
from src.camera_models.usb_camera import description

CAMERA_NAME = "camera"  # namespace of robot_description, as the Orbbec driver's

# --format -> usb_cam pixel_format.
PIXEL_FORMATS = {
    "raw": "yuyv2rgb",  # rgb8 on /image_raw
}


def _stop(signum=None, frame=None):
    raise KeyboardInterrupt


def _stop_on_stdin_eof():
    sys.stdin.read()
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # All default to None: only flags given override usb_cam.yaml.
    ap.add_argument("--format", choices=sorted(PIXEL_FORMATS))
    ap.add_argument("--pixel-format", help="usb_cam pixel_format, overriding --format (e.g. uyvy2rgb, yuyv)")
    ap.add_argument("--device")
    ap.add_argument("--width", type=int, help="0 = auto (see usb_cam.yaml)")
    ap.add_argument("--height", type=int, help="0 = auto (see usb_cam.yaml)")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--max-width", type=int, default=1280, help="largest width auto picks (default 1280)")
    ap.add_argument("--exit-on-stdin-eof", action="store_true", help="stop when stdin closes")
    ap.add_argument("--namespace", default="",
                    help="ROS namespace for this camera, e.g. /scene_camera -> /scene_camera/image_raw (default: none)")
    args = ap.parse_args()

    # "" (no namespace) or "/name": usb_cam's __ns remap and rclpy both want a leading slash.
    ns = "/" + args.namespace.strip("/") if args.namespace.strip("/") else ""

    signal.signal(signal.SIGTERM, _stop)
    if args.exit_on_stdin_eof:
        threading.Thread(target=_stop_on_stdin_eof, daemon=True).start()

    # Settings: usb_cam.yaml, then the flags given.
    with open(usb_cam_params.path()) as f:
        params = dict(yaml.safe_load(f)["/**"]["ros__parameters"])
    flags = {
        "video_device": args.device,
        "pixel_format": args.pixel_format or (PIXEL_FORMATS[args.format] if args.format else None),
        "image_width": args.width,
        "image_height": args.height,
        "framerate": args.fps,
    }
    params.update({k: v for k, v in flags.items() if v is not None})

    # usb_cam mangles symlinks such as /dev/v4l/by-id/... ("/dev/../../video2"); resolve them here.
    device = os.path.realpath(params["video_device"])
    pixel_format = params["pixel_format"]
    fourcc = v4l2_modes.FOURCC.get(pixel_format)
    if fourcc is None:
        print(f"camera: unknown pixel_format {pixel_format!r}", file=sys.stderr)
        return 2
    try:
        available = v4l2_modes.modes(device, fourcc)
        mode = v4l2_modes.select_mode(
            available,
            int(params.get("image_width", 0)),
            int(params.get("image_height", 0)),
            float(params.get("framerate", 30.0)),
            args.max_width,
        )
    except (OSError, ValueError) as e:
        print(f"camera: {device} ({fourcc}): {e}", file=sys.stderr)
        if isinstance(e, ValueError):
            print(f"camera: {device} offers {v4l2_modes.capture_fourccs(device)}", file=sys.stderr)
        return 2
    bpp = v4l2_modes.OUTPUT_BYTES_PER_PIXEL.get(pixel_format)
    bandwidth = "" if bpp is None else f", ~{mode.width * mode.height * bpp * mode.fps / 1e6:.0f} MB/s"
    print(
        f"camera: {device} {fourcc} {mode.width}x{mode.height}@{mode.fps:g} (native aspect "
        f"{v4l2_modes.native_aspect(available):.3f}), publishing {pixel_format}{bandwidth}",
        file=sys.stderr,
        flush=True,
    )
    overrides = {
        "video_device": device,
        "pixel_format": pixel_format,
        "image_width": mode.width,
        "image_height": mode.height,
        "framerate": float(mode.fps),
        "frame_id": description.cameras()[0].frame,  # the model's optical frame
    }
    camera = usb_cam_node.popen(
        "--ros-args",
        *(("-r", f"__ns:={ns}") if ns else ()),
        "--params-file", usb_cam_params.path(),
        *(arg for k, v in overrides.items() for arg in ("-p", f"{k}:={v}")),
    )
    rclpy.init()
    try:
        node = rclpy.create_node("clock_echo", namespace=ns or None)
        # The model (//src/camera_models/usb_camera) on <ns>/camera/robot_description, with its static TF.
        description_node = rclpy.create_node("description", namespace=f"{ns}/{CAMERA_NAME}")
        CameraDescription(description_node, description.urdf_text(), driver_root=description.DRIVER_ROOT)
        pong = node.create_publisher(Header, clock_probe.PONG_TOPIC, clock_probe.QOS)

        def on_ping(ping: Header):
            now = clock_probe.to_time(time.time_ns())
            pong.publish(Header(stamp=now, frame_id=clock_probe.encode_ping_stamp(ping.stamp)))

        node.create_subscription(Header, clock_probe.PING_TOPIC, on_ping, clock_probe.QOS)
        print("camera: clock echo running", file=sys.stderr, flush=True)
        while camera.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"camera: usb_cam exited with code {camera.returncode}", file=sys.stderr)
        return camera.returncode or 1
    except KeyboardInterrupt:
        return 0
    finally:
        if camera.poll() is None:
            camera.send_signal(signal.SIGINT)
            try:
                camera.wait(5)
            except Exception:
                camera.kill()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
