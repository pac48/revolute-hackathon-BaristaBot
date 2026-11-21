"""Measure camera image latency and rate from capture (on the camera's machine) to receipt here.

Usage (with `camera` running, here or on the remote build host):
  bazel run //src/camera_drivers:image_latency -- [--format raw|mjpeg] [--duration 10]
      [--reliability reliable|best_effort] [--json out.json]
Any image topic (e.g. orbbec_camera's), without camera.py's clock echo:
  bazel run //src/camera_drivers:image_latency -- --topic /camera/left_color/image_raw/compressed \
      --assume-synced-clocks
(<topic>/compressed is read as CompressedImage; latency is then only as good as the clock sync.)

Latency per frame = receive time here - capture stamp, with the capture stamp
moved to this machine's clock using the offset measured through camera.py's
clock echo (minimum-round-trip ping, i.e. Cristian's algorithm; the offset is
accurate to about half that round trip, which is reported). Also reports the
received frame rate, frames missing from the stamp sequence, frame size and
throughput.
"""

import argparse
import json
import statistics
import sys
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from src.camera_drivers import clock_probe

TOPICS = {"raw": ("/image_raw", Image), "mjpeg": ("/image_raw/compressed", CompressedImage)}


def measure_clock_offset(node, samples: int, timeout: float) -> tuple[int, int]:
    """(offset, round trip) in ns, offset = camera clock - local clock, from the fastest ping."""
    pub = node.create_publisher(Header, clock_probe.PING_TOPIC, clock_probe.QOS)
    replies = []  # (t0, remote, t1)

    def on_pong(msg: Header):
        t1 = time.time_ns()
        replies.append((clock_probe.decode_ping_stamp(msg.frame_id), clock_probe.to_ns(msg.stamp), t1))

    sub = node.create_subscription(Header, clock_probe.PONG_TOPIC, on_pong, clock_probe.QOS)
    deadline = time.monotonic() + timeout
    # Wait for the echo to be discovered and matched both ways.
    while (pub.get_subscription_count() == 0 or sub.get_publisher_count() == 0) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    sent = 0
    while len(replies) < samples and time.monotonic() < deadline:
        if sent <= len(replies) + 1:  # at most a couple in flight
            t0 = time.time_ns()
            pub.publish(Header(stamp=clock_probe.to_time(t0)))
            sent += 1
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_subscription(sub)
    node.destroy_publisher(pub)
    if not replies:
        raise TimeoutError(f"no clock echo reply on {clock_probe.PONG_TOPIC} within {timeout}s (is camera running?)")
    t0, remote, t1 = min(replies, key=lambda r: r[2] - r[0])
    return remote - (t0 + t1) // 2, t1 - t0


def _stats(values):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, round(q * (len(values) - 1)))]
    return {"n": len(values), "min": values[0], "median": statistics.median(values), "p95": pick(0.95),
            "p99": pick(0.99), "max": values[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=sorted(TOPICS), default="raw", help="must match camera --format")
    ap.add_argument("--topic", help="any image topic instead of --format's; .../compressed is a CompressedImage")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to record (default 10)")
    ap.add_argument("--reliability", choices=["reliable", "best_effort"], default="reliable")
    ap.add_argument("--depth", type=int, default=10, help="subscription queue depth")
    ap.add_argument("--clock-samples", type=int, default=50)
    ap.add_argument("--assume-synced-clocks", action="store_true", help="skip the clock echo, offset 0")
    ap.add_argument("--timeout", type=float, default=15.0, help="for the clock echo and the first frame")
    ap.add_argument("--json", help="also write the per-frame samples and summary here")
    args = ap.parse_args()
    topic, msg_type = TOPICS[args.format]
    if args.topic:
        topic, msg_type = args.topic, CompressedImage if args.topic.endswith("/compressed") else Image

    rclpy.init()
    try:
        node = rclpy.create_node("image_latency")
        if args.assume_synced_clocks:
            offset, rtt = 0, None
        else:
            offset, rtt = measure_clock_offset(node, args.clock_samples, args.timeout)
            print(f"clock offset (camera - here): {offset / 1e6:+.3f} ms, best ping round trip {rtt / 1e6:.3f} ms "
                  f"(offset accurate to about +/-{rtt / 2e6:.3f} ms)", file=sys.stderr, flush=True)

        frames = []  # (receive ns, capture ns on the camera clock, bytes)
        qos = QoSProfile(
            depth=args.depth,
            reliability=ReliabilityPolicy.RELIABLE if args.reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
        )
        shape = {}

        def on_frame(msg):
            frames.append((time.time_ns(), clock_probe.to_ns(msg.header.stamp), len(msg.data)))
            if hasattr(msg, "width"):
                shape.update(size=f"{msg.width}x{msg.height}", encoding=msg.encoding)

        node.create_subscription(msg_type, topic, on_frame, qos)
        deadline = time.monotonic() + args.timeout
        while not frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if not frames:
            print(f"image_latency: no frames on {topic} within {args.timeout}s", file=sys.stderr)
            return 1
        frames.clear()  # start the recording at a frame boundary, after discovery
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        rclpy.try_shutdown()

    if len(frames) < 2:
        print(f"image_latency: only {len(frames)} frames in {args.duration}s", file=sys.stderr)
        return 1
    latencies_ms = [(recv - (stamp - offset)) / 1e6 for recv, stamp, _ in frames]
    stamps = [s for _, s, _ in frames]
    periods = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
    period = statistics.median(periods)
    missing = sum(max(0, round(p / period) - 1) for p in periods)
    span_s = (frames[-1][0] - frames[0][0]) / 1e9
    total_bytes = sum(b for *_, b in frames)
    lat = _stats(latencies_ms)
    summary = {
        "topic": topic,
        "reliability": args.reliability,
        "clock_offset_ms": offset / 1e6,
        "clock_rtt_ms": None if rtt is None else rtt / 1e6,
        "frames": len(frames),
        "camera_rate_hz": 1000 / period,
        "received_rate_hz": (len(frames) - 1) / span_s,
        "missing_frames": missing,
        "frame_bytes": statistics.median(b for *_, b in frames),
        "image": shape,
        "throughput_mb_s": total_bytes / span_s / 1e6,
        "latency_ms": lat,
    }
    print(
        f"\n{topic} ({args.reliability}), {len(frames)} frames in {span_s:.1f} s\n"
        f"  rate:        {summary['received_rate_hz']:.1f} Hz received, camera {summary['camera_rate_hz']:.1f} Hz, "
        f"{missing} missing\n"
        f"  frame size:  {summary['frame_bytes'] / 1e3:.1f} kB"
        + (f" ({shape['size']} {shape['encoding']})" if shape else "")
        + f", throughput {summary['throughput_mb_s']:.1f} MB/s\n"
        f"  latency ms:  min {lat['min']:.1f} / median {lat['median']:.1f} / p95 {lat['p95']:.1f} / "
        f"p99 {lat['p99']:.1f} / max {lat['max']:.1f}"
        + ("" if rtt is None else f"  (clock offset +/-{rtt / 2e6:.2f} ms)")
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "frames": frames}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
