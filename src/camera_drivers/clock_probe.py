"""Clock echo protocol shared by camera.py (echo) and image_latency.py (prober).

The prober publishes a std_msgs/Header on PING_TOPIC with its wall-clock time
in stamp; the echo answers on PONG_TOPIC with its own wall-clock time in stamp
and the ping's stamp in frame_id ("sec.nanosec").
"""

from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy

PING_TOPIC = "/clock_probe/ping"
PONG_TOPIC = "/clock_probe/pong"
QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)


def to_time(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def to_ns(t: Time) -> int:
    return t.sec * 1_000_000_000 + t.nanosec


def encode_ping_stamp(t: Time) -> str:
    return f"{t.sec}.{t.nanosec:09d}"


def decode_ping_stamp(frame_id: str) -> int:
    sec, _, nsec = frame_id.partition(".")
    return int(sec) * 1_000_000_000 + int(nsec)
