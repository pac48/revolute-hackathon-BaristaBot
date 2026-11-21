"""Quest Teleop wire protocol, version 1 (src/teleop/quest_app/PROTOCOL.md).

JSON text messages over one WebSocket; the headset is the client. Every pose is in ROS
convention (x forward, y left, z up), meters, in the robot base frame. Quaternions are
[x, y, z, w].
"""

import json
import struct
from dataclasses import dataclass, field

VERSION = 1

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

LEFT_BUTTONS = ("x", "y", "stick", "menu")
RIGHT_BUTTONS = ("a", "b", "stick")


@dataclass
class Pose:
    p: Vec3 = (0.0, 0.0, 0.0)
    q: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Hand:
    active: bool = False
    pose: Pose | None = None  # None while not tracked
    trigger: float = 0.0
    grip: float = 0.0
    stick: tuple[float, float] = (0.0, 0.0)
    buttons: dict[str, bool] = field(default_factory=dict)


@dataclass
class Controllers:
    seq: int
    t: float  # headset monotonic clock, s
    head: Pose | None
    left: Hand
    right: Hand


class ProtocolError(ValueError):
    pass


@dataclass
class View:
    """A headset view frame (binary message): "VIEW" | u16 width | u16 height | f64 t | JPEG."""

    width: int
    height: int
    t: float  # headset monotonic clock, s
    jpeg: bytes


VIEW_HEADER = struct.Struct("<4sHHd")


def parse_view(data: bytes) -> View:
    if len(data) < VIEW_HEADER.size:
        raise ProtocolError(f"binary message too short ({len(data)} bytes)")
    tag, width, height, t = VIEW_HEADER.unpack_from(data)
    if tag != b"VIEW":
        raise ProtocolError(f"unknown binary message {tag!r}")
    jpeg = bytes(data[VIEW_HEADER.size:])
    if not jpeg.startswith(b"\xff\xd8"):
        raise ProtocolError("view payload is not a JPEG")
    return View(width, height, t, jpeg)


def _floats(value, n: int, what: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != n:
        raise ProtocolError(f"{what}: expected {n} numbers")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"{what}: {e}") from None


def _pose(obj: dict, what: str) -> Pose:
    p = _floats(obj.get("p"), 3, what + ".p")
    q = _floats(obj.get("q"), 4, what + ".q")
    norm = sum(c * c for c in q) ** 0.5
    if norm < 1e-6:
        raise ProtocolError(f"{what}.q: zero quaternion")
    return Pose(p, tuple(c / norm for c in q))


def _hand(obj, names: tuple[str, ...], what: str) -> Hand:
    if not isinstance(obj, dict):
        return Hand(buttons={n: False for n in names})
    active = bool(obj.get("active", False))
    buttons = obj.get("buttons") if isinstance(obj.get("buttons"), dict) else {}
    return Hand(
        active=active,
        pose=_pose(obj, what) if active else None,
        trigger=float(obj.get("trigger", 0.0)),
        grip=float(obj.get("grip", 0.0)),
        stick=_floats(obj.get("stick", [0.0, 0.0]), 2, what + ".stick"),
        buttons={n: bool(buttons.get(n, 0)) for n in names},
    )


def parse(text: str | bytes) -> dict:
    """Decodes one message to a dict with a "type" key; raises ProtocolError."""
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"bad JSON: {e}") from None
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ProtocolError("message must be an object with a string 'type'")
    return msg


def controllers_from(msg: dict) -> Controllers:
    head = msg.get("head")
    return Controllers(
        seq=int(msg.get("seq", 0)),
        t=float(msg.get("t", 0.0)),
        head=_pose(head, "head") if isinstance(head, dict) else None,
        left=_hand(msg.get("left"), LEFT_BUTTONS, "left"),
        right=_hand(msg.get("right"), RIGHT_BUTTONS, "right"),
    )


def encode_gripper_pose(pose: Pose, t: float) -> str:
    return json.dumps(
        {
            "type": "gripper_pose",
            "t": round(t, 6),
            "p": [round(v, 5) for v in pose.p],
            "q": [round(v, 6) for v in pose.q],
        },
        separators=(",", ":"),
    )
