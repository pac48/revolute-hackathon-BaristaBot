"""A camera's URDF model as a ROS topic, plus the TF its driver doesn't publish.

    publisher = CameraDescription(node, urdf_text, driver_root="camera_link")  # or None


publishes `urdf_text` on <node namespace>/robot_description (std_msgs/String,
transient local, like robot_state_publisher) and, on /tf_static, the URDF's
fixed joints outside the subtree under `driver_root`. The driver publishes that
subtree itself from the device's calibration (e.g. orbbec_camera: camera_link ->
color/depth/optical frames), so only the rest of the model (mounting screws,
the bottom screw that parents camera_link) comes from the URDF's nominal values.
With driver_root=None the driver publishes no TF and every joint is published.
The URDF is generated from the camera's MJCF (see camera_models/*_data), which has
a "world" root; its joints from "world" are the model's mounting, not published.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class Joint:
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


def fixed_joints(urdf_text: str) -> list[Joint]:
    joints = []
    for j in ET.fromstring(urdf_text).iter("joint"):
        if j.get("type") != "fixed":
            raise ValueError(f"joint {j.get('name')!r} is {j.get('type')}; a camera model should be rigid")
        origin = j.find("origin")
        parse = lambda attr: tuple(float(v) for v in (origin.get(attr, "0 0 0") if origin is not None else "0 0 0").split())
        joints.append(Joint(j.find("parent").get("link"), j.find("child").get("link"), parse("xyz"), parse("rpy")))
    return joints


def model_only_joints(joints: list[Joint], driver_root: str | None) -> list[Joint]:
    """Joints whose child isn't below `driver_root` (that subtree is the driver's), none of
    them from "world"."""
    joints = [j for j in joints if j.parent != "world"]
    if driver_root is None:
        return joints
    children = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j.child)
    driver_frames, stack = set(), [driver_root]
    while stack:
        frame = stack.pop()
        stack.extend(c for c in children.get(frame, []) if c not in driver_frames)
        driver_frames.add(frame)
    return [j for j in joints if j.parent not in driver_frames]


def quaternion(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """(x, y, z, w) for URDF roll/pitch/yaw (fixed axes X, Y, Z)."""
    (cr, sr), (cp, sp), (cy, sy) = ((math.cos(a / 2), math.sin(a / 2)) for a in rpy)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class CameraDescription:
    def __init__(self, node, urdf_text: str, driver_root: str | None):
        from geometry_msgs.msg import TransformStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._description = node.create_publisher(String, "robot_description", latched)
        self._description.publish(String(data=urdf_text))
        self._tf_static = node.create_publisher(TFMessage, "/tf_static", latched)
        stamp = node.get_clock().now().to_msg()
        transforms = []
        for j in model_only_joints(fixed_joints(urdf_text), driver_root):
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, j.parent, j.child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = j.xyz
            r = t.transform.rotation
            r.x, r.y, r.z, r.w = quaternion(j.rpy)
            transforms.append(t)
        self.frames = [t.child_frame_id for t in transforms]
        self._tf_static.publish(TFMessage(transforms=transforms))
