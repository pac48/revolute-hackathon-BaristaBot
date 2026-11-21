"""The cameras of a camera model's MJCF: where they are and their pinhole intrinsics.

    for cam in cameras(mjcf_text): cam.name, cam.frame, cam.width, cam.height, cam.fovy

MuJoCo `<camera>` elements are the single description of a camera's optics:
`resolution` (nominal image size) and `fovy` (vertical field of view, degrees; pixels
are square). `frame` is the body the camera sits in, which is also the URDF link
generated from the MJCF; the cameras here sit in the ROS optical frame (+Z forward,
+X right, +Y down), so `frame` is the frame image messages are stamped with. MuJoCo's own
camera looks along -Z, so the element is rotated 180 degrees about X inside that body.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    name: str
    frame: str  # body (URDF link) the camera is in
    width: int
    height: int
    fovy: float  # degrees

    def intrinsics(self, width: int | None = None, height: int | None = None) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) in pixels for an image of `width` x `height` (default: nominal).

        The field of view is kept: a different size of the same aspect ratio scales the
        intrinsics, a different aspect ratio crops/extends horizontally (square pixels)."""
        w, h = width or self.width, height or self.height
        f = (h / 2) / math.tan(math.radians(self.fovy) / 2)
        return f, f, w / 2, h / 2


def cameras(mjcf_text: str) -> list[Camera]:
    """Every `<camera>` in the MJCF, in document order (cameras need a `resolution` and `fovy`)."""
    found = []

    def visit(element, body):
        for child in element:
            if child.tag == "camera":
                resolution = child.get("resolution")
                if resolution is None or child.get("fovy") is None:
                    raise ValueError(f"camera {child.get('name')!r} needs resolution and fovy")
                w, h = (int(v) for v in resolution.split())
                found.append(Camera(child.get("name"), body, w, h, float(child.get("fovy"))))
            else:
                visit(child, child.get("name") if child.tag == "body" else body)

    visit(ET.fromstring(mjcf_text), "world")
    return found
