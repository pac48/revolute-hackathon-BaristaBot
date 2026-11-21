"""Enumerate a V4L2 camera's capture modes and pick one in its native aspect ratio.

Pure Python (fcntl ioctls on /dev/videoN), so it runs wherever the camera is,
without v4l2-ctl. select_mode() is the policy camera.py applies when
usb_cam.yaml leaves image_width / image_height at 0 (auto).
"""

import collections
import ctypes
import fcntl
import os

Mode = collections.namedtuple("Mode", "width height fps")

# usb_cam pixel_format -> V4L2 fourcc it captures.
FOURCC = {
    "yuyv": "YUYV",
    "yuyv2rgb": "YUYV",
    "uyvy": "UYVY",
    "uyvy2rgb": "UYVY",
    "mjpeg2rgb": "MJPG",
    "raw_mjpeg": "MJPG",
    "mono8": "GREY",
    "mono16": "Y16 ",
    "y102mono8": "Y10 ",
    "m4202rgb": "M420",
    "rgb8": "RGB3",
}

# usb_cam pixel_format -> bytes per pixel of the published image (None: compressed).
OUTPUT_BYTES_PER_PIXEL = {
    "yuyv": 2,
    "yuyv2rgb": 3,
    "uyvy": 2,
    "uyvy2rgb": 3,
    "mjpeg2rgb": 3,
    "raw_mjpeg": None,
    "mono8": 1,
    "mono16": 2,
    "y102mono8": 1,
    "m4202rgb": 3,
    "rgb8": 3,
}

_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_FRMSIZE_TYPE_DISCRETE = 1


class _FmtDesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _FrmSizeEnum(ctypes.Structure):
    # The union holds discrete {width, height} or stepwise {min_w, max_w, step_w, min_h, max_h, step_h}.
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("u", ctypes.c_uint32 * 6),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class _FrmIvalEnum(ctypes.Structure):
    # The union holds a discrete fraction {numerator, denominator} or stepwise {min, max, step}.
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("u", ctypes.c_uint32 * 6),
        ("reserved", ctypes.c_uint32 * 2),
    ]


def _iowr(nr: int, struct) -> int:
    return (3 << 30) | (ctypes.sizeof(struct) << 16) | (ord("V") << 8) | nr


VIDIOC_ENUM_FMT = _iowr(2, _FmtDesc)
VIDIOC_ENUM_FRAMESIZES = _iowr(74, _FrmSizeEnum)
VIDIOC_ENUM_FRAMEINTERVALS = _iowr(75, _FrmIvalEnum)


def _fourcc_code(fourcc: str) -> int:
    return int.from_bytes(fourcc.encode("ascii"), "little")


def _enum(fd, request, struct):
    """Yield struct for index 0, 1, ... until the driver says there are no more."""
    index = 0
    while True:
        struct.index = index
        try:
            fcntl.ioctl(fd, request, struct)
        except OSError:
            return
        yield struct
        index += 1


def capture_fourccs(device: str) -> list[str]:
    fd = os.open(device, os.O_RDWR)
    try:
        desc = _FmtDesc(type=_V4L2_BUF_TYPE_VIDEO_CAPTURE)
        return [d.pixelformat.to_bytes(4, "little").decode("ascii", "replace") for d in _enum(fd, VIDIOC_ENUM_FMT, desc)]
    finally:
        os.close(fd)


def modes(device: str, fourcc: str) -> list[Mode]:
    """Discrete capture modes of `device` for `fourcc` (e.g. "YUYV"), one per frame rate."""
    code = _fourcc_code(fourcc)
    out = []
    fd = os.open(device, os.O_RDWR)
    try:
        sizes = []
        for s in _enum(fd, VIDIOC_ENUM_FRAMESIZES, _FrmSizeEnum(pixel_format=code)):
            if s.type == _V4L2_FRMSIZE_TYPE_DISCRETE:
                sizes.append((s.u[0], s.u[1]))
            else:  # stepwise/continuous: offer the largest size
                sizes.append((s.u[1], s.u[4]))
        for width, height in sizes:
            ival = _FrmIvalEnum(pixel_format=code, width=width, height=height)
            for i in _enum(fd, VIDIOC_ENUM_FRAMEINTERVALS, ival):
                # u[0]/u[1] is the (discrete or, for stepwise, minimum) frame interval.
                if i.u[0]:
                    out.append(Mode(width, height, i.u[1] / i.u[0]))
    finally:
        os.close(fd)
    return out


def native_aspect(available: list[Mode]) -> float:
    """Aspect ratio of the largest mode, taken as the sensor's native one."""
    largest = max(available, key=lambda m: m.width * m.height)
    return largest.width / largest.height


def select_mode(
    available: list[Mode], width: int = 0, height: int = 0, fps: float = 30.0, max_width: int = 1280
) -> Mode:
    """Pick a supported mode.

    width and height of 0 mean auto: the largest resolution in the camera's
    native aspect ratio with width <= max_width, preferring resolutions that
    reach `fps`. With only width (or height) set, the other follows from the
    native aspect ratio. With both set, that size must be supported. The frame
    rate is the highest supported one not above `fps` (else the lowest).
    Raises ValueError, listing the supported modes, if nothing matches.
    """
    if not available:
        raise ValueError("the camera reports no modes for this pixel format")
    sizes = sorted({(m.width, m.height) for m in available}, key=lambda s: s[0] * s[1])
    aspect = native_aspect(available)

    def is_native(size):
        return abs(size[0] / size[1] - aspect) < 0.02

    def max_fps(size):
        return max(m.fps for m in available if (m.width, m.height) == size)

    if width and height:
        candidates = [s for s in sizes if s == (width, height)]
    elif width or height:
        candidates = [s for s in sizes if (s[0] == width or s[1] == height) and is_native(s)] or [
            s for s in sizes if s[0] == width or s[1] == height
        ]
    else:
        native = [s for s in sizes if is_native(s)]
        candidates = [s for s in native if s[0] <= max_width] or native[:1]
    if not candidates:
        raise ValueError(
            f"no supported mode matches {width or 'auto'}x{height or 'auto'}; supported: "
            + ", ".join(f"{w}x{h}@{max_fps((w, h)):g}" for w, h in sizes)
        )
    fast_enough = [s for s in candidates if max_fps(s) >= fps - 0.5]
    size = (fast_enough or candidates)[-1]
    rates = sorted(m.fps for m in available if (m.width, m.height) == size)
    rate = max((r for r in rates if r <= fps + 0.5), default=rates[0])
    return Mode(size[0], size[1], rate)
