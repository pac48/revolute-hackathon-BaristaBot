"""Orbbec Gemini 305 camera model: its URDF (generated from the MJCF) and cameras (from the MJCF).

    from src.camera_models.gemini_305 import description
    description.urdf_text()    # for <ns>/robot_description
    description.cameras()      # mjcf_cameras.Camera: name, frame, size, fovy
    description.DRIVER_ROOT    # the frame the driver (orbbec_camera) publishes from the device calibration, and below
"""

from src.camera_models.camera_model_utils import mjcf_cameras
from src.camera_models.gemini_305_data import mjcf, urdf

DRIVER_ROOT = "camera_link"


def urdf_text() -> str:
    return urdf.read_text()


def cameras() -> list[mjcf_cameras.Camera]:
    return mjcf_cameras.cameras(mjcf.read_text())
