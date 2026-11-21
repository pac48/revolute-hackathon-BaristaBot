"""Generic USB camera model: its URDF (generated from the MJCF) and cameras (from the MJCF).

    from src.camera_models.usb_camera import description
    description.urdf_text()    # for <ns>/robot_description
    description.cameras()      # mjcf_cameras.Camera: name, frame, size, fovy
    description.DRIVER_ROOT    # None: the usb_cam driver publishes no TF
"""

from src.camera_models.camera_model_utils import mjcf_cameras
from src.camera_models.usb_camera_data import mjcf, urdf

DRIVER_ROOT = None


def urdf_text() -> str:
    return urdf.read_text()


def cameras() -> list[mjcf_cameras.Camera]:
    return mjcf_cameras.cameras(mjcf.read_text())
