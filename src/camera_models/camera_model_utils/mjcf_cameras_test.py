"""mjcf_cameras: cameras with their body and intrinsics."""

import math
import unittest

from src.camera_models.camera_model_utils import mjcf_cameras

MJCF = """<mujoco><worldbody>
  <body name="base"><body name="optical" quat="0.5 -0.5 0.5 -0.5">
    <camera name="color" euler="180 0 0" resolution="1280 720" fovy="60"/>
  </body></body>
</worldbody></mujoco>"""


class MjcfCamerasTest(unittest.TestCase):
    def test_camera_in_its_body(self):
        (cam,) = mjcf_cameras.cameras(MJCF)
        self.assertEqual((cam.name, cam.frame, cam.width, cam.height, cam.fovy), ("color", "optical", 1280, 720, 60.0))

    def test_intrinsics(self):
        (cam,) = mjcf_cameras.cameras(MJCF)
        fx, fy, cx, cy = cam.intrinsics()
        self.assertAlmostEqual(fy, 360 / math.tan(math.radians(30)))
        self.assertEqual((fx, cx, cy), (fy, 640, 360))
        self.assertAlmostEqual(cam.intrinsics(640, 360)[0], fy / 2)  # same view, half the pixels

    def test_missing_optics_rejected(self):
        with self.assertRaises(ValueError):
            mjcf_cameras.cameras('<mujoco><worldbody><camera name="c"/></worldbody></mujoco>')


if __name__ == "__main__":
    unittest.main()
