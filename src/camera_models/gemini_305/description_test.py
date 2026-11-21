"""gemini_305 description: the MJCF's cameras are links of the generated URDF."""

import re
import unittest

from src.camera_models.gemini_305 import description


class DescriptionTest(unittest.TestCase):
    def test_cameras_are_urdf_frames(self):
        links = set(re.findall(r'<link name="([^"]+)"', description.urdf_text()))
        cameras = description.cameras()
        self.assertTrue(cameras)
        for cam in cameras:
            self.assertIn(cam.frame, links)
            self.assertGreater(cam.fovy, 0)
        if description.DRIVER_ROOT is not None:
            self.assertIn(description.DRIVER_ROOT, links)

    def test_camera_names_unique(self):
        names = [c.name for c in description.cameras()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
