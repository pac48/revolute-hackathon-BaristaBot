"""select_mode on the mode lists two real cameras report."""

import unittest

from src.camera_drivers.v4l2_modes import Mode, native_aspect, select_mode


def expand(table):
    return [Mode(w, h, f) for (w, h), rates in table.items() for f in rates]


# Elgato Cam Link 4K, YUYV (16:9 HDMI capture; 720x576/720x480/640x480 are SD modes).
CAM_LINK = expand({
    (1920, 1080): [60, 50, 30, 25], (1280, 720): [60, 50, 30, 25], (720, 576): [60, 50, 30, 25],
    (720, 480): [60, 50, 30, 25], (640, 480): [60, 50, 30, 25],
})
# Lenovo integrated camera, YUYV (16:9 sensor; large sizes are slow).
INTEGRATED = expand({
    (1280, 720): [10], (960, 540): [15], (848, 480): [20], (640, 480): [30], (640, 360): [30],
    (424, 240): [30], (352, 288): [30], (320, 240): [30], (320, 180): [30],
})


class SelectModeTest(unittest.TestCase):
    def test_native_aspect(self):
        self.assertAlmostEqual(native_aspect(CAM_LINK), 16 / 9)
        self.assertAlmostEqual(native_aspect(INTEGRATED), 16 / 9)

    def test_auto_keeps_native_aspect_within_max_width(self):
        self.assertEqual(select_mode(CAM_LINK), Mode(1280, 720, 30))
        self.assertEqual(select_mode(CAM_LINK, max_width=1920, fps=60), Mode(1920, 1080, 60))

    def test_auto_prefers_sizes_that_reach_the_frame_rate(self):
        # 1280x720 is native but only 10 Hz; 640x360 is the largest native size at 30 Hz.
        self.assertEqual(select_mode(INTEGRATED), Mode(640, 360, 30))
        self.assertEqual(select_mode(INTEGRATED, fps=10), Mode(1280, 720, 10))

    def test_width_only_follows_native_aspect(self):
        self.assertEqual(select_mode(CAM_LINK, width=1920), Mode(1920, 1080, 30))
        self.assertEqual(select_mode(INTEGRATED, height=480), Mode(848, 480, 20))

    def test_both_set_must_be_supported(self):
        self.assertEqual(select_mode(CAM_LINK, 640, 480), Mode(640, 480, 30))
        with self.assertRaisesRegex(ValueError, "supported: .*1280x720@60"):
            select_mode(CAM_LINK, 800, 600)

    def test_frame_rate_at_or_below_target(self):
        self.assertEqual(select_mode(CAM_LINK, fps=55), Mode(1280, 720, 50))
        self.assertEqual(select_mode(CAM_LINK, fps=10), Mode(1280, 720, 25))  # none below: lowest


if __name__ == "__main__":
    unittest.main()
