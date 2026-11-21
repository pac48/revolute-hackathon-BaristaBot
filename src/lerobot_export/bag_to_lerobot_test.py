"""bag_to_lerobot's time alignment, quaternion signs and image decoding (no bags, no kinematics)."""

import argparse
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.lerobot_export import bag_to_lerobot as b


class AlignmentTest(unittest.TestCase):
    def test_frame_times_cover_the_overlap(self):
        series = {"a": b.Series(stamps=[1.0, 2.0, 3.0]), "b": b.Series(stamps=[1.22, 2.5])}
        np.testing.assert_allclose(b.frame_times(series, 4), [1.22, 1.47, 1.72, 1.97, 2.22, 2.47])
        series["b"] = b.Series(stamps=[1.0, 3.0])
        self.assertEqual(len(b.frame_times(series, 20)), 41)  # both ends: 2 s at 20 Hz
        with self.assertRaises(ValueError):
            b.frame_times({"a": b.Series(stamps=[0.0, 1.0]), "b": b.Series(stamps=[2.0, 3.0])}, 10)

    def test_nearest(self):
        stamps = [0.0, 0.05, 0.1, 0.15]
        np.testing.assert_array_equal(b.nearest(stamps, np.array([-1.0, 0.02, 0.03, 0.1, 0.2])), [0, 0, 1, 2, 3])
        np.testing.assert_array_equal(b.nearest([5.0], np.array([0.0, 9.0])), [0, 0])

    def test_latest(self):
        stamps = [0.0, 0.02, 0.04]
        np.testing.assert_array_equal(b.latest(stamps, np.array([-0.01, 0.0, 0.019, 0.02, 0.5])), [0, 0, 0, 1, 2])

    def test_interpolate(self):
        values = b.interpolate([0.0, 1.0], [[0.0, 10.0], [1.0, 20.0]], np.array([-1.0, 0.25, 2.0]))
        np.testing.assert_allclose(values, [[0.0, 10.0], [0.25, 12.5], [1.0, 20.0]])


class QuaternionTest(unittest.TestCase):
    def test_signs_follow_the_previous_and_the_reference(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        pose = np.array([[0, 0, 0, *q], [1, 0, 0, *-q], [2, 0, 0, *-q], [3, 0, 0, *q]])
        fixed = b.continuous_quaternions(pose)
        np.testing.assert_array_equal(fixed[:, 3:], np.tile(q, (4, 1)))
        np.testing.assert_array_equal(fixed[:, :3], pose[:, :3])  # positions untouched
        np.testing.assert_array_equal(b.continuous_quaternions(pose, reference=-fixed)[:, 3:], np.tile(-q, (4, 1)))

    def test_vector_values_match_their_features(self):
        n = 3
        state, action = np.zeros((n, 7)), np.ones((n, 7))
        pose = np.tile([0.4, 0, 0.3, 0, 0, 0, 1], (n, 1))
        for space in ("joint", "ee"):
            names = b.vector_features(space)
            values = b.vector_values(space, state, action, pose, pose)
            self.assertEqual(names.keys(), values.keys())
            for key in names:
                self.assertEqual(values[key].shape, (n, len(names[key])), (space, key))
        ee = b.vector_values("ee", state, action, pose, pose)
        np.testing.assert_array_equal(ee["action.joints"], action)
        self.assertEqual(ee["action"][0, -1], 1.0)  # the gripper: the action's joint7


class DecodeTest(unittest.TestCase):
    def test_depth(self):
        depth = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 500)
        ok, png = cv2.imencode(".png", depth)
        self.assertTrue(ok)
        decoded = b.decode_depth(bytes(12) + png.tobytes())  # compressedDepth: a 12-byte config header, then the PNG
        self.assertEqual(decoded.shape, (3, 4, 1))
        np.testing.assert_array_equal(decoded[:, :, 0], depth)
        with self.assertRaises(ValueError):
            b.decode_depth(bytes(40))

    def test_color_is_rgb(self):
        bgr = np.zeros((2, 2, 3), np.uint8)
        bgr[:, :, 0] = 255  # blue
        ok, png = cv2.imencode(".png", bgr)
        rgb = b.decode_color(png.tobytes())
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertTrue((rgb[:, :, 2] == 255).all() and (rgb[:, :, 0] == 0).all())


class ImageSizeTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(b.image_size("360x640"), (360, 640))
        self.assertEqual(b.image_size("240X426"), (240, 426))
        for bad in ("360", "x640", "0x640", "axb"):
            with self.assertRaises(argparse.ArgumentTypeError):
                b.image_size(bad)


class EpisodeBagsTest(unittest.TestCase):
    def test_a_directory_of_episodes_or_one(self):
        root = Path(tempfile.mkdtemp())
        for name in ("episode_0002", "episode_0001", "episode_0003"):
            (root / name).mkdir()
            (root / name / f"0_{name}.mcap").touch()
            if name != "episode_0003":  # still recording
                (root / name / "metadata.yaml").touch()
        (root / "notes").mkdir()
        self.assertEqual([p.parent.name for p in b.episode_bags(root)], ["episode_0001", "episode_0002"])
        self.assertEqual(b.episode_bags(root / "episode_0002"), [root / "episode_0002" / "0_episode_0002.mcap"])
        (root / "episode_0001" / "1_episode_0001.mcap").touch()
        with self.assertRaises(ValueError):
            b.episode_bags(root)


if __name__ == "__main__":
    unittest.main()
