import argparse
import contextlib
import io
import unittest

from src.robot_model.active_model import active_model


def parse(choices, argv):
    ap = argparse.ArgumentParser()
    active_model.add_model_argument(ap, choices)
    return ap.parse_args(argv).model


def parse_error(choices, argv) -> str:
    with contextlib.redirect_stderr(io.StringIO()) as err, unittest.TestCase().assertRaises(SystemExit):
        parse(choices, argv)
    return err.getvalue()


class ActiveModelTest(unittest.TestCase):
    def test_name_is_set(self):
        self.assertRegex(active_model.NAME, r"^[a-z0-9_]+$")

    def test_defaults_to_the_active_model(self):
        self.assertEqual(parse([active_model.NAME, "other"], []), active_model.NAME)

    def test_model_overrides(self):
        self.assertEqual(parse([active_model.NAME, "other"], ["--model", "other"]), "other")

    def test_unsupported_active_model_needs_model(self):
        self.assertEqual(parse(["other"], ["--model", "other"]), "other")
        self.assertIn("active_model.txt", parse_error(["other"], []))

    def test_unknown_model(self):
        self.assertIn("'nope' is not one of other", parse_error(["other"], ["--model", "nope"]))


if __name__ == "__main__":
    unittest.main()
