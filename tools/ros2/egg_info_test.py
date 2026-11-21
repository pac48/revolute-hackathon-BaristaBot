"""egg_info: setup.py -> PKG-INFO and entry_points.txt."""

import unittest

from tools.ros2 import egg_info

SETUP = """
from setuptools import setup
package_name = 'ros2node'
setup(
    name=package_name,
    version='0.40.9',
    entry_points={
        'ros2cli.command': ['node = ros2node.command.node:NodeCommand'],
        'ros2node.verb': [
            'info = ros2node.verb.info:InfoVerb',
            'list=ros2node.verb.list:ListVerb',
        ],
    },
)
"""


class EggInfoTest(unittest.TestCase):
    def test_metadata(self):
        files = egg_info.egg_info(SETUP)
        self.assertIn("Name: ros2node\nVersion: 0.40.9", files["PKG-INFO"])
        self.assertEqual(
            files["entry_points.txt"],
            "[ros2cli.command]\nnode = ros2node.command.node:NodeCommand\n\n"
            "[ros2node.verb]\ninfo = ros2node.verb.info:InfoVerb\nlist = ros2node.verb.list:ListVerb\n",
        )

    def test_entry_points_parse(self):
        import configparser

        parser = configparser.ConfigParser(delimiters=("=",))
        parser.read_string(egg_info.egg_info(SETUP)["entry_points.txt"])
        self.assertEqual(parser["ros2node.verb"]["list"], "ros2node.verb.list:ListVerb")


if __name__ == "__main__":
    unittest.main()
