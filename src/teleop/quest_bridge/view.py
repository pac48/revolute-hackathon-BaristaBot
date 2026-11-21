"""RViz for teleop: the headset's view next to the robot model, its TF (including
gripper_end), the teleop target and commanded tool frames and the quest_* frames.

Usage: bazel run //src/teleop/quest_bridge:view [-- RVIZ_ARGS...]

With no arguments, opens config/quest_view.rviz; any arguments go to rviz2 instead.
"""

import os
import sys

from src.teleop.quest_bridge import quest_view_rviz_config, rviz2


def main():
    args = sys.argv[1:] or ["-d", quest_view_rviz_config.path()]
    os.execv(rviz2.path(), [rviz2.path(), *args])


if __name__ == "__main__":
    main()
