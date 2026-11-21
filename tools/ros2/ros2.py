"""The ros2 command line (ros2cli), e.g. `bazel run //tools/ros2 -- topic list`.

Verbs come from the ros2cli packages in this target's deps (their entry points
are egg-info metadata on the Python path); message types from the interface
packages in its deps.
"""

import sys

# Tab completion: use @pip_ros2_tools' argcomplete, not @pip_ros's older one
# (which crashes on Python 3.12's argparse). Both are on the path; put ours first.
sys.path.sort(key=lambda p: "pip_ros2_tools" not in p)

from ros2cli.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
