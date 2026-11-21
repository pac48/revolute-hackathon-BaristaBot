"""An interactive bash with `ros2` on PATH, without bazel per command.

Usage: [ROS_DOMAIN_ID=N] bazel run //tools/ros2:shell

Runs your normal bash (~/.bashrc included) with:
- `ros2`: a launcher for the //tools/ros2 binary built for this shell, run
  directly (~0.3 s instead of a `bazel run`), from whatever directory you're in;
- tab completion for ros2 (argcomplete);
- ROS_DOMAIN_ID as it was when the shell started (~/.bashrc can't reset it), and
  the prompt prefixed with "(ros2 <domain>)".
The settings `bazel run` adds for ROS processes (.bazelrc --run_env, e.g. the DDS
transports) are given to `ros2` only, not to the shell. `exit` returns. Rebuild
(re-run this) after changing //tools/ros2.
"""

import os
import pathlib
import shlex
import sys
import tempfile

# .bazelrc --run_env values ros2 needs; the rest (LD_LIBRARY_PATH=lib,
# ROS_HOME=.ros, ...) are relative to a runfiles cwd and stay out of the shell.
ROS_ENV = ("ROS_DISTRO", "RMW_IMPLEMENTATION", "FASTDDS_BUILTIN_TRANSPORTS")
DROP = ("LD_LIBRARY_PATH", "ROS_HOME", "QT_QPA_PLATFORM", "PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE")


def main():
    # <runfiles>/_main/tools/ros2/shell.py (not resolved: the runfiles tree is symlinks)
    runfiles = pathlib.Path(os.path.abspath(__file__)).parents[3]
    ros2 = runfiles / "_main" / "tools" / "ros2" / "ros2"
    completion = runfiles / "_main" / "tools" / "ros2" / "ros2_completion.bash"
    if not ros2.exists():
        sys.exit(f"shell: {ros2} not found")

    env = {k: v for k, v in os.environ.items()
           if not (k.startswith(("RUNFILES_", "BUILD_", "BAZEL_")) or k in DROP or k == "JAVA_RUNFILES")}
    ros_env = {k: os.environ[k] for k in ROS_ENV if k in os.environ}
    domain = os.environ.get("ROS_DOMAIN_ID")

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ros2_shell_"))
    (tmp / "bin").mkdir()
    exports = {
        "RUNFILES_DIR": str(runfiles),
        "AMENT_PREFIX_PATH": str(runfiles / "_main"),
        **ros_env,
    }
    launcher = tmp / "bin" / "ros2"
    launcher.write_text(
        "#!/bin/bash\n"
        + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in exports.items())
        + f'export LD_LIBRARY_PATH={shlex.quote(str(runfiles / "_main" / "lib"))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n'
        + f"exec {shlex.quote(str(ros2))} \"$@\"\n"
    )
    launcher.chmod(0o755)

    rc = tmp / "bashrc"
    rc.write_text(
        '[ -f ~/.bashrc ] && source ~/.bashrc\n'
        + (f"export ROS_DOMAIN_ID={shlex.quote(domain)}\n" if domain is not None else "unset ROS_DOMAIN_ID\n")
        + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in ros_env.items())
        + f'export PATH={shlex.quote(str(tmp / "bin"))}:"$PATH"\n'
        + (f"source {shlex.quote(str(completion))}\n" if completion.exists() else "")
        + 'PS1="(ros2 ${ROS_DOMAIN_ID:-0}) $PS1"\n'
        + f"trap 'rm -rf {shlex.quote(str(tmp))}' EXIT\n"
        + 'echo "ros2 shell (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}); exit to leave"\n'
    )
    os.chdir(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()))
    os.execve("/bin/bash", ["bash", "--rcfile", str(rc), "-i"], env)


if __name__ == "__main__":
    main()
