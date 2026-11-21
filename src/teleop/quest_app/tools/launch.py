"""Start the Quest Teleop app on the connected headset, without putting it on.

Usage: bazel run //src/teleop/quest_app:launch [-- options]
       bazel run //src/teleop/quest_app:launch -- --stop

Builds the APK (//src/teleop/quest_app/app:quest_teleop) as part of the run and installs it when it
differs from the one on the headset. Uses the hermetic adb from the Android SDK in MODULE.bazel.

  --wifi HOST[:PORT]   connect the app to HOST over Wi-Fi instead of USB (adb reverse)
  --port PORT          bridge port (default 9090)
  --view-width N       view stream width in pixels, 0 = off (default 720)
  --view-fps N         view stream rate (default 15)
  --view-quality N     JPEG quality 1-100 (default 70)
  --install            reinstall even if the APK looks unchanged
  --no-keep-awake      leave the proximity sensor alone (the app pauses when not worn)
  --logs               follow the app's log after starting (Ctrl-C to stop)
  --stop               stop the app and restore the proximity sensor, then exit

Keeping awake: the Quest sleeps and pauses apps when nobody wears it. This sends the headset's
"proximity sensor covered" broadcast so it keeps rendering on the desk; it lasts until --stop, a
reboot, or `adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable`.

The Quest still refuses to launch VR apps while the controllers are asleep (press a button on
each) or without a boundary set up.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

from src.teleop.quest_app.tools import adb_tool, quest_teleop_apk

PACKAGE = "dev.questteleop"
ACTIVITY = f"{PACKAGE}/android.app.NativeActivity"
APK_PROP = "debug.quest_teleop.apk_sha256"  # sha256 of the APK last installed by this tool


class LaunchError(RuntimeError):
    pass


def say(text: str):
    print(f"[launch] {text}", flush=True)


def adb(*args: str, check: bool = True, timeout: float = 60) -> str:
    """stdout; with check=False, stdout and stderr (adb install reports failures on stderr)."""
    proc = subprocess.run([adb_tool.path(), *args], capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise LaunchError(f"adb {' '.join(args)}: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip() if check else (proc.stdout + proc.stderr).strip()


def shell(*args: str, check: bool = True) -> str:
    return adb("shell", *args, check=check).replace("\r", "")


def wait_for_headset():
    try:
        adb("wait-for-device", timeout=10)
    except subprocess.TimeoutExpired:
        pass
    state = subprocess.run([adb_tool.path(), "get-state"], capture_output=True, text=True)
    out = (state.stdout + state.stderr).strip()
    if state.returncode == 0 and out == "device":
        return
    if "unauthorized" in out:
        raise LaunchError("headset is unauthorized: put it on and accept 'Allow USB debugging' (tick 'Always allow')")
    if "no permissions" in out:
        raise LaunchError("no USB permission for the headset: run `sudo src/teleop/quest_app/tools/setup_udev.sh` and replug")
    raise LaunchError(f"no headset found ({out or 'adb get-state: nothing'}); check the cable and developer mode")


def install_if_changed(force: bool):
    apk = quest_teleop_apk.path()
    digest = hashlib.sha256(open(apk, "rb").read()).hexdigest()
    installed = shell("pm", "path", PACKAGE, check=False).startswith("package:")
    if installed and not force and shell("getprop", APK_PROP, check=False) == digest:
        say("app is up to date")
        return
    out = adb("install", "-r", apk, check=False, timeout=120)
    if "INSTALL_FAILED_UPDATE_INCOMPATIBLE" in out:
        # Installed with a different signing key (e.g. an old Gradle build): replace it.
        say("installed app has a different signature; uninstalling it first")
        adb("uninstall", PACKAGE, check=False)
        out = adb("install", apk, check=False, timeout=120)
    if "Success" not in out:
        raise LaunchError(f"install failed: {out}")
    shell("setprop", APK_PROP, digest)
    say(f"installed {os.path.basename(apk)} ({os.path.getsize(apk) // 1024} KB)")


def start(args):
    adb("logcat", "-c")
    shell("am", "start", "-S", "-n", ACTIVITY)
    focused = False
    for _ in range(40):
        time.sleep(0.25)
        if "session state 5" in adb("logcat", "-d", "-s", "quest_teleop:I"):
            focused = True
            break
    log = adb("logcat", "-d")
    if "launch_blocked_controller_required" in log:
        raise LaunchError("the Quest blocked the launch: controllers are asleep. "
                          "Press a button on each controller and run again.")
    if "launch_blocked" in log:
        reason = next((w for w in log.split() if "app_launch_blocked" in w), "launch blocked")
        raise LaunchError(f"the Quest blocked the launch ({reason}). Check the headset.")
    pid = shell("pidof", PACKAGE, check=False)
    if not pid:
        raise LaunchError("the app didn't start; see `adb logcat -s quest_teleop`")
    say(f"running (pid {pid})" + (", session focused" if focused else
                                  " but the session isn't focused yet; if nothing streams, check the headset"))
    for line in log.splitlines():
        if "quest_teleop" in line and ("host:" in line or "view stream" in line):
            print("[app] " + line.split("quest_teleop: ", 1)[-1], flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wifi", metavar="HOST[:PORT]")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--view-width", type=int, default=720)
    ap.add_argument("--view-fps", type=int, default=15)
    ap.add_argument("--view-quality", type=int, default=70)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--no-keep-awake", action="store_true")
    ap.add_argument("--logs", action="store_true")
    ap.add_argument("--stop", action="store_true")
    args = ap.parse_args(argv)

    try:
        wait_for_headset()
        say(f"headset: {shell('getprop', 'ro.product.model')} ({adb('get-serialno')})")

        if args.stop:
            shell("am", "force-stop", PACKAGE)
            shell("am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable")
            adb("reverse", "--remove", f"tcp:{args.port}", check=False)
            say(f"stopped {PACKAGE}; proximity sensor restored")
            return 0

        install_if_changed(args.install)

        if args.wifi:
            host = args.wifi if ":" in args.wifi else f"{args.wifi}:{args.port}"
            shell("setprop", "debug.quest_teleop.host", host)
            say(f"app -> ws://{host} (Wi-Fi)")
        else:
            adb("reverse", f"tcp:{args.port}", f"tcp:{args.port}")
            shell("setprop", "debug.quest_teleop.host", f"127.0.0.1:{args.port}")
            say(f"app -> ws://127.0.0.1:{args.port} -> this PC's :{args.port} (USB, adb reverse)")
        shell("setprop", "debug.quest_teleop.view_width", str(args.view_width))
        shell("setprop", "debug.quest_teleop.view_fps", str(args.view_fps))
        shell("setprop", "debug.quest_teleop.view_quality", str(args.view_quality))

        if not args.no_keep_awake:
            shell("input", "keyevent", "KEYCODE_WAKEUP")
            shell("am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close")
            say("keeping the headset awake while not worn (undo: bazel run //src/teleop/quest_app:launch -- --stop)")

        start(args)
    except LaunchError as e:
        print(f"[launch] error: {e}", file=sys.stderr)
        return 1

    if args.logs:
        os.execv(adb_tool.path(), [adb_tool.path(), "logcat", "-s", "quest_teleop:*"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
