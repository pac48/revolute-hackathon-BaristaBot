# quest_app

A native Meta Quest app (Quest 2/3/Pro) for robot teleoperation. It's C++ on OpenXR and OpenGL
ES 3, with no game engine. The app streams both controllers and the head pose to the PC over a
WebSocket, draws the robot's gripper pose sent back by the PC, and streams its own view back for
watching without the headset on.

The PC side is `//src/teleop/quest_bridge`, which bridges to ROS 2. `//src/teleop/controller:teleop`
drives the arm from it. The message format is in [PROTOCOL.md](PROTOCOL.md).

```
Quest (this app) ── ws://pc:9090 ──▶ quest_bridge ──▶ /quest/{left,right,head}/pose, /quest/*/joy, /quest/view/…, /tf
                 ◀── gripper_pose ──              ◀── TF base_link → gripper_end
```

## Commands

Build the APK, install it if it changed, and start it on the headset. Nobody needs to wear it:

```bash
bazel run //src/teleop/quest_app:launch
```

Stop the app and let the headset sleep normally again:

```bash
bazel run //src/teleop/quest_app:launch -- --stop
```

Run the tests for the shared core (math, frames, JSON, protocol):

```bash
bazel test //src/teleop/quest_app/...
```

Stand-in for the headset: synthetic controllers, using the app's own networking code:

```bash
bazel run //src/teleop/quest_app/core:fake_headset -- ws://127.0.0.1:9090 10
```

Build only the APK:

```bash
bazel build //src/teleop/quest_app/app:quest_teleop
```

Options of `launch` (pass them after `--`):
- `--wifi HOST` to use Wi-Fi instead of USB.
- `--view-width 0|N`, `--view-fps N`, `--view-quality N` for the view stream.
- `--install` to force a reinstall.
- `--logs` to follow the app's log.
- `--no-keep-awake`.

If the Quest blocks the launch, `launch` says why. The usual causes are asleep controllers (press
a button on each) and a missing boundary.

To keep the app rendering while nobody wears the headset, `launch` sends the Quest's "proximity
sensor covered" broadcast. That lasts until `--stop` or a reboot.

## Toolchain

Everything comes from Bazel, so there's no Android Studio, Gradle, or local SDK.
- **Android SDK 34, build-tools 35.0.0, NDK r27c and platform-tools (adb)** are downloaded at
  pinned versions by `hermetic_android_toolchains`, with `rules_android` and
  `rules_android_ndk` (`MODULE.bazel`).
- **License acceptance** for those versions is in `.bazelrc`
  (`ACCEPTED_ANDROID_SDK_LICENSE_VERSION`, `ACCEPTED_ANDROID_NDK_LICENSE_VERSION`).
- **The Khronos OpenXR loader**: its AAR from Maven Central, with the `.so` and headers imported
  by `//third_party/openxr_loader`.
- **The app** links the C++ runtime statically (the loader needs none), so the APK holds only
  `libquest_teleop.so` and `libopenxr_loader.so`.
- **The core** also builds for this PC with the workspace's LLVM toolchain, for its tests and
  `fake_headset`.

`launch` uses that adb. If you also have another adb installed, keep it at the same protocol
version (41), or the two will restart each other's server.

Building the APK clears the `bazel-bin` convenience symlinks, because `android_binary` builds in
its own configuration. The next ordinary build restores them. `launch` doesn't depend on them.

## Layout

| Path | What |
|---|---|
| `app/main.cpp` | OpenXR session, controller input, robot base anchor, rendering |
| `app/gl_renderer.*` | Flat-shaded boxes (axes, floor grid, gripper) |
| `app/view_capture.*` | View stream: GPU downscale, async readback, JPEG on its own thread |
| `app/AndroidManifest.xml` | NativeActivity, OpenXR permissions, Quest metadata |
| `core/` | Shared code: pose math, XR↔ROS frames, JSON, WebSocket client, reconnecting link; tests and `fake_headset` |
| `tools/launch.py` | `bazel run //src/teleop/quest_app:launch` |
| `tools/setup_udev.sh` | One-time USB permission for adb (Linux) |

## One-time headset setup

1. Let adb open the headset (USB vendor 2833):
   ```bash
   sudo src/teleop/quest_app/tools/setup_udev.sh
   ```
2. Enable developer mode. Register a (free) developer organization at
   <https://developer.oculus.com/manage/organizations/create/>. Then in the Meta Horizon phone
   app: Menu → Devices → your Quest → Headset settings → Developer mode → On. Reboot the headset.
3. Connect it over USB-C, put it on once, and accept "Allow USB debugging" (tick "Always allow").

## Run with the robot

Start the sim (or the real arm):

```bash
bazel run //src/ros2_control_sim:mujoco_arm_sim -- --model rebot_arm
```

Start the bridge:

```bash
bazel run //src/teleop/quest_bridge
```

Start the app:

```bash
bazel run //src/teleop/quest_app:launch
```

Watch the headset's view next to the arm in RViz:

```bash
bazel run //src/teleop/quest_bridge:view
```

Teleoperate (hold the trigger to move the target, A to snap it to the end effector, squeeze the grip to close the gripper):

```bash
bazel run //src/teleop/controller:teleop
```

## In the headset

- **Robot base:** RGB axes (x red = robot forward, z blue = up) with a status cube that is
  green while connected to the bridge and red while not. It starts 0.5 m in front of and
  below your head, facing the way you face. **Click the left thumbstick** to move it to your
  left controller (position and heading).
- **Controllers:** axes of the pose that is sent. Red points forward out of the controller.
- **Gripper:** orange box and fingers at the pose from the bridge; grey if no update for 0.5 s.
- A floor grid at stage height, in 0.5 m steps.

The view stream shows this scene only, not Meta's system dialogs. While the headset lies on the
desk, the Quest reports head tracking as lost, so `/quest/head/pose` pauses until someone wears it.
