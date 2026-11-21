# revolute-hackathon-BaristaBot

Teleoperation, demonstration recording and policy training for the **reBot Arm B601-RS** (one or two
arms), as a Bazel workspace built against the [ROS Central Registry](https://bazel.ros.org)
(ROS 2 Lyrical), with the [BCR](https://bcr.bazel.build) as fallback.

What's here:

| Area | Path | Main targets |
| --- | --- | --- |
| Real-arm driver (SocketCAN, RobStride motors) | `src/ros2_control_sim`, `src/rebot_robstride_protocol`, `third_party/rebot_robstride_protocol` | `//src/ros2_control_sim:real_arm` |
| MuJoCo simulation | `src/ros2_control_sim` | `//src/ros2_control_sim:mujoco_arm_sim` |
| Robot models (URDF/MJCF, kinematics) | `src/robot_model`, `src/camera_models` | `//src/robot_model/<model>:view` |
| VR teleop (Meta Quest app, bridge, controller) | `src/teleop` | `src/teleop/run_teleop.sh`, `//src/teleop/controller:teleop` |
| Demonstration recording (MCAP rosbags, voice cues) | `src/data_recorder` | `//src/data_recorder:record` |
| LeRobot export and training | `src/lerobot_export`, `src/lerobot_policy` | `//src/lerobot_export:bag_to_lerobot`, `//src/lerobot_policy:train`, `:run_policy` |
| Cameras (Orbbec Gemini 305, USB) | `src/camera_drivers` | `//src/camera_drivers:orbbec_camera` |

## Requirements

- [bazelisk](https://github.com/bazelbuild/bazelisk) installed as `bazel` (Bazel 9.x, pinned in `.bazelversion`)
- `git`, `unzip`, `zip`
- For training: an NVIDIA GPU with a recent driver (PyTorch for CUDA comes from PyPI)
- For voice cues while recording: `paplay` (PulseAudio/PipeWire) or `aplay` (ALSA)

No system compiler, ROS or Android SDK is needed: the RCR uses a hermetic LLVM toolchain and Python
interpreter, and the Quest app is built with a hermetic Android SDK/NDK.

The first build is big. Use a remote cache (below) so it isn't compiled from scratch.

## Remote cache

`tools/cache` switches the remote cache for this checkout (it writes the gitignored `.bazelrc.cache`):

| Command | Cache |
| --- | --- |
| `tools/cache use intrinsic` | RCR public cache, read-only (the default) |
| `tools/cache use lan [HOST]` | LAN cache server; read/write, shared between machines |
| `tools/cache use none` | no remote cache |
| `tools/cache status` | show the active cache and whether it's reachable |

A LAN cache server runs on the lab PC (`192.168.7.67`, port 9092) and has everything below already built. On your
machine, run this once:

```bash
tools/cache use lan 192.168.7.67
```

If the LAN cache can't be reached, [`tools/bazel`](tools/bazel) (which bazelisk runs automatically) falls back to the
RCR public cache for that command. To run your own server (Docker): `tools/cache serve 300` and then
`tools/cache use lan localhost` (see `tools/cache help`). The server has no authentication, so keep it on a trusted network.

## Active robot model

`active_model.txt` names the robot model every app uses by default (`mujoco_arm_sim`, `real_arm`, `teleop`,
`run_teleop.sh`, ...). It is **`rebot_dual_arm_gemini_305`**: two arms with a Gemini 305 camera each, the left on
`can0` and the right on `can1`. Other models: `rebot_arm` (one arm), `rebot_arm_gemini_305` (one arm with a camera),
`two_link_arm` (sim only). `--model NAME` overrides it for one run.

In the dual-arm model every joint and frame is prefixed `left_` / `right_` (`left_joint1`, `right_gripper_end`), and one
`joint_trajectory_controller` drives all 14 joints.

## Simulation

```bash
bazel run //src/ros2_control_sim:mujoco_arm_sim                       # the active model, MuJoCo viewer
bazel run //src/ros2_control_sim:mujoco_arm_sim -- --headless
bazel run //src/ros2_control_sim:mujoco_arm_sim -- --model rebot_arm
bazel run //src/robot_model/rebot_dual_arm_gemini_305:view            # just the model
```

Move it: `bazel run //src/trajectory_generator:move_to -- left_joint1=0.5 left_joint2=-0.3`.

## Real arm

```bash
bazel run //src/ros2_control_sim:real_arm                                                        # active model: left on can0, right on can1
bazel run //src/ros2_control_sim:real_arm -- --model rebot_dual_arm_gemini_305 --can-interface can0 can1
bazel run //src/ros2_control_sim:real_arm -- --model rebot_arm_gemini_305 --can-interface can0   # one arm
```

The same controllers and topics as the simulation, with the joints as RobStride motors on SocketCAN buses
(`shared_can_ros2/CanSystem` with `rebot_robstride_protocol`, pinned in `MODULE.bazel`). **It enables the motors**:
the arm holds its position, then follows trajectories. `--help` lists the gripper options.

CAN setup, on the PC with the adapters:

```bash
tools/can/install_udev_rule.sh
```

That installs a udev rule that brings `can*` up at 1 Mbit/s after every boot or replug (it only needs to run once). To read
the motors without any controller:

```bash
bazel test //src/rebot_robstride_protocol:hardware_test --test_output=all --test_arg=can0
```

Troubleshooting:
- `Network is down`: the interface is down; install the udev rule above, or `sudo ip link set can0 up type can bitrate 1000000`.
- Only some motors answer with two arms: power both arms from **one supply** (the PCAN-USB adapters aren't isolated, so two
  supplies give an uncontrolled ground).
- The two adapters report the same serial, so `can0`/`can1` follow the enumeration order: check which arm is which with
  `hardware_test` (one arm per run, with `--test_arg=--no-gripper` if needed), or pin names to USB ports with
  `tools/can/80-rebot-can.rules`.

Cameras (Orbbec Gemini 305): `bazel run //src/camera_drivers:orbbec_camera -- --dual-color`, view with
`bazel run //src/camera_drivers:orbbec_rviz`.

## VR teleop (Meta Quest)

One-time setup on each PC: let adb talk to the headset (USB vendor 2833). Put the headset in developer mode first
(see [the app's README](src/teleop/quest_app/README.md)).

```bash
sudo src/teleop/quest_app/tools/setup_udev.sh
```

Everything in one terminal (sim or real arm, bridge, headset app, teleop node; Ctrl-C sends the arms home and stops it all):

```bash
src/teleop/run_teleop.sh                      # simulation, active model
src/teleop/run_teleop.sh --no-sim             # the real arm (start real_arm first, in another terminal)
src/teleop/run_teleop.sh --view               # also RViz with the headset's view
src/teleop/run_teleop.sh --wifi HEADSET_IP    # headset over Wi-Fi instead of USB
```

Or one at a time:

```bash
bazel run //src/ros2_control_sim:mujoco_arm_sim     # or :real_arm
bazel run //src/teleop/quest_bridge                 # ws://0.0.0.0:9090
bazel run //src/teleop/quest_app:launch             # build, install and start the APK (-- --stop to stop it)
bazel run //src/teleop/controller:teleop            # --help for options
```

Build only the APK: `bazel build //src/teleop/quest_app/app:quest_teleop`.

Controls (dual arm: the left hand drives the left arm, the right hand the right):
- **Hold the trigger** to engage the clutch: the arm follows your hand's motion since then. **Release** to leave it where
  it is; the arm moves only while the trigger is held.
- **Squeeze the grip** to close the gripper.
- Click the **left thumbstick** to place the robot base at the left controller.

The wire protocol is in [`src/teleop/quest_app/PROTOCOL.md`](src/teleop/quest_app/PROTOCOL.md).

## Recording demonstrations

```bash
bazel run //src/data_recorder:record -- -o recordings/grasp                              # one bag until Ctrl-C
bazel run //src/data_recorder:record -- -o recordings/grasp --episodic                   # one bag per episode
bazel run //src/data_recorder:record -- -o recordings/grasp --episodic --use-sim-time    # in simulation
```

It records every controller's state, the camera images (compressed) and `/joint_states`, `/tf`, `/tf_static` to MCAP
rosbags without loss. With `--episodic`, nothing is recorded until something calls the recorder's `~/start` service
(`std_srvs/Trigger`); `~/stop` keeps the episode as `DIR/episode_NNNN`, and a `~/start` during an episode discards it and
starts over. During teleop, **B on the right controller starts an episode and Y on the left controller stops it** (the
teleop node calls `/data_recorder/start` and `/data_recorder/stop`; `--no-recorder-buttons` turns that off). By hand:

```bash
bazel run //tools/ros2 -- service call /data_recorder/start std_srvs/srv/Trigger
```

(`bazel run //tools/ros2 -- service list` shows the recorder's node name.) A voice says "Start demonstration" /
"End demonstration" when recording starts and stops (through `paplay` or `aplay`); `--no-voice` turns it off. Check a bag
for dropped messages with `bazel run //src/data_recorder:bag_check -- BAG_DIR`. All options are at the top of
[`recorder_main.cpp`](src/data_recorder/recorder_main.cpp).

## Training (LeRobot)

```bash
# Episodes -> LeRobot dataset (images at 360x640 by default; --image-size to change)
bazel run //src/lerobot_export:bag_to_lerobot -- recordings/grasp --repo-id local/rebot_grasp

# Train (lerobot-train with this repo's defaults; output in outputs/train/<date>/...)
bazel run //src/lerobot_policy:train -- --dataset.root=recordings/grasp_lerobot [--steps=100000]

# Resume
bazel run //src/lerobot_policy:train -- --config_path=OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json --resume=true

# Run a trained policy on the arm (through the trajectory controller), or on the sim
bazel run //src/lerobot_policy:run_policy -- OUTPUT_DIR/checkpoints/last/pretrained_model
bazel run //src/lerobot_policy:run_policy -- CHECKPOINT --ros-args -p use_sim_time:=true
```

Relative paths are from where you ran `bazel run`.

## Tests

```bash
bazel test //...
```

Runs everything that needs no hardware, display or GPU. Tests that need them are tagged `manual` (for example
`//src/rebot_robstride_protocol:hardware_test`, `//src/lerobot_policy:policy_runner_gpu_test`, and the tracking regression
tests in `//src/ros2_control_sim`).

## ROS 2 command-line tools

```bash
bazel run //tools/ros2:shell                        # a bash with ros2 on PATH
bazel run //tools/ros2 -- topic list -t
bazel run //tools/ros2 -- topic echo /joint_states --once
```

`.bazelrc` sets Fast DDS to its default transports (shared memory + UDPv4) with datagrams up to 1400 bytes. If the shared
memory transport misbehaves (`CacheChangePool` assertions, often from stale `/dev/shm/fastdds_*` segments of killed
processes), add `--config=noshm` to every ROS process of a run.

## Building on another machine

Add `.remote` to a target to build and run it on another machine over ssh (for example the PC with the CAN adapters):

```bash
tools/remote use HOST                                # once per clone; any passwordless ssh host
bazel run //src/ros2_control_sim:real_arm.remote
```

The host builds your current working tree (tracked files, uncommitted changes included). See `tools/remote` for details.

## Layout

- `MODULE.bazel`: dependencies (RCR/BCR modules, the reBot description and CAN driver archives, PyPI hubs, Android toolchains)
- `.bazelrc`: registries, hermetic toolchain, remote cache, Fast DDS and Android settings
- `third_party/`: BUILD files and patches for the external archives; `third_party/registry` is a local registry with the Orbbec
  camera modules
- `src/`: the packages in the table above, plus `trajectory_generator` (trajectory clients, kinematics), `controllers`
  (joint impedance controller), `mujoco_state_msgs`, `sampling_mpc`
- `tools/`: `bazel` (bazelisk wrapper), `cache`, `remote`, `ros2`, `can`, `gdb`

This workspace was ported from the `home-institute` workspace.
