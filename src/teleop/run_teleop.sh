#!/usr/bin/env bash
# Start everything VR teleop needs, in one terminal. Ctrl-C stops it all.
#
#   src/teleop/run_teleop.sh [options] [-- TELEOP_ARGS...]
#
#   --no-sim       don't start the simulation (the real arm is already running)
#   --no-headset   don't build/install/start the Quest app (already running, or using fake_headset)
#   --view         also open RViz with the headset view (//src/teleop/quest_bridge:view)
#   --model NAME   robot model, for the simulation and the teleop node (default: the active model in
#                  active_model.txt;
#                  rebot_dual_arm_gemini_305: both arms, one hand each)
#   --wifi HOST    connect the headset over Wi-Fi instead of USB
#
# Starts: the MuJoCo sim, //src/teleop/quest_bridge, the Quest app (//src/teleop/quest_app:launch)
# and //src/teleop/controller:teleop. Arguments after -- go to the teleop node.
# On exit the headset app is stopped (--stop) so the headset can sleep.
set -uo pipefail

sim=1 headset=1 view=0 model="" wifi=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sim) sim=0 ;;
    --no-headset) headset=0 ;;
    --view) view=1 ;;
    --model) model="${2:?--model needs a value}"; shift ;;
    --wifi) wifi="${2:?--wifi needs a value}"; shift ;;
    -h|--help) sed -n '2,/^set /p' "$0" | sed '$d;s/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
teleop_args=("$@")

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
[[ -n $model ]] || model=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' active_model.txt | grep -m1 .)

targets=(//src/teleop/quest_bridge //src/teleop/controller:teleop)
((sim)) && targets+=(//src/ros2_control_sim:mujoco_arm_sim)
((headset)) && targets+=(//src/teleop/quest_app:launch)
((view)) && targets+=(//src/teleop/quest_bridge:view)

echo "[teleop] building ${targets[*]}"
bazel build "${targets[@]}" || exit 1

pids=()
teleop_pid=
cleaning=0

# Signal the teleop node alone and wait for it: it sends the arm home first (a trajectory to all
# zeros), which needs the sim or driver and the bridge still running. Ctrl-C again skips the wait
# (and aborts the trip). SIGTERM, which the node treats as Ctrl-C (background jobs ignore SIGINT), and only to it, not to
# the `sed` that prefixes its output.
stop_teleop() {
  [[ -n $teleop_pid ]] || return
  local nodes=() p
  for p in $(pgrep -g "$teleop_pid" 2>/dev/null); do
    [[ $(ps -o comm= -p "$p" 2>/dev/null) =~ ^(sed|bash)$ ]] || nodes+=("$p")
  done
  ((${#nodes[@]})) || return
  echo "[teleop] sending the arm home (Ctrl-C again to skip)"
  skip=0
  trap 'skip=1' INT TERM
  kill -TERM "${nodes[@]}" 2>/dev/null
  local i aborted=0
  for ((i = 0; i < 600; i++)); do  # up to 2 minutes
    local alive=0
    for p in "${nodes[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    ((alive)) || break
    if ((skip && !aborted)); then kill -TERM "${nodes[@]}" 2>/dev/null; aborted=1; fi
    sleep 0.2
  done
  trap - INT TERM
}

cleanup() {
  ((cleaning)) && return
  cleaning=1
  echo "[teleop] stopping"
  stop_teleop
  trap - INT TERM
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null; done
  ((headset)) && bazel run //src/teleop/quest_app:launch -- --stop
  wait 2>/dev/null
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

# start NAME BAZEL_RUN_ARGS...: its own process group, output prefixed with NAME.
start() {
  local name=$1; shift
  setsid bash -c 'exec "$@" 2>&1 | sed -u "s/^/['"$name"'] /"' _ bazel run "$@" &
  pids+=($!)
  last=$!
}

((sim)) && { start sim //src/ros2_control_sim:mujoco_arm_sim -- --model "$model"; sleep 5; }
# The headset shows one gripper: the (left) arm's of a robot with two.
bridge_args=()
[[ $model == rebot_dual_arm* ]] && bridge_args=(--tip-frame left_gripper_end)
start bridge //src/teleop/quest_bridge -- "${bridge_args[@]}"; watch=("${pids[@]}")
# the headset launcher exits once the app is up, so it isn't watched
((headset)) && start headset //src/teleop/quest_app:launch -- ${wifi:+--wifi "$wifi"}
((view)) && start view //src/teleop/quest_bridge:view
start teleop //src/teleop/controller:teleop -- --model "$model" "${teleop_args[@]}"; watch+=("$last"); teleop_pid=$last

# Stop when the sim, bridge or teleop node exits.
wait -n "${watch[@]}" 2>/dev/null
echo "[teleop] a process exited"
