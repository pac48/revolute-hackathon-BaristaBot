import json, sys
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

MOVE_THRESH = 0.15   # rad: new waypoint when any arm joint moved this much since the last one
GRIP_THRESH = 0.02   # new waypoint when a gripper (joint7) changes this much (open/close)
START_DELAY = 2.0    # s: time for the sim arms to reach the first waypoint

wps, names, t0, last = [], None, None, None
with open(sys.argv[1], "rb") as f:
    for _, _, msg, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=["/joint_trajectory_controller/controller_state"]):
        q, t = list(m.feedback.positions), msg.log_time * 1e-9
        if names is None:
            names, t0 = list(m.joint_names), t
            grip = [i for i, n in enumerate(names) if n.endswith("joint7")]  # gripper joints
            arm = [i for i in range(len(names)) if i not in grip]          # joints 1-6, both arms
        if last is not None and max(abs(q[i] - last[i]) for i in arm) < MOVE_THRESH \
                and max(abs(q[i] - last[i]) for i in grip) < GRIP_THRESH:
            continue                                                         # too close to the last waypoint
        last = q
        wps.append((t - t0, q))
wps.append((t - t0, q))                                                      # always keep the final pose

# waypoints.json: human-readable list (time in s + 14 joint positions)
json.dump({"joint_names": names,
           "waypoints": [{"t": round(tt, 2), "positions": [round(x, 4) for x in qq]} for tt, qq in wps]},
          open("waypoints.json", "w"), indent=1)

# waypoints_traj.yaml: same waypoints as a JointTrajectory, ready to play in the sim
pts = [{"positions": [round(x, 4) for x in qq],
        "time_from_start": {"sec": int(tt + START_DELAY), "nanosec": int(((tt + START_DELAY) % 1) * 1e9)}}
       for tt, qq in wps]
json.dump({"joint_names": names, "points": pts}, open("waypoints_traj.yaml", "w"))
print(f"{len(wps)} waypoints, joints: {names}")
