import json, sys
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

bag = sys.argv[1]          # path to the .mcap file
rate_hz = 5.0              # downsample so the command line stays small

points, names, t0, last_t = [], None, None, -1e9
with open(bag, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for _, _, msg, ros_msg in reader.iter_decoded_messages(
            topics=["/joint_trajectory_controller/controller_state"]):
        t = msg.log_time * 1e-9                      # message time in seconds
        if t0 is None:
            t0, names = t, list(ros_msg.joint_names) # joint names the controller expects
        if t - last_t < 1.0 / rate_hz:
            continue                                 # skip until next sample
        last_t = t
        tfs = (t - t0) + 2.0                         # +2 s so the arms ease to the start pose
        points.append({
            "positions": [round(p, 4) for p in ros_msg.feedback.positions],  # what the real arms did
            "time_from_start": {"sec": int(tfs), "nanosec": int((tfs % 1) * 1e9)},
        })

print("joints:", names, file=sys.stderr)
print("points:", len(points), file=sys.stderr)
json.dump({"joint_names": names, "points": points}, open("traj.yaml", "w"))  # JSON is valid YAML
