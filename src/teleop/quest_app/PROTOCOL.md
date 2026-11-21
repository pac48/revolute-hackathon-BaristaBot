# Wire protocol (version 1)

One WebSocket connection (`ws://`), headset = client, host = server. Messages are JSON text
frames, except the headset view, which is binary.
Default port 9090.

## Frames and units

Every pose is in **ROS convention** (REP 103: x forward, y left, z up), in meters, expressed
in the **robot base frame**. The headset places this frame in the room (the "robot base
anchor") and converts from OpenXR axes before sending, so the host never sees XR axes.

- `p`: position `[x, y, z]`
- `q`: orientation quaternion `[x, y, z, w]`

The controller pose is the OpenXR *grip* pose, re-expressed so its **+x points forward
out of the controller** (along the handle's pointing direction), +z up when held level.

## Headset → host

`hello`, once after connecting:

```json
{"type": "hello", "version": 1, "device": "quest"}
```

`controllers`, once per rendered frame (72–120 Hz). Only the newest state is sent; the
headset drops stale ones instead of queueing them.

```json
{
  "type": "controllers", "seq": 1234, "t": 52.137,
  "head":  {"p": [..], "q": [..]},
  "left":  {"active": true, "p": [..], "q": [..],
            "trigger": 0.0, "grip": 0.0, "stick": [0.0, 0.0],
            "buttons": {"x": 0, "y": 0, "stick": 0, "menu": 0}},
  "right": {"active": true, "p": [..], "q": [..],
            "trigger": 0.0, "grip": 1.0, "stick": [0.0, 0.0],
            "buttons": {"a": 0, "b": 0, "stick": 0}}
}
```

- `seq` increases by one per message; gaps mean dropped frames.
- `t` is seconds on the headset's monotonic clock (for rate/jitter, not wall time).
- `head` is omitted while head tracking is lost.
- A hand with `"active": false` is not tracked (or the app is not focused) and has no
  `p`/`q`. Its trigger/grip/stick/buttons are zero.
- `trigger`, `grip` are 0..1; `stick` is -1..1 (x right, y forward); buttons are 0/1.
- The left thumbstick click also moves the robot base anchor to the left controller, so
  poses jump when it is pressed.

`VIEW`, a **binary** WebSocket message, at the view stream rate (default 15 Hz; 0 = off).
It carries the headset's left-eye image, downscaled (default 720 px wide) and JPEG-encoded.
The header is little-endian:

| Bytes | Field |
|---|---|
| 0–3 | ASCII `VIEW` |
| 4–5 | u16 width (px) |
| 6–7 | u16 height (px) |
| 8–15 | f64 `t`, headset monotonic clock (s) |
| 16– | JPEG |

Only the newest frame is sent. It always goes after the newest controller state, so a frame
never delays the controls.

## Host → headset

`gripper_pose`, at any rate (the bridge sends ~30 Hz):

```json
{"type": "gripper_pose", "t": 1731650000.123, "p": [0.3, 0.0, 0.2], "q": [0, 0, 0, 1]}
```

The tool frame's +x is the approach direction (reBot `gripper_end`). The headset draws it
grey when no update has arrived for 0.5 s. `t` is informational (host clock, seconds).

Unknown message types are ignored on both sides, so new messages can be added without
breaking older peers.
