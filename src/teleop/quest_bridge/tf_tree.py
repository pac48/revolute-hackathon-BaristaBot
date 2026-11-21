"""Latest-value TF tree: enough to look up one frame in another from /tf and /tf_static.

No interpolation or history (tf2_ros isn't in this workspace's deps, and the bridge only
needs the newest gripper pose).
"""

import threading

from src.teleop.quest_bridge.protocol import Pose


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate(q, v):
    x, y, z, w = q
    # t = 2 u x v; v' = v + w t + u x t
    tx = 2 * (y * v[2] - z * v[1])
    ty = 2 * (z * v[0] - x * v[2])
    tz = 2 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def compose(a: Pose, b: Pose) -> Pose:
    """a * b (T_a_c = T_a_b * T_b_c)."""
    r = rotate(a.q, b.p)
    return Pose((a.p[0] + r[0], a.p[1] + r[1], a.p[2] + r[2]), quat_mul(a.q, b.q))


def inverse(t: Pose) -> Pose:
    qi = (-t.q[0], -t.q[1], -t.q[2], t.q[3])
    r = rotate(qi, t.p)
    return Pose((-r[0], -r[1], -r[2]), qi)


class TfTree:
    def __init__(self):
        self._lock = threading.Lock()
        self._parent: dict[str, tuple[str, Pose]] = {}  # child -> (parent, T_parent_child)

    def set(self, parent: str, child: str, transform: Pose):
        with self._lock:
            self._parent[child] = (parent.lstrip("/"), transform)

    def add_message(self, msg):
        """A tf2_msgs/TFMessage."""
        for tf in msg.transforms:
            t, r = tf.transform.translation, tf.transform.rotation
            self.set(tf.header.frame_id, tf.child_frame_id.lstrip("/"), Pose((t.x, t.y, t.z), (r.x, r.y, r.z, r.w)))

    def _to_root(self, frame: str) -> tuple[str, Pose]:
        """(root, T_root_frame)."""
        t = Pose()
        seen = set()
        while frame in self._parent:
            if frame in seen:
                raise LookupError(f"TF cycle at {frame}")
            seen.add(frame)
            parent, t_parent_frame = self._parent[frame]
            t = compose(t_parent_frame, t)
            frame = parent
        return frame, t

    def lookup(self, target: str, source: str) -> Pose:
        """T_target_source: the pose of `source` in `target`. Raises LookupError."""
        with self._lock:
            root_t, t_root_target = self._to_root(target)
            root_s, t_root_source = self._to_root(source)
        if root_t != root_s:
            raise LookupError(f"{source} and {target} are not connected (roots {root_s}, {root_t})")
        return compose(inverse(t_root_target), t_root_source)
