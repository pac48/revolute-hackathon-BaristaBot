"""What a frame of our LeRobot datasets is: written by bag_to_lerobot, read back by a policy at run time
(//src/lerobot_policy), so that a policy sees on the robot what it was trained on.

The vectors (see bag_to_lerobot's docstring for --space):
  joint space  joint1..joint7: joint1..6 (rad) and the gripper (joint7: its opening, m)
  ee space     EE_NAMES: the tool (gripper_end) in base_link, its rotation as 6D, the gripper (joint7, m)
Images: RGB uint8 (H, W, 3), as decoded from the cameras' CompressedImage topics (CAMERAS).
"""

import cv2
import numpy as np

JOINTS = [f"joint{i}" for i in range(1, 8)]  # the arm's joint1..6 (rad) and the gripper (joint7: its opening, m)
ARM_JOINTS = JOINTS[:6]  # the kinematic chain (base_link -> gripper_end)
POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
# --arms: one arm (joint1..7) or two (left_joint1..7 then right_joint1..7, as rebot_dual_arm_gemini_305 names them).
ARM_PREFIXES = {"single": ("",), "dual": ("left_", "right_")}


def arm_joints(arms: str = "single") -> list[str]:
    """The joints of --arms, in the datasets' order: JOINTS, or each arm's JOINTS with its prefix."""
    return [f"{prefix}{j}" for prefix in ARM_PREFIXES[arms] for j in JOINTS]


def arm_pose_names(arms: str = "single") -> list[str]:
    """POSE_NAMES for each arm of --arms (each arm's tool in its own base_link)."""
    return [f"{prefix}{n}" for prefix in ARM_PREFIXES[arms] for n in POSE_NAMES]
EE_NAMES = ["x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32", "gripper"]  # --space ee: ee_vector
SPACES = {len(JOINTS): "joint", len(EE_NAMES): "ee"}  # observation.state's size -> the space
JOINT_STATES = "/joint_states"
CONTROLLER = "joint_trajectory_controller"  # recorded through, and commanded through at run time
CONTROLLER_STATE = f"/{CONTROLLER}/controller_state"
# Camera name -> (colour topic, depth topic): the one camera the BaristaBot demos are recorded with, as its driver
# publishes it under /camera (the recorder stores color/image_raw as color/image_raw/compressed). The depth topic is
# only read with bag_to_lerobot --depth. The dataset's key of a camera is observation.images.<name>.
CAMERAS = {
    "main": ("/camera/color/image_raw/compressed", "/camera/depth/image_raw/compressedDepth"),
}
IMAGE_KEY_PREFIX = "observation.images."
# The datasets' images (H, W): the cameras' 720 x 1280 halved. ACT at 720p (3 cameras, 2760 image tokens) does not
# train in 24 GB even at batch 8; at 360 x 640 (720 tokens) batch 8 takes 4.3 GB (RTX 4090, ACT's default size).
IMAGE_SIZE = (360, 640)


def resize(image: np.ndarray, size: tuple[int, int], depth: bool = False) -> np.ndarray:
    """image (H, W, C) at size (H, W): area-averaged (colour), or the nearest pixel's (depth: no blended depths).
    How the datasets' images were made smaller, and so how a policy's live images are."""
    if image.shape[:2] == tuple(size):
        return image
    interpolation = cv2.INTER_NEAREST if depth else cv2.INTER_AREA
    resized = cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
    return resized.reshape(*size, image.shape[2])  # cv2 drops a single channel's axis


def decode_color(data: bytes) -> np.ndarray:
    """sensor_msgs/CompressedImage (JPEG/PNG) -> RGB uint8 (H, W, 3)."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode a colour image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rotation_6d(q: np.ndarray) -> np.ndarray:
    """Quaternions (rows of qx, qy, qz, qw) -> the first two columns of their rotation matrices (rows of r11, r21,
    r31, r12, r22, r32): a rotation without the quaternion's sign or a rotation vector's jump at pi (Zhou et al.,
    "On the Continuity of Rotation Representations in Neural Networks"). Back: quaternion_from_6d."""
    x, y, z, w = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
        2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
    ], axis=1).astype(np.float32)


def quaternion_from_6d(r: np.ndarray) -> np.ndarray:
    """A 6D rotation (r11, r21, r31, r12, r22, r32; a policy's output need not be exactly a rotation) -> the
    nearest-by-Gram-Schmidt rotation's quaternion (qx, qy, qz, qw), qw >= 0."""
    a, b = np.asarray(r[:3], np.float64), np.asarray(r[3:6], np.float64)
    c1 = a / np.linalg.norm(a)
    b = b - c1 * (c1 @ b)
    c2 = b / np.linalg.norm(b)
    m = np.stack([c1, c2, np.cross(c1, c2)], axis=1)
    # Shepperd: from the largest of the diagonal's combinations, for precision.
    t = np.trace(m)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def ee_vector(pose: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Rows of EE_NAMES: the tool's position, its rotation (6D) and the gripper's opening (joint7).
    pose: rows of POSE_NAMES; joints: rows of JOINTS."""
    return np.concatenate([pose[:, :3], rotation_6d(pose[:, 3:]), joints[:, 6:7]], axis=1).astype(np.float32)


def vector_features(space: str, arms: str = "single") -> dict[str, list[str]]:
    """The vector features' names, by key. LeRobot's policies take their state and their target from exactly
    observation.state and action (the other keys are in the dataset, for other uses): --space chooses what those are.
    Two arms (--arms dual) are joint space only: both arms' joints, then both tools' poses."""
    if arms != "single":
        if space != "joint":
            raise ValueError(f"--arms {arms} supports only --space joint")
        joints, pose = arm_joints(arms), arm_pose_names(arms)
        return {"observation.state": joints, "observation.ee_pose": pose, "action": joints, "action.ee_pose": pose}
    if space == "ee":
        return {"observation.state": EE_NAMES, "observation.joints": JOINTS, "action": EE_NAMES, "action.joints": JOINTS}
    return {"observation.state": JOINTS, "observation.ee_pose": POSE_NAMES, "action": JOINTS, "action.ee_pose": POSE_NAMES}
