"""Converts data_recorder episodes (MCAP rosbags, one directory per episode) to a LeRobot dataset.

Usage: bazel run //src/lerobot_export:bag_to_lerobot -- recordings/grasp --repo-id local/rebot_grasp_cube
           [--output DIR] [--overwrite] [--task "..."] [--fps 20] [--space joint|ee] [--image-size 360x640] [--jobs N] [--depth]
           [--model rebot_arm_gemini_305]

Each episode becomes one LeRobot episode, sampled on a uniform --fps grid (LeRobot's timestamps are
frame_index / fps) over the time all the streams cover, by the messages' header stamps. The joints:

  state    joint1..joint7 positions (joint1..6 rad; joint7, the gripper, a slide: m), /joint_states interpolated
  action   joint1..joint7 the trajectory controller's reference (its controller_state), the latest at the
           frame's time: what the arm was commanded to

LeRobot's policies learn from exactly observation.state and action (normalizing them by meta/stats.json: the
values are stored raw); --space chooses what they are, the other keys are there for other uses:

  --space joint (default)
    observation.state, action          the joints
    observation.ee_pose, action.ee_pose  the tool (gripper_end) in base_link by FK: x, y, z (m), qx, qy, qz, qw
  --space ee
    observation.state, action          the tool in base_link by FK and the gripper: x, y, z (m), its rotation
                                       matrix's first two columns (r11 r21 r31, r12 r22 r32: continuous, unlike
                                       quaternions or rotation vectors), joint7 (m)
    observation.joints, action.joints  the joints

  observation.images.<camera> the colour image nearest the frame's time (video), at --image-size (default: the
                              cameras' 720 x 1280 halved, conventions.IMAGE_SIZE: ACT does not train on 3 at 720p in 24 GB)
  observation.depth.<camera>  with --depth: the depth image (uint16 mm; a LeRobot depth map, 12-bit HEVC video)

FK is the --model's kinematics (kinematics_client: the model's kinematics CLI).
"""

import argparse
import dataclasses
import multiprocessing
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from src.lerobot_export.conventions import (  # noqa: F401 (re-exported: the frame's conventions)
    CAMERAS,
    CONTROLLER_STATE,
    EE_NAMES,
    JOINT_STATES,
    JOINTS,
    POSE_NAMES,
    IMAGE_SIZE,
    decode_color,
    resize,
    ee_vector,
    rotation_6d,
    vector_features,
)
from src.trajectory_generator import kinematics_client

PNG_SIGNATURE = b"\x89PNG"


@dataclasses.dataclass
class Series:
    """Messages of one topic: header stamps (s) and, per message, a row of values or the image bytes."""

    stamps: list[float] = dataclasses.field(default_factory=list)
    values: list = dataclasses.field(default_factory=list)


def stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def positions(names: list[str], values: list[float], wanted: list[str]) -> list[float]:
    index = {n: i for i, n in enumerate(names)}
    return [values[index[n]] for n in wanted]


def read_bag(mcap_path: Path, depth: bool) -> dict[str, Series]:
    """The episode's joint states, controller references and (compressed) images, by topic."""
    image_topics = [t for color, d in CAMERAS.values() for t in ((color, d) if depth else (color,))]
    series = {t: Series() for t in [JOINT_STATES, CONTROLLER_STATE, *image_topics]}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=list(series)):
            s = series[channel.topic]
            if channel.topic == JOINT_STATES:
                if not set(JOINTS) <= set(msg.name):
                    continue  # e.g. a JointState of another source with only some joints
                s.values.append(positions(msg.name, msg.position, JOINTS))
            elif channel.topic == CONTROLLER_STATE:
                s.values.append(positions(msg.joint_names, msg.reference.positions, JOINTS))
            else:
                s.values.append(bytes(msg.data))
            s.stamps.append(stamp(msg.header))
    for topic, s in series.items():
        if not s.stamps:
            raise ValueError(f"{mcap_path}: no messages on {topic}")
        order = np.argsort(s.stamps, kind="stable")  # recorded in arrival order; stamps can be out of order
        s.stamps = list(np.asarray(s.stamps)[order])
        s.values = [s.values[i] for i in order]
    return series


def frame_times(series: dict[str, Series], fps: float) -> np.ndarray:
    """A uniform grid at fps over the time every topic has messages for."""
    start = max(s.stamps[0] for s in series.values())
    end = min(s.stamps[-1] for s in series.values())
    if end < start:
        raise ValueError("the topics' messages do not overlap in time")
    return start + np.arange(int(np.floor((end - start) * fps + 1e-9)) + 1) / fps


def nearest(stamps: list[float], times: np.ndarray) -> np.ndarray:
    """Index of the stamp nearest each time (stamps sorted)."""
    stamps = np.asarray(stamps)
    if len(stamps) == 1:
        return np.zeros(len(times), dtype=int)
    right = np.clip(np.searchsorted(stamps, times), 1, len(stamps) - 1)
    left = right - 1
    return np.where(times - stamps[left] <= stamps[right] - times, left, right)


def latest(stamps: list[float], times: np.ndarray) -> np.ndarray:
    """Index of the last stamp at or before each time (stamps sorted; the first when none is)."""
    return np.clip(np.searchsorted(np.asarray(stamps), times, side="right") - 1, 0, len(stamps) - 1)


def interpolate(stamps: list[float], values: list, times: np.ndarray) -> np.ndarray:
    """values (rows) linearly interpolated at times, held at the ends."""
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(times, stamps, values[:, j]) for j in range(values.shape[1])], axis=1)


def decode_depth(data: bytes) -> np.ndarray:
    """compressed_depth_image_transport's 16UC1 (a 12-byte config header, then a PNG) -> uint16 mm (H, W, 1)."""
    start = data.find(PNG_SIGNATURE)
    if start < 0:
        raise ValueError("compressedDepth without a PNG (only 16UC1, PNG-compressed depth is supported)")
    depth = cv2.imdecode(np.frombuffer(data[start:], np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError("could not decode a depth image as 16-bit")
    return depth[:, :, None]


def poses(model: str, joints: np.ndarray) -> np.ndarray:
    """FK of each row of joints (JOINTS order): rows of x, y, z, qx, qy, qz, qw."""
    result = kinematics_client.forward_many(model, [dict(zip(JOINTS, map(float, row))) for row in joints], timeout=120.0)
    return np.array([[*p.position, *p.quaternion] for p in result], dtype=np.float32)


def continuous_quaternions(pose: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """pose's quaternions (columns 3:7) with signs chosen so that, q and -q being the same rotation, each is in
    the hemisphere of the one before it, or with a reference (poses of the same frames) in its hemisphere."""
    pose = pose.copy()
    for i in range(len(pose)):
        before = reference[i, 3:] if reference is not None else pose[i - 1, 3:] if i else None
        if before is not None and np.dot(pose[i, 3:], before) < 0:
            pose[i, 3:] *= -1
    return pose


def vector_values(space: str, state: np.ndarray, action: np.ndarray, state_pose: np.ndarray, action_pose: np.ndarray) -> dict[str, np.ndarray]:
    """The vector features' values (rows: frames), by key, as vector_features."""
    if space == "ee":
        return {"observation.state": ee_vector(state_pose, state), "observation.joints": state,
                "action": ee_vector(action_pose, action), "action.joints": action}
    return {"observation.state": state, "observation.ee_pose": state_pose, "action": action, "action.ee_pose": action_pose}


def image_decoders(args) -> dict[str, tuple[str, object]]:
    """Image feature key -> (topic, decode: CompressedImage data -> the image at --image-size)."""
    size = args.image_size
    decoders = {f"observation.images.{camera}": (color, lambda d: resize(decode_color(d), size)) for camera, (color, _) in CAMERAS.items()}
    if args.depth:
        decoders |= {f"observation.depth.{camera}": (topic, lambda d: resize(decode_depth(d), size, depth=True))
                     for camera, (_, topic) in CAMERAS.items()}
    return decoders


def features(series: dict[str, Series], args) -> dict:
    def video(data: bytes, decode, extra: dict) -> dict:
        h, w, c = decode(data).shape
        return {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channels"], **extra}

    result = {key: {"dtype": "float32", "shape": (len(names),), "names": names} for key, names in vector_features(args.space).items()}
    for key, (topic, decode) in image_decoders(args).items():
        extra = {"info": {"is_depth_map": True}} if key.startswith("observation.depth.") else {}
        result[key] = video(series[topic].values[0], decode, extra)
    return result


def episode_bags(path: Path) -> list[Path]:
    """The episodes' MCAP files: path is one episode directory or a directory of them. Episodes without a
    metadata.yaml (rosbag2 writes it when the bag is finished: still recording, or the recorder died) are skipped."""
    if list(path.glob("*.mcap")):
        dirs = [path]
    else:
        dirs = sorted(d for d in path.iterdir() if d.is_dir() and list(d.glob("*.mcap")))
    bags = []
    for d in dirs:
        if not (d / "metadata.yaml").exists():
            print(f"skipping {d}: no metadata.yaml (an unfinished bag)", file=sys.stderr)
            continue
        files = sorted(d.glob("*.mcap"))
        if len(files) > 1:
            raise ValueError(f"{d}: {len(files)} MCAP files; one per episode is supported")
        bags.append(files[0])
    return bags


def convert_episode(dataset, series: dict[str, Series], args) -> int:
    times = frame_times(series, args.fps)
    state = interpolate(series[JOINT_STATES].stamps, series[JOINT_STATES].values, times).astype(np.float32)
    reference = series[CONTROLLER_STATE]
    action = np.asarray(reference.values, dtype=np.float32)[latest(reference.stamps, times)]
    state_pose, action_pose = np.split(poses(args.model, np.concatenate([state, action])), 2)  # one FK call
    state_pose = continuous_quaternions(state_pose)
    action_pose = continuous_quaternions(action_pose, reference=state_pose)
    vectors = vector_values(args.space, state, action, state_pose, action_pose)
    image_keys = image_decoders(args)
    picks = {}
    for key, (topic, _) in image_keys.items():
        s = series[topic]
        picks[key] = nearest(s.stamps, times)
        lag = np.abs(np.asarray(s.stamps)[picks[key]] - times)
        if (late := int(np.sum(lag > 0.5 / args.fps))) > 0:
            print(f"  {topic}: {late} of {len(times)} frames have no image within half a frame (max {lag.max() * 1e3:.0f} ms off)",
                  file=sys.stderr)

    for i in range(len(times)):
        frame = {key: values[i] for key, values in vectors.items()} | {"task": args.task}
        for key, (topic, decode) in image_keys.items():
            frame[key] = decode(series[topic].values[picks[key][i]])
        dataset.add_frame(frame)
    dataset.save_episode()
    return len(times)


def convert_bag(bag: Path, root: Path, args) -> int:
    """One episode into a dataset of its own at root (a process of the pool: datasets have one writer each)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # slow (PyTorch): in the workers, after the arguments

    series = read_bag(bag, args.depth)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features(series, args),
        root=root,
        robot_type=args.model,
        use_videos=True,
    )
    try:
        frames = convert_episode(dataset, series, args)
    finally:
        dataset.finalize()
    print(f"{bag.parent.name}: {frames} frames ({frames / args.fps:.1f} s)", flush=True)
    return frames


def image_size(text: str) -> tuple[int, int]:
    """'360x640' -> (360, 640)."""
    try:
        h, w = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: expected HEIGHTxWIDTH, e.g. 360x640") from None
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: sizes must be positive")
    return h, w


def resolve(path: str) -> Path:
    """Relative paths are from where `bazel run` was run."""
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", "."), path).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="an episode directory (data_recorder --episodic) or a directory of them")
    ap.add_argument("--repo-id", required=True, help="the dataset's name, <user or org>/<name> (e.g. local/rebot_grasp_cube)")
    ap.add_argument("--output", help="the dataset's directory (default: <input>_lerobot next to the input)")
    ap.add_argument("--overwrite", action="store_true", help="replace --output if it exists")
    ap.add_argument("--task", default="Pick up the cube.", help="the episodes' task (language instruction)")
    ap.add_argument("--fps", type=int, default=20, help="frames per second (default 20, the cameras' rate)")
    ap.add_argument("--image-size", type=image_size, default=IMAGE_SIZE, metavar="HxW",
                    help="the images' height x width (default %dx%d: the cameras' halved)" % IMAGE_SIZE)
    ap.add_argument("--space", choices=["joint", "ee"], default="joint",
                    help="what observation.state and action are, the vectors LeRobot's policies learn from: the joints "
                         "(default) or the tool's pose and the gripper (see above)")
    ap.add_argument("--depth", action="store_true",
                    help="also the depth images (12-bit HEVC: slower to encode). LeRobot's policies take every video "
                         "as a colour camera's: these datasets are for a policy that handles depth")
    ap.add_argument("--model", default="rebot_arm_gemini_305", choices=sorted(kinematics_client.KINEMATICS),
                    help="the robot model whose kinematics give the poses (default rebot_arm_gemini_305)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="episodes converted at once, a process each (default: the number of CPUs)")
    args = ap.parse_args()

    source = resolve(args.input)
    output = resolve(args.output) if args.output else source.with_name(source.name + "_lerobot")
    bags = episode_bags(source)
    if not bags:
        raise SystemExit(f"no episodes (directories with an .mcap) in {source}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} exists (--overwrite to replace it)")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Each episode a dataset of its own (in parallel), then aggregated into the output in the episodes' order.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as parts:
        roots = [Path(parts, bag.parent.name) for bag in bags]
        jobs = max(1, min(args.jobs, len(bags)))
        print(f"converting {len(bags)} episodes, {jobs} at a time", flush=True)
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool:
            frames = sum(pool.map(convert_bag, bags, roots, [args] * len(bags)))

        from lerobot.datasets.aggregate import aggregate_datasets

        print(f"aggregating into {output}", flush=True)
        aggregate_datasets(repo_ids=[f"{args.repo_id}_{r.name}" for r in roots], aggr_repo_id=args.repo_id,
                           roots=roots, aggr_root=output)
    print(f"{len(bags)} episodes, {frames} frames ({frames / args.fps:.1f} s) -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
