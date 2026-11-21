"""Runs a LeRobot policy trained on our datasets (//src/lerobot_export:bag_to_lerobot): it is given what the
datasets' frames hold (lerobot_export.conventions) and its actions become joint targets for the trajectory
controller the data was recorded through. No ROS here (run_policy is the node).

    runner = PolicyRunner("outputs/train/<run>/checkpoints/last/pretrained_model")
    runner.reset()                              # at an episode's start (a policy's queued actions are dropped)
    targets = runner.act(joints, images)        # joints: JOINTS' measured positions; images: camera -> RGB (H, W, 3)
    runner.close()

The checkpoint says the rest: its observation.state's size the space (7: joints, 10: the tool's pose and the
gripper, see conventions), its image features the cameras (observation.images.<camera>, CAMERAS) and their size
(live images are resized to it, as the dataset's were).
A policy in the ee space has its poses made joints by IK (the model's kinematics, kept in one process).
"""

from pathlib import Path

import numpy as np

from src.lerobot_export.conventions import ARM_JOINTS, CAMERAS, IMAGE_KEY_PREFIX, JOINTS, SPACES, ee_vector, quaternion_from_6d, resize
from src.trajectory_generator import kinematics_client
from src.trajectory_generator.cartesian_path import Pose
from src.trajectory_generator.kinematics_client import KinematicsError

DEFAULT_TASK = "Pick up the cube."  # bag_to_lerobot's default


class PolicyError(RuntimeError):
    """An action that is not sent: IK has no solution, or a joint would jump."""


class PolicyRunner:
    def __init__(self, path: str | Path, device: str = "cuda", task: str = DEFAULT_TASK,
                 model: str = "rebot_arm_gemini_305", max_joint_step: float = 0.1):
        """max_joint_step: rad an arm joint (joint1..6) may move from where it is in one action; more is refused
        (PolicyError), as is an IK solution that jumps by more."""
        import torch  # slow (PyTorch): here, not at import
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        path = Path(path)
        self.task, self.model, self.max_joint_step = task, model, max_joint_step
        self.device = torch.device(device)
        self.config = PreTrainedConfig.from_pretrained(path)
        self.config.device = device
        self.policy = get_policy_class(self.config.type).from_pretrained(path, config=self.config)
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})

        state, action = self.config.robot_state_feature, self.config.action_feature
        if state is None or action is None:
            raise ValueError(f"{path}: a policy with an observation.state and an action is needed")
        if state.shape[0] not in SPACES or action.shape != state.shape:
            raise ValueError(f"{path}: observation.state {state.shape} and action {action.shape}: expected both "
                             f"(7,) (joints) or (10,) (the tool's pose and the gripper)")
        self.space = SPACES[state.shape[0]]
        self.image_shapes = {}  # camera -> (H, W, 3)
        for key, feature in self.config.image_features.items():
            camera = key.removeprefix(IMAGE_KEY_PREFIX)
            if not key.startswith(IMAGE_KEY_PREFIX) or camera not in CAMERAS:
                raise ValueError(f"{path}: image feature {key}: expected {IMAGE_KEY_PREFIX}<camera>, camera one of {sorted(CAMERAS)}")
            c, h, w = feature.shape
            self.image_shapes[camera] = (h, w, c)
        self.kinematics = kinematics_client.Kinematics(model) if self.space == "ee" else None

    @property
    def cameras(self) -> list[str]:
        return list(self.image_shapes)

    def reset(self):
        """At an episode's start: policies keep state (e.g. ACT's queue of a chunk's actions)."""
        self.policy.reset()

    def observation(self, joints: np.ndarray, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The frame the policy was trained on, from the measured joints (JOINTS) and the cameras' images."""
        joints = np.asarray(joints, np.float32)
        if joints.shape != (len(JOINTS),):
            raise ValueError(f"joints: expected {len(JOINTS)} values ({', '.join(JOINTS)}), got shape {joints.shape}")
        if self.space == "ee":
            pose = self.kinematics.forward(dict(zip(ARM_JOINTS, map(float, joints[:6]))))
            state = ee_vector(np.array([[*pose.position, *pose.quaternion]]), joints[None])[0]
        else:
            state = joints
        frame = {"observation.state": state}
        for camera, shape in self.image_shapes.items():
            if camera not in images:
                raise ValueError(f"no image of camera {camera}")
            image = np.asarray(images[camera])
            if image.ndim != 3 or image.shape[2] != shape[2] or image.dtype != np.uint8:
                raise ValueError(f"camera {camera}: the policy takes {shape[2]}-channel uint8 RGB images, got {image.shape} {image.dtype}")
            frame[IMAGE_KEY_PREFIX + camera] = resize(image, shape[:2])  # as the dataset's were (bag_to_lerobot --image-size)
        return frame

    def act(self, joints: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        """The policy's next action as joint targets (JOINTS), from the measured joints and the images. Raises
        PolicyError for an action not to send (IK, a jump): the arm should hold."""
        from lerobot.common.control_utils import predict_action

        action = predict_action(self.observation(joints, images), self.policy, self.device, self.preprocessor,
                                self.postprocessor, use_amp=False, task=self.task, robot_type=self.model)
        action = action.squeeze(0).float().cpu().numpy()
        targets = self.joints_from_ee(action, joints) if self.space == "ee" else action.astype(np.float64)
        step = np.abs(targets[:6] - np.asarray(joints[:6]))
        worst = int(np.argmax(step))
        if step[worst] > self.max_joint_step:
            raise PolicyError(f"the action moves {JOINTS[worst]} by {step[worst]:.3f} rad in one step (max {self.max_joint_step})")
        return targets

    def joints_from_ee(self, action: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """An ee-space action (EE_NAMES) -> joint targets (JOINTS): IK from the measured joints, the gripper as is."""
        pose = Pose(tuple(map(float, action[:3])), tuple(quaternion_from_6d(action[3:9])))
        try:
            path = self.kinematics.ik_path(dict(zip(ARM_JOINTS, map(float, joints[:6]))), [pose],
                                           max_joint_step=self.max_joint_step)
        except KinematicsError as e:
            raise PolicyError(f"IK: {e}") from None
        arm = dict(zip(path.joint_names, path.positions[0]))
        return np.array([arm[j] for j in ARM_JOINTS] + [float(action[9])])

    def close(self):
        if self.kinematics is not None:
            self.kinematics.close()

    def __enter__(self) -> "PolicyRunner":
        return self

    def __exit__(self, *exc):
        self.close()
