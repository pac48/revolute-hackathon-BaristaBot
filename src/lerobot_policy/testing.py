"""Policy checkpoints for tests: untrained, saved as lerobot-train saves pretrained_model (there is no trained one)."""

from pathlib import Path

import numpy as np

from src.lerobot_export import conventions as c

IMAGE = (48, 64, 3)  # small, for speed (the datasets': DATASET_IMAGE)
DATASET_IMAGE = (*c.IMAGE_SIZE, 3)
CAMERA_IMAGE = (720, 1280, 3)  # the cameras': resized to the policy's by PolicyRunner


def save_checkpoint(path: Path, state_size: int, cameras=("scene",), image=IMAGE, small: bool = True) -> Path:
    """An untrained ACT with normalization stats. small: a tiny transformer and 5-action chunks (the CPU tests');
    else ACT's defaults (as trained: 100-action chunks), with a ResNet-18 backbone either way (untrained)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    h, w, ch = image
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_size,))}
    inputs |= {c.IMAGE_KEY_PREFIX + cam: PolicyFeature(FeatureType.VISUAL, (ch, h, w)) for cam in cameras}
    size = dict(chunk_size=5, n_action_steps=5, dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1,
                n_decoder_layers=1, use_vae=False) if small else {}
    config = ACTConfig(input_features=inputs, output_features={"action": PolicyFeature(FeatureType.ACTION, (state_size,))},
                       device="cpu", pretrained_backbone_weights=None, **size)
    vector = {"mean": np.zeros(state_size), "std": np.full(state_size, 0.01), "min": -np.ones(state_size), "max": np.ones(state_size)}
    stats_image = {"mean": np.full((3, 1, 1), 0.5), "std": np.full((3, 1, 1), 0.25), "min": np.zeros((3, 1, 1)), "max": np.ones((3, 1, 1))}
    stats = {"observation.state": vector, "action": vector} | {c.IMAGE_KEY_PREFIX + cam: stats_image for cam in cameras}
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    for part in (ACTPolicy(config), pre, post):
        part.save_pretrained(path)
    return path
