"""Trains a LeRobot policy (ACT by default) on one of our datasets (//src/lerobot_export:bag_to_lerobot): LeRobot's
lerobot-train, with this repo's defaults. Every lerobot-train option passes through (`-- --help` lists them).

Usage: bazel run //src/lerobot_policy:train -- --dataset.root=recordings/grasp_lerobot [--steps=100000]
           [--batch_size=8] [--output_dir=outputs/train/act_grasp] [--log_freq=200] [--save_freq=20000] [...]
       bazel run //src/lerobot_policy:train -- --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json --resume=true

Defaults added unless given (none when resuming: the checkpoint's config has them):
  --policy.type=act            (not with --policy.path, which fine-tunes a pretrained policy)
  --policy.device=cuda
  --policy.push_to_hub=false   (lerobot's default pushes the policy to the Hugging Face Hub)
  --dataset.repo_id=local/<the dataset directory's name>   (the dataset is read from --dataset.root)
  --job_name=<policy type>_<the dataset directory's name>
Relative paths are from where `bazel run` was run. Without --output_dir: outputs/train/<date>/<time>_<job_name>.
The checkpoints for run_policy: <output_dir>/checkpoints/<step>/pretrained_model (and .../last/...).
"""

import os
import sys
from pathlib import Path


def option(args: list[str], name: str) -> str | None:
    """The value of --name in args (--name=value or --name value), or None."""
    for i, arg in enumerate(args):
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
        if arg == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
    return None


def given(args: list[str], name: str) -> bool:
    """Whether --name or any of its sub-options (--name.x) is in args."""
    return any(a == f"--{name}" or a.startswith((f"--{name}=", f"--{name}.")) for a in args)


def with_defaults(args: list[str]) -> list[str]:
    """args with this repo's defaults (the module's docstring) added where not given."""
    if option(args, "resume") in ("true", "True", "1") or given(args, "help") or "-h" in args:
        return list(args)
    root = option(args, "dataset.root")
    if root is None:
        raise SystemExit("--dataset.root=<a dataset of bag_to_lerobot> is needed (e.g. recordings/grasp_lerobot)")
    name = Path(root).name.removesuffix("_lerobot") or "dataset"
    policy = option(args, "policy.type") or ("pretrained" if option(args, "policy.path") else "act")
    defaults = {
        "policy.device": "cuda",
        "policy.push_to_hub": "false",
        "dataset.repo_id": f"local/{name}",
        "job_name": f"{policy}_{name}",
    }
    if not (given(args, "policy.type") or given(args, "policy.path")):
        defaults["policy.type"] = "act"
    return [f"--{k}={v}" for k, v in defaults.items() if not given(args, k)] + list(args)


def main() -> None:
    args = with_defaults(sys.argv[1:])
    if "BUILD_WORKING_DIRECTORY" in os.environ:
        os.chdir(os.environ["BUILD_WORKING_DIRECTORY"])  # relative paths (dataset, outputs) from where bazel ran
    print("lerobot-train " + " ".join(args), flush=True)
    sys.argv = [sys.argv[0], *args]

    from lerobot.scripts.lerobot_train import main as lerobot_train  # slow (PyTorch): after the arguments

    lerobot_train()


if __name__ == "__main__":
    main()
