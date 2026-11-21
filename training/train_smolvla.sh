#!/bin/bash
# Fine-tune SmolVLA (and optionally an ACT baseline) on the dual-arm BaristaBot dataset, on the HPC.
#
# One-time setup on the HPC LOGIN node (it has internet; compute nodes often don't):
#   python3 -m venv ~/lerobot_env && source ~/lerobot_env/bin/activate
#   pip install "lerobot[smolvla]==0.7.0"        # the SAME LeRobot version the repo exported with
#   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # on a GPU node
#   export HF_HOME=$HOME/hf_cache                # shared cache the compute node can read
#   hf download lerobot/smolvla_base             # pretrained policy
#   hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct   # its vision-language backbone
#
# Then copy the exported dataset folder (bag_to_lerobot's output, e.g. barista_lerobot/) to the HPC and:
#   sbatch train_smolvla.sh            # or: bash train_smolvla.sh   on an interactive GPU node
#
#SBATCH --job-name=smolvla_barista
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=smolvla_barista_%j.log
##SBATCH --partition=<your GPU partition>   # remove one '#' and fill in

set -euo pipefail

# ---- edit these ----
DATASET_ROOT="$HOME/datasets/barista_lerobot"   # the folder with meta/, data/, videos/
REPO_ID="local/barista"                         # any name; must match nothing on the Hub
STEPS=20000                                     # ~20 demos at fixed positions: try checkpoints from 6000 on
BATCH=32                                        # 64 if the GPU has 40 GB or more
SAVE_FREQ=2000
OUT="$HOME/outputs/smolvla_barista_$(date +%m%d_%H%M)"
# ---------------------

source ~/lerobot_env/bin/activate
export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1          # use the downloaded weights; don't try the internet from the compute node

# smolvla_base was pretrained with cameras named camera1/2/3: map our one camera onto camera1, pad the other two as empty.
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --rename_map='{"observation.images.main": "observation.images.camera1"}' \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=100 \
  --output_dir="$OUT" \
  --job_name=smolvla_barista \
  --wandb.enable=false

# Optional ACT baseline (no pretrained weights, no renaming): uncomment to run after SmolVLA.
# lerobot-train \
#   --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
#   --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT" \
#   --batch_size=8 --steps=40000 --save_freq=5000 \
#   --output_dir="${OUT}_act" --job_name=act_barista --wandb.enable=false

echo "checkpoints in $OUT/checkpoints/<step>/pretrained_model"
