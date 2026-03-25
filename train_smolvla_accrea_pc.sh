#!/usr/bin/env bash
set -euo pipefail

# =========================
# ACCREA + SmolVLA finetune
# =========================

# --- Paths ---
BASE_DIR="/home/roboticslab/Documents/MT-amin/02_Data"
LEROBOT_DIR="${BASE_DIR}/lerobot"
DATASET_ROOT="${BASE_DIR}/datasets/accrea_apple_pick_v1"
DATASET_REPO_ID="amindt/accrea_apple_pick_v1"

# --- Output ---
RUN_NAME="accrea_smolvla_bs64_v2_rename"
OUTPUT_DIR="${LEROBOT_DIR}/outputs/train/${RUN_NAME}"

# --- Training ---
BATCH_SIZE=64
STEPS=20000
NUM_WORKERS=8

# --- Logging / saving ---
LOG_FREQ=50
SAVE_FREQ=1000

echo "=========================================="
echo "Starting SmolVLA fine-tuning"
echo "Repo dir     : ${LEROBOT_DIR}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Dataset id   : ${DATASET_REPO_ID}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Steps        : ${STEPS}"
echo "Workers      : ${NUM_WORKERS}"
echo "=========================================="

cd "${LEROBOT_DIR}"

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --log_freq="${LOG_FREQ}" \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable=false \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

echo "=========================================="
echo "Training command finished."
echo "Outputs saved in: ${OUTPUT_DIR}"
echo "=========================================="