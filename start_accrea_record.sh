#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable settings
# =========================
DATASET_REPO_ID="amindt/accrea_apple_pick_v1"
TASK_TEXT="Pick up the apple and place it in the bowl"
NUM_EPISODES=100
EPISODE_TIME_S=35
RESET_TIME_S=8

# Optional local dataset folder
DATASET_ROOT="/home/roboticslab/vla/datasets/accrea_apple_pick_v1"

# Robot / network
ROBOT_ID="accrea_aria_01"
ROBOT_IP="192.168.9.9"
ROBOT_PORT="7777"

# Display / rerun
DISPLAY_IP="192.168.88.243"
DISPLAY_PORT="9877"

# URDF
export ACCREA_URDF_PATH="/home/roboticslab/ALMCollection/RobotDescriptions/accrea_aria_description/urdf/aria_simplified.urdf"

# Camera mapping
# You told me:
#   /dev/video0 = wrist
#   /dev/video2 = top
CAM_WRIST="/dev/video0"
CAM_TOP="/dev/video2"

# =========================
# Run from lerobot root
# =========================
cd /home/roboticslab/vla/lerobot

echo "Starting ACCREA recording..."
echo "Dataset repo id : ${DATASET_REPO_ID}"
echo "Task            : ${TASK_TEXT}"
echo "Episodes        : ${NUM_EPISODES}"
echo "Episode time    : ${EPISODE_TIME_S}s"
echo "Reset time      : ${RESET_TIME_S}s"
echo "Wrist camera    : ${CAM_WRIST}"
echo "Top camera      : ${CAM_TOP}"
echo "URDF            : ${ACCREA_URDF_PATH}"


# Remove previous local dataset folder if it already exists
if [ -d "${DATASET_ROOT}" ]; then
  echo "Removing existing dataset folder: ${DATASET_ROOT}"
  rm -rf "${DATASET_ROOT}"
fi

lerobot-record \
  --robot.type=accrea_follower \
  --robot.id="${ROBOT_ID}" \
  --robot.robot_ip="${ROBOT_IP}" \
  --robot.robot_port="${ROBOT_PORT}" \
  --robot.max_delta_per_step_rad=0.15 \
  --robot.require_user_confirmation=false \
  --robot.cameras="{
    wrist: {index_or_path: \"${CAM_WRIST}\", width: 640, height: 480, fps: 30},
    top:   {index_or_path: \"${CAM_TOP}\",   width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=accrea_gamepad_cartesian \
  --teleop.id=xbox_0 \
  --teleop.hz=60 \
  --teleop.lin_vel_mps=1.7 \
  --teleop.roll_vel_rps=1.7 \
  --teleop.pitch_vel_rps=1.7 \
  --teleop.yaw_vel_rps=1.7 \
  --teleop.damping_lambda=0.04 \
  --teleop.max_qd_rad_s=2.4 \
  --teleop.ee_link=tcp \
  --display_data=true \
  --display_ip="${DISPLAY_IP}" \
  --display_port="${DISPLAY_PORT}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK_TEXT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false