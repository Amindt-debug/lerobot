#!/usr/bin/env bash
set -e

export ACCREA_URDF_PATH=/home/roboticslab/ALMCollection/RobotDescriptions/accrea_aria_description/urdf/aria_simplified.urdf

lerobot-teleoperate \
  --robot.type=accrea_follower \
  --robot.id=accrea_aria_01 \
  --robot.robot_ip=192.168.9.9 \
  --robot.robot_port=7777 \
  --robot.max_delta_per_step_rad=0.07 \
  --robot.require_user_confirmation=false \
  --robot.cameras='{
    top:   {index_or_path: "/dev/video0", width: 640, height: 480, fps: 30},
    wrist: {index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=accrea_gamepad_cartesian \
  --teleop.id=xbox_0 \
  --teleop.hz=60 \
  --fps=30 \
  --teleop.lin_vel_mps=0.7 \
  --teleop.roll_vel_rps=0.7 \
  --teleop.pitch_vel_rps=0.7 \
  --teleop.yaw_vel_rps=0.7 \
  --teleop.damping_lambda=0.04 \
  --teleop.max_qd_rad_s=1.4 \
  --teleop.ee_link=tcp \
  --display_data=true \
  --display_ip=192.168.88.243 \
  --display_port=9877
