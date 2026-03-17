lerobot-teleoperate \
  --robot.type=accrea_follower \
  --robot.id=accrea_aria_01 \
  --robot.robot_ip=192.168.9.9 \
  --robot.robot_port=7777 \
  --robot.max_delta_per_step_rad=0.02 \
  --teleop.type=accrea_gamepad_joints \
  --teleop.id=xbox_0 \
  --fps=30


## Teleoperation by moving TCP: DOF = 3T + 3R (rpy) = 6 DOF 

export ACCREA_URDF_PATH=/home/roboticslab/ALMCollection/RobotDescriptions/accrea_aria_description/urdf/aria_simplified.urdf

lerobot-teleoperate \
  --robot.type=accrea_follower \
  --robot.id=accrea_aria_01 \
  --robot.robot_ip=192.168.9.9 \
  --robot.robot_port=7777 \
  --robot.max_delta_per_step_rad=0.07 \
  --robot.require_user_confirmation=false \
  --teleop.type=accrea_gamepad_cartesian \
  --teleop.id=xbox_0 \
  --teleop.hz=60 \
  --fps=60 \
  --teleop.lin_vel_mps=0.7 \
  --teleop.roll_vel_rps=0.7 \
  --teleop.pitch_vel_rps=0.7 \
  --teleop.yaw_vel_rps=0.7 \
  --teleop.damping_lambda=0.04 \
  --teleop.max_qd_rad_s=1.4 \
  --teleop.ee_link=tcp \
  --display_data=false


## Teleoperation by moving TCP - Camera Included 

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
  --display_data=false

  ## Teleoperation by moving TCP - to port it and open the GUI ##


  # 1) Run this in 1st terminal in pc

    rerun --serve-grpc --bind 0.0.0.0 --port 9877

  # 2) Run this in the Orin terminal in /home/roboticslab/vla/lerobot after `conda activate lerobot` or directly run ./start_accrea_teleop.sh

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
  --teleop.lin_vel_mps=1.0 \
  --teleop.roll_vel_rps=1.0 \
  --teleop.pitch_vel_rps=1.0 \
  --teleop.yaw_vel_rps=1.0 \
  --teleop.damping_lambda=0.04 \
  --teleop.max_qd_rad_s=1.6 \
  --teleop.ee_link=tcp \
  --display_data=true \
  --display_ip=192.168.88.243 \
  --display_port=9877

  # 3) Run in the 2nd terminal in pc 
  rerun --connect rerun+http://127.0.0.1:9877/proxy --expect-data-soon
  

## If DNS breaks again and I can not push to Git ( You only had to run the DNS fix because initially your system couldn’t resolve github.com. Now that you set DNS via NetworkManager, it should stay fixed — but if it ever breaks again, repeat these two lines )

sudo nmcli connection modify "Wired connection 2" ipv4.dns "1.1.1.1 8.8.8.8"
sudo nmcli connection modify "Wired connection 2" ipv4.ignore-auto-dns yes


## ** ToDo: Adapting URDF file with respect to my setup ** ##


## Recording 
lerobot-record \
  --robot.type=accrea_follower \
  --robot.id=accrea_aria_01 \
  --robot.robot_ip=192.168.9.9 \
  --robot.robot_port=7777 \
  --robot.max_delta_per_step_rad=0.07 \
  --robot.require_user_confirmation=false \
  --robot.cameras='{
    top: {index_or_path: "/dev/video0", width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=accrea_gamepad_cartesian \
  --teleop.id=xbox_0 \
  --teleop.hz=30 \
  --teleop.exit_btn=6 \
  --teleop.lin_vel_mps=0.10 \
  --teleop.roll_vel_rps=0.20 \
  --teleop.pitch_vel_rps=0.20 \
  --teleop.yaw_vel_rps=0.20 \
  --teleop.damping_lambda=0.06 \
  --teleop.max_qd_rad_s=1.0 \
  --teleop.ee_link=tcp \
  --dataset.repo_id=local/debug_cam_check_headless_ok \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=1 \
  --dataset.fps=30 \
  --dataset.single_task="camera check" \
  --display_data=false

## sometimes i need to remove and the again record
rm -rf /home/roboticslab/.cache/huggingface/lerobot/local/debug_cam_check