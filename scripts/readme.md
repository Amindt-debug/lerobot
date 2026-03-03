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
  --robot.max_delta_per_step_rad=0.05 \
  --robot.require_user_confirmation=false \
  --teleop.type=accrea_gamepad_cartesian \
  --teleop.id=xbox_0 \
  --teleop.hz=60 \
  --fps=60 \
  --teleop.lin_vel_mps=0.5 \
  --teleop.roll_vel_rps=0.5 \
  --teleop.pitch_vel_rps=0.5 \
  --teleop.yaw_vel_rps=0.5 \
  --teleop.damping_lambda=0.04 \
  --teleop.max_qd_rad_s=1.2 \
  --teleop.ee_link=tcp \
  --display_data=false
