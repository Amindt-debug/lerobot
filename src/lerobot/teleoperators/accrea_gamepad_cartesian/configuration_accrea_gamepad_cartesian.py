from dataclasses import dataclass
from lerobot.teleoperators.teleoperator import TeleoperatorConfig

@dataclass
class AccreaGamepadCartesianTeleopConfig(TeleoperatorConfig):
    # loop
    hz: int = 30

    # gamepad
    joystick_index: int = 0
    deadman_btn: int = 5     # RB
    exit_btn: int = 7        # START

    # axes/buttons (xbox mapping usually)
    lx_axis: int = 0
    ly_axis: int = 1
    rx_axis: int = 3
    ry_axis: int = 4
    lt_axis: int = 2
    rt_axis: int = 5
    hat_idx: int = 0

    deadzone: float = 0.12

    # TCP control speeds
    lin_vel_mps: float = 0.05     # max translation speed at full stick
    yaw_vel_rps: float = 0.5      # yaw rate (rad/s) at full stick

    # gripper
    gripper_rate: float = 1.5     # units/s in [0..1]

    # IK settings
    damping_lambda: float = 0.05  # DLS damping
    max_qd_rad_s: float = 0.6    # clip joint speeds

    # kinematics frames (URDF link names)
    base_link: str = "link_0"
    ee_link: str = "tcp"          # your added tcp link
