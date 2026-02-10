#!/usr/bin/env python
from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

from lerobot.teleoperators.config import TeleoperatorConfig



@TeleoperatorConfig.register_subclass("accrea_gamepad_joints")
@dataclass
class AccreaGamepadJointsTeleopConfig(TeleoperatorConfig):
    # Control loop
    hz: float = 30.0
    deadzone: float = 0.12

    # Buttons (pygame indices on Linux Xbox)
    deadman_btn: int = 5  # RB
    exit_btn: int = 7     # START

    # Axes mapping
    left_stick_x_axis: int = 0
    left_stick_y_axis: int = 1
    right_stick_x_axis: int = 3
    right_stick_y_axis: int = 4

    # D-pad hat index (preferred). If no hats, fallback uses axes 6/7.
    dpad_hat_index: int = 0
    dpad_fallback_x_axis: int = 6
    dpad_fallback_y_axis: int = 7

    # Trigger mapping
    lt_axis: int = 2
    rt_axis: int = 5
    combined_triggers_on_lt_axis: bool = True  # if rt axis missing, interpret lt axis +/- as combined

    # Joint motion rate (rad/s at full deflection)
    joint_rate: list[float] = field(default_factory=lambda: [0.6, 0.6, 0.6, 0.8, 0.8, 1.0])

    # Gripper motion rate (units/s)
    gripper_rate: float = 0.8

    # Sign flips if you want to fix “reversed” directions cleanly (no code edits)
    invert_lx: bool = False
    invert_ly: bool = False
    invert_rx: bool = False
    invert_ry: bool = False
    invert_hatx: bool = False
    invert_haty: bool = False
