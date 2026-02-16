#!/usr/bin/env python3
import os
import time
import numpy as np

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig
from lerobot.teleoperators.accrea_gamepad_cartesian import (
    AccreaGamepadCartesianTeleop,
    AccreaGamepadCartesianTeleopConfig,
)
from lerobot.teleoperators.utils import TeleopEvents


def main():
    # --- IMPORTANT: set URDF path for the cartesian teleop (Pinocchio model build) ---
    # Change this path if your URDF lives somewhere else.
    os.environ.setdefault(
        "ACCREA_URDF_PATH",
        "/home/roboticslab/ALMCollection/RobotDescriptions/accrea_aria_description/urdf/aria_simplified.urdf",
    )

    # --- Robot ---
    robot = AccreaFollower(
        AccreaFollowerConfig(
            robot_ip="192.168.9.9",
            robot_port=7777,
            require_user_confirmation=True,
            max_delta_per_step_rad=0.02,
        )
    )
    robot.connect()

    # Read initial robot state
    obs = robot.get_observation()
    q = np.array([obs[f"joint_{i}.pos"] for i in range(6)], dtype=float)
    g = float(obs.get("gripper.pos", 0.0))

    # --- Teleop (Cartesian) ---
    teleop = AccreaGamepadCartesianTeleop(
        AccreaGamepadCartesianTeleopConfig(
            hz=30,
            lin_vel_mps=0.06,     # start conservative
            yaw_vel_rps=0.8,      # start conservative
            damping_lambda=0.06,  # a bit more damping for stability
            max_qd_rad_s=1.0,
            ee_link="tcp",
            base_link="link_0",
        )
    )
    teleop.connect()

    # Initialize targets (we keep them anchored to robot state)
    # If your Teleoperator base class doesn't have set_initial_targets, we just rely on obs in get_action
    hz = teleop.config.hz
    dt = 1.0 / float(hz)

    print("Cartesian teleop running.")
    print("Hold RB (deadman) to move. Left stick = XY, D-pad up/down = Z, Right stick X = yaw, triggers = gripper. START to quit.")

    try:
        while True:
            obs = robot.get_observation()

            # Teleop uses obs to compute action (and internally maintains q_target)
            action = teleop.get_action(obs)

            # exit requested?
            events = teleop.get_events()
            if getattr(events, "should_exit", False):
                break

            # If action is empty (e.g., exit), skip
            if action:
                robot.send_action(action)

            time.sleep(dt)

    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
