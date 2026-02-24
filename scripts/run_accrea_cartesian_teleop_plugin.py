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

    # Read initial robot state (optional, but nice)
    obs = robot.get_observation()
    q0 = np.array([obs[f"joint_{i}.pos"] for i in range(6)], dtype=float)
    g0 = float(obs.get("gripper.pos", 0.0))

    # --- Teleop (Cartesian) ---
    teleop = AccreaGamepadCartesianTeleop(
        AccreaGamepadCartesianTeleopConfig(
            hz=30,
            lin_vel_mps=0.08,      # conservative start
            yaw_vel_rps=0.08,      # conservative start
            damping_lambda=0.06,
            max_qd_rad_s=0.9,
            ee_link="tcp",
            base_link="link_0",
        )
    )
    teleop.connect()

    # (optional) initialize targets (your cartesian teleop now anchors to q_now anyway)
    try:
        teleop.set_initial_targets(q0, g0)
    except Exception:
        pass

    hz = teleop.config.hz
    dt = 1.0 / float(hz)

    print("Cartesian teleop running.")
    print("Hold RB (deadman) to move. START to quit.")

    try:
        while True:
            obs = robot.get_observation()

            # print("\nCurrent robot state:")
            # for k, v in obs.items():
            #     print(f"  {k}: {v}")

            action = teleop.get_action(obs)

            # print("\nTeleop action command:")
            # for k, v in action.items():
            #     print(f"  {k}: {v}")

            # Correct way in this codebase: check dict events
            ev = teleop.get_teleop_events()
            if ev.get(TeleopEvents.TERMINATE_EPISODE, False):
                break

            # Only send when deadman is held (your cartesian teleop returns {} otherwise)
            if action:
                robot.send_action(action)

            time.sleep(dt)

    finally:
        try:
            teleop.disconnect()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()