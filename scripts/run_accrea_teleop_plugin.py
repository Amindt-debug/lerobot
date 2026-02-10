#!/usr/bin/env python3
import time
import numpy as np

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig
from lerobot.teleoperators.accrea_gamepad_joints import AccreaGamepadJointsTeleop, AccreaGamepadJointsTeleopConfig


def main():
    robot = AccreaFollower(AccreaFollowerConfig(require_user_confirmation=True, max_delta_per_step_rad=0.02))
    robot.connect()

    obs = robot.get_observation()
    q = np.array([obs[f"joint_{i}.pos"] for i in range(6)], dtype=float)
    g = float(obs.get("gripper.pos", 0.0))

    teleop = AccreaGamepadJointsTeleop(AccreaGamepadJointsTeleopConfig())
    teleop.connect()
    teleop.set_initial_targets(q, g)

    hz = teleop.config.hz
    dt = 1.0 / hz

    print("Teleop plugin running. Hold RB to move. START to quit.")
    try:
        while True:
            action = teleop.get_action()

            # If teleop requested termination
            events = teleop.get_teleop_events()
            if events.get("terminate_episode", False):
                break

            robot.send_action(action)
            time.sleep(dt)
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
