import time
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig

from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from lerobot.teleoperators.utils import TeleopEvents


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    # --- Cameras ---
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30),
        "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
    }

    # --- Robot config ---
    robot_cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras=camera_config,
        max_delta_per_step_rad=0.02,        # safety clamp
        require_user_confirmation=True,     # asks once before first move
    )

    robot = AccreaFollower(robot_cfg)

    print("\n✅ Connecting robot...")
    robot.connect()

    # --- Gamepad controller (official LeRobot) ---
    # These step sizes are not meters anymore for us. We treat them as joint increments scaling.
    # You can tune these.
    gamepad = GamepadController(
        x_step_size=0.03,  # rad per tick for joint0
        y_step_size=0.03,  # rad per tick for joint1
        z_step_size=0.03,  # rad per tick for joint2
        deadzone=0.1,
    )
    print("\n✅ Connecting gamepad...")
    gamepad.start()

    try:
        # Initialize targets from current robot state
        obs = robot.get_observation()
        dof = int(robot._dof)

        q_target = np.array([obs[f"joint_{i}.pos"] for i in range(dof)], dtype=np.float32)
        g_target = float(obs.get("gripper.pos", 0.5))

        print("\n==============================")
        print("✅ ACCREA JOINT TELEOP MODE")
        print("==============================")
        print("Left stick controls joint_0, joint_1")
        print("Right stick vertical controls joint_2")
        print("RB/LB (or whatever mapped) controls gripper open/close (LeRobot uses buttons 7/6)")
        print("B/Circle exits (LeRobot mapping).")
        print("Y success, A failure, X rerecord (LeRobot mapping).")
        print("==============================\n")

        dt = 0.05  # 20 Hz control loop

        while gamepad.running:
            gamepad.update()

            # --- Exit / episode events ---
            status = gamepad.get_episode_end_status()
            if status is not None:
                print(f"[Teleop] Episode event: {status}")
                if status in [TeleopEvents.FAILURE, TeleopEvents.RERECORD_EPISODE]:
                    print("[Teleop] Exiting teleop loop...")
                    break

            # --- Read joystick deltas from official GamepadController ---
            dx, dy, dz = gamepad.get_deltas()

            # ✅ Map joystick deltas to JOINT target increments
            # You can change this mapping any time.
            q_target[0] += float(dx)
            q_target[1] += float(dy)
            q_target[2] += float(dz)

            # OPTIONAL: add extra joints slowly drifting to 0
            # q_target[3] = q_target[3]
            # q_target[4] = q_target[4]
            # q_target[5] = q_target[5]

            # --- Gripper control from official controller flags ---
            grip_cmd = gamepad.gripper_command()  # "open", "close", "stay"
            if grip_cmd == "open":
                g_target = clamp(g_target + 0.03, 0.0, 1.0)
            elif grip_cmd == "close":
                g_target = clamp(g_target - 0.03, 0.0, 1.0)

            # Build action (absolute joint targets)
            action = {f"joint_{i}.pos": float(q_target[i]) for i in range(dof)}
            action["gripper.pos"] = float(g_target)

            # Send action to robot
            robot.send_action(action)

            time.sleep(dt)

    finally:
        print("\n✅ Disconnecting robot + gamepad...")
        try:
            gamepad.stop()
        except Exception:
            pass
        robot.disconnect()
        print("✅ Done.")


if __name__ == "__main__":
    main()
