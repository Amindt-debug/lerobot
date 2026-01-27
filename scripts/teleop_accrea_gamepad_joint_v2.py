import time
import numpy as np

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig
from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from lerobot.teleoperators.utils import TeleopEvents


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    # ✅ Disable cameras for teleop stability (enable later for Pi0 policy)
    camera_config = {}

    robot_cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras=camera_config,
        max_delta_per_step_rad=0.08,        # safety clamp
        require_user_confirmation=False,     # asks once before first move
    )

    robot = AccreaFollower(robot_cfg)

    print("\n✅ Connecting robot...")
    robot.connect()

    obs = robot.get_observation()
    joint_pos_keys = sorted([k for k in obs.keys() if k.startswith("joint_") and k.endswith(".pos")])

    q0 = np.array([obs[k] for k in joint_pos_keys], dtype=np.float32)
    q1 = q0.copy()
    q1[5] += np.deg2rad(10)   # move joint 5 by 10 degrees

    action = {joint_pos_keys[i]: float(q1[i]) for i in range(len(q1))}
    action["gripper.pos"] = float(obs.get("gripper.pos", 0.5))

    print("Sending 10deg move on joint 5...")
    robot.send_action(action)

    time.sleep(1.0)
    obs2 = robot.get_observation()
    print("joint5 before:", q0[5], "after:", obs2[joint_pos_keys[5]])


    gamepad = GamepadController(
        x_step_size=0.20,
        y_step_size=0.20,
        z_step_size=0.20,
        deadzone=0.3,
    )

    print("\n✅ Connecting gamepad...")
    gamepad.start()

    try:
        obs = robot.get_observation()

        # ✅ detect joint pos keys automatically
        joint_pos_keys = sorted([k for k in obs.keys() if k.startswith("joint_") and k.endswith(".pos")])

        print("Detected joints:", joint_pos_keys)

        q_target = np.array([obs[k] for k in joint_pos_keys], dtype=np.float32)
        dof = len(joint_pos_keys)

        g_target = float(obs.get("gripper.pos", 0.5))


        print("\n==============================")
        print("✅ ACCREA JOINT TELEOP MODE")
        print("==============================")
        print("Left stick controls joint_0, joint_1")
        print("Right stick vertical controls joint_2")
        print("RB/LB controls gripper open/close")
        print("B exits")
        print("==============================\n")

        dt = 0.05  # 20 Hz

        # ✅ IMPORTANT: robot will pause and ask for ENTER (safety)
        # When you see "Press ENTER to allow motion..." -> press ENTER.

        while gamepad.running:
            gamepad.update()

            print("should_intervene:", gamepad.should_intervene())
            dx, dy, dz = gamepad.get_deltas()
            print("dx,dy,dz:", dx, dy, dz)

            status = gamepad.get_episode_end_status()
            if status is not None:
                print(f"[Teleop] Episode event: {status}")
                if status in [TeleopEvents.FAILURE, TeleopEvents.RERECORD_EPISODE]:
                    print("[Teleop] Exiting teleop loop...")
                    break

            dx, dy, dz = gamepad.get_deltas()

            # ✅ Debug: show joystick deltas when moving
            if abs(dx) > 1e-4 or abs(dy) > 1e-4 or abs(dz) > 1e-4:
                print(f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

            # Map joystick to joints
            q_target[0] += float(dx)
            q_target[1] += float(dy)
            q_target[2] += float(dz)

            # Gripper
            grip_cmd = gamepad.gripper_command()
            if grip_cmd == "open":
                g_target = clamp(g_target + 0.03, 0.0, 1.0)
            elif grip_cmd == "close":
                g_target = clamp(g_target - 0.03, 0.0, 1.0)

            action = {joint_pos_keys[i]: float(q_target[i]) for i in range(dof)}
            action["gripper.pos"] = float(g_target)

            robot.send_action(action)
            if int(time.time() * 10) % 10 == 0:  # ~1Hz
                obs2 = robot.get_observation()
                print("joint2 now:", obs2.get(joint_pos_keys[2], None), " target:", q_target[2])


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
