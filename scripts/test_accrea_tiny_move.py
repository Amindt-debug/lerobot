import time
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig


def main():
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30),
        "wrist_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
    }

    cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras=camera_config,
        max_delta_per_step_rad=0.01,       # try 0.01 rad per step
        require_user_confirmation=False,   # ✅ NO prompt
    )

    robot = AccreaFollower(cfg)

    print("\n✅ Connecting...")
    robot.connect()

    try:
        print("\nDriver type:", type(robot._driver))
        methods = [m for m in dir(robot._driver) if ("joint" in m) or ("move" in m)]
        print("\n--- safe_robot_driver methods (move/joint) ---")
        print(methods)

        obs = robot.get_observation()
        dof = int(robot._dof)

        q_now = np.array([obs[f"joint_{i}.pos"] for i in range(dof)], dtype=np.float32)
        print("\n--- Current q ---")
        print("DOF =", dof)
        print("q_now =", np.round(q_now, 6))

        joint_to_move = 0
        step = 0.01

        q_target = q_now.copy()
        q_target[joint_to_move] += step

        action = {f"joint_{i}.pos": float(q_target[i]) for i in range(dof)}

        print("\n--- Sending action ---")
        print("joint =", joint_to_move, " step =", step)
        robot.send_action(action)

        time.sleep(1.0)

        obs2 = robot.get_observation()
        q_after = np.array([obs2[f"joint_{i}.pos"] for i in range(dof)], dtype=np.float32)

        print("\n--- After ---")
        print("q_after =", np.round(q_after, 6))
        print("delta   =", np.round(q_after - q_now, 6))

    finally:
        print("\n✅ Disconnecting...")
        robot.disconnect()
        print("✅ Done.")


if __name__ == "__main__":
    main()
