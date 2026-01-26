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
        max_delta_per_step_rad=0.01,
        require_user_confirmation=False,
    )

    robot = AccreaFollower(cfg)

    print("\n✅ Connecting...")
    robot.connect()

    try:
        obs = robot.get_observation()
        dof = int(robot._dof)

        q_now = np.array([obs[f"joint_{i}.pos"] for i in range(dof)], dtype=np.float32)
        g_now = obs.get("gripper.pos", None)

        print("\nq_now =", np.round(q_now, 4))
        print("gripper_now =", g_now)

        # 🔥 Try open/close
        for g in [0.0, 0.5, 1.0, 0.2]:
            action = {f"joint_{i}.pos": float(q_now[i]) for i in range(dof)}
            action["gripper.pos"] = float(g)

            print(f"\nSending gripper.pos={g}")
            robot.send_action(action)
            time.sleep(1.5)

            obs2 = robot.get_observation()
            print("reported gripper.pos =", obs2.get("gripper.pos", None))

    finally:
        print("\n✅ Disconnecting...")
        robot.disconnect()
        print("✅ Done.")


if __name__ == "__main__":
    main()
