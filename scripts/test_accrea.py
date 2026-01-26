import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig


def main():
    # Use the camera paths you detected with lerobot-find-cameras
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30),
        "wrist_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
    }

    cfg = AccreaFollowerConfig(
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras=camera_config,
        require_user_confirmation=False,  # NO motion in this test
    )

    robot = AccreaFollower(cfg)
    robot.connect()
    print("✅ Connected to AccreaFollower")

    obs = robot.get_observation()
    print("\n--- Observation keys ---")
    for k in obs.keys():
        print(" ", k)

    dof = robot._dof
    q_now = np.array([obs[f"joint_{i}.pos"] for i in range(dof)], dtype=float)

    print("\n--- Joint state ---")
    print("DOF =", dof)
    print("q_now =", np.round(q_now, 4))

    print("\n--- Camera frames ---")
    for cam_key in camera_config.keys():
        frame = obs[cam_key]
        # frame is typically numpy array (H,W,3)
        print(f"{cam_key}: type={type(frame)}, shape={getattr(frame, 'shape', None)}, dtype={getattr(frame, 'dtype', None)}")

    robot.disconnect()
    print("\n✅ Done (observation only).")


if __name__ == "__main__":
    main()
