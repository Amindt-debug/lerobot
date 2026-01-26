import time
import torch
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/pi0_base"

    print("✅ Loading PI0:", model_id)
    model = PI0Policy.from_pretrained(model_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ✅ Pi0 expects base_0_rgb + left_wrist_0_rgb + right_wrist_0_rgb
    # You only have one wrist cam => we feed it as left_wrist and duplicate it for right_wrist in the adapter.
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30),
        "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
        # NOTE: no right_wrist device; the adapter duplicates frames
    }

    robot_cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras=camera_config,
        max_delta_per_step_rad=0.01,       # ✅ very safe
        require_user_confirmation=True,    # ✅ prompts before real motion
    )

    robot = AccreaFollower(robot_cfg)

    print("\n✅ Connecting robot...")
    robot.connect()

    # dataset_features mapping (important!)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    task = "move slightly"   # you can put your real instruction later
    robot_type = "accrea_follower"

    try:
        for step in range(10):
            obs = robot.get_observation()

            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=task,
                robot_type=robot_type,
            )

            obs_proc = preprocess(obs_frame)

            with torch.no_grad():
                raw_action = model.select_action(obs_proc)

            raw_action = postprocess(raw_action)

            # ✅ Important: PI0 sometimes outputs action CHUNKS (horizon H).
            # We only execute the first action each step (safe + simple).
            # This handler is robust to both (D,) and (H,D) outputs.
            if isinstance(raw_action, dict) and "action" in raw_action:
                a = raw_action["action"]
                if hasattr(a, "ndim") and a.ndim == 2:
                    raw_action["action"] = a[0]

            action = make_robot_action(raw_action, dataset_features)

            print(f"\n--- Step {step} ---")
            print("Sending action keys:", list(action.keys()))

            robot.send_action(action)
            time.sleep(0.25)

    finally:
        print("\n✅ Disconnecting...")
        robot.disconnect()
        print("✅ Done.")


if __name__ == "__main__":
    main()
