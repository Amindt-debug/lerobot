import torch
import time

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.datasets.utils import hw_to_dataset_features

from lerobot.robots.accrea_follower.accrea_follower import AccreaFollower
from lerobot.robots.accrea_follower.config_accrea_follower import AccreaFollowerConfig

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fine-tuned SmolVLA checkpoint for inference
    model_dir = "/home/roboticslab/vla/lerobot/outputs/train/020000/pretrained_model"

    print(f"Using device: {device}")
    print(f"Loading model from: {model_dir}")

    # Load fine-tuned policy
    model = SmolVLAPolicy.from_pretrained(model_dir)
    model.eval()

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_dir,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print("Expected image keys:", model.config.image_features)

    robot_cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        require_user_confirmation=False,
        max_delta_per_step_rad=0.02,
        cameras={
            "top": OpenCVCameraConfig(
                index_or_path="/dev/video2",
                width=640,
                height=480,
                fps=30,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                width=640,
                height=480,
                fps=30,
            ),
        },
    )

    robot = AccreaFollower(robot_cfg)
    robot.connect()

    try:
        task = "Pick up the apple and place it in the bowl"
        robot_type = "accrea_follower"

        # Build dataset-style features from robot hardware interface
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}

        # Match inference keys to the keys used during training
        rename_map = {
            "observation.images.top": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        }

        for ep in range(MAX_EPISODES):
            print(f"Starting episode {ep + 1}/{MAX_EPISODES}")

            # Very important: reset model action queue at episode start
            # model.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                obs = robot.get_observation()

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=task,
                    robot_type=robot_type,
                )

                renamed_obs_frame = {
                    rename_map.get(k, k): v for k, v in obs_frame.items()
                }

                if ep == 0 and step == 0:
                    print("Observation keys before rename:", list(obs_frame.keys()))
                    print("Observation keys after rename:", list(renamed_obs_frame.keys()))

                proc_obs = preprocess(renamed_obs_frame)

                t0 = time.time()
                action = model.select_action(proc_obs)
                t1 = time.time()
                print(f"select_action took {t1 - t0:.3f} s")
                action = postprocess(action)
                action = make_robot_action(action, dataset_features)

                robot.send_action(action)

            print("Episode finished.")

    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()