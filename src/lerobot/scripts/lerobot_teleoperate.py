# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
import numpy as np

from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


def _hold_action_from_obs(obs: dict) -> dict:
    hold = {f"joint_{i}.pos": float(obs[f"joint_{i}.pos"]) for i in range(6)}
    # keep gripper stable if available
    if "gripper.pos" in obs:
        hold["gripper.pos"] = float(obs["gripper.pos"])
    return hold    


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # --- DEBUG: stage marker ---
        logging.debug("[loop] start")

        # 1) observation
        try:
            obs = robot.get_observation()
        except Exception:
            logging.exception("[loop] robot.get_observation() failed")
            raise

        # Print obs keys once
        if not hasattr(teleop_loop, "_printed_obs_keys"):
            teleop_loop._printed_obs_keys = True
            logging.info("[debug] obs keys (first 40): %s", sorted(list(obs.keys()))[:40])

        # Optional: keep targets synced (your block) — guard KeyError
        if hasattr(teleop, "update_from_robot_state"):
            try:
                q = np.array([obs[f"joint_{i}.pos"] for i in range(6)], dtype=float)
                g = float(obs.get("gripper.pos", 0.0))
                teleop.update_from_robot_state(q, g)
            except KeyError:
                logging.error("[loop] Missing expected joint keys joint_0.pos..joint_5.pos in obs")
                logging.error("[loop] Available keys (first 60): %s", list(obs.keys())[:60])
                raise

        # 2) teleop action
        try:
            raw_action = teleop.get_action(obs)
        except Exception:
            logging.exception("[loop] teleop.get_action(obs) failed")
            raise

        # Print teleop raw keys once
        if not hasattr(teleop_loop, "_printed_raw_action"):
            teleop_loop._printed_raw_action = True
            logging.info("[debug] raw_action type=%s keys=%s",
                        type(raw_action), list(raw_action.keys()) if isinstance(raw_action, dict) else None)

        # 3) processors + HOLD fallback when deadman not pressed
        try:
            if isinstance(raw_action, dict) and len(raw_action) == 0:
                # Deadman not pressed => do NOT send empty dict. Hold current pose.
                teleop_action = raw_action  # keep for logging/visualization
                robot_action_to_send = _hold_action_from_obs(obs)
            else:
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action_to_send = robot_action_processor((teleop_action, obs))
        except Exception:
            logging.exception("[loop] action processors failed")
            raise

        # Print action features / sent keys once (keep your existing block here)

        # 4) send action
        try:
            _ = robot.send_action(robot_action_to_send)
        except Exception:
            logging.exception("[loop] robot.send_action(...) failed")
            raise

        # 4) send action
        try:
            _ = robot.send_action(robot_action_to_send)
        except Exception:
            logging.exception("[loop] robot.send_action(...) failed")
            raise

        # 5) optional: allow START to exit cleanly
        if hasattr(teleop, "get_teleop_events"):
            ev = teleop.get_teleop_events()
            # This depends on what TeleopEvents constants actually are; we’ll refine if needed.
            # For now, also support a plain string key.
            terminate = False
            if isinstance(ev, dict):
                terminate = bool(ev.get("terminate_episode", False)) or bool(ev.get("TERMINATE_EPISODE", False))
            if terminate:
                logging.info("[loop] Teleop requested termination. Exiting.")
                return

        ...

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()
    # --- Accrea gamepad teleop safety: initialize targets from current robot state ---
    obs0 = robot.get_observation()
    q0 = np.array([obs0[f"joint_{i}.pos"] for i in range(6)], dtype=float)
    g0 = float(obs0.get("gripper.pos", 0.0))

    if hasattr(teleop, "set_initial_targets"):
        teleop.set_initial_targets(q0, g0)
    # -------------------------------------------------------------------------------


    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
