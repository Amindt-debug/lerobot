import logging
import time
import numpy as np
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_accrea_follower import AccreaFollowerConfig

import safe_robot_driver as srd

logger = logging.getLogger(__name__)


class AccreaFollower(Robot):
    """
    Accrea Aria robot adapter for LeRobot using safe_robot_driver.
    Observation: joints + cameras
    Action: joint position targets (internally executed via joint position command OR speed fallback).
    """

    config_class = AccreaFollowerConfig
    name = "accrea_follower"

    def __init__(self, config: AccreaFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._driver = None
        self._dof = None
        self._first_send_action = True

    # ---------------- Features ----------------
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        dof = self._dof if self._dof is not None else 6
        joint_ft = {f"joint_{i}.pos": float for i in range(dof)}
        gripper_ft = {"gripper.pos": float}
        cam_ft = {
            k: (self.config.cameras[k].height, self.config.cameras[k].width, 3)
            for k in self.cameras
        }
        return {**joint_ft, **gripper_ft, **cam_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        dof = self._dof if self._dof is not None else 6
        ft = {f"joint_{i}.pos": float for i in range(dof)}
        ft["gripper.pos"] = float
        return ft


    # ---------------- Connection ----------------
    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values() if cam is not None) if self.cameras else True
        return (self._driver is not None) and cams_ok

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        cfg = srd.SafeRobotDriverConfig()

        # --- Network ---
        cfg.robot_ip = self.config.robot_ip
        cfg.robot_port = self.config.robot_port

        # --- URDF / SRDF / Environment ---
        cfg.robot_urdf = str(self.config.robot_urdf)
        cfg.robot_srdf = str(self.config.robot_srdf)
        cfg.environment_urdf = str(self.config.environment_urdf)

        cfg.robot_package_directory = str(self.config.robot_package_directory)
        cfg.environment_package_directory = str(self.config.environment_package_directory)

        # Optional fields only if exist
        if hasattr(cfg, "robot_port_real_time"):
            cfg.robot_port_real_time = 0
        if hasattr(cfg, "command_timeout"):
            cfg.command_timeout = 0.5
        if hasattr(cfg, "robot_control_dt"):
            cfg.robot_control_dt = 0.008

        robot_type = srd.RobotType.ACCREA_ARIA

        self._driver = srd.create_robot_driver(robot_type, cfg)
        self._driver.connect()
        self._driver.start()

        self._dof = int(self._driver.dof())

        # Cameras
        for name, cam in self.cameras.items():
            try:
                cam.connect()
                print(f"✅ Camera connected: {name}")
            except Exception as e:
                print(f"⚠️ Camera {name} failed: {e}")
                print("⚠️ Continuing without this camera.")
                self.cameras[name] = None



        self.configure()

        logger.info(f"{self} connected (dof={self._dof}).")

        # Helpful debug
        methods = [m for m in dir(self._driver) if ("joint" in m) or ("move" in m)]
        logger.info(f"[AccreaFollower] driver type: {type(self._driver)}")
        logger.info(f"[AccreaFollower] driver methods (joint/move): {methods}")

    # ---------------- Required by base Robot ----------------
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(f"{self} calibrate(): skipped (not required).")

    def configure(self) -> None:
        logger.info(f"{self} configure(): skipped (not required).")

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam is None:
                continue
            cam.disconnect()

        try:
            self._driver.disconnect()
        except Exception as e:
            logger.warning(f"{self} disconnect(): driver disconnect raised: {e}")

        self._driver = None
        logger.info(f"{self} disconnected.")

    # ---------------- IO ----------------
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}

        q = np.array(self._driver.joint_positions(), dtype=np.float32)
        if hasattr(self._driver, "gripper_position"):
            obs["gripper.pos"] = float(self._driver.gripper_position())
        else:
            obs["gripper.pos"] = 0.0

        for i in range(len(q)):
            obs[f"joint_{i}.pos"] = float(q[i])

        # Read cameras
        for cam_key, cam in self.cameras.items():
            if cam_key is None:
                continue
            if cam is None:
                continue
            obs[cam_key] = cam.async_read()

        # ✅ Pi0 expects BOTH wrist cameras. If you have only one wrist cam,
        # we duplicate it so inference does not break.
        if "left_wrist_0_rgb" in obs and "right_wrist_0_rgb" not in obs:
            obs["right_wrist_0_rgb"] = obs["left_wrist_0_rgb"]

        if "right_wrist_0_rgb" in obs and "left_wrist_0_rgb" not in obs:
            obs["left_wrist_0_rgb"] = obs["right_wrist_0_rgb"]


        return obs

    # ---------------- Motion helpers ----------------
    def _send_joint_positions_best_effort(self, q_target: np.ndarray) -> None:
        """
        Teleop-safe execution:
        Always stream joint speeds toward the target (servo behavior),
        because move_to_joint_positions() is not meant for high-rate teleop updates.
        """
        # --- Prefer speed streaming (teleop servo) ---
        if hasattr(self._driver, "command_joint_speeds"):
            q_now = np.array(self._driver.joint_positions(), dtype=np.float32)
            err = q_target - q_now

            kp = 6.0             # stronger response
            max_speed = 0.8      # rad/s
            tol = 0.0005         # rad

            if float(np.max(np.abs(err))) < tol:
                qd = np.zeros_like(err)
            else:
                qd = kp * err
                qd = np.clip(qd, -max_speed, max_speed)

            self._driver.command_joint_speeds([float(x) for x in qd])
            return

        # --- Fallback to position methods if speeds not available ---
        q_list = [float(x) for x in q_target]

        if hasattr(self._driver, "move_to_joint_positions"):
            self._driver.move_to_joint_positions(q_list)
            return

        if hasattr(self._driver, "command_joint_positions"):
            self._driver.command_joint_positions(q_list)
            return

        if hasattr(self._driver, "set_joint_positions"):
            self._driver.set_joint_positions(q_list)
            return

        candidates = [m for m in dir(self._driver) if ("joint" in m) or ("move" in m)]
        raise AttributeError(
            "No supported motion interface found.\n"
            f"Available related methods: {candidates}"
    )




    def _move_via_joint_speeds(self, q_target: np.ndarray) -> None:
        """
        Safe fallback: drive toward q_target using command_joint_speeds().
        This is slow and safe, but reliable when move_to_joint_positions is not exposed.
        """

        # Safe parameters (tunable)
        kp = 2.0                 # proportional gain
        max_speed = 0.25         # rad/s clamp (safe)
        dt = 0.02                # 50 Hz loop
        timeout = 0.6            # seconds
        tol = 0.002              # rad

        t0 = time.time()

        while True:
            q_now = np.array(self._driver.joint_positions(), dtype=np.float32)
            err = q_target - q_now

            if float(np.max(np.abs(err))) < tol:
                break

            qd = kp * err
            qd = np.clip(qd, -max_speed, max_speed)

            self._driver.command_joint_speeds([float(x) for x in qd])

            time.sleep(dt)

            if (time.time() - t0) > timeout:
                break

        # Stop speeds (important!)
        self._driver.command_joint_speeds([0.0] * int(self._dof))

    # ---------------- Main action ----------------
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._first_send_action and getattr(self.config, "require_user_confirmation", True):
            try:
                input(
                    "\n[AccreaFollower] READY TO MOVE THE REAL ROBOT.\n"
                    "Make sure workspace is clear and E-Stop is reachable.\n"
                    "Press ENTER to allow motion...\n"
                )
            except Exception as e:
                logger.warning(f"[AccreaFollower] user confirmation input failed: {e}")
            self._first_send_action = False

        if "gripper.pos" in action and hasattr(self._driver, "move_to_gripper_position"):
            try:
                self._driver.move_to_gripper_position(float(action["gripper.pos"]))
            except Exception as e:
                logger.warning(f"[AccreaFollower] gripper move failed: {e}")
    

        dof = int(self._dof)

        q_target = np.array(
            [float(action[f"joint_{i}.pos"]) for i in range(dof)],
            dtype=np.float32
        )

        q_now = np.array(self._driver.joint_positions(), dtype=np.float32)
        dq = q_target - q_now

        max_dq = float(self.config.max_delta_per_step_rad)
        dq = np.clip(dq, -max_dq, max_dq)
        q_safe = q_now + dq

        # ✅ Best-effort motion command
        self._send_joint_positions_best_effort(q_safe)

        out = {f"joint_{i}.pos": float(q_safe[i]) for i in range(dof)}
        out["gripper.pos"] = float(action.get("gripper.pos", 0.0))
        return out

