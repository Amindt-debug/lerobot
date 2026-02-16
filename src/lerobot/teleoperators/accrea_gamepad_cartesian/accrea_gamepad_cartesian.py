#!/usr/bin/env python3
import os
from typing import Dict, Any, Optional

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.robots.accrea_follower import AccreaFollower  # only for type hints (optional)
from .configuration_accrea_gamepad_cartesian import AccreaGamepadCartesianTeleopConfig
from ..utils import TeleopEvents


def apply_deadzone(x: float, dz: float) -> float:
    return 0.0 if abs(x) < dz else float(x)


class AccreaGamepadCartesianTeleop(Teleoperator):
    """
    Gamepad -> Cartesian (TCP) jogging -> Jacobian DLS -> absolute joint targets.

    Outputs absolute targets:
      - joint_0.pos ... joint_5.pos
      - gripper.pos

    Start with 3D translation + yaw (about base Z).
    """
    config_class = AccreaGamepadCartesianTeleopConfig
    name = "accrea_gamepad_cartesian"

    def __init__(self, config: AccreaGamepadCartesianTeleopConfig):
        super().__init__(config)
        self.config = config

        self._dt = 1.0 / float(config.hz)

        # pygame joystick
        self._pygame = None
        self._js = None

        # internal targets
        self._q_target: Optional[np.ndarray] = None
        self._g_target: float = 0.0

        # pinocchio model/data
        self._pin = None
        self._model = None
        self._data = None
        self._ee_frame_id = None

        self._exit_requested = False

    # ---------- lifecycle ----------
    def connect(self) -> None:
        import pygame  # local import so package is optional at import time
        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= self.config.joystick_index:
            raise RuntimeError(
                f"No joystick at index {self.config.joystick_index}. "
                f"Detected {pygame.joystick.get_count()} joystick(s)."
            )
        self._js = pygame.joystick.Joystick(self.config.joystick_index)
        self._js.init()

        # Pinocchio import
        try:
            import pinocchio as pin
        except Exception as e:
            raise RuntimeError(
                "Pinocchio is required for accrea_gamepad_cartesian teleop but is not available "
                "in this environment. Install it or we’ll switch backend."
            ) from e

        self._pin = pin

        # Load URDF from robot config via env var (set by runner) OR default path.
        # In LeRobot, best practice is: we pass the robot's urdf path into the teleop config or env.
        urdf_path = os.environ.get("ACCREA_URDF_PATH", "")
        if not urdf_path:
            raise RuntimeError(
                "ACCREA_URDF_PATH env var not set. "
                "Set it to the URDF file path so the teleop can build kinematics."
            )

        # Build model
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()

        # Find ee frame
        ee = self.config.ee_link
        if not self._model.existFrame(ee):
            raise RuntimeError(f"EE frame '{ee}' not found in URDF model frames. Check urdf and ee_link.")
        self._ee_frame_id = self._model.getFrameId(ee)

    def disconnect(self) -> None:
        try:
            if self._js is not None:
                self._js.quit()
        except Exception:
            pass
        try:
            if self._pygame is not None:
                self._pygame.joystick.quit()
                self._pygame.quit()
        except Exception:
            pass

    # ---------- required by Teleoperator ----------
    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        obs contains joint positions from the robot adapter.
        We compute q_target incrementally using Jacobian DLS.
        """
        self._pygame.event.pump()

        # exit
        if self._js.get_button(self.config.exit_btn):
            self._exit_requested = True
            return {}

        # read current robot joints from obs
        q_now = []
        for i in range(6):
            q_now.append(float(obs[f"joint_{i}.pos"]))
        q_now = np.asarray(q_now, dtype=float)

        # initialize targets to current
        if self._q_target is None:
            self._q_target = q_now.copy()

        # gripper read/initialize
        g_now = float(obs.get("gripper.pos", 0.0))
        if self._g_target is None:
            self._g_target = g_now

        # deadman
        if not self._js.get_button(self.config.deadman_btn):
            return self._pack_action(self._q_target, self._g_target)

        # gamepad axes
        lx = apply_deadzone(self._js.get_axis(self.config.lx_axis), self.config.deadzone)
        ly = apply_deadzone(self._js.get_axis(self.config.ly_axis), self.config.deadzone)
        rx = apply_deadzone(self._js.get_axis(self.config.rx_axis), self.config.deadzone)
        # ry available if you want (e.g., z)
        ry = apply_deadzone(self._js.get_axis(self.config.ry_axis), self.config.deadzone)

        # hats (dpad) for z
        hatx, haty = 0, 0
        if self._js.get_numhats() > 0:
            hatx, haty = self._js.get_hat(self.config.hat_idx)

        # --- Desired Cartesian velocity (base frame) ---
        # map:
        #   left stick: x/y
        #   dpad up/down: z
        #   right stick x: yaw
        vx = self.config.lin_vel_mps * lx
        vy = self.config.lin_vel_mps * (-ly)   # invert so up is +y (feel free to flip)
        vz = self.config.lin_vel_mps * float(haty)

        wz = self.config.yaw_vel_rps * rx

        # 6D twist in base frame: [vx,vy,vz, wx,wy,wz]
        v = np.array([vx, vy, vz, 0.0, 0.0, wz], dtype=float)

        # compute qdot via Jacobian DLS at current q_now
        qdot = self._jacobian_dls(q_now, v)

        # integrate target and clip speed
        qdot = np.clip(qdot, -self.config.max_qd_rad_s, self.config.max_qd_rad_s)
        self._q_target = self._q_target + qdot * self._dt

        # gripper from triggers
        lt = (self._js.get_axis(self.config.lt_axis) + 1.0) / 2.0  # [0,1]
        rt = (self._js.get_axis(self.config.rt_axis) + 1.0) / 2.0
        # open = LT, close = RT (swap if you prefer)
        self._g_target = float(np.clip(self._g_target + (lt - rt) * self.config.gripper_rate * self._dt, 0.0, 1.0))

        return self._pack_action(self._q_target, self._g_target)

    def get_events(self) -> TeleopEvents:
        events = TeleopEvents()
        if self._exit_requested:
            events.should_exit = True
        return events

    # ---------- helpers ----------
    def _pack_action(self, q: np.ndarray, g: float) -> Dict[str, Any]:
        out = {f"joint_{i}.pos": float(q[i]) for i in range(6)}
        out["gripper.pos"] = float(g)
        return out

    def _jacobian_dls(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Damped least squares:
          qdot = J^T (J J^T + λ^2 I)^-1 v
        """
        pin = self._pin
        model = self._model
        data = self._data
        ee_id = self._ee_frame_id

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # 6xN Jacobian in WORLD frame
        J = pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.WORLD)

        lam = float(self.config.damping_lambda)
        JJt = J @ J.T
        A = JJt + (lam * lam) * np.eye(6)
        qdot = J.T @ np.linalg.solve(A, v)
        return qdot
