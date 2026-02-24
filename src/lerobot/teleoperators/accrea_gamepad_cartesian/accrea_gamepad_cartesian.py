#!/usr/bin/env python3
import os
from typing import Dict, Any, Optional

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
# from lerobot.robots.accrea_follower import AccreaFollower  # only for type hints (optional)
from .configuration_accrea_gamepad_cartesian import AccreaGamepadCartesianTeleopConfig
from ..utils import TeleopEvents

""""" command line example for teleoperating  
lerobot-teleoperate \
  --robot.type=accrea_follower \
  --robot.id=accrea_aria_01 \
  --robot.robot_ip=192.168.9.9 \
  --robot.robot_port=7777 \
  --robot.max_delta_per_step_rad=0.02 \
  --teleop.type=accrea_gamepad_cartesian \
  --teleop.id=xbox_0 \
  --fps=30
"""""

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

        # internal targets (kept for compatibility with runner/scripts, but not used for motion anymore)
        self._q_target: Optional[np.ndarray] = None
        self._g_target: Optional[float] = None

        # pinocchio model/data
        self._pin = None
        self._model = None
        self._data = None
        self._ee_frame_id = None

        self._exit_requested = False

    # ---------- Teleoperator interface (match joint teleop pattern) ----------
    @property
    def action_features(self) -> dict:
        # absolute targets: 6 joints + 1 gripper
        return {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "joint_0.pos": 0,
                "joint_1.pos": 1,
                "joint_2.pos": 2,
                "joint_3.pos": 3,
                "joint_4.pos": 4,
                "joint_5.pos": 5,
                "gripper.pos": 6,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def calibrate(self) -> None:
        # No calibration needed for gamepad teleop
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        # No extra configuration required
        pass

    def send_feedback(self, feedback: dict) -> None:
        # No haptics/feedback used for now
        pass

    @property
    def is_connected(self) -> bool:
        return self._js is not None

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

        self._js = None
        self._pygame = None

    # ---------- required by Teleoperator ----------
    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        obs contains joint positions from the robot adapter.
        We compute q_target anchored to q_now each loop (same philosophy as joint teleop).
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

        # gripper current
        g_now = float(obs.get("gripper.pos", 0.0))

        # initialize stored targets (kept for compatibility; does not drive motion)
        if self._q_target is None:
            self._q_target = q_now.copy()
        if self._g_target is None:
            self._g_target = g_now

        # deadman: NO motion when not held (joint teleop style)
        if not self._js.get_button(self.config.deadman_btn):
            return {}

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
        qdot = np.clip(qdot, -self.config.max_qd_rad_s, self.config.max_qd_rad_s)

        # IMPORTANT: anchor to current q_now each loop (joint teleop style)
        q_target = q_now + qdot * self._dt

        # gripper from triggers (anchored to g_now each loop)
        lt = (self._js.get_axis(self.config.lt_axis) + 1.0) / 2.0  # [0,1]
        rt = (self._js.get_axis(self.config.rt_axis) + 1.0) / 2.0
        # open = LT, close = RT (swap if you prefer)
        g_target = float(np.clip(g_now + (lt - rt) * self.config.gripper_rate * self._dt, 0.0, 1.0))

        # also update stored targets (optional; keeps state consistent for external callers)
        self._q_target = q_target.copy()
        self._g_target = g_target

        return self._pack_action(q_target, g_target)

    # match joint teleop naming for events (processor expects this)
    def get_teleop_events(self) -> dict[str, Any]:
        # minimal: allow pipeline to stop if START pressed
        exit_requested = getattr(self, "_exit_requested", False)
        return {
            TeleopEvents.IS_INTERVENTION: True,
            TeleopEvents.TERMINATE_EPISODE: exit_requested,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    # Keep an alias for older callers (harmless)
    def get_events(self) -> TeleopEvents:
        events = TeleopEvents()
        if self._exit_requested:
            events.should_exit = True
        return events

    # same pattern as joint teleop: allow runner to initialize targets explicitly
    def set_initial_targets(self, q: np.ndarray, g: float) -> None:
        """Call once after connecting robot: sets initial joint/gripper targets to current state."""
        self._q_target = np.array(q, dtype=float).copy()
        self._g_target = float(g)

    def update_from_robot_state(self, q: np.ndarray, g: float) -> None:
        """
        Keep internal targets aligned with the real robot state.
        This prevents 'target drift' that can cause the robot to keep moving
        after the joystick returns to center.
        """
        self._q_target = np.array(q, dtype=float).copy()
        self._g_target = float(g)

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
        J = pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        lam = float(self.config.damping_lambda)
        JJt = J @ J.T
        A = JJt + (lam * lam) * np.eye(6)
        qdot = J.T @ np.linalg.solve(A, v)
        return qdot