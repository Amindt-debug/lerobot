#!/usr/bin/env python
import sys
from typing import Any

import numpy as np

from lerobot.utils.decorators import check_if_not_connected
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_accrea_gamepad_joints import AccreaGamepadJointsTeleopConfig


def apply_deadzone(x: float, dz: float) -> float:
    return 0.0 if abs(x) < dz else float(x)


class AccreaGamepadJointsTeleop(Teleoperator):
    """
    Teleoperator for Accrea joint-space jogging using an Xbox controller.

    Outputs absolute joint/gripper targets:
      - joint_0.pos ... joint_5.pos
      - gripper.pos
    """
    config_class = AccreaGamepadJointsTeleopConfig
    name = "accrea_gamepad_joints"

    def __init__(self, config: AccreaGamepadJointsTeleopConfig):
        super().__init__(config)
        self.config = config

        self._pygame = None
        self._js = None
        self._use_hat = False

        self._dt = 1.0 / float(config.hz)

        # Internal targets (start as None; first get_action() must be given robot state externally)
        self._q_target = None
        self._g_target = None

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

    def connect(self) -> None:
        # match GamepadTeleop behavior: pygame on non-mac
        if sys.platform == "darwin":
            raise RuntimeError("This teleop uses pygame; macOS may need HID mode.")

        import pygame
        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Check connection and /dev/input/js0.")

        js = pygame.joystick.Joystick(0)
        js.init()

        self._js = js
        self._use_hat = js.get_numhats() > 0

        # Print once for visibility
        print("Gamepad:", js.get_name())
        print("Axes:", js.get_numaxes(), "Buttons:", js.get_numbuttons(), "Hats:", js.get_numhats())
        print("HOLD RB to move. Press START to quit.")

    def _trigger_to_01(self, v: float) -> float:
        # support [-1,1] or [0,1]
        if v < -0.2:
            v = (v + 1.0) / 2.0
        return float(np.clip(v, 0.0, 1.0))

    @check_if_not_connected
    def get_action(self) -> dict:
        """
        Returns an action dict. Requires that the caller sets initial targets once
        by calling set_initial_targets(q, g) after robot connection.
        """
        if self._q_target is None or self._g_target is None:
            # teleop must be initialized with robot state first
            return {
                "joint_0.pos": 0.0,
                "joint_1.pos": 0.0,
                "joint_2.pos": 0.0,
                "joint_3.pos": 0.0,
                "joint_4.pos": 0.0,
                "joint_5.pos": 0.0,
                "gripper.pos": 0.0,
            }

        cfg = self.config
        js = self._js
        pg = self._pygame

        pg.event.pump()

        # Exit button
        if js.get_button(cfg.exit_btn):
            self._exit_requested = True

        deadman = bool(js.get_button(cfg.deadman_btn))
        if not deadman:
            # no motion: just return current targets
            return {f"joint_{i}.pos": float(self._q_target[i]) for i in range(6)} | {"gripper.pos": float(self._g_target)}

        # Read sticks
        lx = apply_deadzone(js.get_axis(cfg.left_stick_x_axis), cfg.deadzone)
        ly = apply_deadzone(js.get_axis(cfg.left_stick_y_axis), cfg.deadzone)
        rx = apply_deadzone(js.get_axis(cfg.right_stick_x_axis), cfg.deadzone) if js.get_numaxes() > cfg.right_stick_x_axis else 0.0
        ry = apply_deadzone(js.get_axis(cfg.right_stick_y_axis), cfg.deadzone) if js.get_numaxes() > cfg.right_stick_y_axis else 0.0

        if cfg.invert_lx: lx = -lx
        if cfg.invert_ly: ly = -ly
        if cfg.invert_rx: rx = -rx
        if cfg.invert_ry: ry = -ry

        # D-pad
        if self._use_hat:
            hatx, haty = js.get_hat(cfg.dpad_hat_index)
        else:
            hatx = int(np.sign(js.get_axis(cfg.dpad_fallback_x_axis))) if js.get_numaxes() > cfg.dpad_fallback_x_axis else 0
            haty = int(np.sign(js.get_axis(cfg.dpad_fallback_y_axis))) if js.get_numaxes() > cfg.dpad_fallback_y_axis else 0

        if cfg.invert_hatx: hatx = -hatx
        if cfg.invert_haty: haty = -haty

        # Map to joint velocities (KEEP SAME SIGNS AS YOUR SCRIPT)
        joint_rate = np.array(cfg.joint_rate, dtype=float)
        qd = np.zeros(6, dtype=float)
        qd[0] = -lx * joint_rate[0]
        qd[1] = -ly * joint_rate[1]
        qd[2] = -rx * joint_rate[2]
        qd[3] = -ry * joint_rate[3]
        qd[4] = float(haty) * joint_rate[4]
        qd[5] = float(hatx) * joint_rate[5]

        self._q_target = self._q_target + qd * self._dt

        # Gripper LT/RT
        trig_close = 0.0
        trig_open = 0.0
        if js.get_numaxes() > cfg.lt_axis:
            a_lt = js.get_axis(cfg.lt_axis)
            if js.get_numaxes() > cfg.rt_axis:
                lt = self._trigger_to_01(a_lt)
                rt = self._trigger_to_01(js.get_axis(cfg.rt_axis))
                trig_close = lt
                trig_open = rt
            elif cfg.combined_triggers_on_lt_axis:
                trig_close = float(np.clip(-a_lt, 0.0, 1.0))
                trig_open = float(np.clip(a_lt, 0.0, 1.0))

        self._g_target = float(np.clip(self._g_target + (trig_open - trig_close) * cfg.gripper_rate * self._dt, 0.0, 1.0))

        return {f"joint_{i}.pos": float(self._q_target[i]) for i in range(6)} | {"gripper.pos": float(self._g_target)}

    def get_teleop_events(self) -> dict[str, Any]:
        # minimal: allow pipeline to stop if START pressed
        exit_requested = getattr(self, "_exit_requested", False)
        return {
            TeleopEvents.IS_INTERVENTION: True,  # deadman implies intervention; good enough for now
            TeleopEvents.TERMINATE_EPISODE: exit_requested,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

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


    def disconnect(self) -> None:
        if self._pygame is not None:
            try:
                if self._js is not None:
                    self._js.quit()
            except Exception:
                pass
            try:
                self._pygame.joystick.quit()
                self._pygame.quit()
            except Exception:
                pass
        self._js = None
        self._pygame = None

    @property
    def is_connected(self) -> bool:
        return self._js is not None
