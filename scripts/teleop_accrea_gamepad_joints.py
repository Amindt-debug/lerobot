#!/usr/bin/env python3
import time
import numpy as np

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig

# We reuse the existing pygame-based controller style used by LeRobot
import pygame


def apply_deadzone(x: float, dz: float = 0.12) -> float:
    return 0.0 if abs(x) < dz else x


def main():
    # -----------------------------
    # Robot setup
    # -----------------------------
    cfg = AccreaFollowerConfig(
        require_user_confirmation=True,   # you already implemented this safety gate
        max_delta_per_step_rad=0.10,      # keep conservative for first tests
    )
    robot = AccreaFollower(cfg)
    robot.connect()

    # Read initial state
    obs = robot.get_observation()
    dof = 6
    q = np.array([obs[f"joint_{i}.pos"] for i in range(dof)], dtype=float)
    g = float(obs.get("gripper.pos", 0.0))

    print("\nConnected. HOLD RB to move. Press START to quit.\n")

    # -----------------------------
    # Gamepad init
    # -----------------------------
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected. Check /dev/input/js0 and connection.")

    js = pygame.joystick.Joystick(0)
    js.init()
    print("Gamepad:", js.get_name())
    print("Axes:", js.get_numaxes(), "Buttons:", js.get_numbuttons(), "Hats:", js.get_numhats())

    # -----------------------------
    # Control parameters
    # -----------------------------
    hz = 30.0
    dt = 1.0 / hz

    # joint jogging scale (radians per second at full stick deflection)
    joint_rate = np.array([0.6, 0.6, 0.6, 0.8, 0.8, 1.0], dtype=float)

    # gripper jogging scale (units depend on driver; keep small)
    gripper_rate = 0.8

    # Button mapping (common Xbox on Linux):
    # RB is usually button 5 in pygame; jstest shows BtnTR exists.
    DEADMAN_BTN = 5
    START_BTN = 7  # jstest shows BtnStart exists; pygame often uses 7

    # Triggers: often axes (Z/Rz) on Linux; sometimes buttons.
    # We'll support both patterns safely.
    # D-pad: either hat (preferred) or axes (Hat0X/Hat0Y appear in jstest as axes 6,7).
    use_hat = js.get_numhats() > 0

    try:
        last_print = time.time()

        while True:
            t0 = time.time()
            pygame.event.pump()

            # Exit
            if js.get_button(START_BTN):
                print("START pressed -> exiting.")
                break

            deadman = bool(js.get_button(DEADMAN_BTN))

            # Always refresh current joint state (so we don't drift)
            obs = robot.get_observation()
            q = np.array([obs[f"joint_{i}.pos"] for i in range(dof)], dtype=float)
            g = float(obs.get("gripper.pos", g))

            if not deadman:
                # No motion if deadman not held
                # (still allow reading obs)
                time.sleep(max(0.0, dt - (time.time() - t0)))
                continue

            # -----------------------------
            # Read axes
            # -----------------------------
            # Left stick: axes 0,1
            lx = apply_deadzone(js.get_axis(0))
            ly = apply_deadzone(js.get_axis(1))

            # Right stick: axes 3,4 (sometimes 2,3). We'll start with 3,4.
            rx = apply_deadzone(js.get_axis(3))
            ry = apply_deadzone(js.get_axis(4)) if js.get_numaxes() > 4 else 0.0

            # D-pad for last two joints:
            if use_hat:
                hatx, haty = js.get_hat(0)  # (-1,0,1)
            else:
                # fallback: jstest shows Hat0X/Hat0Y as axes 6,7
                hatx = int(np.sign(js.get_axis(6))) if js.get_numaxes() > 6 else 0
                haty = int(np.sign(js.get_axis(7))) if js.get_numaxes() > 7 else 0

            # -----------------------------
            # Map to joint velocities (rad/s)
            # -----------------------------
            qd = np.zeros(dof, dtype=float)

            # Example mapping:
            # joints 0-1 from left stick
            qd[0] = -lx * joint_rate[0]
            qd[1] = -ly * joint_rate[1]

            # joints 2-3 from right stick
            qd[2] = -rx * joint_rate[2]
            qd[3] = -ry * joint_rate[3]

            # joints 4-5 from D-pad
            # haty up/down, hatx left/right
            qd[4] = float(haty) * joint_rate[4]
            qd[5] = float(hatx) * joint_rate[5]

            # Integrate to get target positions
            q_target = q + qd * dt

            # -----------------------------
            # Gripper (LT/RT triggers)
            # -----------------------------
            # LT -> CLOSE, RT -> OPEN
            #
            # Common mappings:
            #   - Separate axes: LT=2, RT=5
            #   - Combined on axis 2: LT negative, RT positive
            trig_close = 0.0
            trig_open = 0.0

            if js.get_numaxes() > 2:
                a2 = js.get_axis(2)

                if js.get_numaxes() > 5:
                    # Separate RT axis exists
                    lt = a2
                    rt = js.get_axis(5)

                    # normalize each to [0,1] (supports [-1,1] or [0,1])
                    lt = (lt + 1.0) / 2.0 if lt < -0.2 else lt
                    rt = (rt + 1.0) / 2.0 if rt < -0.2 else rt

                    trig_close = float(np.clip(lt, 0.0, 1.0))
                    trig_open = float(np.clip(rt, 0.0, 1.0))
                else:
                    # Combined triggers on axis 2: LT is negative, RT is positive
                    trig_close = float(np.clip(-a2, 0.0, 1.0))
                    trig_open = float(np.clip(a2, 0.0, 1.0))

            g_target = g + (trig_open - trig_close) * gripper_rate * dt
            g_target = float(np.clip(g_target, 0.0, 1.0))

            # -----------------------------
            # Send action (absolute targets)
            # -----------------------------
            action = {f"joint_{i}.pos": float(q_target[i]) for i in range(dof)}
            action["gripper.pos"] = float(g_target)

            robot.send_action(action)

            # Small status print
            if time.time() - last_print > 1.0:
                print(
                    "deadman=ON  q:", np.round(q, 3),
                    "g:", round(g, 3),
                    "LT(close):", round(trig_close, 2),
                    "RT(open):", round(trig_open, 2),
                )
                last_print = time.time()

            # loop timing
            time.sleep(max(0.0, dt - (time.time() - t0)))

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        pygame.quit()
        print("Disconnected cleanly.")


if __name__ == "__main__":
    main()
