#!/usr/bin/env python3
import time
import numpy as np

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig
from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.model.kinematics import RobotKinematics


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def se3_translate(dx, dy, dz):
    """4x4 homogeneous translation matrix."""
    T = np.eye(4, dtype=float)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    return T


def main():
    robot_cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras={},                      # ✅ no cameras
        max_delta_per_step_rad=0.12,     # smoother streaming
        require_user_confirmation=False,
    )

    robot = AccreaFollower(robot_cfg)
    print("✅ Connecting robot...")
    robot.connect()

    # Optional: clear faults & disable freedrive (if supported)
    drv = getattr(robot, "_driver", None)
    if drv is not None:
        if hasattr(drv, "clear_faults"):
            try:
                drv.clear_faults()
                print("✅ clear_faults() done")
            except Exception as e:
                print(f"⚠️ clear_faults() failed: {e}")
        if hasattr(drv, "set_freedrive"):
            try:
                drv.set_freedrive(False)
                print("✅ set_freedrive(False) done")
            except Exception as e:
                print(f"⚠️ set_freedrive(False) failed: {e}")

    # ---- Kinematics (LeRobot/placo) ----
    # Your simplified URDF ends at link_6 -> use that as EE frame
    kin = RobotKinematics(
        urdf_path=str(robot_cfg.robot_urdf),
        target_frame_name="link_6",   # ✅ matches aria_simplified.urdf end link
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    )

    # ---- Gamepad ----
    # Hold RB (button 5) to enable motion in LeRobot’s controller implementation.
    gamepad = GamepadController(x_step_size=0.003, y_step_size=0.003, z_step_size=0.003, deadzone=0.10)
    gamepad.start()

    print("\n=== CONTROLS ===")
    print("Hold RB to enable motion (intervention).")
    print("Left stick: X/Y translation. Right stick vertical: Z translation.")
    print("Ctrl+C to exit.\n")

    record = []
    t0 = time.time()

    hz = 50.0
    dt = 1.0 / hz

    try:
        while True:
            gamepad.update()
            is_intervention = gamepad.should_intervene()

            end_status = gamepad.get_episode_end_status()
            if end_status in [TeleopEvents.SUCCESS, TeleopEvents.FAILURE, TeleopEvents.RERECORD_EPISODE]:
                print(f"Episode end status: {end_status}. Stopping.")
                break

            obs = robot.get_observation()
            joint_keys = [f"joint_{i}.pos" for i in range(1, 7)]
            q_now_rad = np.array([obs[k] for k in joint_keys], dtype=float)

            # LeRobot RobotKinematics expects degrees input/output :contentReference[oaicite:2]{index=2}
            q_now_deg = np.rad2deg(q_now_rad)

            T_now = kin.forward_kinematics(q_now_deg)

            dx, dy, dz = gamepad.get_deltas()

            # HOLD if not intervening or no stick motion
            if (not is_intervention) or (dx == 0.0 and dy == 0.0 and dz == 0.0):
                action_hold = {joint_keys[i]: float(q_now_rad[i]) for i in range(6)}
                action_hold["gripper.pos"] = float(obs.get("gripper.pos", 0.0))
                robot.send_action(action_hold)

                record.append({
                    "t": time.time() - t0,
                    "q_now_rad": q_now_rad.copy(),
                    "q_cmd_rad": q_now_rad.copy(),
                    "T_now": T_now.copy(),
                    "T_goal": T_now.copy(),
                    "dx": dx, "dy": dy, "dz": dz,
                    "intervene": bool(is_intervention),
                    "ik_success": True,
                })
                time.sleep(dt)
                continue

            # Goal pose: translate in base frame
            T_goal = se3_translate(dx, dy, dz) @ T_now

            # IK (position-only): orientation_weight=0.0 :contentReference[oaicite:3]{index=3}
            try:
                q_cmd_deg = kin.inverse_kinematics(
                    current_joint_pos=q_now_deg,
                    desired_ee_pose=T_goal,
                    position_weight=1.0,
                    orientation_weight=0.0,
                )
                ik_success = True
            except Exception as e:
                print(f"⚠️ IK exception (holding): {e}")
                q_cmd_deg = q_now_deg.copy()
                ik_success = False

            q_cmd_rad = np.deg2rad(q_cmd_deg)

            # Gripper (optional, if gamepad flags exist)
            gr = float(obs.get("gripper.pos", 0.0))
            if getattr(gamepad, "open_gripper_command", False) and not getattr(gamepad, "close_gripper_command", False):
                gr = clamp(gr + 0.02, 0.0, 1.0)
            elif getattr(gamepad, "close_gripper_command", False) and not getattr(gamepad, "open_gripper_command", False):
                gr = clamp(gr - 0.02, 0.0, 1.0)

            action = {joint_keys[i]: float(q_cmd_rad[i]) for i in range(6)}
            action["gripper.pos"] = float(gr)
            robot.send_action(action)

            record.append({
                "t": time.time() - t0,
                "q_now_rad": q_now_rad.copy(),
                "q_cmd_rad": q_cmd_rad.copy(),
                "T_now": T_now.copy(),
                "T_goal": T_goal.copy(),
                "dx": dx, "dy": dy, "dz": dz,
                "intervene": bool(is_intervention),
                "ik_success": bool(ik_success),
            })

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt -> exiting.")

    finally:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = f"accrea_ee_teleop_record_{ts}.npz"
        if record:
            np.savez(
                out,
                t=np.array([r["t"] for r in record], dtype=float),
                q_now_rad=np.stack([r["q_now_rad"] for r in record], axis=0),
                q_cmd_rad=np.stack([r["q_cmd_rad"] for r in record], axis=0),
                T_now=np.stack([r["T_now"] for r in record], axis=0),
                T_goal=np.stack([r["T_goal"] for r in record], axis=0),
                intervene=np.array([r["intervene"] for r in record], dtype=bool),
                ik_success=np.array([r["ik_success"] for r in record], dtype=bool),
            )
            print(f"✅ Saved recording to: {out}")
        else:
            print("⚠️ No samples recorded.")

        try:
            gamepad.stop()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
