#!/usr/bin/env python3
import time
import re
import numpy as np
import xml.etree.ElementTree as ET

from lerobot.robots.accrea_follower import AccreaFollower, AccreaFollowerConfig
from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.model.kinematics import RobotKinematics  # uses placo


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def se3_translate(dx, dy, dz):
    T = np.eye(4, dtype=float)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    return T


def parse_arm_names_from_urdf(urdf_path: str):
    """
    Returns:
      joint_names_arm: list[str] (6 joints)
      ee_link_name: str  (child link of joint_6 / Joint_6)
    """
    root = ET.parse(urdf_path).getroot()

    # Collect joint names
    all_joint_names = [j.attrib.get("name", "") for j in root.findall("joint")]

    # Prefer patterns joint_1..6 or Joint_1..6
    def pick_arm(pattern):
        out = []
        for i in range(1, 7):
            cand = [n for n in all_joint_names if re.fullmatch(pattern.format(i=i), n)]
            if not cand:
                return None
            out.append(cand[0])
        return out

    arm = pick_arm(r"joint_{i}") or pick_arm(r"Joint_{i}")
    if arm is None:
        raise RuntimeError(
            f"Could not find 6 arm joints in URDF. Found joints: {all_joint_names[:20]}..."
        )

    # Find EE link as child of joint_6 / Joint_6
    joint6_name = arm[5]
    ee_link = None
    for j in root.findall("joint"):
        if j.attrib.get("name") == joint6_name:
            child = j.find("child")
            if child is not None:
                ee_link = child.attrib.get("link")
            break

    if ee_link is None:
        raise RuntimeError(f"Could not find child link of {joint6_name} in URDF.")

    return arm, ee_link


def main():
    # --- Robot config: NO cameras ---
    robot_cfg = AccreaFollowerConfig(
        id="accrea_aria_01",
        robot_ip="192.168.9.9",
        robot_port=7777,
        cameras={},
        max_delta_per_step_rad=0.12,
        require_user_confirmation=False,
    )

    robot = AccreaFollower(robot_cfg)
    print("✅ Connecting robot...")
    robot.connect()

    # Clear faults + freedrive off (if available)
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

    # --- URDF used for kinematics ---
    # IMPORTANT: keep this consistent with what safe_robot_driver uses.
    urdf_path = str(robot_cfg.robot_urdf)

    joint_names_arm, ee_link_name = parse_arm_names_from_urdf(urdf_path)
    print(f"✅ URDF: {urdf_path}")
    print(f"✅ Arm joints: {joint_names_arm}")
    print(f"✅ EE/Flange link: {ee_link_name}")

    # LeRobot RobotKinematics expects joint angles in DEGREES (internally converts to rad) :contentReference[oaicite:3]{index=3}
    kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=ee_link_name,
        joint_names=joint_names_arm,
    )

    # --- Gamepad ---
    # LeRobot GamepadController requires holding RB (button 5) to "intervene"/enable motion.
    gamepad = GamepadController(x_step_size=0.003, y_step_size=0.003, z_step_size=0.003, deadzone=0.10)
    gamepad.start()

    print("\n=== CONTROLS ===")
    print("Hold RB to enable motion (intervention).")
    print("Left stick: X/Y translation. Right stick vertical: Z translation.")
    print("Ctrl+C to exit.\n")

    # --- Recording ---
    record = []
    t0 = time.time()

    hz = 50.0
    dt = 1.0 / hz

    # Observation keys from your AccreaFollower are joint_1.pos ... joint_6.pos (LeRobot naming)
    obs_joint_keys = [f"joint_{i}.pos" for i in range(1, 7)]

    try:
        while True:
            gamepad.update()
            is_intervention = gamepad.should_intervene()

            end_status = gamepad.get_episode_end_status()
            if end_status in [TeleopEvents.SUCCESS, TeleopEvents.FAILURE, TeleopEvents.RERECORD_EPISODE]:
                print(f"Episode end status: {end_status}. Stopping.")
                break

            obs = robot.get_observation()

            # Current robot joints (radians)
            q_now_rad = np.array([obs[k] for k in obs_joint_keys], dtype=float)
            q_now_deg = np.rad2deg(q_now_rad)

            # FK flange pose
            T_now = kin.forward_kinematics(q_now_deg)

            dx, dy, dz = gamepad.get_deltas()

            # HOLD if not enabled or no stick motion
            if (not is_intervention) or (dx == 0.0 and dy == 0.0 and dz == 0.0):
                action_hold = {obs_joint_keys[i]: float(q_now_rad[i]) for i in range(6)}
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

            # Goal: translate flange in BASE frame
            T_goal = se3_translate(dx, dy, dz) @ T_now

            # IK (position-only)
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

            # Optional gripper via gamepad flags
            gr = float(obs.get("gripper.pos", 0.0))
            if getattr(gamepad, "open_gripper_command", False) and not getattr(gamepad, "close_gripper_command", False):
                gr = clamp(gr + 0.02, 0.0, 1.0)
            elif getattr(gamepad, "close_gripper_command", False) and not getattr(gamepad, "open_gripper_command", False):
                gr = clamp(gr - 0.02, 0.0, 1.0)

            # Send joint targets (robot wrapper will clamp + stream)
            action = {obs_joint_keys[i]: float(q_cmd_rad[i]) for i in range(6)}
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
