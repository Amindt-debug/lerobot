from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@dataclass
class AccreaFollowerConfig:
    # REQUIRED by LeRobot base class
    id: str = "accrea_aria_01"
    calibration_dir: Optional[Path] = None

    # Robot connection
    robot_ip: str = "192.168.9.9"
    robot_port: int = 7777

    # REQUIRED by safe_robot_driver (same as your minimalExample)
    robot_urdf: Path = Path("/home/roboticslab/ALMCollection/RobotDescriptions/accrea_aria_description/urdf/aria_simplified.urdf")
    robot_srdf: Path = Path("/home/roboticslab/ALMCollection/RobotDescriptions/accrea_aria_description/srdf/aria_simplified.srdf")
    environment_urdf: Path = Path("/home/roboticslab/ALMCollection/roboticslab.urdf")
    robot_package_directory: Path = Path("/home/roboticslab/ALMCollection/RobotDescriptions")
    environment_package_directory: Path = Path("/home/roboticslab/ALMCollection/RobotDescriptions")

    # Cameras
    cameras: Dict[str, OpenCVCameraConfig] = field(default_factory=dict)

    # Safety
    max_delta_per_step_rad: float = 0.02
    require_user_confirmation: bool = True

    # ✅ NEW: teleop prefers joint speed streaming
    teleop_stream_speeds: bool = True