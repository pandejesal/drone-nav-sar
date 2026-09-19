#!/usr/bin/env python3
"""Mock YOLOv8 victim detector for DroneNav-SAR (Sprint 7).

SAR-only: detects victims (people needing med-kit) in camera FOV.
Mock returns GT bbox + noise when victim in FOV; real uses YOLOv8n ONNX.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """Single detection result."""
    class_name: str  # "victim"
    bbox: Tuple[float, float, float, float]  # x, y, w, h in normalized [0,1]
    confidence: float  # 0.0-1.0
    position_3d: Optional[Tuple[float, float, float]] = None  # world coords if known


class MockYOLODetector:
    """Mock victim detector for simulation.

    In mock mode: returns ground-truth victim bbox with noise when victim
    is within camera FOV and range. In real mode: would load YOLOv8n ONNX.
    """

    def __init__(
        self,
        camera_fov_deg: float = 90.0,
        camera_resolution: Tuple[int, int] = (64, 64),
        max_range: float = 10.0,
        conf_threshold: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.camera_fov_deg = camera_fov_deg
        self.camera_resolution = camera_resolution
        self.max_range = max_range
        self.conf_threshold = conf_threshold
        self._rng = rng or np.random.default_rng()
        self._victims: List[np.ndarray] = []  # list of (x, y, z) positions

    def set_victims(self, victims: List[np.ndarray]) -> None:
        """Update known victim positions (world coordinates)."""
        self._victims = [np.asarray(v, dtype=np.float32) for v in victims]

    def detect(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        camera_frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Detect victims in current camera view.

        Args:
            drone_pos: (3,) drone position in world
            drone_quat: (4,) drone orientation quaternion [w,x,y,z]
            camera_frame: optional RGB frame (unused in mock)

        Returns:
            List of Detection objects for victims in FOV.
        """
        detections = []
        forward = self._quat_forward(drone_quat)

        for victim_pos in self._victims:
            to_victim = victim_pos - drone_pos
            dist = float(np.linalg.norm(to_victim))

            if dist > self.max_range:
                continue

            # Check if in FOV (angle between forward and to_victim)
            to_victim_norm = to_victim / (dist + 1e-6)
            cos_angle = float(np.dot(forward, to_victim_norm))
            half_fov = np.deg2rad(self.camera_fov_deg) / 2.0

            if cos_angle < np.cos(half_fov):
                continue  # outside FOV

            # Project to image coordinates
            # Simplified: assume camera points along forward, image plane perpendicular
            rel = to_victim - forward * dist
            lateral = float(np.linalg.norm(rel))
            angle_off = np.arctan2(lateral, dist)

            # Normalized image coords (0-1)
            u = 0.5 + angle_off / np.deg2rad(self.camera_fov_deg) * np.sign(float(np.cross(forward, to_victim_norm)[2]))
            v = 0.5  # simplified: assume level camera

            # Bbox size decreases with distance
            bbox_size = max(0.05, 0.3 * (5.0 / dist))
            u = np.clip(u, bbox_size/2, 1 - bbox_size/2)
            v = np.clip(v, bbox_size/2, 1 - bbox_size/2)

            # Confidence: higher when closer and centered
            conf = float(np.clip(0.9 - 0.05 * dist + 0.1 * cos_angle, 0.3, 0.99))
            conf += float(self._rng.normal(0, 0.05))
            conf = np.clip(conf, 0.0, 1.0)

            if conf >= self.conf_threshold:
                detections.append(Detection(
                    class_name="victim",
                    bbox=(u - bbox_size/2, v - bbox_size/2, bbox_size, bbox_size),
                    confidence=conf,
                    position_3d=tuple(victim_pos.tolist()),
                ))

        return detections

    def _quat_forward(self, quat: np.ndarray) -> np.ndarray:
        """Get forward vector from quaternion [w,x,y,z]."""
        w, x, y, z = quat
        return np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x),
            1 - 2*(x*x + y*y),
        ], dtype=np.float32)


class RealYOLODetector:
    """Real YOLOv8n ONNX detector (stub for hardware deployment)."""

    def __init__(self, onnx_path: str, conf_threshold: float = 0.5):
        self.onnx_path = onnx_path
        self.conf_threshold = conf_threshold
        self._session = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8n on RGB frame. Returns detections."""
        self._load()
        # Preprocess: resize to 640x640, normalize, NCHW
        # Run inference
        # Postprocess: NMS, scale bboxes
        # Return Detection objects
        raise NotImplementedError("Real YOLO detector requires ONNX model and preprocessing pipeline")


def create_detector(mode: str = "mock", **kwargs) -> MockYOLODetector:
    """Factory for detector instances."""
    if mode == "mock":
        return MockYOLODetector(**kwargs)
    elif mode == "real":
        return RealYOLODetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector mode: {mode}")