#!/usr/bin/env python3
"""Multi-sensor fusion for DroneNav-SAR (Sprint 15).

SAR-only: fuses RGB-D (YOLO victims + metric depth), IMU, and optical
flow into obstacle-avoidance waypoints + victim world positions for the
policy. Budget: <20ms end-to-end on Jetson Orin.

Sensors @ each tick:
  RGB 640x480 @ 30Hz -> YOLOv8n victims -> bearing
  Depth (metric)      -> obstacle sectors + victim range
  IMU 200Hz           -> VIO pose (via VIOSLAM)
  Optical flow 30Hz   -> VIO translation delta
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FusionOutput:
    """Fused perception snapshot consumed by the policy bridge."""

    pose_position: Tuple[float, float, float]
    victim_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    victim_confidences: List[float] = field(default_factory=list)
    waypoint: Tuple[float, float, float] = (0.0, 0.0, 1.5)
    obstacle_stop: bool = False
    sectors: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


class MultiSensorFusion:
    """RGB-D + IMU + optical-flow fusion with latency accounting.

    Args:
        danger_range: obstacle sector threshold (m).
        cruise_altitude: default waypoint altitude (m).
        victim_altitude: assumed victim ground plane (m).
    """

    LATENCY_BUDGET_MS = 20.0

    def __init__(
        self,
        danger_range: float = 2.0,
        cruise_altitude: float = 1.5,
        victim_altitude: float = 0.0,
    ) -> None:
        self.danger_range = danger_range
        self.cruise_altitude = cruise_altitude
        self.victim_altitude = victim_altitude
        self.last_latency_ms: float = 0.0

    def fuse(
        self,
        detections,
        depth: np.ndarray,
        pose_position: np.ndarray,
        pose_quat: Optional[np.ndarray] = None,
        fx: float = 520.0,
    ) -> FusionOutput:
        """Fuse one synchronized sensor tick into a policy-ready snapshot."""
        t0 = time.perf_counter()
        pos = np.asarray(pose_position, dtype=np.float64).ravel()
        h, w = depth.shape[:2]

        # Obstacle sectors from depth thirds.
        thirds = np.array_split(np.asarray(depth), 3, axis=1)
        sectors = {s: float(np.min(t)) for s, t in
                   zip(("left", "center", "right"), thirds)}
        obstacle_stop = bool(sectors["center"] < self.danger_range)

        # Victims: back-project bbox center ray + depth range -> world.
        victims: List[Tuple[float, float, float]] = []
        confs: List[float] = []
        for det in detections:
            x, y, bw, bh = det.bbox
            cx, cy = x + bw / 2.0, y + bh / 2.0
            px, py = int(np.clip(cx * w, 0, w - 1)), int(np.clip(cy * h, 0, h - 1))
            rng = float(np.clip(depth[py, px], 0.2, 20.0))
            # Bearing in camera frame (forward = +z/depth axis).
            dx = (cx - 0.5) * w / fx
            bearing = np.array([dx * rng, 0.0, rng])
            world = pos + bearing  # level-camera approx (SAR indoor)
            world[2] = self.victim_altitude
            victims.append((float(world[0]), float(world[1]), float(world[2])))
            confs.append(float(det.confidence))

        # Waypoint: advance forward unless blocked; sidestep toward free sector.
        step = 1.0
        if obstacle_stop:
            side = 1.0 if sectors["left"] > sectors["right"] else -1.0
            waypoint = (float(pos[0]), float(pos[1] + side * step),
                        float(self.cruise_altitude))
        else:
            waypoint = (float(pos[0] + step), float(pos[1]),
                        float(self.cruise_altitude))

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return FusionOutput(
            pose_position=(float(pos[0]), float(pos[1]), float(pos[2])),
            victim_positions=victims, victim_confidences=confs,
            waypoint=waypoint, obstacle_stop=obstacle_stop,
            sectors=sectors, latency_ms=self.last_latency_ms)

    @property
    def within_budget(self) -> bool:
        return self.last_latency_ms < self.LATENCY_BUDGET_MS


def create_fusion(**kwargs) -> MultiSensorFusion:
    return MultiSensorFusion(**kwargs)
