#!/usr/bin/env python3
"""Perception -> policy bridge for DroneNav-SAR (Sprint 15).

SAR-only: converts fused perception (victim positions, obstacle
waypoints, SLAM pose) into task embeddings consumed by LocalPolicy.

Task embedding layout (8-dim, float32):
  [victim_present, victim_dx, victim_dy, victim_conf,
   obstacle_stop, free_left, free_center, free_right]
Positions are normalized by `norm_range` (default 10m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PolicyInput:
    """Policy-ready observation fragment from perception."""

    task_embedding: np.ndarray  # (8,) float32
    waypoint: Tuple[float, float, float]
    victim_detected: bool
    victim_position: Optional[Tuple[float, float, float]]
    obstacle_stop: bool


class PerceptionBridge:
    """Maps FusionOutput (+ SLAM pose) to LocalPolicy inputs."""

    EMBEDDING_DIM = 8

    def __init__(self, norm_range: float = 10.0,
                 default_altitude: float = 1.5) -> None:
        self.norm_range = norm_range
        self.default_altitude = default_altitude

    def to_policy_input(self, fusion, drone_pos: np.ndarray) -> PolicyInput:
        """Convert a FusionOutput into a PolicyInput."""
        pos = np.asarray(drone_pos, dtype=np.float64).ravel()
        emb = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        victim_detected = len(fusion.victim_positions) > 0
        victim_position: Optional[Tuple[float, float, float]] = None
        if victim_detected:
            # Highest-confidence victim drives the embedding.
            best = int(np.argmax(fusion.victim_confidences))
            victim_position = fusion.victim_positions[best]
            conf = float(fusion.victim_confidences[best])
            dx = (victim_position[0] - float(pos[0])) / self.norm_range
            dy = (victim_position[1] - float(pos[1])) / self.norm_range
            emb[0] = 1.0
            emb[1] = float(np.clip(dx, -1, 1))
            emb[2] = float(np.clip(dy, -1, 1))
            emb[3] = float(np.clip(conf, 0, 1))

        emb[4] = 1.0 if fusion.obstacle_stop else 0.0
        for i, sector in enumerate(("left", "center", "right")):
            rng = float(fusion.sectors.get(sector, self.norm_range))
            emb[5 + i] = float(np.clip(rng / self.norm_range, 0, 1))

        return PolicyInput(
            task_embedding=emb, waypoint=fusion.waypoint,
            victim_detected=victim_detected,
            victim_position=victim_position,
            obstacle_stop=bool(fusion.obstacle_stop))

    def to_detection_task(self, detections) -> Dict[str, object]:
        """Legacy helper: raw YOLO detections -> {'detect': embedding}."""
        from src.perception.fusion import MultiSensorFusion
        import numpy as _np
        depth = _np.full((48, 64), 5.0, dtype=_np.float32)
        fusion = MultiSensorFusion().fuse(
            detections, depth, _np.zeros(3))
        policy_in = self.to_policy_input(fusion, _np.zeros(3))
        return {"detect": policy_in.task_embedding,
                "victims": list(fusion.victim_positions)}


def create_perception_bridge(**kwargs) -> PerceptionBridge:
    return PerceptionBridge(**kwargs)
