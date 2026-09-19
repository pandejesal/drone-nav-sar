#!/usr/bin/env python3
"""Ground Contact Detector — accel spike + pressure + depth fusion (Sprint 17)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ContactConfig:
    accel_spike_thresh: float = 4.0
    pressure_jump_pa: float = 15.0
    depth_thresh_m: float = 0.15
    votes_required: int = 1


class GroundContactDetector:
    def __init__(self, config: Optional[ContactConfig] = None):
        self.config = config or ContactConfig()
        self.contacted = False
        self.votes = 0

    def reset(self) -> None:
        self.contacted = False
        self.votes = 0

    def step(self, accel: Optional[np.ndarray] = None,
             pressure_delta_pa: Optional[float] = None,
             depth_m: Optional[float] = None,
             baseline_accel: float = 9.81) -> Dict:
        c = self.config
        votes = 0
        reasons = []
        if accel is not None:
            a = np.asarray(accel, dtype=np.float64).reshape(-1)
            spike = float(np.max(np.abs(a - baseline_accel))) if a.size else 0.0
            if spike >= c.accel_spike_thresh:
                votes += 1
                reasons.append(f"accel_spike:{spike:.2f}")
        if pressure_delta_pa is not None and abs(float(pressure_delta_pa)) >= c.pressure_jump_pa:
            votes += 1
            reasons.append(f"pressure:{pressure_delta_pa:.1f}Pa")
        if depth_m is not None and float(depth_m) <= c.depth_thresh_m:
            votes += 1
            reasons.append(f"depth:{depth_m:.2f}m")
        self.votes = votes
        if votes >= c.votes_required:
            self.contacted = True
        return {"contact": bool(self.contacted), "votes": votes,
                "reasons": reasons}
