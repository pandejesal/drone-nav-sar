#!/usr/bin/env python3
"""Deployment Controller — altitude/velocity-triggered chute actuation (Sprint 17).

Arms at altitude, fires servo when release conditions hold, confirms via
accelerometer spike (deceleration jolt). SAR-only: drop_payload gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class DeployState(Enum):
    STOWED = "stowed"
    ARMED = "armed"
    DEPLOYED = "deployed"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class DeployConfig:
    deploy_altitude: float = 50.0
    min_altitude: float = 10.0
    max_speed_ms: float = 12.0
    accel_spike_thresh: float = 3.0  # m/s^2 above baseline
    confirm_window_s: float = 2.0


class DeploymentController:
    def __init__(self, config: Optional[DeployConfig] = None):
        self.config = config or DeployConfig()
        self.state = DeployState.STOWED
        self.deploy_time: Optional[float] = None
        self.log: List[Dict] = []

    def reset(self) -> None:
        self.state = DeployState.STOWED
        self.deploy_time = None
        self.log.clear()

    def arm(self) -> Dict:
        if self.state == DeployState.STOWED:
            self.state = DeployState.ARMED
        return {"ok": True, "state": self.state.value}

    def should_deploy(self, altitude: float, speed: float) -> bool:
        c = self.config
        return bool(self.state == DeployState.ARMED
                    and altitude >= c.min_altitude
                    and speed <= c.max_speed_ms)

    def deploy(self, t: float = 0.0) -> Dict:
        if self.state != DeployState.ARMED:
            return {"ok": False, "state": self.state.value,
                    "reason": "not_armed"}
        self.state = DeployState.DEPLOYED
        self.deploy_time = float(t)
        self.log.append({"event": "deploy", "t": float(t)})
        return {"ok": True, "state": self.state.value}

    def confirm(self, accel_history: np.ndarray, baseline: float = 9.81) -> Dict:
        """Accel spike detection: max |a - baseline| over window."""
        a = np.asarray(accel_history, dtype=np.float64).reshape(-1)
        spike = float(np.max(np.abs(a - baseline))) if a.size else 0.0
        if spike >= self.config.accel_spike_thresh:
            self.state = DeployState.CONFIRMED
            return {"ok": True, "confirmed": True, "spike": spike,
                    "state": self.state.value}
        return {"ok": False, "confirmed": False, "spike": spike,
                "state": self.state.value}

    def check_timeout(self, t: float) -> Dict:
        if (self.state == DeployState.DEPLOYED and self.deploy_time is not None
                and float(t) - self.deploy_time > self.config.confirm_window_s):
            self.state = DeployState.FAILED
        return {"state": self.state.value}

    def drop_skill(self, x: float, y: float, z: float) -> Dict:
        """SAR-only drop_payload command (call after arm)."""
        return {"skill": "drop_payload",
                "params": {"x": float(x), "y": float(y), "z": float(z),
                           "yaw_deg": 0.0},
                "constraints": {"timeout_s": 5.0}}
