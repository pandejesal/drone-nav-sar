#!/usr/bin/env python3
"""Terminal Guidance — proportional navigation + augmented PN (Sprint 16).

Guides the release/approach onto the predicted target point: PN acceleration
command from line-of-sight rate, impact-angle shaping via biased PN.
SAR-only: outputs velocity corrections applied to navigate_to/hover goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class GuidanceCommand:
    accel: np.ndarray
    closing_speed: float
    los_rate: float
    go: bool


class TerminalGuidance:
    def __init__(self, nav_gain: float = 3.0, commit_radius: float = 1.0,
                 max_accel: float = 6.0, desired_impact_deg: float = 90.0):
        self.N = float(nav_gain)
        self.commit_radius = float(commit_radius)
        self.max_accel = float(max_accel)
        self.desired_impact = float(np.deg2rad(desired_impact_deg))

    def step(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
             target_pos: np.ndarray, target_vel: np.ndarray,
             dt: float = 0.05) -> "GuidanceCommand":
        d = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        dv = np.asarray(drone_vel, dtype=np.float64).reshape(3)
        t = np.asarray(target_pos, dtype=np.float64).reshape(3)
        tv = np.asarray(target_vel, dtype=np.float64).reshape(3)
        rel = t - d
        rng = float(np.linalg.norm(rel)) + 1e-9
        rel_vel = tv - dv
        closing = float(-np.dot(rel_vel, rel) / rng)
        los = rel / rng
        
        # LOS rate (perpendicular to LOS)
        los_rate_vec = (rel_vel - np.dot(rel_vel, los) * los) / rng
        los_rate = float(np.linalg.norm(los_rate_vec))
        
        omega = np.cross(los, rel_vel) / rng
        accel = self.N * max(0.0, closing) * np.cross(omega, los)
        
        # Impact-angle bias: pull vertical toward desired impact angle
        if self.desired_impact != np.pi / 2:
            bias = (self.desired_impact - np.pi / 2.0) * 0.1
            accel[2] += bias * max(0.0, closing) * 0.2
        
        # Feedforward: target velocity component perpendicular to LOS
        v_parallel = np.dot(rel_vel, los) * los
        v_perp = rel_vel - v_parallel
        accel += v_perp * 0.5
        
        norm = float(np.linalg.norm(accel))
        if norm > self.max_accel:
            accel = accel / norm * self.max_accel
        
        go = bool(rng <= self.commit_radius or (closing > 0 and rng < 5.0))
        return GuidanceCommand(accel=accel, closing_speed=closing,
                               los_rate=los_rate, go=go)

    def corrected_goal(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
                       target_pos: np.ndarray, target_vel: np.ndarray,
                       dt: float = 0.05) -> Dict:
        """SAR-only hover correction toward the homing point."""
        cmd = self.step(drone_pos, drone_vel, target_pos, target_vel, dt)
        goal = np.asarray(target_pos, dtype=np.float64).reshape(3)
        corr = goal + cmd.accel * dt * dt * 0.5
        return {"skill": "hover",
                "params": {"x": float(corr[0]), "y": float(corr[1]),
                           "z": float(max(0.5, corr[2])),
                           "yaw_deg": 0.0, "timeout_s": 5.0},
                "constraints": {"timeout_s": 5.0},
                "closing_speed": cmd.closing_speed, "go": cmd.go}


from dataclasses import dataclass

@dataclass
class GuidanceCommand:
    accel: np.ndarray
    closing_speed: float
    los_rate: float
    go: bool