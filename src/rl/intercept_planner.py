#!/usr/bin/env python3
"""Intercept Planner — minimum-time intercept with wind compensation (Sprint 16).

Time-to-go from free-fall ballistics, lead point = predicted target pos at
impact + wind drift correction, waypoint trajectory with obstacle push-out.
Outputs are consumed as navigate_to goals; SAR-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

G = 9.81


def fall_time(altitude_m: float) -> float:
    """Free-fall time from altitude (s)."""
    h = max(0.0, float(altitude_m))
    return float(np.sqrt(2.0 * h / G))


def time_to_go(drone_pos: np.ndarray, target_pos: np.ndarray,
               drone_speed: float, drop_altitude: float = 0.0) -> float:
    """Transit time + payload fall time."""
    d = np.asarray(drone_pos, dtype=np.float64).reshape(3)
    t = np.asarray(target_pos, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(d - t))
    transit = dist / max(0.2, float(drone_speed))
    return transit + fall_time(float(drop_altitude))


def lead_point(target_pos: np.ndarray, target_vel: np.ndarray,
               t_go: float, wind: Optional[np.ndarray] = None,
               drag_factor: float = 0.0) -> np.ndarray:
    """Aim point: target propagated by t_go, minus wind drift of payload."""
    p = np.asarray(target_pos, dtype=np.float64).reshape(3)
    v = np.asarray(target_vel, dtype=np.float64).reshape(3)
    lead = p + v * float(t_go)
    if wind is not None:
        w = np.asarray(wind, dtype=np.float64).reshape(3)
        lead = lead - w * float(t_go) * (1.0 - float(drag_factor))
    return lead


@dataclass
class InterceptPlan:
    release_point: np.ndarray
    t_go: float
    waypoints: List[np.ndarray]
    target_pred: np.ndarray


class InterceptPlanner:
    """Minimum-time intercept planner with collision avoidance en route."""

    def __init__(self, drone_speed: float = 8.0, drop_altitude: float = 50.0,
                 safe_margin: float = 2.0):
        self.drone_speed = float(drone_speed)
        self.drop_altitude = float(drop_altitude)
        self.safe_margin = float(safe_margin)

    def plan(self, drone_pos: np.ndarray, target_pos: np.ndarray,
             target_vel: np.ndarray, wind: Optional[np.ndarray] = None,
             obstacles: Optional[List[np.ndarray]] = None) -> InterceptPlan:
        d = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        t = np.asarray(target_pos, dtype=np.float64).reshape(3)
        v = np.asarray(target_vel, dtype=np.float64).reshape(3)
        # Fixed-point iteration so t_go is consistent with the release point.
        t_go = time_to_go(d, t, self.drone_speed, self.drop_altitude)
        release = lead_point(t, v, t_go, wind)
        release[2] = self.drop_altitude
        for _ in range(8):
            t_go = time_to_go(d, release, self.drone_speed, self.drop_altitude)
            release = lead_point(t, v, t_go, wind)
            release[2] = self.drop_altitude
        target_pred = (t + v * t_go).astype(np.float64)
        waypoints = self._waypoints(d, release, obstacles or [])
        return InterceptPlan(release_point=release, t_go=float(t_go),
                             waypoints=waypoints, target_pred=target_pred)

    def _waypoints(self, start: np.ndarray, goal: np.ndarray,
                   obstacles: List[np.ndarray]) -> List[np.ndarray]:
        mid = (start + goal) / 2.0
        # Push mid-point out of obstacle spheres (collision avoidance).
        for ob in obstacles:
            o = np.asarray(ob, dtype=np.float64).reshape(3)
            diff = mid - o
            dist = float(np.linalg.norm(diff)) + 1e-9
            if dist < self.safe_margin + 3.0:
                push = (self.safe_margin + 3.0 - dist)
                mid = mid + (diff / dist) * push + np.array([0.0, 0.0, 2.0])
        return [start.copy(), mid, goal.copy()]

    def plan_skill(self, drone_pos: np.ndarray, target_pos: np.ndarray,
                   target_vel: np.ndarray, wind: Optional[np.ndarray] = None,
                   obstacles: Optional[List[np.ndarray]] = None) -> Dict:
        """SAR-only navigate_to command to the release point."""
        plan = self.plan(drone_pos, target_pos, target_vel, wind, obstacles)
        r = plan.release_point
        return {"skill": "navigate_to",
                "params": {"x": float(r[0]), "y": float(r[1]), "z": float(r[2]),
                           "yaw_deg": 0.0, "speed_ms": self.drone_speed},
                "constraints": {"timeout_s": 60.0},
                "t_go": plan.t_go}
