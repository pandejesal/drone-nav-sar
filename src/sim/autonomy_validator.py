#!/usr/bin/env python3
"""Sprint 22: formal autonomy validation — reachability, safety, liveness.

Implements lightweight, dependency-free checks in the spirit of HJ/SOS
reachability, barrier-function safety, and LTL liveness. SAR-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ValidatorConfig:
    geofence_xy: float = 50.0      # m, square half-width
    min_altitude: float = 0.3      # m AGL
    max_altitude: float = 120.0    # m
    max_speed: float = 15.0        # m/s
    goal_radius: float = 1.0       # m, liveness acceptance ball
    obstacle_margin: float = 1.0   # m, CBF safety margin
    n_reach_bins: int = 8          # per-axis coverage grid resolution


@dataclass
class EpisodeResult:
    reached: bool
    safety_violations: List[str]
    min_obstacle_dist: float
    coverage: float


class AutonomyValidator:
    """Formal-style checks over recorded SAR episodes."""

    def __init__(self, config: Optional[ValidatorConfig] = None,
                 obstacles: Optional[List[np.ndarray]] = None,
                 obstacle_radius: float = 1.0):
        self.cfg = config or ValidatorConfig()
        self.obstacles = [np.asarray(o, dtype=np.float64).reshape(3) for o in (obstacles or [])]
        self.obstacle_radius = obstacle_radius

    # -- safety invariants (barrier functions) ---------------------------
    def verify_safety_invariants(self, states: np.ndarray) -> List[str]:
        """Return list of violated invariants; empty means 100% safe."""
        S = np.asarray(states, dtype=np.float64)
        pos = S[:, 0:3] if S.ndim == 2 and S.shape[1] >= 3 else S.reshape(-1, 3)[:, 0:3]
        vel = S[:, 3:6] if S.ndim == 2 and S.shape[1] >= 6 else np.zeros_like(pos)
        violations: List[str] = []
        c = self.cfg
        if np.any(np.abs(pos[:, 0]) > c.geofence_xy) or np.any(np.abs(pos[:, 1]) > c.geofence_xy):
            violations.append("geofence")
        if np.any(pos[:, 2] < c.min_altitude):
            violations.append("min_altitude")
        if np.any(pos[:, 2] > c.max_altitude):
            violations.append("max_altitude")
        if np.any(np.linalg.norm(vel, axis=1) > c.max_speed):
            violations.append("max_speed")
        for o in self.obstacles:
            d = np.linalg.norm(pos - o, axis=1)
            if np.any(d < (self.obstacle_radius + c.obstacle_margin)):
                violations.append("obstacle_cbf")
                break
        return sorted(set(violations))

    def barrier_margin(self, position: np.ndarray) -> float:
        """Min control-barrier margin: >0 safe, <0 violated."""
        p = np.asarray(position, dtype=np.float64).reshape(3)
        c = self.cfg
        margins = [c.geofence_xy - abs(p[0]), c.geofence_xy - abs(p[1]),
                   p[2] - c.min_altitude, c.max_altitude - p[2]]
        for o in self.obstacles:
            margins.append(float(np.linalg.norm(p - o)) - self.obstacle_radius - c.obstacle_margin)
        return float(min(margins))

    def verify_cbf(self, positions: np.ndarray) -> Tuple[bool, float]:
        margins = [self.barrier_margin(p) for p in np.asarray(positions).reshape(-1, 3)]
        return (all(m >= 0 for m in margins), float(min(margins)) if margins else float("inf"))

    # -- reachability -----------------------------------------------------
    def check_reachability(self, trajectories: List[np.ndarray],
                           target: np.ndarray) -> Dict[str, float]:
        """Grid coverage of visited states + fraction of trajs hitting target.

        Approximates HJ/SOS reachable-set coverage on a coarse grid.
        """
        tgt = np.asarray(target, dtype=np.float64).reshape(3)
        c = self.cfg
        grid = np.zeros((c.n_reach_bins, c.n_reach_bins, c.n_reach_bins), dtype=bool)
        span = 2 * c.geofence_xy
        hit = 0
        for traj in trajectories:
            T = np.asarray(traj, dtype=np.float64)
            P = T[:, 0:3] if T.ndim == 2 and T.shape[1] >= 3 else T.reshape(-1, 3)
            idx = np.clip(((P + c.geofence_xy) / span * c.n_reach_bins).astype(int),
                          0, c.n_reach_bins - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            if np.any(np.linalg.norm(P - tgt, axis=1) <= c.goal_radius):
                hit += 1
        coverage = float(grid.mean())
        return {"coverage": coverage,
                "hit_rate": float(hit / len(trajectories)) if trajectories else 0.0,
                "cells_visited": int(grid.sum()),
                "cells_total": int(grid.size)}

    # -- liveness (LTL: F goal within T, G safe) ---------------------------
    def check_liveness(self, positions: np.ndarray, goal: np.ndarray,
                       max_steps: Optional[int] = None) -> Dict[str, object]:
        P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        G = np.asarray(goal, dtype=np.float64).reshape(3)
        T = len(P) if max_steps is None else min(len(P), max_steps)
        dists = np.linalg.norm(P[:T] - G, axis=1)
        hit_idx = int(np.argmin(dists))
        reached = bool(np.any(dists <= self.cfg.goal_radius))
        return {"reached": reached,
                "steps_to_goal": hit_idx if reached else None,
                "final_distance": float(dists[-1]) if len(dists) else float("inf")}

    # -- episode / batch ---------------------------------------------------
    def validate_episode(self, states: np.ndarray, goal: np.ndarray,
                         target: Optional[np.ndarray] = None) -> EpisodeResult:
        S = np.asarray(states, dtype=np.float64)
        P = S[:, 0:3] if S.ndim == 2 and S.shape[1] >= 3 else S.reshape(-1, 3)
        violations = self.verify_safety_invariants(S)
        live = self.check_liveness(P, goal)
        reach = self.check_reachability([S], target if target is not None else goal)
        min_obs = float("inf")
        for o in self.obstacles:
            min_obs = min(min_obs, float(np.min(np.linalg.norm(P - o, axis=1)))) if len(P) else min_obs
        return EpisodeResult(reached=bool(live["reached"]),
                             safety_violations=violations,
                             min_obstacle_dist=min_obs,
                             coverage=float(reach["coverage"]))

    def validate_batch(self, episodes: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, object]:
        """episodes: list of (states, goal). Returns aggregate metrics."""
        results = [self.validate_episode(s, g) for s, g in episodes]
        n = len(results) or 1
        return {"episodes": len(results),
                "reachability": sum(r.reached for r in results) / n,
                "safety_violations": sum(len(r.safety_violations) for r in results),
                "violation_free_rate": sum(not r.safety_violations for r in results) / n}
