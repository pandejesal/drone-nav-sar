#!/usr/bin/env python3
"""Sprint 24: Hamilton-Jacobi reachability — backward/forward reachable sets.

Level-set HJ PDE for single-integrator dynamics ẋ = u, |u| <= vmax:
  dV/dt + min_u ∇V·f = 0  →  Hamiltonian H = -vmax·|∇V| (backward min-time).
Backward integration grows the backward reachable set (BRS) of the target
while carving out the avoid set. NumPy-only (SAR-only, no ROC-HJ dep).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class ReachabilityConfig:
    geofence_xy: float = 50.0
    min_altitude: float = 0.3
    max_altitude: float = 120.0
    max_speed: float = 15.0
    n_bins: int = 16
    dt: float = 0.1
    horizon: float = 2.0


@dataclass
class TargetSet:
    center: np.ndarray
    radius: float = 1.0

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        return np.linalg.norm(P - c, axis=-1) - float(self.radius)


@dataclass
class AvoidSet:
    centers: List[np.ndarray]
    radius: float = 1.0
    margin: float = 1.0

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        if not self.centers:
            return np.full(P.shape[:-1], np.inf)
        d = np.stack([np.linalg.norm(P - np.asarray(o).reshape(3), axis=-1)
                      for o in self.centers], axis=0)
        return d.min(axis=0) - float(self.radius) - float(self.margin)


class HJReachability:
    """Grid level-set HJ reachability over a 3-D position box."""

    def __init__(self, config: ReachabilityConfig | None = None,
                 target: TargetSet | None = None,
                 avoid: AvoidSet | None = None):
        self.cfg = config or ReachabilityConfig()
        self.target = target or TargetSet(np.zeros(3))
        self.avoid = avoid or AvoidSet([])
        self._build_grid()
        self.value = self._init_value()

    def _build_grid(self) -> None:
        c = self.cfg
        n = c.n_bins
        self.xs = np.linspace(-c.geofence_xy, c.geofence_xy, n)
        self.ys = np.linspace(-c.geofence_xy, c.geofence_xy, n)
        self.zs = np.linspace(c.min_altitude, c.max_altitude, n)
        X, Y, Z = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.grid_points = np.stack([X, Y, Z], axis=-1)  # (n,n,n,3)

    def _init_value(self) -> np.ndarray:
        """V0(x) = dist-to-target; avoid cells forced positive (unsafe)."""
        V = self.target.signed_distance(self.grid_points)
        d_avoid = self.avoid.signed_distance(self.grid_points)
        V = np.where(d_avoid < 0.0, np.abs(d_avoid) + V.max() + 1.0, V)
        return V

    def _hamiltonian(self, V: np.ndarray) -> np.ndarray:
        """H = -vmax·|∇V| via upwind central differences."""
        c = self.cfg
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]
        dz = self.zs[1] - self.zs[0]
        gx, gy, gz = np.gradient(V, dx, dy, dz, edge_order=1)
        return -c.max_speed * np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    def compute_backward_reachable_set(self, horizon: float | None = None) -> np.ndarray:
        """Backward HJ PDE integration; returns value grid (<=0 reachable)."""
        T = self.horizon if horizon is None else float(horizon)
        V = self._init_value().copy()
        steps = max(1, int(T / self.cfg.dt))
        for _ in range(steps):
            H = self._hamiltonian(V)
            V = V + self.cfg.dt * np.minimum(H, 0.0)
            # Pin avoid-set cells unsafe for all time (reach-avoid game).
            d_avoid = self.avoid.signed_distance(self.grid_points)
            V = np.where(d_avoid < 0.0, np.abs(V).max() + 1.0, V)
        self.value = V
        return V

    @property
    def horizon(self) -> float:
        return float(self.cfg.horizon)

    def backward_reachable_fraction(self) -> float:
        return float(np.mean(self.value <= 0.0))

    def is_backward_reachable(self, position: np.ndarray) -> bool:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        c = self.cfg
        ix = int(np.clip(np.searchsorted(self.xs, p[0]), 0, c.n_bins - 1))
        iy = int(np.clip(np.searchsorted(self.ys, p[1]), 0, c.n_bins - 1))
        iz = int(np.clip(np.searchsorted(self.zs, p[2]), 0, c.n_bins - 1))
        return bool(self.value[ix, iy, iz] <= 0.0)

    def compute_forward_reachable_set(self, start: np.ndarray,
                                      horizon: float | None = None) -> Dict[str, float]:
        """Forward HJ: single-integrator ball growth r(t) = r0 + vmax·t."""
        T = self.horizon if horizon is None else float(horizon)
        p0 = np.asarray(start, dtype=np.float64).reshape(3)
        radius = float(self.cfg.max_speed * T)
        d_avoid = (min(float(np.linalg.norm(p0 - np.asarray(o).reshape(3)))
                       for o in self.avoid.centers)
                   if self.avoid.centers else float("inf"))
        return {"center": p0, "radius": radius,
                "touches_avoid": bool(d_avoid <= radius + self.avoid.radius + self.avoid.margin),
                "min_avoid_dist": float(d_avoid)}

    def coverage_of_trajectories(self, trajectories: List[np.ndarray],
                                 target: np.ndarray) -> Dict[str, float]:
        tgt = np.asarray(target, dtype=np.float64).reshape(3)
        c = self.cfg
        grid = np.zeros((c.n_bins, c.n_bins, c.n_bins), dtype=bool)
        span = 2 * c.geofence_xy
        hit = 0
        for traj in trajectories:
            T = np.asarray(traj, dtype=np.float64)
            P = T[:, 0:3] if T.ndim == 2 and T.shape[1] >= 3 else T.reshape(-1, 3)
            idx = np.clip(((P - (-c.geofence_xy)) / span * c.n_bins).astype(int),
                          0, c.n_bins - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            if np.any(np.linalg.norm(P - tgt, axis=1) <= self.target.radius):
                hit += 1
        return {"coverage": float(grid.mean()),
                "hit_rate": float(hit / len(trajectories)) if trajectories else 0.0,
                "cells_visited": int(grid.sum()),
                "cells_total": int(grid.size)}
