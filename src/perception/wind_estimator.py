#!/usr/bin/env python3
"""Wind Drift Estimator — online wind from GPS/IMU drift (Sprint 17).

Ensemble-mean wind estimate with exponential blending; predicts payload drift
for a given descent profile so the release point can be shifted upwind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class WindConfig:
    ensemble_n: int = 8
    process_noise: float = 0.05
    meas_noise: float = 0.3
    blend: float = 0.2


class WindEstimator:
    """2-D horizontal wind estimator (ensemble Kalman-lite)."""

    def __init__(self, config: WindConfig | None = None, seed: int = 0):
        self.config = config or WindConfig()
        self._rng = np.random.default_rng(seed)
        self.ensemble = self._rng.normal(0.0, 1.0, size=(self.config.ensemble_n, 2))
        self.wind = np.zeros(2, dtype=np.float64)

    def update(self, gps_vel: np.ndarray, air_vel: np.ndarray) -> np.ndarray:
        """Wind = ground velocity - air-relative velocity (horizontal)."""
        g = np.asarray(gps_vel, dtype=np.float64).reshape(-1)[:2]
        a = np.asarray(air_vel, dtype=np.float64).reshape(-1)[:2]
        meas = g - a
        c = self.config
        self.ensemble = self.wind + self._rng.normal(
            0.0, c.process_noise, size=self.ensemble.shape)
        dists = np.linalg.norm(self.ensemble - meas, axis=1)
        w = np.exp(-0.5 * (dists / max(1e-6, c.meas_noise)) ** 2)
        if float(w.sum()) <= 0.0:
            w = np.full_like(w, 1.0 / len(w))  # STRIKEPKGFIX far-field uniform
        else:
            w = w / w.sum()
        ens_mean = (self.ensemble * w[:, None]).sum(axis=0)
        innovation = 0.7 * (meas - self.wind) + 0.3 * (ens_mean - self.wind)
        self.wind = self.wind + c.blend * innovation
        self.ensemble = (self.ensemble - self.ensemble.mean(axis=0)) + self.wind
        return self.wind.copy()

    def drift(self, descent_time_s: float, chute_factor: float = 1.0) -> np.ndarray:
        """Predicted horizontal drift over descent (m)."""
        return (self.wind * float(descent_time_s) * float(chute_factor)).astype(np.float64)

    def compensated_release(self, target: np.ndarray,
                            descent_time_s: float) -> np.ndarray:
        """Upwind-shifted release point for zero-drift landing."""
        t = np.asarray(target, dtype=np.float64).reshape(3)
        out = t.copy()
        out[:2] = out[:2] - self.drift(descent_time_s)[:2]
        return out

    def status(self) -> Dict:
        return {"wind_ms": [float(self.wind[0]), float(self.wind[1])],
                "speed": float(np.linalg.norm(self.wind))}
