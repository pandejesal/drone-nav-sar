#!/usr/bin/env python3
"""Target Tracker — multi-hypothesis Kalman filter (CV + CT) with JPDA (Sprint 16).

SAR-only context: tracks moving targets (boats/vehicles/people at 5-20 m/s)
so a drop planner can lead them. No weapons logic; outputs position/velocity
estimates used to cue navigate_to / hover / drop_payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

G = 9.81


@dataclass
class Track:
    """Single Kalman track, constant-velocity model with coordinated-turn blend."""

    track_id: int
    state: np.ndarray  # (6,) x,y,z,vx,vy,vz
    cov: np.ndarray    # (6,6)
    turn_rate: float = 0.0
    age: int = 0
    hits: int = 0
    misses: int = 0
    history: List[np.ndarray] = field(default_factory=list)

    def predict(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # Coordinated-turn blend: rotate horizontal velocity slightly.
        if abs(self.turn_rate) > 1e-9:
            w = self.turn_rate * dt
            c, s = np.cos(w), np.sin(w)
            R = np.array([[c, -s], [s, c]])
            self.state[3:5] = R @ self.state[3:5]
        self.state = F @ self.state
        q = 0.5
        Q = np.eye(6, dtype=np.float64) * q * dt
        self.cov = F @ self.cov @ F.T + Q
        self.age += 1
        return self.predicted_position()

    def update(self, meas: np.ndarray, meas_noise: float = 1.0) -> None:
        z = np.asarray(meas, dtype=np.float64).reshape(3)
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * float(meas_noise)
        y = z - H @ self.state
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.cov = (np.eye(6) - K @ H) @ self.cov
        self.hits += 1
        self.misses = 0
        self.history.append(self.state[:3].copy())

    def predicted_position(self, horizon: float = 0.0) -> np.ndarray:
        p = self.state[:3] + self.state[3:6] * float(horizon)
        return p.astype(np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].astype(np.float64)


class TargetTracker:
    """Multi-target tracker with JPDA-style gating + nearest association."""

    def __init__(self, gate_m: float = 5.0, meas_noise: float = 0.5,
                 max_misses: int = 5):
        self.gate_m = float(gate_m)
        self.meas_noise = float(meas_noise)
        self.max_misses = int(max_misses)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def _new_track(self, meas: np.ndarray) -> Track:
        state = np.zeros(6)
        state[:3] = np.asarray(meas, dtype=np.float64).reshape(3)
        cov = np.eye(6) * 2.0
        tr = Track(track_id=self._next_id, state=state, cov=cov)
        tr.hits = 1
        tr.history.append(state[:3].copy())
        self.tracks[self._next_id] = tr
        self._next_id += 1
        return tr

    def predict(self, dt: float) -> None:
        for tr in self.tracks.values():
            tr.predict(dt)

    def update(self, detections: List[np.ndarray], dt: float) -> Dict[int, int]:
        """Predict then JPDA-gated associate. Returns {track_id: det_idx}."""
        self.predict(dt)
        dets = [np.asarray(d, dtype=np.float64).reshape(3) for d in detections]
        assoc: Dict[int, int] = {}
        used = set()
        # Cost matrix with gating.
        for tid, tr in self.tracks.items():
            best, best_d = None, float("inf")
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = float(np.linalg.norm(d - tr.state[:3]))
                if dist <= self.gate_m and dist < best_d:
                    best, best_d = i, dist
            if best is not None:
                tr.update(dets[best], self.meas_noise)
                assoc[tid] = best
                used.add(best)
            else:
                tr.misses += 1
        # Spawn tracks for unassociated detections.
        for i, d in enumerate(dets):
            if i not in used:
                tr = self._new_track(d)
                assoc[tr.track_id] = i
        # Prune stale tracks.
        dead = [t for t, tr in self.tracks.items() if tr.misses > self.max_misses]
        for t in dead:
            del self.tracks[t]
        return assoc

    def estimate(self, track_id: int, horizon: float = 0.0) -> Optional[np.ndarray]:
        tr = self.tracks.get(track_id)
        if tr is None:
            return None
        return tr.predicted_position(horizon)

    def best_track_for(self, point: np.ndarray) -> Optional[int]:
        if not self.tracks:
            return None
        p = np.asarray(point, dtype=np.float64).reshape(3)
        return min(self.tracks, key=lambda t: float(np.linalg.norm(self.tracks[t].state[:3] - p)))

    def all_estimates(self) -> Dict[int, np.ndarray]:
        return {t: tr.state[:3].copy() for t, tr in self.tracks.items()}
