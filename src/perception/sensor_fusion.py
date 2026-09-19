#!/usr/bin/env python3
"""Sprint 20: Multi-source Sensor Fusion for SAR COP.

Fuses RGB-D, thermal, LiDAR, radar, RF detections into unified tracks.
Pipeline: Detection -> JPDA association -> IMM-lite fusion -> fused tracks.
SAR-only: entity types are victim / hazard / landing_zone / teammate.

Sensors have per-modality noise models; fusion uses covariance
intersection + probabilistic (JPDA) weighting of gated detections.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SAR_ENTITY_TYPES = ("victim", "hazard", "landing_zone", "teammate")

# Per-sensor position noise (m, 1-sigma) and class confidence scale.
SENSOR_MODELS: Dict[str, Dict[str, float]] = {
    "rgb": {"pos_noise": 0.30, "conf": 1.0},
    "thermal": {"pos_noise": 0.45, "conf": 0.95},
    "lidar": {"pos_noise": 0.15, "conf": 1.0},
    "radar": {"pos_noise": 0.80, "conf": 0.85},
    "rf": {"pos_noise": 1.50, "conf": 0.60},
}


@dataclass
class Detection:
    """Single-sensor detection with covariance."""

    pos: np.ndarray  # (3,)
    cov: np.ndarray  # (3,3)
    sensor: str
    entity_type: str = "victim"
    confidence: float = 0.9
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(3)
        self.cov = np.asarray(self.cov, dtype=np.float64).reshape(3, 3)
        if self.entity_type not in SAR_ENTITY_TYPES:
            self.entity_type = "victim"


def make_detection(
    pos,
    sensor: str,
    entity_type: str = "victim",
    confidence: float = 0.9,
    timestamp: Optional[float] = None,
) -> Detection:
    """Build a Detection with the per-sensor noise model applied."""
    model = SENSOR_MODELS.get(sensor, {"pos_noise": 0.5, "conf": 1.0})
    noise = float(model["pos_noise"])
    cov = np.eye(3) * noise**2
    conf = float(confidence) * float(model["conf"])
    return Detection(
        pos=np.asarray(pos, dtype=np.float64).reshape(3),
        cov=cov,
        sensor=sensor,
        entity_type=entity_type,
        confidence=min(1.0, max(0.0, conf)),
        timestamp=time.time() if timestamp is None else float(timestamp),
    )


@dataclass
class FusedTrack:
    """Fused track: Kalman CV state + provenance."""

    track_id: int
    state: np.ndarray  # (6,) pos + vel
    cov: np.ndarray  # (6,6)
    entity_type: str = "victim"
    confidence: float = 0.9
    hits: int = 0
    misses: int = 0
    age: int = 0
    source_sensors: List[str] = field(default_factory=list)
    last_update: float = 0.0

    @property
    def pos(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.state[3:6].copy()

    def predict(self, dt: float) -> None:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        Q = np.eye(6) * 0.3 * max(dt, 1e-6)
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        self.age += 1

    def jpda_update(self, dets: List[Detection], weights: List[float]) -> None:
        """JPDA update: weighted combination of gated detections."""
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        z_bar = sum(w * d.pos for d, w in zip(dets, weights))
        R_bar = sum(w * d.cov for d, w in zip(dets, weights))
        R_bar = R_bar + np.eye(3) * 1e-6
        y = z_bar - H @ self.state
        S = H @ self.cov @ H.T + R_bar
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.cov = (np.eye(6) - K @ H) @ self.cov
        # Fuse classification by confidence-weighted vote.
        tot = sum(weights) + 1e-9
        type_scores: Dict[str, float] = {}
        for d, w in zip(dets, weights):
            type_scores[d.entity_type] = type_scores.get(d.entity_type, 0.0) + w * d.confidence
        if type_scores:
            self.entity_type = max(type_scores, key=type_scores.get)
            self.confidence = min(1.0, max(type_scores.values()) / tot)
        for d in dets:
            if d.sensor not in self.source_sensors:
                self.source_sensors.append(d.sensor)
        self.hits += 1
        self.misses = 0
        self.last_update = max((d.timestamp for d in dets), default=self.last_update)


class SensorFusion:
    """Multi-source fusion with JPDA gating + covariance-intersection init."""

    def __init__(
        self,
        gate_m: float = 4.0,
        max_misses: int = 5,
        confirm_hits: int = 2,
        beta_new: float = 0.1,
    ):
        self.gate_m = float(gate_m)
        self.max_misses = int(max_misses)
        self.confirm_hits = int(confirm_hits)
        self.beta_new = float(beta_new)  # new-target prior for JPDA
        self.tracks: Dict[int, FusedTrack] = {}
        self._next_id = 0

    # -- association ---------------------------------------------------
    def _gate(self, track: FusedTrack, det: Detection) -> Tuple[bool, float]:
        dist = float(np.linalg.norm(det.pos - track.state[:3]))
        pos_std = float(np.sqrt(max(np.trace(track.cov[:3, :3]) / 3.0, 1e-6)))
        det_std = float(np.sqrt(max(np.trace(det.cov) / 3.0, 1e-6)))
        gate = self.gate_m * (pos_std + det_std)
        return dist <= gate, dist

    def _jpda_weights(self, dists: List[float]) -> List[float]:
        """Soft-max over negative Mahalanobis-ish distances + new-target prior."""
        if not dists:
            return []
        scores = np.exp(-np.asarray(dists) / 2.0)
        denom = scores.sum() + self.beta_new
        return [float(s / denom) for s in scores]

    # -- lifecycle -----------------------------------------------------
    def _new_track(self, det: Detection) -> FusedTrack:
        state = np.zeros(6)
        state[:3] = det.pos
        cov = np.eye(6) * 2.0
        cov[:3, :3] = det.cov + np.eye(3) * 0.25
        tr = FusedTrack(
            track_id=self._next_id,
            state=state,
            cov=cov,
            entity_type=det.entity_type,
            confidence=det.confidence,
            hits=1,
            source_sensors=[det.sensor],
            last_update=det.timestamp,
        )
        self.tracks[self._next_id] = tr
        self._next_id += 1
        return tr

    def step(self, detections: List[Detection], dt: float) -> Dict[int, FusedTrack]:
        """Predict all tracks, JPDA-associate, update, spawn, prune."""
        for tr in self.tracks.values():
            tr.predict(max(dt, 1e-6))
        used = set()
        for tr in list(self.tracks.values()):
            gated = []
            for i, d in enumerate(detections):
                if i in used:
                    continue
                ok, dist = self._gate(tr, d)
                if ok:
                    gated.append((i, d, dist))
            if gated:
                dets = [g[1] for g in gated]
                weights = self._jpda_weights([g[2] for g in gated])
                tr.jpda_update(dets, weights)
                # All gated detections are consumed by this JPDA update
                # (multi-sensor views of one target must not spawn dupes).
                for j in [g[0] for g in gated]:
                    used.add(j)
            else:
                tr.misses += 1
        for i, d in enumerate(detections):
            if i not in used:
                # Only spawn if no existing track of the same type gates it
                # (protects against out-of-order track iteration).
                gated_elsewhere = False
                for tr in self.tracks.values():
                    ok, _ = self._gate(tr, d)
                    if ok and tr.entity_type == d.entity_type:
                        gated_elsewhere = True
                        break
                if not gated_elsewhere:
                    self._new_track(d)
        self._merge_close_tracks(threshold_m=0.8)
        dead = [t for t, tr in self.tracks.items() if tr.misses > self.max_misses]
        for t in dead:
            del self.tracks[t]
        return dict(self.tracks)

    def _merge_close_tracks(self, threshold_m: float = 0.8) -> None:
        """Merge same-type tracks within threshold (keeps lowest track id)."""
        ids = sorted(self.tracks.keys())
        merged = set()
        for i, a in enumerate(ids):
            if a in merged or a not in self.tracks:
                continue
            for b in ids[i + 1:]:
                if b in merged or b not in self.tracks:
                    continue
                ta, tb = self.tracks[a], self.tracks[b]
                if ta.entity_type != tb.entity_type:
                    continue
                if float(np.linalg.norm(ta.pos - tb.pos)) <= threshold_m:
                    # Fuse b into a: confidence-weighted state, union sensors.
                    wa = max(ta.confidence, 1e-6)
                    wb = max(tb.confidence, 1e-6)
                    ta.state = (wa * ta.state + wb * tb.state) / (wa + wb)
                    ta.confidence = min(1.0, max(ta.confidence, tb.confidence))
                    ta.hits += tb.hits
                    ta.misses = min(ta.misses, tb.misses)
                    for s in tb.source_sensors:
                        if s not in ta.source_sensors:
                            ta.source_sensors.append(s)
                    merged.add(b)
        for b in merged:
            del self.tracks[b]

    # -- metrics -------------------------------------------------------
    def track_purity(self, ground_truth: List[np.ndarray], tol_m: float = 1.0) -> float:
        """Fraction of confirmed tracks within tol of a ground-truth point."""
        confirmed = [t for t in self.tracks.values() if t.hits >= self.confirm_hits]
        if not confirmed:
            return 0.0
        good = 0
        for tr in confirmed:
            if any(float(np.linalg.norm(tr.pos - np.asarray(g).reshape(3))) <= tol_m for g in ground_truth):
                good += 1
        return good / len(confirmed)

    def false_track_rate(self, ground_truth: List[np.ndarray], tol_m: float = 1.0) -> float:
        return 1.0 - self.track_purity(ground_truth, tol_m)

    def confirmed_tracks(self) -> List[FusedTrack]:
        return [t for t in self.tracks.values() if t.hits >= self.confirm_hits]
