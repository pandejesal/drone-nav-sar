#!/usr/bin/env python3
"""Sprint 20: COP Engine — entity registry, temporal sync, uncertainty.

Consumes FusedTracks from SensorFusion and maintains a time-aligned
Common Operating Picture. SAR-only entity types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class COPEntity:
    """Serialized COP entity (matches sprint entity schema)."""

    entity_id: str
    type: str
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    cov: np.ndarray  # (3,3) position covariance
    confidence: float = 0.9
    timestamp: float = 0.0
    source_sensors: List[str] = field(default_factory=list)

    @property
    def uncertainty(self) -> Dict[str, float]:
        pos_m = float(np.sqrt(max(np.trace(self.cov) / 3.0, 0.0)))
        return {"pos_m": pos_m, "vel_mps": pos_m * 0.4}

    def to_dict(self) -> Dict:
        u = self.uncertainty
        return {
            "entity_id": self.entity_id,
            "type": self.type,
            "state": {
                "pos": [float(v) for v in self.pos],
                "vel": [float(v) for v in self.vel],
                "cov": [[float(c) for c in row] for row in self.cov],
            },
            "classification": {"type": self.type, "confidence": float(self.confidence)},
            "timestamp": float(self.timestamp),
            "source_sensors": list(self.source_sensors),
            "uncertainty": u,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "COPEntity":
        st = d.get("state", {})
        return cls(
            entity_id=str(d.get("entity_id", "unknown")),
            type=str(d.get("type", d.get("classification", {}).get("type", "victim"))),
            pos=np.asarray(st.get("pos", [0, 0, 0]), dtype=np.float64).reshape(3),
            vel=np.asarray(st.get("vel", [0, 0, 0]), dtype=np.float64).reshape(3),
            cov=np.asarray(st.get("cov", np.eye(3).tolist()), dtype=np.float64).reshape(3, 3),
            confidence=float(d.get("classification", {}).get("confidence", 0.9)),
            timestamp=float(d.get("timestamp", 0.0)),
            source_sensors=list(d.get("source_sensors", [])),
        )


class COPEngine:
    """Entity registry with temporal alignment + uncertainty propagation."""

    def __init__(self, max_staleness_s: float = 2.0):
        self.max_staleness_s = float(max_staleness_s)
        self.entities: Dict[str, COPEntity] = {}
        self._track_to_entity: Dict[int, str] = {}
        self.last_update_time: float = 0.0
        self.last_latency_ms: float = 0.0

    def update_from_tracks(self, tracks, now: Optional[float] = None) -> Dict[str, COPEntity]:
        """Ingest fused tracks; extrapolate each to common time `now`."""
        t0 = time.perf_counter()
        now = time.time() if now is None else float(now)
        if hasattr(tracks, "values"):
            tracks = list(tracks.values())
        for tr in tracks:
            eid = self._track_to_entity.get(tr.track_id)
            if eid is None:
                eid = f"{tr.entity_type}_{tr.track_id:03d}"
                self._track_to_entity[tr.track_id] = eid
            dt = max(0.0, now - float(getattr(tr, "last_update", now)))
            # Temporal sync: constant-velocity extrapolation + covariance growth.
            pos = tr.state[:3] + tr.state[3:6] * dt
            cov3 = np.asarray(tr.cov[:3, :3]) + np.eye(3) * 0.1 * dt
            self.entities[eid] = COPEntity(
                entity_id=eid,
                type=tr.entity_type,
                pos=pos.astype(np.float64),
                vel=tr.state[3:6].astype(np.float64),
                cov=cov3.astype(np.float64),
                confidence=float(tr.confidence),
                timestamp=now,
                source_sensors=list(getattr(tr, "source_sensors", [])),
            )
        # Drop stale entities.
        stale = [k for k, e in self.entities.items() if now - e.timestamp > self.max_staleness_s]
        for k in stale:
            del self.entities[k]
        self.last_update_time = now
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return dict(self.entities)

    def snapshot(self) -> Dict:
        return {
            "timestamp": self.last_update_time,
            "entities": [e.to_dict() for e in self.entities.values()],
            "latency_ms": self.last_latency_ms,
        }

    @staticmethod
    def serialize(snapshot: Dict) -> Dict:
        return dict(snapshot)

    def victims(self, min_conf: float = 0.5) -> List[COPEntity]:
        return [e for e in self.entities.values() if e.type == "victim" and e.confidence >= min_conf]

    def __len__(self) -> int:
        return len(self.entities)
