#!/usr/bin/env python3
"""Payload Manager — registry + weight/CoG tracking + jettison (Sprint 18)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PayloadSpec:
    payload_id: str
    kind: str  # e.g. medkit, sensor, flare, chute
    weight_kg: float
    slot: int
    deploy_altitude: float = 50.0
    chute: bool = True


class PayloadManager:
    def __init__(self, empty_mass_kg: float = 2.0):
        self.empty_mass = float(empty_mass_kg)
        self.registry: Dict[str, PayloadSpec] = {}
        self.dropped: List[str] = []
        self.jettisoned: List[str] = []

    def register(self, spec: PayloadSpec) -> None:
        self.registry[spec.payload_id] = spec

    def drop(self, payload_id: str) -> Dict:
        spec = self.registry.get(payload_id)
        if spec is None:
            return {"ok": False, "reason": "unknown_payload"}
        self.dropped.append(payload_id)
        del self.registry[payload_id]
        return {"ok": True, "payload_id": payload_id, "slot": spec.slot}

    def emergency_jettison(self) -> Dict:
        ids = list(self.registry.keys())
        self.jettisoned.extend(ids)
        self.registry.clear()
        return {"ok": True, "jettisoned": ids}

    def total_mass(self) -> float:
        return self.empty_mass + sum(s.weight_kg for s in self.registry.values())

    def cog_offset(self, slot_radius: float = 0.12, n_slots: int = 6) -> np.ndarray:
        """CoG offset from asymmetric slot loading (mirrors carousel geometry)."""
        total = self.total_mass()
        mx = my = 0.0
        for s in self.registry.values():
            a = 2.0 * np.pi * (int(s.slot) % n_slots) / n_slots
            mx += s.weight_kg * slot_radius * np.cos(a)
            my += s.weight_kg * slot_radius * np.sin(a)
        return np.array([mx / max(1e-9, total), my / max(1e-9, total)])

    def remaining(self) -> List[str]:
        return list(self.registry.keys())

    def drop_skill(self, payload_id: str, x: float, y: float, z: float) -> Dict:
        """SAR-only drop_payload command for one registered payload."""
        return {"skill": "drop_payload",
                "params": {"x": float(x), "y": float(y), "z": float(z),
                           "yaw_deg": 0.0, "payload_id": payload_id},
                "constraints": {"timeout_s": 5.0}}
