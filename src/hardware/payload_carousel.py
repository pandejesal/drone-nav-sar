#!/usr/bin/env python3
"""Payload Carousel — 6-slot rotary mechanism (Sprint 18).

Servo/stepper indexing, presence + weight sensing per slot, CoG tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

N_SLOTS = 6


@dataclass
class Slot:
    index: int
    angle_deg: float
    present: bool = False
    weight_kg: float = 0.0
    label: str = ""


class PayloadCarousel:
    def __init__(self, radius_m: float = 0.12, tare_kg: float = 0.4):
        self.radius = float(radius_m)
        self.tare = float(tare_kg)
        self.slots: List[Slot] = [
            Slot(i, i * 360.0 / N_SLOTS) for i in range(N_SLOTS)]
        self.active_slot = 0
        self.rotations = 0

    def load(self, slot: int, weight_kg: float, label: str = "") -> Dict:
        s = self.slots[int(slot) % N_SLOTS]
        s.present = True
        s.weight_kg = float(weight_kg)
        s.label = str(label)
        return {"ok": True, "slot": s.index}

    def rotate_to(self, slot: int) -> Dict:
        target = int(slot) % N_SLOTS
        steps = (target - self.active_slot) % N_SLOTS
        self.active_slot = target
        self.rotations += 1
        return {"ok": True, "active_slot": self.active_slot,
                "steps": steps,
                "angle_deg": self.slots[target].angle_deg}

    def next_full_slot(self) -> Optional[int]:
        for k in range(N_SLOTS):
            idx = (self.active_slot + k) % N_SLOTS
            if self.slots[idx].present:
                return idx
        return None

    def release_active(self) -> Dict:
        s = self.slots[self.active_slot]
        if not s.present:
            return {"ok": False, "reason": "slot_empty",
                    "slot": self.active_slot}
        w = s.weight_kg
        s.present = False
        s.weight_kg = 0.0
        label = s.label
        s.label = ""
        return {"ok": True, "slot": self.active_slot, "weight_kg": w,
                "label": label}

    def cog(self) -> np.ndarray:
        """Planar CoG offset (x,y) from slot weights; tare at center."""
        total = self.tare + sum(s.weight_kg for s in self.slots)
        if total <= 1e-9:
            return np.zeros(2)
        mx = my = 0.0
        for s in self.slots:
            a = np.deg2rad(s.angle_deg)
            mx += s.weight_kg * self.radius * np.cos(a)
            my += s.weight_kg * self.radius * np.sin(a)
        return np.array([mx / total, my / total])

    def total_weight(self) -> float:
        return self.tare + sum(s.weight_kg for s in self.slots)

    def status(self) -> Dict:
        return {"active_slot": self.active_slot,
                "present": [s.present for s in self.slots],
                "total_weight_kg": self.total_weight(),
                "cog_m": [float(v) for v in self.cog()]}
