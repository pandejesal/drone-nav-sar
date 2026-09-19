#!/usr/bin/env python3
"""Mission Sequencer — waypoint + payload sequencing, CoG-aware (Sprint 18).

Sequences N drops to N locations; replans transit when CoG offset grows;
emergency jettison funnels to return_home. SAR-only outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DropTask:
    payload_id: str
    target: np.ndarray  # (3,)
    allen_s: float = 20.0  # budgeted leg time


class MissionSequencer:
    def __init__(self, home: Optional[np.ndarray] = None,
                 cruise_speed: float = 8.0, cog_limit_m: float = 0.03):
        self.home = np.asarray(home if home is not None else [0, 0, 50.0],
                               dtype=np.float64).reshape(3)
        self.cruise = float(cruise_speed)
        self.cog_limit = float(cog_limit_m)
        self.tasks: List[DropTask] = []
        self.cursor = 0
        self.elapsed = 0.0
        self.completed: List[str] = []

    def add_task(self, task: DropTask) -> None:
        self.tasks.append(task)

    def reset(self) -> None:
        self.cursor = 0
        self.elapsed = 0.0
        self.completed.clear()

    def advance_time(self, dt: float) -> None:
        self.elapsed += float(dt)

    def current(self) -> Optional[DropTask]:
        if self.cursor < len(self.tasks):
            return self.tasks[self.cursor]
        return None

    def mark_dropped(self) -> Optional[DropTask]:
        task = self.current()
        if task is None:
            return None
        self.completed.append(task.payload_id)
        self.cursor += 1
        return task

    def needs_replan(self, cog_offset_m: float) -> bool:
        return float(cog_offset_m) > self.cog_limit

    def leg_speed(self, cog_offset_m: float) -> float:
        """CoG-aware cruise derate: heavy asymmetry -> slower transit."""
        if self.needs_replan(cog_offset_m):
            return max(2.0, self.cruise * 0.6)
        return self.cruise

    def next_command(self, drone_pos: np.ndarray,
                     cog_offset_m: float = 0.0) -> Dict:
        """Next SAR-only skill: navigate_to leg target / drop / return_home."""
        task = self.current()
        if task is None:
            h = self.home
            return {"skill": "return_home",
                    "params": {"x": float(h[0]), "y": float(h[1]),
                               "z": float(h[2]), "speed_ms": self.leg_speed(cog_offset_m)},
                    "constraints": {"timeout_s": 60.0}, "done": True}
        d = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        t = np.asarray(task.target, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(d - t)) < 1.0:
            return {"skill": "drop_payload",
                    "params": {"x": float(t[0]), "y": float(t[1]),
                               "z": float(t[2]), "yaw_deg": 0.0,
                               "payload_id": task.payload_id},
                    "constraints": {"timeout_s": 5.0}, "done": False}
        return {"skill": "navigate_to",
                "params": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
                           "yaw_deg": 0.0,
                           "speed_ms": self.leg_speed(cog_offset_m)},
                "constraints": {"timeout_s": 60.0}, "done": False}

    def jettison_command(self) -> Dict:
        h = self.home
        return {"skill": "return_home",
                "params": {"x": float(h[0]), "y": float(h[1]),
                           "z": float(h[2]), "speed_ms": 3.0,
                           "jettison": True},
                "constraints": {"timeout_s": 60.0}, "done": False}

    def summary(self) -> Dict:
        return {"total": len(self.tasks), "done": len(self.completed),
                "cursor": self.cursor, "elapsed_s": self.elapsed}
