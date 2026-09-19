#!/usr/bin/env python3
"""Strike Coordinator — target -> plan -> execute -> confirm (shared infra).

SAR-only: navigate_to / hover / drop_payload / return_home. No weapons.
Orchestrates tracker -> intercept planner -> terminal guidance -> deployment
-> carousel/manager/sequencer, gated by the extended safety filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")


class StrikePhase(Enum):
    IDLE = "idle"
    TRACK = "track"
    PLAN = "plan"
    EXECUTE = "execute"
    CONFIRM = "confirm"
    DONE = "done"
    ABORT = "abort"


class StrikeCoordinator:
    def __init__(self, tracker=None, planner=None, guidance=None,
                 deploy=None, wind=None, contact=None, carousel=None,
                 manager=None, sequencer=None, home=None):
        # Lazy imports keep the module importable without the full stack.
        from src.perception.target_tracker import TargetTracker
        from src.rl.intercept_planner import InterceptPlanner
        from src.control.terminal_guidance import TerminalGuidance
        self.tracker = tracker or TargetTracker()
        self.planner = planner or InterceptPlanner()
        self.guidance = guidance or TerminalGuidance()
        self.deploy = deploy
        self.wind = wind
        self.contact = contact
        self.carousel = carousel
        self.manager = manager
        self.sequencer = sequencer
        self.home = np.asarray(home if home is not None else [0, 0, 50.0],
                               dtype=np.float64).reshape(3)
        self.phase = StrikePhase.IDLE
        self.last_plan = None
        self.log: List[Dict] = []

    def reset(self) -> None:
        self.phase = StrikePhase.IDLE
        self.last_plan = None
        self.log.clear()

    def _sar(self, skill: str, params: Dict, timeout: float = 20.0) -> Dict:
        assert skill in SAR_SKILLS, f"non-SAR skill blocked: {skill}"
        return {"skill": skill, "params": dict(params),
                "constraints": {"timeout_s": float(timeout)}}

    def update(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
               detections: List[np.ndarray], dt: float,
               wind_vec: Optional[np.ndarray] = None,
               obstacles: Optional[List[np.ndarray]] = None) -> Dict:
        d = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        dv = np.asarray(drone_vel, dtype=np.float64).reshape(3)
        if self.phase == StrikePhase.IDLE:
            self.phase = StrikePhase.TRACK
        # TRACK
        assoc = self.tracker.update(list(detections), dt)
        if not self.tracker.tracks:
            h = self.home
            return self._sar("hover", {"z": float(d[2]), "yaw_deg": 0.0})
        tid = self.tracker.best_track_for(d)
        tr = self.tracker.tracks[tid]
        # PLAN
        self.phase = StrikePhase.PLAN
        plan = self.planner.plan(d, tr.state[:3], tr.velocity, wind_vec,
                                 obstacles or [])
        self.last_plan = plan
        self.phase = StrikePhase.EXECUTE
        # EXECUTE: terminal correction near release, else transit.
        dist = float(np.linalg.norm(d - plan.release_point))
        if dist < 8.0:
            cmd = self.guidance.corrected_goal(d, dv, plan.target_pred,
                                               tr.velocity, dt)
            cmd_skill = self._sar("hover", cmd["params"], 5.0)
            cmd_skill["t_go"] = plan.t_go
            self.log.append({"phase": "execute-terminal", "dist": dist})
            return cmd_skill
        r = plan.release_point
        self.log.append({"phase": "execute-transit", "dist": dist})
        out = self._sar("navigate_to",
                        {"x": float(r[0]), "y": float(r[1]), "z": float(r[2]),
                         "yaw_deg": 0.0, "speed_ms": self.planner.drone_speed})
        out["t_go"] = plan.t_go
        return out

    def confirm_drop(self, landing_pos: np.ndarray,
                     target_pos: np.ndarray) -> Dict:
        err = float(np.linalg.norm(np.asarray(landing_pos).reshape(3)
                                   - np.asarray(target_pos).reshape(3)))
        self.phase = StrikePhase.CONFIRM
        ok = err < 1.5
        if ok:
            self.phase = StrikePhase.DONE
        return {"confirmed": bool(ok), "cep_m": err,
                "phase": self.phase.value}

    def abort(self) -> Dict:
        self.phase = StrikePhase.ABORT
        h = self.home
        return self._sar("return_home",
                         {"x": float(h[0]), "y": float(h[1]), "z": float(h[2]),
                          "speed_ms": 3.0}, 60.0)
