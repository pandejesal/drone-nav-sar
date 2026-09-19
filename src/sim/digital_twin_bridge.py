#!/usr/bin/env python3
"""Sprint 22: Twin <-> hardware bridge with state sync + latency compensation.

Predicts hardware state forward by the measured HIL latency so the twin
compares against the *current* (not stale) airframe state. SAR-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.sim.digital_twin import DigitalTwin, TwinConfig, TwinState
from src.sim.hil_interface import HILConfig, HILInterface


@dataclass
class BridgeConfig:
    compensate_latency: bool = True
    max_latency_s: float = 0.020  # 20 ms budget
    control_clip: float = 5.0     # m/s^2 per axis


class DigitalTwinBridge:
    """Couples a DigitalTwin to hardware through a HILInterface."""

    def __init__(self, twin: Optional[DigitalTwin] = None,
                 hil: Optional[HILInterface] = None,
                 config: Optional[BridgeConfig] = None):
        self.twin = twin or DigitalTwin(TwinConfig())
        self.hil = hil or HILInterface(HILConfig())
        self.cfg = config or BridgeConfig()
        self.cycle_log: List[Dict[str, Any]] = []

    def connect(self) -> bool:
        return self.hil.connect()

    def compensate(self, hw_state: TwinState, latency_s: float) -> TwinState:
        """Constant-velocity forward prediction over latency horizon."""
        if not self.cfg.compensate_latency:
            return hw_state
        dt = float(np.clip(latency_s, 0.0, self.cfg.max_latency_s))
        pred = TwinState(position=hw_state.position + hw_state.velocity * dt,
                         velocity=hw_state.velocity.copy(),
                         orientation=hw_state.orientation.copy(),
                         angular_velocity=hw_state.angular_velocity.copy(),
                         timestamp=hw_state.timestamp + dt)
        return pred

    def cycle(self, control_accel: np.ndarray) -> Dict[str, Any]:
        u = np.clip(np.asarray(control_accel, dtype=np.float64).reshape(3),
                    -self.cfg.control_clip, self.cfg.control_clip)
        # 1. advance twin physics
        self.twin.step(u)
        # 2. HIL round-trip: control out, hardware state back
        res = self.hil.loop_once(u)
        hw = TwinState.from_array(np.asarray(res["state"], dtype=np.float64))
        latency_s = float(res["roundtrip_ms"]) / 1e3
        # 3. latency-compensated sync of twin toward hardware
        pred_hw = self.compensate(hw, latency_s)
        sync = self.twin.sync_from_hardware(pred_hw)
        pos_err, vel_err = self.twin.sync_error(pred_hw)
        entry = {"roundtrip_ms": res["roundtrip_ms"],
                 "within_budget": res["within_budget"],
                 "pos_err_m": pos_err, "vel_err": vel_err,
                 "sync": sync}
        self.cycle_log.append(entry)
        return entry

    def run(self, controls: List[np.ndarray]) -> Dict[str, Any]:
        for u in controls:
            self.cycle(u)
        pos_errs = [e["pos_err_m"] for e in self.cycle_log]
        rts = [e["roundtrip_ms"] for e in self.cycle_log]
        return {"cycles": len(self.cycle_log),
                "mean_sync_error_m": float(np.mean(pos_errs)) if pos_errs else 0.0,
                "max_sync_error_m": float(np.max(pos_errs)) if pos_errs else 0.0,
                "mean_roundtrip_ms": float(np.mean(rts)) if rts else 0.0,
                "max_roundtrip_ms": float(np.max(rts)) if rts else 0.0,
                "all_within_budget": all(e["within_budget"] for e in self.cycle_log)}
