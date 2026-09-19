#!/usr/bin/env python3
"""Safety monitor for DroneNav-SAR hardware ops (Sprint 14).

Non-negotiable, hardcoded limits (mirrors PX4 params + skill_schema.json):
- Geofence: 4 x 4 x 2.5 m  (x in [0,4], y in [0,4], z in [0.2,2.5])
- Altitude ceiling: 2.5 m max
- Battery failsafe: RTL at 3.3 V/cell
- RC override: always active; loss of RC kills offboard (-> RTL)
- Signal watchdog: heartbeat age > timeout triggers RTL
- Emergency stop: latching kill switch (physical button + software)

evaluate(state) -> {safe, violations, action}; action in
{"nominal", "hover", "rtl", "land", "kill"}. Fail closed.
"""

from typing import Any, Dict, List, Optional

GEOFENCE_X = (0.0, 4.0)
GEOFENCE_Y = (0.0, 4.0)
GEOFENCE_Z = (0.2, 2.5)
ALTITUDE_CEILING_M = 2.5
BATTERY_MIN_PER_CELL_V = 3.3


class SafetyMonitor:
    """Stateless-limit checker + latching e-stop / failsafe state."""

    def __init__(self,
                 geofence_x=GEOFENCE_X,
                 geofence_y=GEOFENCE_Y,
                 geofence_z=GEOFENCE_Z,
                 ceiling_m: float = ALTITUDE_CEILING_M,
                 battery_min_per_cell_v: float = BATTERY_MIN_PER_CELL_V,
                 cells: int = 1,
                 rc_timeout_s: float = 1.0,
                 signal_timeout_s: float = 1.0) -> None:
        self.geofence_x = tuple(geofence_x)
        self.geofence_y = tuple(geofence_y)
        self.geofence_z = tuple(geofence_z)
        self.ceiling_m = float(ceiling_m)
        self.battery_min_v = float(battery_min_per_cell_v) * int(cells)
        self.cells = int(cells)
        self.rc_timeout_s = float(rc_timeout_s)
        self.signal_timeout_s = float(signal_timeout_s)
        self.rc_connected = True
        self._estop_latched = False
        self._failsafe_reason: Optional[str] = None
        self.violations_log: List[Dict[str, Any]] = []

    # -- individual checks ------------------------------------------------
    def check_position(self, position) -> Dict[str, Any]:
        try:
            x, y, z = float(position[0]), float(position[1]), float(position[2])
        except Exception:
            return {"ok": False, "reason": "position_invalid"}
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = (
            self.geofence_x, self.geofence_y, self.geofence_z)
        if not (xlo <= x <= xhi and ylo <= y <= yhi and zlo <= z <= zhi):
            return {"ok": False, "reason": f"geofence_breach: {[x, y, z]}"}
        if z > self.ceiling_m:
            return {"ok": False, "reason": f"above_ceiling: {z} > {self.ceiling_m}"}
        return {"ok": True, "reason": "position_ok"}

    def check_battery(self, voltage_v: float) -> Dict[str, Any]:
        try:
            v = float(voltage_v)
        except Exception:
            return {"ok": False, "reason": "battery_invalid"}
        if v < self.battery_min_v:
            return {"ok": False,
                    "reason": f"battery_low: {v:.2f}V < {self.battery_min_v:.2f}V"}
        return {"ok": True, "reason": "battery_ok"}

    def check_rc(self, rc_connected: Optional[bool] = None,
                 rc_override: bool = False) -> Dict[str, Any]:
        connected = self.rc_connected if rc_connected is None else bool(rc_connected)
        if rc_override:
            return {"ok": False, "reason": "rc_override_active"}
        if not connected:
            return {"ok": False, "reason": "rc_link_loss"}
        return {"ok": True, "reason": "rc_ok"}

    def check_signal(self, heartbeat_age_s: float) -> Dict[str, Any]:
        try:
            age = float(heartbeat_age_s)
        except Exception:
            return {"ok": False, "reason": "signal_invalid"}
        if age > self.signal_timeout_s:
            return {"ok": False, "reason": f"signal_loss: age {age:.2f}s"}
        return {"ok": True, "reason": "signal_ok"}

    # -- combined evaluation -----------------------------------------------
    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a telemetry/skill state. Always fails closed.

        Accepted keys: position [x,y,z], battery_v, rc_connected,
        rc_override, heartbeat_age_s, estop.
        """
        if self._estop_latched or state.get("estop", False):
            self._estop_latched = True
            return {"safe": False, "violations": ["emergency_stop"],
                    "action": "kill"}
        violations: List[str] = []
        action = "nominal"

        if "position" in state:
            r = self.check_position(state["position"])
            if not r["ok"]:
                violations.append(r["reason"])
                action = "rtl"
        if "battery_v" in state:
            r = self.check_battery(state["battery_v"])
            if not r["ok"]:
                violations.append(r["reason"])
                action = "rtl"
        if "rc_connected" in state or "rc_override" in state:
            r = self.check_rc(state.get("rc_connected", self.rc_connected),
                              bool(state.get("rc_override", False)))
            if not r["ok"]:
                violations.append(r["reason"])
                action = "rtl"  # RC loss kills offboard -> RTL
        if "heartbeat_age_s" in state:
            r = self.check_signal(state["heartbeat_age_s"])
            if not r["ok"]:
                violations.append(r["reason"])
                action = "rtl"
        if self._failsafe_reason is not None and not violations:
            # Latched failsafe persists until explicit reset.
            return {"safe": False, "violations": [self._failsafe_reason],
                    "action": "rtl"}

        safe = len(violations) == 0
        if not safe:
            self._failsafe_reason = violations[0]
            self.violations_log.append({"violations": list(violations),
                                        "action": action})
            if action == "nominal":
                action = "hover"
        return {"safe": safe, "violations": violations, "action": action}

    # -- failsafe controls ----------------------------------------------------
    def trigger_failsafe(self, reason: str = "manual") -> Dict[str, Any]:
        self._failsafe_reason = str(reason)
        self.violations_log.append({"violations": [self._failsafe_reason],
                                    "action": "rtl"})
        return {"safe": False, "action": "rtl", "reason": self._failsafe_reason}

    def emergency_stop(self) -> Dict[str, Any]:
        self._estop_latched = True
        return {"safe": False, "action": "kill", "reason": "emergency_stop"}

    def reset(self) -> None:
        self._estop_latched = False
        self._failsafe_reason = None

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched
