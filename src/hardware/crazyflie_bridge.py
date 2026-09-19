#!/usr/bin/env python3
"""Crazyflie bridge for DroneNav-SAR (Sprint 14).

cflib wrapper with guarded import so CI runs on MockCrazyflieLink.
SAR-only: navigate_to / hover / drop_payload / return_home. No weapons.
Reject = hover in place (via safety_filter.allow).

Features: connect/disconnect, param sync, telemetry logging, setpoint send.
"""

from typing import Any, Dict, List, Optional

try:  # optional real backend; absent in CI
    import cflib  # type: ignore  # noqa: F401
    from cflib.crazyflie import Crazyflie  # type: ignore  # noqa: F401
    _CFLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    cflib = None  # type: ignore
    Crazyflie = None  # type: ignore
    _CFLIB_AVAILABLE = False

from src.control.safety_filter import allow as _safety_allow

SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")


class MockCrazyflieLink:
    """In-memory Crazyradio link recording setpoints + params (CI default)."""

    def __init__(self) -> None:
        self.setpoints: List[Dict[str, Any]] = []
        self.params: Dict[str, Any] = {}
        self._connected = False
        self._log: List[Dict[str, Any]] = []

    def connect(self, uri: str = "radio://0/80/2M") -> bool:
        self._connected = True
        self.uri = uri
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_setpoint(self, roll: float, pitch: float,
                      yawrate: float, thrust: int) -> Dict[str, Any]:
        sp = {"roll": float(roll), "pitch": float(pitch),
              "yawrate": float(yawrate), "thrust": int(thrust)}
        self.setpoints.append(sp)
        return sp

    def set_param(self, name: str, value: Any) -> None:
        self.params[name] = value

    def get_param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)


class CrazyflieBridge:
    """cflib wrapper: param sync, logging, SAR skill -> setpoint mapping."""

    def __init__(self, uri: str = "radio://0/80/2M", link=None) -> None:
        self.uri = uri
        self.link = link or MockCrazyflieLink()
        self._log_active = False
        self._log_buffer: List[Dict[str, Any]] = []
        # Default Crazyflie 2.1 params (synced to firmware on connect)
        self._params: Dict[str, Any] = {
            "stabilizer.estimator": 2,  # EKF2
            "commander.enHighLevel": 1,
            "loco.mode": 0,
        }

    # -- connection -----------------------------------------------------
    def connect(self) -> bool:
        ok = bool(self.link.connect(self.uri))
        if ok:
            self.sync_params(self._params)
        return ok

    def disconnect(self) -> None:
        self.stop_logging()
        self.link.disconnect()

    @property
    def is_connected(self) -> bool:
        return bool(self.link.is_connected)

    @property
    def cflib_available(self) -> bool:
        return _CFLIB_AVAILABLE

    # -- param sync ------------------------------------------------------
    def sync_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Push params to firmware link; returns echoed param dict."""
        for k, v in params.items():
            self._params[k] = v
            try:
                self.link.set_param(k, v)
            except Exception:
                pass
        return dict(self._params)

    def get_params(self) -> Dict[str, Any]:
        synced = {}
        for k in self._params:
            try:
                synced[k] = self.link.get_param(k, self._params[k])
            except Exception:
                synced[k] = self._params[k]
        return synced

    # -- logging ----------------------------------------------------------
    def start_logging(self, period_ms: int = 100) -> None:
        self._log_active = True
        self._log_period_ms = int(period_ms)

    def stop_logging(self) -> None:
        self._log_active = False

    def log_telemetry(self, entry: Dict[str, Any]) -> None:
        """Record one telemetry sample (stateEstimate, battery, rssi)."""
        if self._log_active:
            self._log_buffer.append(dict(entry))

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self._log_buffer)

    # -- setpoints ---------------------------------------------------------
    def send_setpoint(self, roll: float = 0.0, pitch: float = 0.0,
                      yawrate: float = 0.0, thrust: int = 0) -> Dict[str, Any]:
        thrust_i = int(max(0, min(60000, thrust)))
        return self.link.send_setpoint(float(roll), float(pitch),
                                       float(yawrate), thrust_i)

    def execute(self, skill_cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Map a SAR skill command to a Crazyflie setpoint. Reject -> hover."""
        verdict = _safety_allow(skill_cmd)
        if not verdict["ok"]:
            sp = self.send_setpoint(0.0, 0.0, 0.0, 10000)  # safe hover thrust
            return {"ok": False, "reason": verdict["reason"],
                    "fallback": "hover", "setpoint": sp}
        skill = skill_cmd.get("skill")
        params = skill_cmd.get("params", {}) or {}
        if skill == "navigate_to":
            # Proportional nudge toward target from params (mock-level mapping;
            # real deck uses high-level commander goto in production).
            dx = float(params.get("x", 2.0)) - 2.0
            dy = float(params.get("y", 2.0)) - 2.0
            pitch = max(-20.0, min(20.0, dx * 5.0))
            roll = max(-20.0, min(20.0, -dy * 5.0))
            sp = self.send_setpoint(roll, pitch, 0.0, 35000)
        elif skill == "hover":
            sp = self.send_setpoint(0.0, 0.0, 0.0, 35000)
        elif skill == "drop_payload":
            sp = self.send_setpoint(0.0, 0.0, 0.0, 30000)  # descend-hold, release
            sp["payload_release"] = True
        elif skill == "return_home":  # RTL: gentle level hold, yaw home
            sp = self.send_setpoint(0.0, 0.0, 30.0, 35000)
        else:  # fail closed (unreachable after allow(), belt + suspenders)
            sp = self.send_setpoint(0.0, 0.0, 0.0, 10000)
            return {"ok": False, "reason": f"skill_not_allowlisted: {skill!r}",
                    "fallback": "hover", "setpoint": sp}
        self.log_telemetry({"skill": skill, "setpoint": dict(sp)})
        return {"ok": True, "reason": "allow", "skill": skill, "setpoint": sp}

    def heartbeat(self) -> bool:
        return self.is_connected
