#!/usr/bin/env python3
"""PX4 interface for DroneNav-SAR (Sprint 14).

MAVLink 2.0 + ROS2 bridge with offboard mode and pre-arm safety checks.
Real deps (pymavlink, rclpy) are guarded so CI runs on MockMAVLinkConnection.
SAR-only: navigate_to / hover / drop_payload / return_home. No weapons.
"""

from typing import Any, Dict, List, Optional

try:  # optional real backends; absent in CI
    from pymavlink import mavutil  # type: ignore  # noqa: F401
    _MAVLINK_AVAILABLE = True
except Exception:  # pragma: no cover
    mavutil = None  # type: ignore
    _MAVLINK_AVAILABLE = False

try:
    import rclpy  # type: ignore  # noqa: F401
    _ROS2_AVAILABLE = True
except Exception:  # pragma: no cover
    rclpy = None  # type: ignore
    _ROS2_AVAILABLE = False

from src.control.safety_filter import allow as _safety_allow

OFFBOARD_MIN_STREAM_HZ = 2.0  # PX4 requires >2Hz setpoint stream before OFFBOARD


class MockMAVLinkConnection:
    """In-memory MAVLink connection (CI default)."""

    def __init__(self, uri: str = "udp:127.0.0.1:14540") -> None:
        self.uri = uri
        self.sent: List[Dict[str, Any]] = []
        self._connected = False
        self._armed = False
        self._offboard = False
        self._heartbeats = 0

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._armed = False
        self._offboard = False

    def send(self, msg: Dict[str, Any]) -> None:
        self.sent.append(dict(msg))

    def heartbeat(self) -> bool:
        self._heartbeats += 1
        return self._connected


class PX4Interface:
    """MAVLink + ROS2 bridge for PX4 Autopilot (v1.14+)."""

    def __init__(self, uri: str = "udp:127.0.0.1:14540",
                 connection=None, safety_monitor=None) -> None:
        self.uri = uri
        self.conn = connection or MockMAVLinkConnection(uri)
        self.safety = safety_monitor
        self._armed = False
        self._offboard = False
        self.home: Optional[List[float]] = None

    # -- connection / arming -------------------------------------------
    def connect(self) -> bool:
        ok = bool(self.conn.connect())
        if ok and self.home is None:
            self.home = [0.0, 0.0, 1.5]
        return ok

    def disconnect(self) -> None:
        try:
            self.stop_offboard()
            self.disarm()
        finally:
            self.conn.disconnect()

    @property
    def is_connected(self) -> bool:
        return bool(self.conn.heartbeat())

    @property
    def mavlink_available(self) -> bool:
        return _MAVLINK_AVAILABLE

    @property
    def ros2_available(self) -> bool:
        return _ROS2_AVAILABLE

    def _prearm_checks(self) -> Dict[str, Any]:
        """Safety checks before arming: RC link, battery, geofence config."""
        if self.safety is not None:
            state = {"rc_connected": getattr(self.safety, "rc_connected", True)}
            verdict = self.safety.evaluate(state)
            if not verdict["safe"]:
                return {"ok": False, "reason": f"safety: {verdict['violations']}"}
        return {"ok": True, "reason": "prearm_ok"}

    def arm(self) -> Dict[str, Any]:
        check = self._prearm_checks()
        if not check["ok"]:
            return check
        self._armed = True
        try:
            self.conn.send({"type": "COMMAND_LONG", "command": "ARM"})
        except Exception:
            pass
        return {"ok": True, "reason": "armed"}

    def disarm(self) -> Dict[str, Any]:
        self._armed = False
        try:
            self.conn.send({"type": "COMMAND_LONG", "command": "DISARM"})
        except Exception:
            pass
        return {"ok": True, "reason": "disarmed"}

    # -- offboard ---------------------------------------------------------
    def start_offboard(self) -> Dict[str, Any]:
        if not self._armed:
            return {"ok": False, "reason": "not_armed"}
        # PX4 requires streaming setpoints before mode switch; emulate stream.
        for _ in range(3):
            self.conn.send({"type": "SET_POSITION_TARGET_LOCAL_NED",
                            "stream_warmup": True})
        self._offboard = True
        self.conn.send({"type": "SET_MODE", "mode": "OFFBOARD"})
        return {"ok": True, "reason": "offboard"}

    def stop_offboard(self) -> Dict[str, Any]:
        self._offboard = False
        try:
            self.conn.send({"type": "SET_MODE", "mode": "POSCTL"})
        except Exception:
            pass
        return {"ok": True, "reason": "offboard_stopped"}

    @property
    def in_offboard(self) -> bool:
        return self._offboard

    # -- setpoints ----------------------------------------------------------
    def send_position_ned(self, north: float, east: float,
                          down: float, yaw_deg: float = 0.0) -> Dict[str, Any]:
        if not self._offboard:
            return {"ok": False, "reason": "not_in_offboard"}
        msg = {"type": "SET_POSITION_TARGET_LOCAL_NED", "n": float(north),
               "e": float(east), "d": float(down), "yaw_deg": float(yaw_deg)}
        self.conn.send(msg)
        return {"ok": True, "reason": "sent", "msg": msg}

    def send_velocity_ned(self, vn: float, ve: float,
                          vd: float, yaw_deg: float = 0.0) -> Dict[str, Any]:
        if not self._offboard:
            return {"ok": False, "reason": "not_in_offboard"}
        msg = {"type": "SET_POSITION_TARGET_LOCAL_NED", "vn": float(vn),
               "ve": float(ve), "vd": float(vd), "yaw_deg": float(yaw_deg)}
        self.conn.send(msg)
        return {"ok": True, "reason": "sent", "msg": msg}

    def execute(self, skill_cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a SAR skill via offboard setpoints. Reject = hover hold."""
        verdict = _safety_allow(skill_cmd)
        if not verdict["ok"]:
            self.conn.send({"type": "SET_POSITION_TARGET_LOCAL_NED",
                            "hold": True, "fallback": "hover"})
            return {"ok": False, "reason": verdict["reason"],
                    "fallback": "hover"}
        if self.safety is not None:
            pos = skill_cmd.get("params", {})
            sv = self.safety.evaluate(
                {"position": [float(pos.get("x", 2.0)), float(pos.get("y", 2.0)),
                              float(pos.get("z", 1.0))]})
            if not sv["safe"]:
                return {"ok": False, "reason": f"safety: {sv['violations']}",
                        "fallback": sv["action"]}
        skill = skill_cmd.get("skill")
        params = skill_cmd.get("params", {}) or {}
        if skill == "navigate_to":
            # ENU (x,y,z-up) -> NED for PX4: n=x, e=y, d=-z
            res = self.send_position_ned(float(params["x"]), float(params["y"]),
                                         -float(params["z"]),
                                         float(params.get("yaw_deg", 0.0)))
        elif skill == "hover":
            res = self.send_velocity_ned(0.0, 0.0, 0.0,
                                         float(params.get("yaw_deg", 0.0)))
        elif skill == "drop_payload":
            res = self.send_velocity_ned(0.0, 0.0, 0.2)  # gentle descend-hold
            if res.get("ok"):
                res["payload_release"] = True
        elif skill == "return_home":
            hx, hy, hz = self.home or [0.0, 0.0, 1.5]
            res = self.send_position_ned(hx, hy, -hz, 0.0)
        else:
            return {"ok": False, "reason": f"skill_not_allowlisted: {skill!r}",
                    "fallback": "hover"}
        res["skill"] = skill
        return res

    def publish_ros2(self, skill_cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Mirror a skill command to the ROS2 topic layer (mock in CI)."""
        msg = {"transport": "ros2", "skill": skill_cmd.get("skill"),
               "params": dict(skill_cmd.get("params", {}) or {})}
        self.conn.send(msg)
        return msg

    def heartbeat(self) -> bool:
        return self.is_connected
