#!/usr/bin/env python3
"""ROS 2 bridge alternative for DroneNav-SAR (Sprint C1).

Same filtered-skill contract as mavlink_bridge; rclpy guarded so CI
runs on MockTransport. Mock transport in CI, real node on hardware.
"""

from typing import Dict, List

try:  # optional real backend (R11); absent in CI
    import rclpy  # type: ignore  # noqa: F401
    _REAL_AVAILABLE = True
except Exception:  # pragma: no cover
    rclpy = None  # type: ignore
    _REAL_AVAILABLE = False


class MockTransport:
    """In-memory transport recording published skill messages."""

    def __init__(self) -> None:
        self.sent: List[Dict] = []
        self._heartbeats = 0

    def send(self, msg: Dict) -> None:
        self.sent.append(dict(msg))

    def heartbeat(self) -> bool:
        self._heartbeats += 1
        return True


class Ros2Bridge:
    """Filtered-skill -> ROS 2 message bridge (mock transport default)."""

    def __init__(self, transport=None) -> None:
        self.transport = transport or MockTransport()

    def execute(self, skill_cmd: Dict) -> Dict:
        skill = skill_cmd.get("skill")
        params = skill_cmd.get("params", {})
        msg = {"transport": "ros2", "skill": skill, "params": dict(params)}
        self.transport.send(msg)
        return msg

    def heartbeat(self) -> bool:
        return bool(self.transport.heartbeat())

    @property
    def real_available(self) -> bool:
        return _REAL_AVAILABLE
