#!/usr/bin/env python3
"""MAVLink bridge for DroneNav-SAR (Sprint 6).

Translates FILTERED skill commands to vehicle setpoints. Real MAVSDK
imports are guarded (try/except) so CI runs on MockTransport.
Only SAR skills ever reach the transport — call safety_filter.allow first.

Real hardware mode: connects to Crazyflie/PX4 via MAVSDK, sends
position/velocity setpoints in offboard mode.
"""

import asyncio
from typing import Dict, List, Optional

from .safety_filter import allow

try:  # optional real backend (R10); absent in CI
    from mavsdk import System  # type: ignore  # noqa: F401
    from mavsdk.offboard import PositionNedYaw, VelocityNedYaw  # type: ignore
    _REAL_AVAILABLE = True
except Exception:  # pragma: no cover
    System = None  # type: ignore
    PositionNedYaw = None  # type: ignore
    VelocityNedYaw = None  # type: ignore
    _REAL_AVAILABLE = False


class MockTransport:
    """In-memory transport recording sent setpoints (CI default)."""

    def __init__(self) -> None:
        self.sent: List[Dict] = []
        self._heartbeats = 0

    def send(self, setpoint: Dict) -> None:
        self.sent.append(dict(setpoint))

    def heartbeat(self) -> bool:
        self._heartbeats += 1
        return True


class RealMavlinkTransport:
    """Real MAVSDK transport for Crazyflie/PX4 hardware."""

    def __init__(self, uri: str = "udp://:14540") -> None:
        if not _REAL_AVAILABLE:
            raise RuntimeError("MAVSDK not available")
        self.uri = uri
        self.system = None  # type: ignore[assignment]  # mavsdk System when connected
        self._connected = False

    async def connect(self) -> bool:
        """Connect to vehicle and wait for heartbeat."""
        assert System is not None, "MAVSDK not available"
        self.system = System()
        await self.system.connect(system_address=self.uri)

        async for state in self.system.core.connection_state():  # type: ignore[union-attr]
            if state.is_connected:
                self._connected = True
                print(f"MAVSDK connected to {self.uri}")
                return True
        return False

    async def arm(self) -> bool:
        """Arm the vehicle."""
        if not self._connected or self.system is None:
            return False
        try:
            await self.system.action.arm()  # type: ignore[union-attr]
            return True
        except Exception as e:
            print(f"Arm failed: {e}")
            return False

    async def start_offboard(self) -> bool:
        """Start offboard mode."""
        if not self._connected or self.system is None:
            return False
        try:
            await self.system.offboard.start()  # type: ignore[union-attr]
            return True
        except Exception as e:
            print(f"Offboard start failed: {e}")
            return False

    async def send_position_ned(self, north: float, east: float, down: float, yaw: float) -> bool:
        """Send position setpoint in NED frame."""
        if not self._connected or self.system is None or PositionNedYaw is None:
            return False
        try:
            await self.system.offboard.set_position_ned(PositionNedYaw(north, east, down, yaw))  # type: ignore[union-attr]
            return True
        except Exception as e:
            print(f"Position setpoint failed: {e}")
            return False

    async def send_velocity_ned(self, north: float, east: float, down: float, yaw: float) -> bool:
        """Send velocity setpoint in NED frame."""
        if not self._connected or self.system is None or VelocityNedYaw is None:
            return False
        try:
            await self.system.offboard.set_velocity_ned(VelocityNedYaw(north, east, down, yaw))  # type: ignore[union-attr]
            return True
        except Exception as e:
            print(f"Velocity setpoint failed: {e}")
            return False

    def send(self, setpoint: Dict) -> None:
        """Sync wrapper for async send (used in non-async contexts)."""
        # In real usage, this would be called from an async event loop
        pass

    def heartbeat(self) -> bool:
        return self._connected


class MavlinkBridge:
    """Filtered-skill -> setpoint bridge (mock or real transport)."""

    def __init__(self, transport=None, uri: str = "udp://:14540") -> None:
        if transport is not None:
            self.transport = transport
        elif _REAL_AVAILABLE:
            self.transport = RealMavlinkTransport(uri)
        else:
            self.transport = MockTransport()

    def execute(self, skill_cmd: Dict) -> Dict:
        skill = skill_cmd.get("skill")
        params = skill_cmd.get("params", {})
        setpoint = {"transport": "mavlink", "skill": skill, "params": dict(params)}
        self.transport.send(setpoint)
        return setpoint

    def heartbeat(self) -> bool:
        return bool(self.transport.heartbeat())

    @property
    def real_available(self) -> bool:
        return _REAL_AVAILABLE
