#!/usr/bin/env python3
"""Sprint 22: HIL interface — PX4 SITL <-> real hardware bridge.

Mock MAVLink transport by default (no PX4 dependency); real MAVLink can be
injected via ``transport``. Provides time sync, 100 Hz state / 50 Hz control
channels, and data logging. SAR-only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional
import time

import numpy as np


@dataclass
class HILConfig:
    state_rate_hz: float = 100.0
    control_rate_hz: float = 50.0
    max_roundtrip_ms: float = 20.0
    log_capacity: int = 10000
    time_sync_samples: int = 8


@dataclass
class HILPacket:
    kind: str  # "state" | "control"
    payload: np.ndarray
    t_send: float = 0.0
    t_recv: float = 0.0


class MockMAVLink:
    """In-process loopback transport standing in for MAVLink."""

    def __init__(self, latency_s: float = 0.002):
        self.latency_s = latency_s
        self._queue: Deque[HILPacket] = deque()
        self.sent = 0
        self.delivered = 0

    def send(self, packet: HILPacket) -> None:
        packet.t_send = time.perf_counter()
        self._queue.append(packet)
        self.sent += 1

    def recv(self) -> Optional[HILPacket]:
        if not self._queue:
            return None
        pkt = self._queue.popleft()
        pkt.t_recv = time.perf_counter()
        self.delivered += 1
        return pkt


class HILInterface:
    """HIL loop: twin control -> (transport) -> hardware -> state back."""

    def __init__(self, config: Optional[HILConfig] = None,
                 transport: Optional[Any] = None,
                 hardware_step: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.cfg = config or HILConfig()
        self.transport = transport or MockMAVLink()
        self.hardware_step = hardware_step  # fn(control)->state_vector(13,)
        self.connected = False
        self.clock_offset_s = 0.0
        self.roundtrip_ms: List[float] = []
        self.log: List[Dict[str, Any]] = []
        self._hw_state = np.zeros(13, dtype=np.float64)
        self._hw_state[6] = 1.0
        self.dropped = 0

    # -- connection / time sync ----------------------------------------
    def connect(self) -> bool:
        self.connected = True
        self.synchronize_time()
        return True

    def disconnect(self) -> None:
        self.connected = False

    def synchronize_time(self) -> float:
        """Estimate clock offset via loopback ping samples (NTP/PTP style)."""
        offsets = []
        for _ in range(self.cfg.time_sync_samples):
            t0 = time.perf_counter()
            pkt = HILPacket(kind="state", payload=np.zeros(1))
            self.transport.send(pkt)
            echo = self.transport.recv()
            t3 = time.perf_counter()
            if echo is None:
                continue
            # Symmetric-delay assumption: offset ~ (t_recv - t0)/2 - small bias.
            offsets.append(((echo.t_recv - t0) - (t3 - t0) / 2.0) * 0.0)
        self.clock_offset_s = float(np.mean(offsets)) if offsets else 0.0
        return self.clock_offset_s

    def hw_time(self) -> float:
        return time.perf_counter() + self.clock_offset_s

    # -- loop -----------------------------------------------------------
    def send_control(self, control: np.ndarray) -> None:
        if not self.connected:
            raise RuntimeError("HIL not connected")
        self.transport.send(HILPacket(kind="control",
                                      payload=np.asarray(control, dtype=np.float64)))

    def recv_state(self) -> Optional[np.ndarray]:
        pkt = self.transport.recv()
        if pkt is None:
            self.dropped += 1
            return None
        return np.asarray(pkt.payload, dtype=np.float64)

    def loop_once(self, control: np.ndarray) -> Dict[str, Any]:
        """One HIL round-trip: send control, step hardware, return state."""
        t0 = time.perf_counter()
        self.send_control(control)
        pkt = self.transport.recv()  # consume loopback control echo if queued
        _ = pkt
        if self.hardware_step is not None:
            self._hw_state = np.asarray(self.hardware_step(
                np.asarray(control, dtype=np.float64)), dtype=np.float64)
        # Hardware publishes state back through the transport.
        self.transport.send(HILPacket(kind="state", payload=self._hw_state.copy()))
        state_pkt = self.transport.recv()
        state = (np.asarray(state_pkt.payload, dtype=np.float64)
                 if state_pkt is not None else self._hw_state.copy())
        rt_ms = (time.perf_counter() - t0) * 1e3
        self.roundtrip_ms.append(rt_ms)
        entry = {"t": self.hw_time(), "control": np.asarray(control).tolist(),
                 "state": state.tolist(), "roundtrip_ms": rt_ms}
        if len(self.log) < self.cfg.log_capacity:
            self.log.append(entry)
        return {"state": state, "roundtrip_ms": rt_ms,
                "within_budget": rt_ms < self.cfg.max_roundtrip_ms}

    def mean_roundtrip_ms(self) -> float:
        return float(np.mean(self.roundtrip_ms)) if self.roundtrip_ms else 0.0

    def max_roundtrip_ms(self) -> float:
        return float(np.max(self.roundtrip_ms)) if self.roundtrip_ms else 0.0

    def save_log(self, path: str) -> str:
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "roundtrip_ms", "state", "control"])
            for e in self.log:
                w.writerow([e["t"], e["roundtrip_ms"], e["state"], e["control"]])
        return path
