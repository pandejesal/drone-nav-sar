#!/usr/bin/env python3
"""Sprint 33: HIL Orchestrator - PX4 SITL <-> real hardware, time sync, data logging."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import time
import threading
import uuid

import numpy as np

from src.sim.hil_interface import HILInterface, HILConfig, MockMAVLink, HILPacket
from src.sim.digital_twin import DigitalTwin, TwinConfig, TwinState


@dataclass
class DroneHILConfig:
    drone_id: int
    transport_type: str = "mock"
    transport_config: Dict[str, Any] = field(default_factory=dict)
    hil_config: Optional[HILConfig] = None
    twin_config: Optional[TwinConfig] = None
    hardware_step_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass
class HILMetrics:
    drone_id: int
    mean_roundtrip_ms: float
    max_roundtrip_ms: float
    mean_sync_error_m: float
    max_sync_error_m: float
    packets_sent: int
    packets_delivered: int
    packets_dropped: int
    clock_offset_s: float
    within_budget_rate: float


class TimeSyncManager:
    def __init__(self, max_drift_ms: float = 1.0, sync_interval_s: float = 10.0):
        self.max_drift_ms = max_drift_ms
        self.sync_interval_s = sync_interval_s
        self.last_sync_time = 0.0
        self.clock_offsets: Dict[int, float] = {}
        self.drift_history: Dict[int, List[float]] = {}

    def synchronize_all(self, interfaces: Dict[int, HILInterface]) -> Dict[int, float]:
        offsets = {}
        for drone_id, hil in interfaces.items():
            offset = hil.synchronize_time()
            self.clock_offsets[drone_id] = offset
            offsets[drone_id] = offset
            if drone_id not in self.drift_history:
                self.drift_history[drone_id] = []
            self.drift_history[drone_id].append(offset)
        self.last_sync_time = time.perf_counter()
        return offsets

    def check_drift(self, interfaces: Dict[int, HILInterface]) -> Dict[int, bool]:
        drift_exceeded = {}
        for drone_id, hil in interfaces.items():
            new_offset = hil.synchronize_time()
            old_offset = self.clock_offsets.get(drone_id, 0.0)
            drift_ms = abs(new_offset - old_offset) * 1000
            drift_exceeded[drone_id] = drift_ms > self.max_drift_ms
            if drift_exceeded[drone_id]:
                self.clock_offsets[drone_id] = new_offset
        return drift_exceeded

    def get_master_time(self, interfaces: Dict[int, HILInterface]) -> float:
        times = [hil.hw_time() for hil in interfaces.values()]
        return float(np.mean(times)) if times else time.perf_counter()


class DataLogger:
    def __init__(self, log_dir: str = "logs/hil", max_entries_per_drone: int = 100000):
        self.log_dir = log_dir
        self.max_entries = max_entries_per_drone
        self.logs: Dict[int, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def log_entry(self, drone_id: int, entry: Dict[str, Any]) -> None:
        with self._lock:
            if drone_id not in self.logs:
                self.logs[drone_id] = []
            if len(self.logs[drone_id]) < self.max_entries:
                self.logs[drone_id].append(entry)

    def log_batch(self, drone_id: int, entries: List[Dict[str, Any]]) -> None:
        with self._lock:
            if drone_id not in self.logs:
                self.logs[drone_id] = []
            remaining = self.max_entries - len(self.logs[drone_id])
            if remaining > 0:
                self.logs[drone_id].extend(entries[:remaining])

    def save_logs(self, prefix: str = "hil") -> Dict[int, str]:
        import csv
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        paths = {}
        with self._lock:
            for drone_id, entries in self.logs.items():
                if not entries:
                    continue
                path = os.path.join(self.log_dir, "{}_drone_{}_{}.csv".format(prefix, drone_id, uuid.uuid4().hex[:8]))
                with open(path, "w", newline="") as f:
                    if entries:
                        writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                        writer.writeheader()
                        writer.writerows(entries)
                paths[drone_id] = path
        return paths

    def get_log_stats(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            return {
                drone_id: {
                    "entries": len(entries),
                    "duration_s": entries[-1].get("t", 0) - entries[0].get("t", 0) if entries else 0,
                }
                for drone_id, entries in self.logs.items()
            }


class HILOrchestrator:
    def __init__(
        self,
        drone_configs: List[DroneHILConfig],
        time_sync_config: Optional[Dict[str, Any]] = None,
        log_config: Optional[Dict[str, Any]] = None,
    ):
        self.drone_configs = {c.drone_id: c for c in drone_configs}
        self.interfaces: Dict[int, HILInterface] = {}
        self.twins: Dict[int, DigitalTwin] = {}
        self.hardware_steps: Dict[int, Callable[[np.ndarray], np.ndarray]] = {}

        sync_cfg = time_sync_config or {}
        self.time_sync = TimeSyncManager(
            max_drift_ms=sync_cfg.get("max_drift_ms", 1.0),
            sync_interval_s=sync_cfg.get("sync_interval_s", 10.0),
        )

        log_cfg = log_config or {}
        self.logger = DataLogger(
            log_dir=log_cfg.get("log_dir", "logs/hil"),
            max_entries_per_drone=log_cfg.get("max_entries", 100000),
        )

        self.connected = False
        self.running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())[:8]

        self._initialize_hardware()

    def _initialize_hardware(self) -> None:
        for drone_id, config in self.drone_configs.items():
            if config.transport_type == "mock":
                latency = config.transport_config.get("latency_s", 0.002)
                transport = MockMAVLink(latency_s=latency)
            elif config.transport_type == "mavlink":
                transport = MockMAVLink(latency_s=0.005)
            elif config.transport_type == "px4_sitl":
                transport = MockMAVLink(latency_s=0.01)
            else:
                transport = MockMAVLink()

            hil_config = config.hil_config or HILConfig()
            hardware_step = config.hardware_step_fn
            hil = HILInterface(hil_config, transport=transport, hardware_step=hardware_step)
            self.interfaces[drone_id] = hil
            self.hardware_steps[drone_id] = hardware_step or (lambda u: np.zeros(13))

            twin_config = config.twin_config or TwinConfig()
            self.twins[drone_id] = DigitalTwin(twin_config)

    def connect_all(self) -> bool:
        all_connected = True
        for drone_id, hil in self.interfaces.items():
            if not hil.connect():
                all_connected = False
        if all_connected:
            self.time_sync.synchronize_all(self.interfaces)
            self.connected = True
        return all_connected

    def disconnect_all(self) -> None:
        for hil in self.interfaces.values():
            hil.disconnect()
        self.connected = False

    def start_loop(self, control_provider: Callable[[int], np.ndarray], rate_hz: float = 50.0) -> None:
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._control_loop,
            args=(control_provider, rate_hz),
            daemon=True,
        )
        self._loop_thread.start()

    def stop_loop(self) -> None:
        self.running = False
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)

    def _control_loop(self, control_provider: Callable[[int], np.ndarray], rate_hz: float) -> None:
        dt = 1.0 / rate_hz
        next_sync = time.perf_counter() + self.time_sync.sync_interval_s
        while self.running and not self._stop_event.is_set():
            loop_start = time.perf_counter()
            if loop_start >= next_sync:
                self.time_sync.synchronize_all(self.interfaces)
                next_sync = loop_start + self.time_sync.sync_interval_s
            for drone_id, hil in self.interfaces.items():
                try:
                    control = control_provider(drone_id)
                    result = hil.loop_once(control)
                    hw_state = TwinState.from_array(result["state"], timestamp=hil.hw_time())
                    sync_result = self.twins[drone_id].sync_from_hardware(hw_state)
                    log_entry = {
                        "t": hil.hw_time(),
                        "drone_id": drone_id,
                        "control": control.tolist(),
                        "state": result["state"].tolist(),
                        "roundtrip_ms": result["roundtrip_ms"],
                        "within_budget": result["within_budget"],
                        "sync_pos_err_m": sync_result["pos_err_after_m"],
                        "sync_vel_err": sync_result["vel_err_after"],
                    }
                    self.logger.log_entry(drone_id, log_entry)
                except Exception as e:
                    self.logger.log_entry(drone_id, {
                        "t": time.perf_counter(),
                        "drone_id": drone_id,
                        "error": str(e),
                    })
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def step_once(self, controls: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        results = {}
        for drone_id, hil in self.interfaces.items():
            control = controls.get(drone_id, np.zeros(4))
            result = hil.loop_once(control)
            hw_state = TwinState.from_array(result["state"], timestamp=hil.hw_time())
            sync_result = self.twins[drone_id].sync_from_hardware(hw_state)
            results[drone_id] = {
                "state": result["state"],
                "roundtrip_ms": result["roundtrip_ms"],
                "within_budget": result["within_budget"],
                "sync_pos_err_m": sync_result["pos_err_after_m"],
                "sync_vel_err": sync_result["vel_err_after"],
            }
            self.logger.log_entry(drone_id, {
                "t": hil.hw_time(),
                "drone_id": drone_id,
                "control": control.tolist(),
                "state": result["state"].tolist(),
                "roundtrip_ms": result["roundtrip_ms"],
                "within_budget": result["within_budget"],
                "sync_pos_err_m": sync_result["pos_err_after_m"],
                "sync_vel_err": sync_result["vel_err_after"],
            })
        return results

    def get_metrics(self) -> Dict[int, HILMetrics]:
        metrics = {}
        for drone_id, hil in self.interfaces.items():
            twin = self.twins[drone_id]
            roundtrips = hil.roundtrip_ms
            sync_errors = twin.sync_errors_m
            within_budget = sum(1 for rt in roundtrips if rt < hil.cfg.max_roundtrip_ms)
            total = len(roundtrips)
            metrics[drone_id] = HILMetrics(
                drone_id=drone_id,
                mean_roundtrip_ms=float(np.mean(roundtrips)) if roundtrips else 0.0,
                max_roundtrip_ms=float(np.max(roundtrips)) if roundtrips else 0.0,
                mean_sync_error_m=float(np.mean(sync_errors)) if sync_errors else 0.0,
                max_sync_error_m=float(np.max(sync_errors)) if sync_errors else 0.0,
                packets_sent=hil.transport.sent if hasattr(hil.transport, 'sent') else 0,
                packets_delivered=hil.transport.delivered if hasattr(hil.transport, 'delivered') else 0,
                packets_dropped=hil.dropped,
                clock_offset_s=hil.clock_offset_s,
                within_budget_rate=within_budget / total if total > 0 else 0.0,
            )
        return metrics

    def get_twin_states(self) -> Dict[int, TwinState]:
        return {d: t.get_twin_state() for d, t in self.twins.items()}

    def check_sync_health(self) -> Dict[int, bool]:
        health = {}
        for drone_id, twin in self.twins.items():
            health[drone_id] = twin.converged()
        return health

    def save_session_logs(self, prefix: str = None) -> Dict[int, str]:
        prefix = prefix or "hil_session_{}".format(self.session_id)
        return self.logger.save_logs(prefix)

    def run_scenario(
        self,
        scenario_steps: int,
        control_provider: Callable[[int, int], np.ndarray],
        rate_hz: float = 50.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        results = {d: [] for d in self.interfaces.keys()}
        dt = 1.0 / rate_hz
        for step in range(scenario_steps):
            controls = {d: control_provider(d, step) for d in self.interfaces.keys()}
            step_results = self.step_once(controls)
            for drone_id, result in step_results.items():
                result["step"] = step
                results[drone_id].append(result)
            time.sleep(dt)
        return results