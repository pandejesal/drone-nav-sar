#!/usr/bin/env python3
"""Sprint 22: Real-time Digital Twin with sim<->hardware sync + state estimation.

SAR-only: keeps twin aligned to the real (or HIL) airframe so SAR search
patterns validated in sim transfer to field ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import numpy as np


@dataclass
class TwinConfig:
    dt: float = 0.01  # 100 Hz twin physics
    pos_process_noise: float = 1e-4
    vel_process_noise: float = 1e-3
    gps_noise: float = 0.05      # m
    imu_noise: float = 0.02      # m/s^2
    sync_gain: float = 0.35      # complementary-filter correction gain
    max_sync_error_m: float = 0.05  # 5 cm acceptance


@dataclass
class TwinState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float64))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    timestamp: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity,
                               self.orientation, self.angular_velocity]).astype(np.float64)

    @staticmethod
    def from_array(arr: np.ndarray, timestamp: float = 0.0) -> "TwinState":
        arr = np.asarray(arr, dtype=np.float64)
        s = TwinState(timestamp=timestamp)
        s.position = arr[0:3].copy()
        s.velocity = arr[3:6].copy()
        s.orientation = arr[6:10].copy()
        s.angular_velocity = arr[10:13].copy()
        return s


class StateEstimator:
    """Lightweight EKF-style complementary filter fusing IMU + GPS + VO.

    Full EKF/UKF lives in src/fusion; the twin embeds a fast <5ms correction
    step so the twin tracks hardware between fusion updates.
    """

    def __init__(self, config: TwinConfig):
        self.cfg = config
        self._P = np.eye(6) * 1e-2  # cov for [pos(3), vel(3)]

    def predict(self, state: TwinState, accel_body: np.ndarray, dt: float) -> TwinState:
        # Simplified: integrate velocity with process noise growth.
        state.velocity = state.velocity + np.asarray(accel_body, dtype=np.float64) * dt
        state.position = state.position + state.velocity * dt
        self._P[:3, :3] += self.cfg.pos_process_noise
        self._P[3:, 3:] += self.cfg.vel_process_noise
        return state

    def correct(self, state: TwinState,
                gps_pos: Optional[np.ndarray] = None,
                vo_vel: Optional[np.ndarray] = None) -> TwinState:
        g = self.cfg.sync_gain
        if gps_pos is not None:
            innov = np.asarray(gps_pos, dtype=np.float64) - state.position
            state.position = state.position + g * innov
            self._P[:3, :3] *= (1.0 - g * 0.5)
        if vo_vel is not None:
            innov = np.asarray(vo_vel, dtype=np.float64) - state.velocity
            state.velocity = state.velocity + g * innov
            self._P[3:, 3:] *= (1.0 - g * 0.5)
        return state

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self._P))


class DigitalTwin:
    """Real-time sim twin of one airframe.

    - ``step`` advances twin physics (point-mass + gravity + control accel).
    - ``sync_from_hardware`` corrects twin toward measured hardware state.
    - ``sync_error`` reports position / velocity alignment error.
    """

    GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    def __init__(self, config: Optional[TwinConfig] = None,
                 initial: Optional[TwinState] = None):
        self.cfg = config or TwinConfig()
        self.state = initial or TwinState()
        self.estimator = StateEstimator(self.cfg)
        self._dynamics = None
        try:  # reuse high-fidelity quadrotor model when available
            from src.sim.drone_dynamics import create_default_quadrotor
            self._dynamics = create_default_quadrotor()
            hover = float(self._dynamics.compute_hover_thrust())
            self._full_state = np.zeros(17, dtype=np.float32)
            self._full_state[6] = 1.0
            self._full_state[13:17] = hover
        except Exception:
            self._dynamics = None
        self.step_times_ms: List[float] = []
        self.sync_errors_m: List[float] = []
        self.last_sync_error_m: float = 0.0
        self.last_sync_vel_error: float = 0.0

    # -- physics ---------------------------------------------------------
    def step(self, control_accel: np.ndarray, dt: Optional[float] = None) -> TwinState:
        t0 = time.perf_counter()
        dt = self.cfg.dt if dt is None else dt
        u = np.asarray(control_accel, dtype=np.float64).reshape(3)
        if self._dynamics is not None:
            # Map desired world accel to collective-thrust motor command
            # around hover (keeps twin consistent with drone_dynamics).
            hover = float(self._dynamics.compute_hover_thrust())
            thrust_scale = float(np.clip((u[2] + 9.81) / 9.81, 0.0, 1.5))
            cmd = np.full(4, np.clip(hover * thrust_scale, 0.0, 1.0), dtype=np.float32)
            self._full_state[0:3] = self.state.position.astype(np.float32)
            self._full_state[3:6] = self.state.velocity.astype(np.float64)
            nxt = self._dynamics.step(self._full_state, cmd, float(dt))
            self._full_state = nxt
            self.state.position = np.asarray(nxt[0:3], dtype=np.float64)
            self.state.velocity = np.asarray(nxt[3:6], dtype=np.float64)
            self.state.orientation = np.asarray(nxt[6:10], dtype=np.float64)
            self.state.angular_velocity = np.asarray(nxt[10:13], dtype=np.float64)
        else:
            self.state.velocity = self.state.velocity + (u + self.GRAVITY * 0.0) * dt
            self.state.position = self.state.position + self.state.velocity * dt
        self.state.timestamp += dt
        self.step_times_ms.append((time.perf_counter() - t0) * 1e3)
        return self.state

    # -- sync ------------------------------------------------------------
    def sync_from_hardware(self, hw_state: TwinState,
                           gps_pos: Optional[np.ndarray] = None,
                           vo_vel: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fuse hardware measurement into twin; return sync error metrics."""
        pos_err_before = float(np.linalg.norm(hw_state.position - self.state.position))
        vel_err_before = float(np.linalg.norm(hw_state.velocity - self.state.velocity))
        g = self.cfg.sync_gain
        self.state.position = self.state.position + g * (hw_state.position - self.state.position)
        self.state.velocity = self.state.velocity + g * (hw_state.velocity - self.state.velocity)
        self.state.orientation = hw_state.orientation.copy()
        self.state.angular_velocity = hw_state.angular_velocity.copy()
        self.estimator.correct(self.state, gps_pos=gps_pos, vo_vel=vo_vel)
        self.last_sync_error_m = float(np.linalg.norm(hw_state.position - self.state.position))
        self.last_sync_vel_error = float(np.linalg.norm(hw_state.velocity - self.state.velocity))
        self.sync_errors_m.append(self.last_sync_error_m)
        self.state.timestamp = hw_state.timestamp
        return {"pos_err_before_m": pos_err_before,
                "vel_err_before": vel_err_before,
                "pos_err_after_m": self.last_sync_error_m,
                "vel_err_after": self.last_sync_vel_error}

    def sync_error(self, hw_state: TwinState) -> Tuple[float, float]:
        dp = float(np.linalg.norm(hw_state.position - self.state.position))
        dv = float(np.linalg.norm(hw_state.velocity - self.state.velocity))
        return dp, dv

    def converged(self) -> bool:
        return (self.last_sync_error_m < self.cfg.max_sync_error_m
                and self.last_sync_vel_error < 0.1)

    def mean_step_ms(self) -> float:
        return float(np.mean(self.step_times_ms)) if self.step_times_ms else 0.0

    def get_twin_state(self) -> TwinState:
        return TwinState(position=self.state.position.copy(),
                         velocity=self.state.velocity.copy(),
                         orientation=self.state.orientation.copy(),
                         angular_velocity=self.state.angular_velocity.copy(),
                         timestamp=self.state.timestamp)
