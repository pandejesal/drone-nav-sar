#!/usr/bin/env python3
"""
Shared mock physics backend for DroneNav-SAR sim envs.

Used automatically when the real simulator for a backend is unavailable
(no Isaac Sim install, no AirSim/Unreal endpoint, no ROS2/Gazebo daemon).
Integrates the local QuadrotorDynamics model, applies DomainRandomizer
scales, and synthesizes headless RGB/depth frames.

Every backend env delegates reset/step/render here in mock mode, so the
Gymnasium interface, reward shaping and randomization are identical
across backends and testable without GPU/simulator installs.

Mock mode is explicit: info["mock_backend"] is True and each env exposes
.use_mock. Real-backend hooks live in the per-backend env modules.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .base_env import DroneState, SimConfig
from .drone_dynamics import QuadrotorDynamics, QuadrotorParams
from .domain_randomization import DomainRandomizer


class MockDroneBackend:
    """Headless quadrotor backend: dynamics + wind + synthetic camera."""

    def __init__(
        self,
        config: SimConfig,
        backend_name: str = "mock",
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self.backend_name = backend_name
        self._rng = rng or np.random.default_rng()

        self._nominal_mass = float(config.mass)
        self._nominal_thrust = float(config.max_thrust)
        self._nominal_tau = 0.02

        params = QuadrotorParams(
            mass=config.mass,
            max_thrust=config.max_thrust,
            max_rpm=config.max_rpm,
            motor_time_constant=self._nominal_tau,
        )
        self.dynamics = QuadrotorDynamics(params)
        self.randomizer = DomainRandomizer(config, rng=self._rng)
        self._state_vec: Optional[np.ndarray] = None
        self.dr_params: Dict = {}

    # -- lifecycle ------------------------------------------------------

    def reseed(self, rng: np.random.Generator) -> None:
        """Adopt the env's current RNG (env calls this on every reset)."""
        self._rng = rng
        self.randomizer.reseed(rng)

    def apply_randomization(self) -> Dict:
        """Sample episode params and push physics scales into dynamics."""
        self.dr_params = self.randomizer.apply_all()
        phys = self.dr_params["physics"]
        self.dynamics.params.mass = self._nominal_mass * phys.mass_scale
        self.dynamics.params.max_thrust = self._nominal_thrust * phys.thrust_scale
        self.dynamics.params.motor_time_constant = (
            self._nominal_tau * phys.motor_tau_scale
        )
        self.dynamics.gravity = np.array(
            [0.0, 0.0, -9.81 * phys.gravity_scale], dtype=np.float32
        )
        return self.dr_params

    def reset(self) -> DroneState:
        """Build the initial hover state (nominal physics, fresh pose)."""
        # Restore nominal physics first; env applies episode scales after.
        self.dynamics.params.mass = self._nominal_mass
        self.dynamics.params.max_thrust = self._nominal_thrust
        self.dynamics.params.motor_time_constant = self._nominal_tau
        self.dynamics.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        pos, quat = self.randomizer.randomize_initial_pose()
        hover = self.dynamics.compute_hover_thrust()
        state = DroneState(
            position=pos,
            velocity=np.zeros(3, dtype=np.float32),
            orientation=quat,
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, hover, dtype=np.float32),
        )
        self._state_vec = state.to_array()
        self.randomizer.sample_wind(0.0)  # reset OU process
        return state

    def step(
        self,
        action: np.ndarray,
        dt: float,
        goal: Optional[np.ndarray],
    ) -> Tuple[DroneState, float, bool, Dict]:
        """
        Integrate one control step. Returns (state, reward, done, info).
        Termination (crash/goal/bounds) is handled by BaseDroneEnv.step;
        this backend only reports physics-level done=False.
        """
        motor_cmd = np.clip(action, 0.0, 1.0).astype(np.float32)
        self._state_vec = self.dynamics.step(self._state_vec, motor_cmd, dt)

        # Wind drag: push position/velocity along the wind vector.
        wind = self.randomizer.sample_wind(dt)
        self._state_vec[0:3] += wind * dt * 0.3
        self._state_vec[3:6] += wind * dt * 0.1

        state = DroneState.from_array(self._state_vec)

        # Shaped reward: goal progress + hover-effort penalty + alive bonus.
        hover = self.dynamics.compute_hover_thrust()
        effort = float(np.sum((motor_cmd - hover) ** 2))
        if goal is not None:
            dist = float(np.linalg.norm(state.position - goal))
        else:
            dist = 0.0
        reward = -0.1 * dist - 0.01 * effort + 0.1

        info = {
            "backend": self.backend_name,
            "mock_backend": True,
            "wind": wind.copy(),
            "dist_to_goal": dist,
        }
        return state, float(reward), False, info

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Synthesize a headless camera frame at config resolution."""
        res = tuple(getattr(self.config, "camera_resolution", (64, 64)))
        h, w = int(res[1]), int(res[0])
        alt = float(self._state_vec[2]) if self._state_vec is not None else 1.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2.0
        shade = np.clip(0.4 + 0.1 * alt + 0.2 * base, 0.0, 1.0)
        noise = self._rng.standard_normal((h, w)).astype(np.float32) * 0.03
        if mode == "depth_array":
            depth = np.clip(
                alt + (base - 0.5) * 2.0 + noise, 0.1, 100.0
            ).astype(np.float32)
            return depth
        # rgb_array default
        frame = np.stack([shade + noise, shade, shade - noise], axis=-1)
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
