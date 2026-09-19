#!/usr/bin/env python3
"""
Base Environment Interface for DroneNav-SAR
Gymnasium-like interface for all simulation backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class DroneState:
    """Drone state representation."""
    position: np.ndarray      # (3,) x, y, z in world frame
    velocity: np.ndarray      # (3,) vx, vy, vz in world frame
    orientation: np.ndarray   # (4,) quaternion (w, x, y, z)
    angular_velocity: np.ndarray  # (3,) wx, wy, wz in body frame
    motor_speeds: np.ndarray  # (4,) motor RPMs or normalized thrusts

    def to_array(self) -> np.ndarray:
        """Flatten to single array for observations."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity,
            self.motor_speeds,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "DroneState":
        """Reconstruct from flat array."""
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            orientation=arr[6:10],
            angular_velocity=arr[10:13],
            motor_speeds=arr[13:17],
        )

    @property
    def dim(self) -> int:
        return 17  # 3 + 3 + 4 + 3 + 4


@dataclass
class SimConfig:
    """Simulation configuration."""
    # Environment
    mesh_path: str
    headless: bool = True
    device: str = "cuda:0"

    # Drone
    drone_model: str = "quadrotor"
    mass: float = 1.0
    max_thrust: float = 20.0  # N per motor
    max_rpm: float = 15000.0

    # Physics
    dt: float = 0.02  # 50 Hz control
    gravity: float = 9.81
    wind_enabled: bool = False
    wind_max_speed: float = 2.0  # m/s

    # Sensors
    camera_enabled: bool = True
    camera_resolution: Tuple[int, int] = (64, 64)
    camera_fov: float = 90.0
    depth_enabled: bool = True
    imu_noise: float = 0.01
    gps_noise: float = 0.1

    # Domain randomization
    randomize_lighting: bool = True
    randomize_textures: bool = True
    randomize_physics: bool = True
    randomize_initial_pose: bool = True

    # Initial state randomization (Sprint 8 - mock backend compatibility)
    initial_state_position_bounds: Tuple[float, float, float, float, float, float] = (-3.0, 3.0, -3.0, 3.0, 1.0, 5.0)

    # Task
    goal_position: Optional[np.ndarray] = None
    goal_tolerance: float = 0.5
    max_episode_steps: int = 1000

    @property
    def initial_state(self):
        """Compatibility shim for DomainRandomizer mock-backend API."""
        from types import SimpleNamespace
        return SimpleNamespace(
            position_bounds=self.initial_state_position_bounds
        )


class BaseDroneEnv(ABC, gym.Env):
    """
    Abstract base class for drone simulation environments.
    All backends (Isaac Sim, AirSim, Gazebo) must implement this interface.
    """

    metadata = {"render_modes": ["rgb_array", "depth_array"], "render_fps": 30}

    def __init__(self, config: SimConfig):
        super().__init__()
        self.config = config
        self.current_step = 0
        self.episode_reward = 0.0

        # State
        self._state: Optional[DroneState] = None
        self._goal: Optional[np.ndarray] = None

        # Spaces (defined by subclasses)
        self.action_space: spaces.Box
        self.observation_space: spaces.Box

        # Domain randomization
        self._rng = np.random.default_rng()

    @abstractmethod
    def _initialize_sim(self) -> None:
        """Initialize simulation backend (Isaac Sim, AirSim, Gazebo)."""
        pass

    @abstractmethod
    def _reset_sim(self) -> DroneState:
        """Reset simulation and return initial state."""
        pass

    @abstractmethod
    def _step_sim(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Step simulation with action, return (next_state, reward, done, info)."""
        pass

    @abstractmethod
    def _render_sim(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render frame from simulation."""
        pass

    @abstractmethod
    def _close_sim(self) -> None:
        """Clean up simulation resources."""
        pass

    @abstractmethod
    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization to current episode."""
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self.episode_reward = 0.0

        # Reset simulation
        self._state = self._reset_sim()

        # Set goal
        if self.config.goal_position is not None:
            self._goal = self.config.goal_position.copy()
        else:
            # Random goal within bounds
            self._goal = self._sample_random_goal()

        # Apply domain randomization
        self._apply_domain_randomization()

        # Get observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment with action."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step simulation
        self._state, reward, done, info = self._step_sim(action)

        self.current_step += 1
        self.episode_reward += reward

        # Check termination conditions
        truncated = self.current_step >= self.config.max_episode_steps

        # Goal reached
        if self._goal is not None:
            dist_to_goal = np.linalg.norm(self._state.position - self._goal)
            if dist_to_goal < self.config.goal_tolerance:
                done = True
                reward += 100.0  # Goal bonus
                info["goal_reached"] = True

        # Crash detection
        if self._state.position[2] < 0.1:  # Below ground
            done = True
            reward -= 50.0
            info["crashed"] = True

        # Out of bounds
        if np.linalg.norm(self._state.position[:2]) > 100.0:
            done = True
            reward -= 50.0
            info["out_of_bounds"] = True

        obs = self._get_obs()
        info.update(self._get_info())

        return obs, reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation from state."""
        # Base observation: drone state (17 dims)
        obs_parts = [self._state.to_array()]

        # Goal relative position (3 dims)
        if self._goal is not None:
            rel_goal = self._goal - self._state.position
            obs_parts.append(rel_goal)
        else:
            obs_parts.append(np.zeros(3))

        # Time remaining (1 dim)
        time_remaining = 1.0 - (self.current_step / self.config.max_episode_steps)
        obs_parts.append(np.array([time_remaining]))

        return np.concatenate(obs_parts).astype(np.float32)

    def _get_info(self) -> Dict:
        """Get info dict."""
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "position": self._state.position.copy() if self._state is not None else None,
            "velocity": self._state.velocity.copy() if self._state is not None else None,
        }
        if self._goal is not None:
            info["goal"] = self._goal.copy()
            info["dist_to_goal"] = float(np.linalg.norm(self._state.position - self._goal))
        return info

    def _sample_random_goal(self) -> np.ndarray:
        """Sample random goal position within reasonable bounds."""
        # Sample in cylinder around origin
        radius = self._rng.uniform(3.0, 15.0)
        angle = self._rng.uniform(0, 2 * np.pi)
        height = self._rng.uniform(1.0, 10.0)
        return np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height,
        ], dtype=np.float32)

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render frame."""
        return self._render_sim(mode)

    def close(self) -> None:
        """Close environment."""
        self._close_sim()

    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []


class DiscreteActionWrapper(gym.ActionWrapper):
    """Wrapper to convert continuous actions to discrete."""

    def __init__(self, env: BaseDroneEnv, num_discrete: int = 9):
        super().__init__(env)
        self.num_discrete = num_discrete
        # Discrete actions: hover, up, down, forward, back, left, right, yaw_left, yaw_right
        self.action_space = spaces.Discrete(num_discrete)

    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous motor commands."""
        # Base hover thrust (counteract gravity)
        hover_thrust = self.env.config.mass * self.env.config.gravity / 4.0
        hover_normalized = hover_thrust / self.env.config.max_thrust

        actions = {
            0: np.array([hover_normalized] * 4),  # hover
            1: np.array([hover_normalized + 0.2] * 4),  # up
            2: np.array([hover_normalized - 0.2] * 4),  # down
            3: np.array([hover_normalized, hover_normalized,
                         hover_normalized + 0.15, hover_normalized + 0.15]),  # forward
            4: np.array([hover_normalized + 0.15, hover_normalized + 0.15,
                         hover_normalized, hover_normalized]),  # back
            5: np.array([hover_normalized + 0.15, hover_normalized,
                         hover_normalized, hover_normalized + 0.15]),  # left
            6: np.array([hover_normalized, hover_normalized + 0.15,
                         hover_normalized + 0.15, hover_normalized]),  # right
            7: np.array([hover_normalized + 0.1, hover_normalized - 0.1,
                         hover_normalized + 0.1, hover_normalized - 0.1]),  # yaw left
            8: np.array([hover_normalized - 0.1, hover_normalized + 0.1,
                         hover_normalized - 0.1, hover_normalized + 0.1]),  # yaw right
        }

        return np.clip(actions.get(action, actions[0]), 0.0, 1.0)


def make_env(
    backend: str,
    mesh_path: str,
    headless: bool = True,
    **kwargs,
) -> BaseDroneEnv:
    """Factory function to create environment by backend name."""
    config = SimConfig(mesh_path=mesh_path, headless=headless, **kwargs)

    if backend.lower() == "isaac":
        from src.sim.isaac_sim_env import IsaacSimEnv
        return IsaacSimEnv(config)
    elif backend.lower() == "airsim":
        from src.sim.airsim_env import AirSimEnv
        return AirSimEnv(config)
    elif backend.lower() == "gazebo":
        from src.sim.gazebo_env import GazeboEnv
        return GazeboEnv(config)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose: isaac, airsim, gazebo")