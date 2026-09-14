#!/usr/bin/env python3
"""
Domain Randomization for DroneNav-SAR
Randomizes lighting, textures, physics, sensors, and environment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import random


class RandomizationType(Enum):
    LIGHTING = "lighting"
    TEXTURES = "textures"
    PHYSICS = "physics"
    SENSORS = "sensors"
    ENVIRONMENT = "environment"
    INITIAL_STATE = "initial_state"


@dataclass
class LightingRandomization:
    """Lighting randomization parameters."""
    enabled: bool = True
    intensity_range: tuple = (300, 3000)  # lux
    angle_range: tuple = (0.05, 0.8)  # radians
    color_temp_range: tuple = (3000, 6500)  # Kelvin
    direction_randomize: bool = True
    shadows_enabled: bool = True
    num_lights_range: tuple = (1, 3)


@dataclass
class TextureRandomization:
    """Texture/material randomization parameters."""
    enabled: bool = True
    color_jitter_range: tuple = (0.5, 1.5)  # multiplier
    roughness_range: tuple = (0.1, 0.9)
    metallic_range: tuple = (0.0, 0.5)
    albedo_noise_std: float = 0.1
    normal_noise_std: float = 0.05
    use_procedural_textures: bool = True
    texture_library_path: Optional[str] = None


@dataclass
class PhysicsRandomization:
    """Physics parameter randomization."""
    enabled: bool = True
    mass_range: tuple = (0.8, 1.2)  # multiplier
    inertia_range: tuple = (0.8, 1.2)  # multiplier
    friction_range: tuple = (0.5, 1.5)  # multiplier
    restitution_range: tuple = (0.0, 0.3)
    gravity_range: tuple = (9.5, 10.1)  # m/s^2
    drag_coeff_range: tuple = (0.5, 2.0)  # multiplier
    motor_thrust_range: tuple = (0.9, 1.1)  # multiplier
    motor_time_constant_range: tuple = (0.5, 2.0)  # multiplier


@dataclass
class SensorRandomization:
    """Sensor noise and bias randomization."""
    enabled: bool = True
    imu_accel_noise_range: tuple = (0.01, 0.1)  # m/s^2
    imu_gyro_noise_range: tuple = (0.001, 0.01)  # rad/s
    imu_accel_bias_range: tuple = (-0.1, 0.1)  # m/s^2
    imu_gyro_bias_range: tuple = (-0.01, 0.01)  # rad/s
    gps_noise_range: tuple = (0.05, 0.5)  # m
    gps_bias_range: tuple = (-0.2, 0.2)  # m
    camera_noise_std: tuple = (0.0, 0.05)  # pixel intensity
    depth_noise_std: tuple = (0.001, 0.02)  # m


@dataclass
class EnvironmentRandomization:
    """Environment layout randomization."""
    enabled: bool = True
    obstacle_count_range: tuple = (0, 10)
    obstacle_size_range: tuple = (0.5, 3.0)  # meters
    obstacle_position_bounds: tuple = (-15, 15, -15, 15, 0.5, 5)  # x_min, x_max, y_min, y_max, z_min, z_max
    moving_obstacle_prob: float = 0.1
    moving_obstacle_speed_range: tuple = (0.5, 2.0)  # m/s
    wind_enabled: bool = True
    wind_speed_range: tuple = (0, 3.0)  # m/s
    wind_direction_change_rate: float = 0.1  # Hz


@dataclass
class InitialStateRandomization:
    """Initial state randomization."""
    enabled: bool = True
    position_bounds: tuple = (-5, 5, -5, 5, 1, 5)  # x_min, x_max, y_min, y_max, z_min, z_max
    velocity_range: tuple = (-1, 1)  # m/s per axis
    orientation_yaw_range: tuple = (-np.pi, np.pi)
    orientation_tilt_max: float = 0.2  # rad


@dataclass
class DomainRandomizationConfig:
    """Complete domain randomization configuration."""
    lighting: LightingRandomization = field(default_factory=LightingRandomization)
    textures: TextureRandomization = field(default_factory=TextureRandomization)
    physics: PhysicsRandomization = field(default_factory=PhysicsRandomization)
    sensors: SensorRandomization = field(default_factory=SensorRandomization)
    environment: EnvironmentRandomization = field(default_factory=EnvironmentRandomization)
    initial_state: InitialStateRandomization = field(default_factory=InitialStateRandomization)

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_steps: int = 10000  # steps to reach full randomization
    current_step: int = 0

    def get_curriculum_factor(self) -> float:
        """Get curriculum factor [0, 1] based on current step."""
        if not self.curriculum_enabled:
            return 1.0
        return min(1.0, self.current_step / self.curriculum_steps)

    def step_curriculum(self):
        """Advance curriculum step."""
        self.current_step += 1


class DomainRandomizer:
    """
    Applies domain randomization to simulation environments.
    Supports Isaac Sim, AirSim, Gazebo, and generic environments.
    """

    def __init__(self, config: Optional[DomainRandomizationConfig] = None):
        self.config = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng()
        self._randomization_callbacks: Dict[RandomizationType, List[Callable]] = {
            rt: [] for rt in RandomizationType
        }

    def register_callback(self, rand_type: RandomizationType, callback: Callable):
        """Register a callback for a randomization type."""
        self._randomization_callbacks[rand_type].append(callback)

    def randomize_all(self, env: Any) -> Dict[str, Any]:
        """Apply all enabled randomizations."""
        applied = {}

        factor = self.config.get_curriculum_factor()

        # Lighting
        if self.config.lighting.enabled:
            applied["lighting"] = self._randomize_lighting(env, factor)

        # Textures
        if self.config.textures.enabled:
            applied["textures"] = self._randomize_textures(env, factor)

        # Physics
        if self.config.physics.enabled:
            applied["physics"] = self._randomize_physics(env, factor)

        # Sensors
        if self.config.sensors.enabled:
            applied["sensors"] = self._randomize_sensors(env, factor)

        # Environment
        if self.config.environment.enabled:
            applied["environment"] = self._randomize_environment(env, factor)

        # Initial state
        if self.config.initial_state.enabled:
            applied["initial_state"] = self._randomize_initial_state(env, factor)

        # Call registered callbacks
        for rand_type, callbacks in self._randomization_callbacks.items():
            for callback in callbacks:
                try:
                    callback(env, factor)
                except Exception as e:
                    print(f"Randomization callback failed: {e}")

        self.config.step_curriculum()
        return applied

    def _randomize_lighting(self, env: Any, factor: float) -> Dict:
        """Randomize lighting."""
        cfg = self.config.lighting
        result = {}

        # Intensity
        intensity = self._rng.uniform(*cfg.intensity_range)
        result["intensity"] = intensity

        # Direction
        if cfg.direction_randomize:
            theta = self._rng.uniform(0, 2 * np.pi)
            phi = self._rng.uniform(*cfg.angle_range)
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                -np.cos(phi),
            ])
            result["direction"] = direction

        # Color temperature
        color_temp = self._rng.uniform(*cfg.color_temp_range)
        result["color_temp"] = color_temp

        # Number of lights
        num_lights = self._rng.integers(*cfg.num_lights_range)
        result["num_lights"] = num_lights

        # Apply to environment (callback will handle actual application)
        return result

    def _randomize_textures(self, env: Any, factor: float) -> Dict:
        """Randomize textures/materials."""
        cfg = self.config.textures
        result = {}

        # Color jitter
        color_mult = self._rng.uniform(*cfg.color_jitter_range)
        result["color_multiplier"] = color_mult

        # Roughness
        roughness = self._rng.uniform(*cfg.roughness_range)
        result["roughness"] = roughness

        # Metallic
        metallic = self._rng.uniform(*cfg.metallic_range)
        result["metallic"] = metallic

        return result

    def _randomize_physics(self, env: Any, factor: float) -> Dict:
        """Randomize physics parameters."""
        cfg = self.config.physics
        result = {}

        # Mass
        mass_mult = self._rng.uniform(*cfg.mass_range)
        result["mass_multiplier"] = mass_mult

        # Inertia
        inertia_mult = self._rng.uniform(*cfg.inertia_range)
        result["inertia_multiplier"] = inertia_mult

        # Friction
        friction_mult = self._rng.uniform(*cfg.friction_range)
        result["friction_multiplier"] = friction_mult

        # Restitution
        restitution = self._rng.uniform(*cfg.restitution_range)
        result["restitution"] = restitution

        # Gravity
        gravity = self._rng.uniform(*cfg.gravity_range)
        result["gravity"] = gravity

        # Drag
        drag_mult = self._rng.uniform(*cfg.drag_coeff_range)
        result["drag_multiplier"] = drag_mult

        # Motor thrust
        thrust_mult = self._rng.uniform(*cfg.motor_thrust_range)
        result["motor_thrust_multiplier"] = thrust_mult

        # Motor time constant
        motor_tau_mult = self._rng.uniform(*cfg.motor_time_constant_range)
        result["motor_tau_multiplier"] = motor_tau_mult

        return result

    def _randomize_sensors(self, env: Any, factor: float) -> Dict:
        """Randomize sensor parameters."""
        cfg = self.config.sensors
        result = {}

        # IMU
        result["imu_accel_noise"] = self._rng.uniform(*cfg.imu_accel_noise_range)
        result["imu_gyro_noise"] = self._rng.uniform(*cfg.imu_gyro_noise_range)
        result["imu_accel_bias"] = self._rng.uniform(*cfg.imu_accel_bias_range, size=3)
        result["imu_gyro_bias"] = self._rng.uniform(*cfg.imu_gyro_bias_range, size=3)

        # GPS
        result["gps_noise"] = self._rng.uniform(*cfg.gps_noise_range)
        result["gps_bias"] = self._rng.uniform(*cfg.gps_bias_range, size=3)

        # Camera
        result["camera_noise"] = self._rng.uniform(*cfg.camera_noise_std)
        result["depth_noise"] = self._rng.uniform(*cfg.depth_noise_std)

        return result

    def _randomize_environment(self, env: Any, factor: float) -> Dict:
        """Randomize environment layout."""
        cfg = self.config.environment
        result = {}

        # Obstacle count
        num_obstacles = self._rng.integers(*cfg.obstacle_count_range)
        result["num_obstacles"] = num_obstacles

        # Obstacle positions and sizes
        obstacles = []
        for i in range(num_obstacles):
            x = self._rng.uniform(cfg.obstacle_position_bounds[0], cfg.obstacle_position_bounds[1])
            y = self._rng.uniform(cfg.obstacle_position_bounds[2], cfg.obstacle_position_bounds[3])
            z = self._rng.uniform(cfg.obstacle_position_bounds[4], cfg.obstacle_position_bounds[5])
            size = self._rng.uniform(*cfg.obstacle_size_range)

            moving = self._rng.random() < cfg.moving_obstacle_prob
            move_speed = 0.0
            move_dir = None
            if moving:
                move_speed = self._rng.uniform(*cfg.moving_obstacle_speed_range)
                move_angle = self._rng.uniform(0, 2 * np.pi)
                move_dir = np.array([np.cos(move_angle), np.sin(move_angle), 0])

            obstacles.append({
                "position": np.array([x, y, z]),
                "size": size,
                "moving": moving,
                "move_speed": move_speed,
                "move_direction": move_dir,
            })

        result["obstacles"] = obstacles

        # Wind
        if cfg.wind_enabled:
            wind_speed = self._rng.uniform(*cfg.wind_speed_range)
            wind_dir = self._rng.uniform(0, 2 * np.pi)
            result["wind"] = np.array([
                wind_speed * np.cos(wind_dir),
                wind_speed * np.sin(wind_dir),
                0,
            ])

        return result

    def _randomize_initial_state(self, env: Any, factor: float) -> Dict:
        """Randomize initial state."""
        cfg = self.config.initial_state
        result = {}

        # Position
        x = self._rng.uniform(cfg.position_bounds[0], cfg.position_bounds[1])
        y = self._rng.uniform(cfg.position_bounds[2], cfg.position_bounds[3])
        z = self._rng.uniform(cfg.position_bounds[4], cfg.position_bounds[5])
        result["position"] = np.array([x, y, z])

        # Velocity
        vel = self._rng.uniform(*cfg.velocity_range, size=3)
        result["velocity"] = vel

        # Orientation
        yaw = self._rng.uniform(*cfg.orientation_yaw_range)
        tilt_x = self._rng.uniform(-cfg.orientation_tilt_max, cfg.orientation_tilt_max)
        tilt_y = self._rng.uniform(-cfg.orientation_tilt_max, cfg.orientation_tilt_max)

        # Convert to quaternion
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(tilt_y * 0.5), np.sin(tilt_y * 0.5)
        cr, sr = np.cos(tilt_x * 0.5), np.sin(tilt_x * 0.5)

        quat = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float32)

        result["orientation"] = quat

        return result

    def reset_curriculum(self):
        """Reset curriculum learning."""
        self.config.current_step = 0

    def set_curriculum_step(self, step: int):
        """Set curriculum step directly."""
        self.config.current_step = step

    def get_config_dict(self) -> Dict:
        """Get configuration as dictionary."""
        return {
            "lighting": self.config.lighting.__dict__,
            "textures": self.config.textures.__dict__,
            "physics": self.config.physics.__dict__,
            "sensors": self.config.sensors.__dict__,
            "environment": self.config.environment.__dict__,
            "initial_state": self.config.initial_state.__dict__,
            "curriculum_enabled": self.config.curriculum_enabled,
            "curriculum_steps": self.config.curriculum_steps,
            "current_step": self.config.current_step,
        }


def create_default_randomizer() -> DomainRandomizer:
    """Create default domain randomizer with sensible defaults."""
    config = DomainRandomizationConfig(
        curriculum_enabled=True,
        curriculum_steps=50000,
    )

    # Slightly less aggressive defaults for early training
    config.lighting.intensity_range = (500, 2000)
    config.textures.color_jitter_range = (0.7, 1.3)
    config.physics.mass_range = (0.9, 1.1)
    config.physics.friction_range = (0.7, 1.3)
    config.sensors.imu_accel_noise_range = (0.01, 0.05)
    config.sensors.gps_noise_range = (0.05, 0.2)
    config.environment.obstacle_count_range = (0, 5)
    config.environment.wind_speed_range = (0, 2.0)
    config.initial_state.position_bounds = (-3, 3, -3, 3, 1, 4)

    return DomainRandomizer(config)


def create_aggressive_randomizer() -> DomainRandomizer:
    """Create aggressive domain randomizer for robust sim2real."""
    config = DomainRandomizationConfig(
        curriculum_enabled=True,
        curriculum_steps=100000,
    )
    return DomainRandomizer(config)


if __name__ == "__main__":
    # Test
    randomizer = create_default_randomizer()

    class MockEnv:
        pass

    env = MockEnv()

    for i in range(5):
        applied = randomizer.randomize_all(env)
        print(f"Episode {i}: {list(applied.keys())}")
        if "physics" in applied:
            print(f"  Physics: mass_mult={applied['physics'].get('mass_multiplier', 'N/A'):.3f}")
        if "wind" in applied.get("environment", {}):
            print(f"  Wind: {applied['environment']['wind']}")