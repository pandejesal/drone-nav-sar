#!/usr/bin/env python3
"""
AirSim Environment for DroneNav-SAR
Connects to external AirSim/Unreal Engine instance
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
import time

from src.sim.base_env import BaseDroneEnv, SimConfig, DroneState
from src.sim.drone_dynamics import QuadrotorDynamics, create_default_quadrotor


class AirSimEnv(BaseDroneEnv):
    """
    AirSim environment for drone navigation.
    Connects to external AirSim instance running in Unreal Engine.
    """

    def __init__(self, config: SimConfig):
        # Setup spaces
        self._setup_spaces(config)

        # Dynamics model (for local simulation when not connected)
        self.dynamics = create_default_quadrotor()
        self.dynamics.params.mass = config.mass
        self.dynamics.params.max_thrust = config.max_thrust
        self.dynamics.params.max_rpm = config.max_rpm

        # AirSim client
        self._client = None
        self._vehicle_name = "Drone1"
        self._connected = False

        # Camera
        self._camera_name = "front_center"

        # Mock mode flag
        self._mock_mode = False

        super().__init__(config)

    def _setup_spaces(self, config: SimConfig):
        """Setup action and observation spaces."""
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        base_obs_dim = 21
        if config.camera_enabled:
            cam_h, cam_w = config.camera_resolution
            cam_obs_dim = cam_h * cam_w * 3
            obs_dim = base_obs_dim + cam_obs_dim
        else:
            obs_dim = base_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _initialize_sim(self) -> None:
        """Initialize AirSim connection."""
        try:
            import airsim

            # Connect to AirSim
            self._client = airsim.MultirotorClient()
            self._client.confirmConnection()

            # Enable API control
            self._client.enableApiControl(True, self._vehicle_name)
            self._client.armDisarm(True, self._vehicle_name)

            # Check if we can get state
            state = self._client.getMultirotorState(self._vehicle_name)
            print(f"Connected to AirSim. Vehicle state: {state}")

            self._connected = True

        except ImportError:
            print("AirSim Python client not installed. Running in mock mode.")
            self._mock_mode = True
        except Exception as e:
            print(f"Could not connect to AirSim: {e}")
            print("Running in mock mode. Start AirSim in Unreal Engine first.")
            self._mock_mode = True

    def _reset_sim(self) -> DroneState:
        """Reset AirSim vehicle to initial state."""
        if self._mock_mode or not self._connected:
            return self._mock_reset()

        try:
            # Reset vehicle
            self._client.reset()
            self._client.enableApiControl(True, self._vehicle_name)
            self._client.armDisarm(True, self._vehicle_name)

            # Take off to initial height
            init_z = -2.0  # NED frame: negative Z is up
            self._client.takeoffAsync(vehicle_name=self._vehicle_name).join()

            # Move to initial position
            if self.config.randomize_initial_pose:
                init_x = np.random.uniform(-5, 5)
                init_y = np.random.uniform(-5, 5)
                init_z = -np.random.uniform(2, 5)
                init_yaw = np.random.uniform(-np.pi, np.pi)
            else:
                init_x, init_y, init_z = 0, 0, -3
                init_yaw = 0

            self._client.moveToPositionAsync(
                init_x, init_y, init_z, 2.0,
                vehicle_name=self._vehicle_name
            ).join()

            # Rotate to initial yaw
            self._client.rotateToYawAsync(
                np.degrees(init_yaw), 5.0,
                vehicle_name=self._vehicle_name
            ).join()

            time.sleep(0.5)

            # Get state
            state = self._get_airsim_state()
            return state

        except Exception as e:
            print(f"AirSim reset failed: {e}, falling back to mock")
            self._mock_mode = True
            return self._mock_reset()

    def _mock_reset(self) -> DroneState:
        """Mock reset for testing."""
        init_pos = np.array([0, 0, 2.0], dtype=np.float32)
        return DroneState(
            position=init_pos,
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),
        )

    def _step_sim(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Step AirSim with motor commands."""
        if self._mock_mode or not self._connected:
            return self._mock_step(action)

        try:
            # Convert motor commands to AirSim velocity command
            # AirSim uses velocity control, not direct motor control
            # We'll use moveByVelocityAsync with body-frame velocities

            # Get current state for dynamics
            current_state = self._get_airsim_state()

            # Use dynamics to compute desired acceleration
            state_array = current_state.to_array()
            next_state_array = self.dynamics.step(state_array, action, self.config.dt)
            next_state = DroneState.from_array(next_state_array)

            # Compute desired velocity (simple integration)
            desired_vel = next_state.velocity

            # Send velocity command in body frame
            # Convert world velocity to body frame
            R = self.dynamics.quat_to_rot_matrix(current_state.orientation)
            body_vel = R.T @ desired_vel

            # Send command
            self._client.moveByVelocityAsync(
                body_vel[0], body_vel[1], body_vel[2],
                self.config.dt,
                vehicle_name=self._vehicle_name
            )

            time.sleep(self.config.dt)

            # Get new state
            new_state = self._get_airsim_state()

            reward = self._compute_reward(new_state)

            return new_state, reward, False, {}

        except Exception as e:
            print(f"AirSim step failed: {e}, falling back to mock")
            self._mock_mode = True
            return self._mock_step(action)

    def _mock_step(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Mock step using local dynamics."""
        state_array = self._state.to_array()
        next_state_array = self.dynamics.step(state_array, action, self.config.dt)
        next_state = DroneState.from_array(next_state_array)

        reward = self._compute_reward(next_state)
        return next_state, reward, False, {}

    def _get_airsim_state(self) -> DroneState:
        """Get current state from AirSim."""
        try:
            import airsim

            state = self._client.getMultirotorState(self._vehicle_name)

            # Position (NED to ENU)
            pos_ned = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val,
            ], dtype=np.float32)
            pos_enu = np.array([pos_ned[1], pos_ned[0], -pos_ned[2]], dtype=np.float32)

            # Velocity (NED to ENU)
            vel_ned = np.array([
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val,
            ], dtype=np.float32)
            vel_enu = np.array([vel_ned[1], vel_ned[0], -vel_ned[2]], dtype=np.float32)

            # Orientation (quaternion)
            q = state.kinematics_estimated.orientation
            # AirSim uses NED, convert to ENU
            quat_ned = np.array([q.w_val, q.x_val, q.y_val, q.z_val], dtype=np.float32)
            # Quaternion from NED to ENU: swap x,y and negate z
            quat_enu = np.array([
                quat_ned[0], quat_ned[2], quat_ned[1], -quat_ned[3]
            ], dtype=np.float32)

            # Angular velocity
            ang_vel_ned = np.array([
                state.kinematics_estimated.angular_velocity.x_val,
                state.kinematics_estimated.angular_velocity.y_val,
                state.kinematics_estimated.angular_velocity.z_val,
            ], dtype=np.float32)
            ang_vel_enu = np.array([
                ang_vel_ned[1], ang_vel_ned[0], -ang_vel_ned[2]
            ], dtype=np.float32)

            return DroneState(
                position=pos_enu,
                velocity=vel_enu,
                orientation=quat_enu / np.linalg.norm(quat_enu),
                angular_velocity=ang_vel_enu,
                motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),
            )

        except Exception as e:
            print(f"Failed to get AirSim state: {e}")
            raise

    def _render_sim(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Get camera image from AirSim."""
        if not self.config.camera_enabled or self._mock_mode or not self._connected:
            return None

        try:
            import airsim

            responses = self._client.simGetImages([
                airsim.ImageRequest(
                    self._camera_name,
                    airsim.ImageType.Scene,
                    False,  # pixels_as_float
                    False,  # compress
                )
            ], vehicle_name=self._vehicle_name)

            if responses:
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                return img_rgb.astype(np.float32) / 255.0

        except Exception as e:
            print(f"AirSim render failed: {e}")

        return None

    def _close_sim(self) -> None:
        """Disconnect from AirSim."""
        if self._client and self._connected:
            try:
                self._client.enableApiControl(False, self._vehicle_name)
                self._client.armDisarm(False, self._vehicle_name)
            except Exception:
                pass
            self._connected = False

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization via AirSim RPC."""
        if self._mock_mode or not self._connected:
            return

        try:
            import airsim

            # Randomize lighting
            if self.config.randomize_lighting:
                self._client.simSetTimeOfDay(
                    np.random.uniform(6, 18),  # 6am to 6pm
                    is_started=True,
                    vehicle_name=self._vehicle_name
                )

            # Randomize weather
            if self.config.randomize_textures:
                weather = np.random.choice(["Clear", "Rain", "Snow", "Fog"])
                self._client.simSetWeatherParameter(
                    airsim.WeatherParameter.Rain, 
                    1.0 if weather == "Rain" else 0.0,
                    vehicle_name=self._vehicle_name
                )

            # Randomize wind
            if self.config.wind_enabled:
                wind = airsim.Vector3r(
                    np.random.uniform(-self.config.wind_max_speed, self.config.wind_max_speed),
                    np.random.uniform(-self.config.wind_max_speed, self.config.wind_max_speed),
                    0,
                )
                self._client.simSetWind(wind, vehicle_name=self._vehicle_name)

        except Exception as e:
            print(f"AirSim domain randomization failed: {e}")

    def _compute_reward(self, state: DroneState) -> float:
        """Compute reward."""
        reward = 0.0

        if self._goal is not None:
            dist = np.linalg.norm(state.position - self._goal)
            reward -= dist * 0.1
            reward -= np.linalg.norm(state.velocity) * 0.01

        reward += 0.1  # Survival

        # Upright bonus
        up_vec = self.dynamics.quat_rotate(state.orientation, np.array([0, 0, 1]))
        reward += up_vec[2] * 0.5

        return float(reward)


def create_airsim_env(config: SimConfig) -> AirSimEnv:
    """Create AirSim environment."""
    return AirSimEnv(config)