#!/usr/bin/env python3
"""
Gazebo Harmonic + ROS2 + PX4 SITL Environment for DroneNav-SAR
Realistic physics simulation with PX4 autopilot
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import subprocess
import time
import os
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

from src.sim.base_env import BaseDroneEnv, SimConfig, DroneState
from src.sim.drone_dynamics import QuadrotorDynamics, create_default_quadrotor


class GazeboEnv(BaseDroneEnv):
    """
    Gazebo Harmonic environment with PX4 SITL for realistic drone simulation.
    Uses ROS2 for communication with PX4.
    """

    def __init__(self, config: SimConfig):
        self._setup_spaces(config)

        # Dynamics model (fallback)
        self.dynamics = create_default_quadrotor()
        self.dynamics.params.mass = config.mass
        self.dynamics.params.max_thrust = config.max_thrust
        self.dynamics.params.max_rpm = config.max_rpm

        # Gazebo/PX4 processes
        self._gz_process = None
        self._px4_process = None
        self._ros2_process = None
        self._bridge_process = None

        # ROS2 nodes
        self._ros2_node = None
        self._odom_sub = None
        self._imu_sub = None
        self._cmd_vel_pub = None
        self._current_odom = None
        self._current_imu = None

        # Mock mode
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
        """Initialize Gazebo + PX4 + ROS2."""
        try:
            # Check if Gazebo is available
            result = subprocess.run(["gz", "sim", "--versions"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("Gazebo not found")

            # Check ROS2
            result = subprocess.run(["ros2", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("ROS2 not found")

            print("Gazebo + ROS2 available. Starting simulation...")

            # Start Gazebo with world
            self._start_gazebo()

            # Start PX4 SITL
            self._start_px4()

            # Start ROS2-Gazebo bridge
            self._start_bridge()

            # Initialize ROS2 node
            self._init_ros2()

            self._mock_mode = False

        except Exception as e:
            print(f"Gazebo/PX4 initialization failed: {e}")
            print("Running in mock mode.")
            self._mock_mode = True

    def _start_gazebo(self):
        """Start Gazebo with the environment world."""
        # Create SDF world file from mesh
        world_file = self._create_world_sdf()

        # Start Gazebo headless
        cmd = [
            "gz", "sim", "-v", "4",
            "-r", world_file,  # -r = run on start
            "--headless-rendering",
        ]

        self._gz_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "GZ_SIM_RESOURCE_PATH": "/workspace/assets/gazebo_models"}
        )

        # Wait for Gazebo to start
        time.sleep(5)

    def _create_world_sdf(self) -> str:
        """Create SDF world file from mesh."""
        world_path = "/tmp/drone_nav_world.sdf"

        mesh_path = Path(self.config.mesh_path)
        use_mesh = mesh_path.exists() and mesh_path.suffix in [".sdf", ".urdf", ".obj", ".stl"]

        sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="drone_nav_world">
    <physics name="default_physics" default="0" type="ignition">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <plugin filename="gz-sim-physics-system" name="ignition::gazebo::systems::Physics">
      <gravity>0 0 -9.81</gravity>
    </plugin>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Sun light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <linear>0.01</linear>
        <constant>0.9</constant>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Environment mesh if available -->
"""

        if use_mesh:
            sdf_content += f"""
    <model name="environment">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>file://{mesh_path.absolute()}</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://{mesh_path.absolute()}</uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
"""
        else:
            sdf_content += """
    <!-- Default room -->
    <model name="room_walls">
      <static>true</static>
      <link name="wall1">
        <pose>0 10 5 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>20 0.1 10</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.1 10</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient></material>
        </visual>
      </link>
      <link name="wall2">
        <pose>0 -10 5 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>20 0.1 10</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.1 10</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient></material>
        </visual>
      </link>
      <link name="wall3">
        <pose>10 0 5 0 0 1.57</pose>
        <collision name="collision">
          <geometry><box><size>20 0.1 10</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.1 10</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient></material>
        </visual>
      </link>
      <link name="wall4">
        <pose>-10 0 5 0 0 1.57</pose>
        <collision name="collision">
          <geometry><box><size>20 0.1 10</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.1 10</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient></material>
        </visual>
      </link>
    </model>
"""

        sdf_content += """
    <!-- PX4 SITL spawn point -->
    <model name="px4_sitl_spawn">
      <pose>0 0 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><sphere><radius>0.05</radius></sphere></geometry>
          <material><ambient>1 0 0 1</ambient></material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
"""

        with open(world_path, "w") as f:
            f.write(sdf_content)

        return world_path

    def _start_px4(self):
        """Start PX4 SITL."""
        # PX4 SITL typically started via make px4_sitl gz_x500
        # For now, we assume PX4 is started separately or via ROS2 launch
        # This is a placeholder - real implementation would use PX4's launch system
        pass

    def _start_bridge(self):
        """Start ROS2-Gazebo bridge."""
        # ros_gz_bridge for topic bridging
        cmd = [
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/model/px4_sitl/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/model/px4_sitl/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/model/px4_sitl/camera@sensor_msgs/msg/Image@gz.msgs.Image",
            "/model/px4_sitl/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
        ]

        self._bridge_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(2)

    def _init_ros2(self):
        """Initialize ROS2 node for communication."""
        try:
            import rclpy
            from rclpy.node import Node
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Imu
            from geometry_msgs.msg import Twist

            rclpy.init()

            class DroneNode(Node):
                def __init__(self):
                    super().__init__('drone_nav_sar')
                    self.odom_sub = self.create_subscription(
                        Odometry, '/model/px4_sitl/odometry', self.odom_callback, 10)
                    self.imu_sub = self.create_subscription(
                        Imu, '/model/px4_sitl/imu', self.imu_callback, 10)
                    self.cmd_vel_pub = self.create_publisher(
                        Twist, '/model/px4_sitl/cmd_vel', 10)
                    self.current_odom = None
                    self.current_imu = None

                def odom_callback(self, msg):
                    self.current_odom = msg

                def imu_callback(self, msg):
                    self.current_imu = msg

            self._ros2_node = DroneNode()

            # Spin in background thread
            self._ros2_thread = threading.Thread(target=self._spin_ros2, daemon=True)
            self._ros2_thread.start()

            # Wait for first messages
            time.sleep(1)

        except Exception as e:
            print(f"ROS2 initialization failed: {e}")
            self._mock_mode = True

    def _spin_ros2(self):
        """Spin ROS2 node."""
        import rclpy
        rclpy.spin(self._ros2_node)

    def _reset_sim(self) -> DroneState:
        """Reset Gazebo + PX4."""
        if self._mock_mode:
            return self._mock_reset()

        # Reset Gazebo world
        # This would require gz service calls or world reset
        # For now, use mock
        return self._mock_reset()

    def _mock_reset(self) -> DroneState:
        """Mock reset."""
        init_pos = np.array([0, 0, 2.0], dtype=np.float32)
        return DroneState(
            position=init_pos,
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),
        )

    def _step_sim(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Step simulation."""
        if self._mock_mode:
            return self._mock_step(action)

        # Send command to PX4 via ROS2
        # Convert motor commands to velocity setpoint
        # This is simplified - real PX4 integration uses MAVLink/ROS2 offboard mode

        try:
            # Get current state from ROS2
            if self._ros2_node.current_odom is not None:
                state = self._odom_to_state(self._ros2_node.current_odom)
            else:
                state = self._mock_step(action)[0]

            # Compute reward
            reward = self._compute_reward(state)

            return state, reward, False, {}

        except Exception as e:
            print(f"Gazebo step failed: {e}")
            return self._mock_step(action)

    def _mock_step(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Mock step."""
        state_array = self._state.to_array()
        next_state_array = self.dynamics.step(state_array, action, self.config.dt)
        next_state = DroneState.from_array(next_state_array)

        reward = self._compute_reward(next_state)
        return next_state, reward, False, {}

    def _odom_to_state(self, odom_msg) -> DroneState:
        """Convert ROS2 Odometry to DroneState."""
        pos = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
        ], dtype=np.float32)

        vel = np.array([
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z,
        ], dtype=np.float32)

        q = odom_msg.pose.pose.orientation
        quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

        ang_vel = np.array([
            odom_msg.twist.twist.angular.x,
            odom_msg.twist.twist.angular.y,
            odom_msg.twist.twist.angular.z,
        ], dtype=np.float32)

        return DroneState(
            position=pos,
            velocity=vel,
            orientation=quat / np.linalg.norm(quat),
            angular_velocity=ang_vel,
            motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),
        )

    def _render_sim(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Get camera image from Gazebo via ROS2."""
        if not self.config.camera_enabled or self._mock_mode:
            return None
        # Would subscribe to camera topic
        return None

    def _close_sim(self) -> None:
        """Clean up processes."""
        for proc in [self._gz_process, self._px4_process, self._bridge_process]:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if self._ros2_node:
            import rclpy
            self._ros2_node.destroy_node()
            rclpy.shutdown()

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization."""
        if self._mock_mode:
            return

        # Randomize physics via Gazebo services
        # Randomize lighting via SDF modification
        pass

    def _compute_reward(self, state: DroneState) -> float:
        """Compute reward."""
        reward = 0.0

        if self._goal is not None:
            dist = np.linalg.norm(state.position - self._goal)
            reward -= dist * 0.1
            reward -= np.linalg.norm(state.velocity) * 0.01

        reward += 0.1

        up_vec = self.dynamics.quat_rotate(state.orientation, np.array([0, 0, 1]))
        reward += up_vec[2] * 0.5

        return float(reward)


def create_gazebo_env(config: SimConfig) -> GazeboEnv:
    """Create Gazebo environment."""
    return GazeboEnv(config)