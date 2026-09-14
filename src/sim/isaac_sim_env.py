#!/usr/bin/env python3
"""
Isaac Sim Environment for DroneNav-SAR
Headless Isaac Sim with quadrotor dynamics and domain randomization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from src.sim.base_env import BaseDroneEnv, SimConfig, DroneState
from src.sim.drone_dynamics import QuadrotorDynamics, create_default_quadrotor


class IsaacSimEnv(BaseDroneEnv):
    """
    Isaac Sim environment for drone navigation.
    Uses Isaac Sim's physics engine with custom quadrotor dynamics.
    """

    def __init__(self, config: SimConfig):
        # Initialize spaces before parent init
        self._setup_spaces(config)

        # Dynamics model
        self.dynamics = create_default_quadrotor()
        self.dynamics.params.mass = config.mass
        self.dynamics.params.max_thrust = config.max_thrust
        self.dynamics.params.max_rpm = config.max_rpm

        # Isaac Sim objects
        self._sim = None
        self._world = None
        self._drone_prim = None
        self._stage = None
        self._mesh_prim = None

        # Camera
        self._camera = None
        self._camera_annotators = {}

        super().__init__(config)

    def _setup_spaces(self, config: SimConfig):
        """Setup action and observation spaces."""
        # Action: 4 motor thrusts normalized [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation: state(17) + goal_rel(3) + time(1) = 21
        # If camera enabled: add camera obs
        base_obs_dim = 21
        if config.camera_enabled:
            cam_h, cam_w = config.camera_resolution
            cam_obs_dim = cam_h * cam_w * (3 if config.depth_enabled else 3)
            obs_dim = base_obs_dim + cam_obs_dim
        else:
            obs_dim = base_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _initialize_sim(self) -> None:
        """Initialize Isaac Sim."""
        try:
            import omni.isaac.core as isaac_core
            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.objects import DynamicCuboid
            from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema

            # Create world
            self._world = World(
                stage_units_in_meters=1.0,
                physics_dt=self.config.dt,
                rendering_dt=self.config.dt,
                backend="numpy",
            )
            self._stage = self._world.stage

            # Load environment mesh
            self._load_environment_mesh()

            # Create drone
            self._create_drone()

            # Setup camera
            if self.config.camera_enabled:
                self._setup_camera()

            # Reset world
            self._world.reset()

        except ImportError as e:
            print(f"Isaac Sim not available: {e}")
            print("Running in mock mode for testing")
            self._mock_mode = True
        except Exception as e:
            print(f"Isaac Sim initialization error: {e}")
            self._mock_mode = True

    def _load_environment_mesh(self):
        """Load the reconstructed environment mesh."""
        from pxr import Usd, UsdGeom, Gf, Sdf

        mesh_path = Path(self.config.mesh_path)
        if not mesh_path.exists():
            print(f"Mesh not found: {mesh_path}, creating default room")
            self._create_default_room()
            return

        # Import mesh as USD
        if mesh_path.suffix == ".usd" or mesh_path.suffix == ".usda":
            # Direct USD reference
            self._mesh_prim = self._stage.DefinePrim("/World/Environment", "Xform")
            self._mesh_prim.GetReferences().AddReference(str(mesh_path))
        elif mesh_path.suffix == ".glb" or mesh_path.suffix == ".gltf":
            # Need to convert or use Isaac Sim's GLB importer
            # For now, create collision mesh from GLB
            self._create_collision_from_glb(mesh_path)
        else:
            self._create_default_room()

        # Add physics collision
        from omni.isaac.core.utils.prims import set_prim_visibility
        import omni.physx as physx

        # Enable collision on mesh
        pass  # Physics setup handled by Isaac Sim automatically for imported USD

    def _create_collision_from_glb(self, glb_path: Path):
        """Create collision mesh from GLB using trimesh."""
        try:
            import trimesh
            mesh = trimesh.load(str(glb_path))
            # Create convex hull for collision
            convex = mesh.convex_hull
            # Export as OBJ for Isaac Sim
            obj_path = glb_path.with_suffix(".obj")
            convex.export(str(obj_path))
            print(f"Created collision mesh: {obj_path}")
        except Exception as e:
            print(f"Could not create collision from GLB: {e}")
            self._create_default_room()

    def _create_default_room(self):
        """Create a simple default room for testing."""
        from pxr import Usd, UsdGeom, Gf, Sdf
        from omni.isaac.core.objects import DynamicCuboid

        # Floor
        floor = DynamicCuboid(
            prim_path="/World/Environment/Floor",
            position=np.array([0, 0, -0.05]),
            scale=np.array([20, 20, 0.1]),
            color=np.array([0.5, 0.5, 0.5]),
        )

        # Walls
        walls = [
            ("Wall1", np.array([0, 10, 5]), np.array([20, 0.1, 10])),
            ("Wall2", np.array([0, -10, 5]), np.array([20, 0.1, 10])),
            ("Wall3", np.array([10, 0, 5]), np.array([0.1, 20, 10])),
            ("Wall4", np.array([-10, 0, 5]), np.array([0.1, 20, 10])),
        ]
        for name, pos, scale in walls:
            DynamicCuboid(
                prim_path=f"/World/Environment/{name}",
                position=pos,
                scale=scale,
                color=np.array([0.7, 0.7, 0.7]),
            )

        # Add some obstacles
        for i in range(5):
            DynamicCuboid(
                prim_path=f"/World/Environment/Obstacle_{i}",
                position=np.array([
                    np.random.uniform(-8, 8),
                    np.random.uniform(-8, 8),
                    np.random.uniform(1, 3),
                ]),
                scale=np.array([1, 1, np.random.uniform(1, 3)]),
                color=np.array([0.8, 0.3, 0.3]),
            )

    def _create_drone(self):
        """Create drone prim with quadrotor properties."""
        from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
        from omni.isaac.core.objects import DynamicCuboid

        # Drone body (simple cuboid for now)
        self._drone_prim = DynamicCuboid(
            prim_path="/World/Drone",
            position=np.array([0, 0, 2.0]),
            scale=np.array([0.4, 0.4, 0.15]),
            color=np.array([0.2, 0.6, 0.9]),
            mass=self.config.mass,
        )

        # Add physics properties
        prim = self._stage.GetPrimAtPath("/World/Drone")
        if prim:
            # Rigid body API
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(self.config.mass)

            # Set center of mass
            com = Gf.Vec3f(0, 0, 0)
            mass_api.CreateCenterOfMassAttr(com)

    def _setup_camera(self):
        """Setup onboard camera for visual observations."""
        from omni.isaac.sensor import Camera
        from pxr import Gf

        cam_h, cam_w = self.config.camera_resolution

        self._camera = Camera(
            prim_path="/World/Drone/Camera",
            position=Gf.Vec3f(0.1, 0, 0),  # Forward-facing
            orientation=Gf.Quatf(1, 0, 0, 0),  # No rotation relative to drone
            resolution=(cam_w, cam_h),
        )
        self._camera.initialize()

        # Add annotators for RGB and depth
        self._camera_annotators["rgb"] = self._camera.get_rgb()
        if self.config.depth_enabled:
            self._camera_annotators["depth"] = self._camera.get_depth()

    def _reset_sim(self) -> DroneState:
        """Reset simulation to initial state."""
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return self._mock_reset()

        # Reset world
        self._world.reset()

        # Randomize initial pose if enabled
        if self.config.randomize_initial_pose:
            init_pos = np.array([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                np.random.uniform(1, 3),
            ], dtype=np.float32)
            init_yaw = np.random.uniform(-np.pi, np.pi)
            init_quat = np.array([
                np.cos(init_yaw / 2), 0, 0, np.sin(init_yaw / 2)
            ], dtype=np.float32)
        else:
            init_pos = np.array([0, 0, 2.0], dtype=np.float32)
            init_quat = np.array([1, 0, 0, 0], dtype=np.float32)

        # Set drone pose
        from pxr import Gf
        self._drone_prim.set_world_pose(
            position=init_pos,
            orientation=Gf.Quatf(init_quat[0], init_quat[1], init_quat[2], init_quat[3]),
        )

        # Initialize state
        state = DroneState(
            position=init_pos,
            velocity=np.zeros(3, dtype=np.float32),
            orientation=init_quat,
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),
        )

        return state

    def _mock_reset(self) -> DroneState:
        """Mock reset for testing without Isaac Sim."""
        init_pos = np.array([0, 0, 2.0], dtype=np.float32)
        return DroneState(
            position=init_pos,
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),
        )

    def _step_sim(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Step simulation with motor commands."""
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return self._mock_step(action)

        # Get current state from simulation
        pos, orient = self._drone_prim.get_world_pose()
        vel = self._drone_prim.get_linear_velocity()
        ang_vel = self._drone_prim.get_angular_velocity()

        # Convert to our state format
        current_state = DroneState(
            position=np.array(pos, dtype=np.float32),
            velocity=np.array(vel, dtype=np.float32),
            orientation=np.array([orient.real, orient.imag[0], orient.imag[1], orient.imag[2]], dtype=np.float32),
            angular_velocity=np.array(ang_vel, dtype=np.float32),
            motor_speeds=np.full(4, self.dynamics.compute_hover_thrust(), dtype=np.float32),  # Approx
        )

        # Use dynamics model to compute next state
        state_array = current_state.to_array()
        next_state_array = self.dynamics.step(state_array, action, self.config.dt)
        next_state = DroneState.from_array(next_state_array)

        # Apply to simulation (simplified - in reality would apply forces)
        from pxr import Gf
        self._drone_prim.set_world_pose(
            position=next_state.position,
            orientation=Gf.Quatf(
                next_state.orientation[0], next_state.orientation[1],
                next_state.orientation[2], next_state.orientation[3]
            ),
        )
        self._drone_prim.set_linear_velocity(next_state.velocity)
        self._drone_prim.set_angular_velocity(next_state.angular_velocity)

        # Step world
        self._world.step(render=self.config.headless is False)

        # Compute reward
        reward = self._compute_reward(next_state)

        return next_state, reward, False, {}

    def _mock_step(self, action: np.ndarray) -> Tuple[DroneState, float, bool, Dict]:
        """Mock step for testing."""
        # Use dynamics model directly
        state_array = self._state.to_array()
        next_state_array = self.dynamics.step(state_array, action, self.config.dt)
        next_state = DroneState.from_array(next_state_array)

        reward = self._compute_reward(next_state)
        return next_state, reward, False, {}

    def _compute_reward(self, state: DroneState) -> float:
        """Compute reward for current state."""
        reward = 0.0

        # Goal distance reward
        if self._goal is not None:
            dist = np.linalg.norm(state.position - self._goal)
            reward -= dist * 0.1  # Penalty for distance
            reward -= np.linalg.norm(state.velocity) * 0.01  # Penalty for speed

        # Survival bonus
        reward += 0.1

        # Upright bonus
        up_vec = self.dynamics.quat_rotate(state.orientation, np.array([0, 0, 1]))
        reward += up_vec[2] * 0.5  # Reward for staying upright

        return float(reward)

    def _render_sim(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render camera view."""
        if not self.config.camera_enabled or self._camera is None:
            return None

        if mode == "rgb_array" and "rgb" in self._camera_annotators:
            rgb = self._camera_annotators["rgb"]()
            return rgb[:, :, :3] if rgb is not None else None
        elif mode == "depth_array" and "depth" in self._camera_annotators:
            depth = self._camera_annotators["depth"]()
            return depth if depth is not None else None

        return None

    def _close_sim(self) -> None:
        """Clean up Isaac Sim resources."""
        if self._world is not None:
            self._world.clear()
            self._world = None

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization for current episode."""
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return

        # Randomize lighting
        if self.config.randomize_lighting:
            self._randomize_lighting()

        # Randomize textures
        if self.config.randomize_textures:
            self._randomize_textures()

        # Randomize physics parameters
        if self.config.randomize_physics:
            self._randomize_physics()

        # Apply wind
        if self.config.wind_enabled:
            self._apply_wind()

    def _randomize_lighting(self):
        """Randomize environment lighting."""
        try:
            from pxr import UsdLux, Gf
            import omni.usd

            # Get or create distant light
            light_prim = self._stage.GetPrimAtPath("/World/Light")
            if not light_prim:
                light_prim = UsdLux.DistantLight.Define(self._stage, "/World/Light")

            # Randomize intensity and angle
            intensity = np.random.uniform(500, 2000)
            light_prim.GetIntensityAttr().Set(intensity)

            angle = np.random.uniform(0.1, 0.5)
            light_prim.GetAngleAttr().Set(angle)

            # Randomize direction
            dir_x = np.random.uniform(-1, 1)
            dir_y = np.random.uniform(-1, 1)
            dir_z = -np.random.uniform(0.5, 1)
            light_prim.GetDirectionAttr().Set(Gf.Vec3f(dir_x, dir_y, dir_z))
        except Exception as e:
            print(f"Lighting randomization failed: {e}")

    def _randomize_textures(self):
        """Randomize material textures."""
        # For now, just randomize colors of obstacles
        try:
            from pxr import UsdGeom, Gf
            import omni.usd

            for prim in self._stage.Traverse():
                if "Obstacle" in prim.GetPath().pathString or "Wall" in prim.GetPath().pathString:
                    color = np.random.uniform(0.2, 0.8, 3)
                    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        except Exception as e:
            print(f"Texture randomization failed: {e}")

    def _randomize_physics(self):
        """Randomize physics parameters."""
        # Randomize mass slightly
        mass_variation = np.random.uniform(0.9, 1.1)
        self.dynamics.params.mass = self.config.mass * mass_variation

        # Randomize thrust
        thrust_variation = np.random.uniform(0.95, 1.05)
        self.dynamics.params.max_thrust = self.config.max_thrust * thrust_variation

        # Randomize drag
        drag_variation = np.random.uniform(0.8, 1.2)
        self.dynamics.params.drag_coeff *= drag_variation

    def _apply_wind(self):
        """Apply random wind force."""
        # Wind as constant force in world frame
        wind_speed = np.random.uniform(0, self.config.wind_max_speed)
        wind_dir = np.random.uniform(0, 2 * np.pi)
        wind_force = np.array([
            wind_speed * np.cos(wind_dir),
            wind_speed * np.sin(wind_dir),
            0,
        ], dtype=np.float32)

        # Apply as external force to drone (would need PhysX force API)
        # For now, store for dynamics
        self._wind_force = wind_force


# Factory function for easy import
def create_isaac_env(config: SimConfig) -> IsaacSimEnv:
    """Create Isaac Sim environment."""
    return IsaacSimEnv(config)