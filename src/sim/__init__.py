# DroneNav-SAR Simulation Environments
# Isaac Sim, AirSim, Gazebo + unified interface (Sprint 3).
# Backend libs (isaacsim / airsim / rclpy) are optional: each env falls
# back to the shared mock physics backend when its simulator is absent.

from .base_env import (
    BaseDroneEnv,
    DroneState,
    SimConfig,
    DiscreteActionWrapper,
    make_env,
)
from .drone_dynamics import (
    QuadrotorDynamics,
    QuadrotorParams,
    create_default_quadrotor,
)
from .domain_randomization import (
    DomainRandomizer,
    LightingParams,
    TextureParams,
    PhysicsParams,
)
from .mock_backend import MockDroneBackend
from .isaac_sim_env import IsaacSimEnv
from .airsim_env import AirSimEnv
from .gazebo_env import GazeboEnv

__all__ = [
    "BaseDroneEnv",
    "DroneState",
    "SimConfig",
    "DiscreteActionWrapper",
    "make_env",
    "QuadrotorDynamics",
    "QuadrotorParams",
    "create_default_quadrotor",
    "DomainRandomizer",
    "LightingParams",
    "TextureParams",
    "PhysicsParams",
    "MockDroneBackend",
    "IsaacSimEnv",
    "AirSimEnv",
    "GazeboEnv",
]
