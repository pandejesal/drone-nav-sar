"""DroneNav-SAR reinforcement learning (Sprint 4 small slice).

SAR-only: navigate_to / hover / drop_payload / return_home.
No targeting, no kinetic, no missile guidance.
"""

from src.rl.policies import ActorCritic, ACT_DIM, OBS_DIM
from src.rl.vec_env import DroneVecEnv, coerce_config

__all__ = ["ActorCritic", "ACT_DIM", "OBS_DIM", "DroneVecEnv", "coerce_config"]
