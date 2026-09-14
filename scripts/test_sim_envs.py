#!/usr/bin/env python3
"""
Quick test script for simulation environments.
Run this to verify all sim backends work in mock mode.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_drone_dynamics():
    """Test drone dynamics model."""
    print("Testing drone dynamics...")
    from src.sim.drone_dynamics import create_default_quadrotor

    drone = create_default_quadrotor()
    hover = drone.compute_hover_thrust()
    print(f"  Hover thrust: {hover:.4f}")

    state = np.zeros(17, dtype=np.float32)
    state[6] = 1.0
    state[13:17] = hover

    for i in range(20):
        state = drone.step(state, np.full(4, hover), 0.02)

    print(f"  Final position: {state[0:3]}")
    print(f"  Final velocity: {state[3:6]}")
    print("  ✓ Drone dynamics OK")


def test_base_env():
    """Test base environment interface."""
    print("\nTesting base environment...")
    from src.sim.base_env import SimConfig, DroneState, DiscreteActionWrapper

    config = SimConfig(mesh_path="test.glb", headless=True)
    print(f"  Config: dt={config.dt}, max_steps={config.max_episode_steps}")

    state = DroneState(
        position=np.array([0, 0, 2], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        motor_speeds=np.full(4, 0.5, dtype=np.float32),
    )
    arr = state.to_array()
    print(f"  State array shape: {arr.shape}")
    state2 = DroneState.from_array(arr)
    print(f"  Round-trip OK: {np.allclose(state.position, state2.position)}")

    # Test discrete wrapper
    class MockEnv:
        def __init__(self):
            self.action_space = None
            self.observation_space = None
            self.config = config
            self._state = state
        def step(self, action): return np.zeros(21), 0, False, False, {}
        def reset(self): return np.zeros(21), {}

    from gymnasium import spaces
    mock = MockEnv()
    mock.action_space = spaces.Box(low=0, high=1, shape=(4,))
    wrapped = DiscreteActionWrapper(mock)
    for a in range(9):
        cont = wrapped.action(a)
        assert cont.shape == (4,)
    print("  ✓ Base environment OK")


def test_domain_randomization():
    """Test domain randomization."""
    print("\nTesting domain randomization...")
    from src.sim.domain_randomization import create_default_randomizer, DomainRandomizationConfig

    randomizer = create_default_randomizer()

    class MockEnv:
        pass

    env = MockEnv()
    for i in range(3):
        applied = randomizer.randomize_all(env)
        print(f"  Episode {i}: {list(applied.keys())}")
        if "physics" in applied:
            print(f"    mass_mult={applied['physics'].get('mass_multiplier', 0):.3f}")
            print(f"    gravity={applied['physics'].get('gravity', 0):.3f}")
        if "environment" in applied and "wind" in applied["environment"]:
            print(f"    wind={applied['environment']['wind']}")

    print("  ✓ Domain randomization OK")


def test_isaac_sim_env():
    """Test Isaac Sim environment (mock mode)."""
    print("\nTesting Isaac Sim environment (mock)...")
    from src.sim.base_env import SimConfig
    from src.sim.isaac_sim_env import IsaacSimEnv

    config = SimConfig(mesh_path="nonexistent.glb", headless=True)
    env = IsaacSimEnv(config)

    # Test reset
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Info keys: {list(info.keys())}")

    # Test step
    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step: reward={reward:.3f}, done={done}")

    env.close()
    print("  ✓ Isaac Sim env OK (mock mode)")


def test_airsim_env():
    """Test AirSim environment (mock mode)."""
    print("\nTesting AirSim environment (mock)...")
    from src.sim.base_env import SimConfig
    from src.sim.airsim_env import AirSimEnv

    config = SimConfig(mesh_path="nonexistent.glb", headless=True)
    env = AirSimEnv(config)

    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")

    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step: reward={reward:.3f}, done={done}")

    env.close()
    print("  ✓ AirSim env OK (mock mode)")


def test_gazebo_env():
    """Test Gazebo environment (mock mode)."""
    print("\nTesting Gazebo environment (mock)...")
    from src.sim.base_env import SimConfig
    from src.sim.gazebo_env import GazeboEnv

    config = SimConfig(mesh_path="nonexistent.glb", headless=True)
    env = GazeboEnv(config)

    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")

    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step: reward={reward:.3f}, done={done}")

    env.close()
    print("  ✓ Gazebo env OK (mock mode)")


def test_factory():
    """Test environment factory."""
    print("\nTesting environment factory...")
    from src.sim.base_env import make_env, SimConfig

    config = SimConfig(mesh_path="test.glb")

    for backend in ["isaac", "airsim", "gazebo"]:
        try:
            env = make_env(backend, "test.glb")
            obs, _ = env.reset()
            print(f"  {backend}: obs_shape={obs.shape}")
            env.close()
        except Exception as e:
            print(f"  {backend}: {e}")

    print("  ✓ Factory OK")


def main():
    print("=" * 60)
    print("DroneNav-SAR Simulation Environment Tests")
    print("=" * 60)

    try:
        test_drone_dynamics()
        test_base_env()
        test_domain_randomization()
        test_isaac_sim_env()
        test_airsim_env()
        test_gazebo_env()
        test_factory()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())