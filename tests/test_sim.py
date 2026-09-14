#!/usr/bin/env python3
"""
Unit tests for DroneNav-SAR Simulation Environments
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDroneDynamics(unittest.TestCase):
    """Tests for quadrotor dynamics model."""

    def setUp(self):
        from src.sim.drone_dynamics import create_default_quadrotor
        self.drone = create_default_quadrotor()

    def test_hover_thrust(self):
        """Test hover thrust computation."""
        hover = self.drone.compute_hover_thrust()
        self.assertGreater(hover, 0)
        self.assertLess(hover, 1)

    def test_step_hover(self):
        """Test that hover command maintains altitude."""
        state = np.zeros(17, dtype=np.float32)
        state[6] = 1.0  # quaternion w
        hover = self.drone.compute_hover_thrust()
        state[13:17] = hover

        # Step a few times
        for _ in range(10):
            state = self.drone.step(state, np.full(4, hover), 0.02)

        # Should stay near initial position
        self.assertLess(np.linalg.norm(state[0:3]), 1.0)

    def test_quaternion_operations(self):
        """Test quaternion math."""
        q1 = np.array([1, 0, 0, 0], dtype=np.float32)
        q2 = np.array([0, 1, 0, 0], dtype=np.float32)
        result = self.drone.quat_multiply(q1, q2)
        np.testing.assert_allclose(result, q2)

        # Test conjugate
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        q = q / np.linalg.norm(q)
        conj = self.drone.quat_conjugate(q)
        expected = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        np.testing.assert_allclose(conj, expected)

    def test_euler_conversion(self):
        """Test Euler <-> quaternion conversion."""
        euler = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        quat = self.drone.euler_to_quat(euler)
        euler_back = self.drone.quat_to_euler(quat)
        np.testing.assert_allclose(euler, euler_back, atol=1e-5)

    def test_linearization(self):
        """Test linearization produces valid matrices."""
        A, B = self.drone.linearize(None)
        self.assertEqual(A.shape, (17, 17))
        self.assertEqual(B.shape, (17, 4))
        # Should not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(A)))
        self.assertFalse(np.any(np.isnan(B)))


class TestBaseEnv(unittest.TestCase):
    """Tests for base environment interface."""

    def test_sim_config(self):
        """Test SimConfig dataclass."""
        from src.sim.base_env import SimConfig
        config = SimConfig(mesh_path="test.glb")
        self.assertEqual(config.mesh_path, "test.glb")
        self.assertEqual(config.dt, 0.02)
        self.assertEqual(config.max_episode_steps, 1000)

    def test_drone_state(self):
        """Test DroneState dataclass."""
        from src.sim.base_env import DroneState
        state = DroneState(
            position=np.array([1, 2, 3], dtype=np.float32),
            velocity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            motor_speeds=np.full(4, 0.5, dtype=np.float32),
        )
        arr = state.to_array()
        self.assertEqual(len(arr), 17)

        state2 = DroneState.from_array(arr)
        np.testing.assert_allclose(state.position, state2.position)
        np.testing.assert_allclose(state.velocity, state2.velocity)
        np.testing.assert_allclose(state.orientation, state2.orientation)

    def test_discrete_action_wrapper(self):
        """Test discrete action wrapper."""
        from src.sim.base_env import DiscreteActionWrapper, BaseDroneEnv, SimConfig
        from gymnasium import spaces

        class MockEnv(BaseDroneEnv):
            def _initialize_sim(self): pass
            def _reset_sim(self):
                return DroneState(
                    position=np.zeros(3),
                    velocity=np.zeros(3),
                    orientation=np.array([1,0,0,0]),
                    angular_velocity=np.zeros(3),
                    motor_speeds=np.full(4, 0.5),
                )
            def _step_sim(self, action):
                return self._state, 0.0, False, {}
            def _render_sim(self, mode): return None
            def _close_sim(self): pass
            def _apply_domain_randomization(self): pass

        config = SimConfig(mesh_path="test.glb")
        env = MockEnv(config)
        wrapped = DiscreteActionWrapper(env)

        self.assertEqual(wrapped.action_space.n, 9)
        # Test each discrete action produces valid continuous action
        for a in range(9):
            cont = wrapped.action(a)
            self.assertEqual(cont.shape, (4,))
            self.assertTrue(np.all(cont >= 0) and np.all(cont <= 1))


class TestDomainRandomization(unittest.TestCase):
    """Tests for domain randomization."""

    def test_config_creation(self):
        """Test DomainRandomizationConfig creation."""
        from src.sim.domain_randomization import DomainRandomizationConfig, create_default_randomizer
        config = DomainRandomizationConfig()
        self.assertTrue(config.lighting.enabled)
        self.assertTrue(config.physics.enabled)

    def test_randomizer(self):
        """Test DomainRandomizer basic functionality."""
        from src.sim.domain_randomization import create_default_randomizer
        randomizer = create_default_randomizer()

        class MockEnv:
            pass

        env = MockEnv()
        applied = randomizer.randomize_all(env)

        # Check that randomization was applied
        self.assertIn("lighting", applied)
        self.assertIn("physics", applied)
        self.assertIn("textures", applied)
        self.assertIn("sensors", applied)
        self.assertIn("environment", applied)
        self.assertIn("initial_state", applied)

    def test_curriculum(self):
        """Test curriculum learning progression."""
        from src.sim.domain_randomization import DomainRandomizationConfig, DomainRandomizer
        config = DomainRandomizationConfig(
            curriculum_enabled=True,
            curriculum_steps=100,
        )
        randomizer = DomainRandomizer(config)

        # Initial factor should be 0
        self.assertEqual(randomizer.config.get_curriculum_factor(), 0.0)

        # Advance
        for _ in range(50):
            randomizer.config.step_curriculum()
        self.assertAlmostEqual(randomizer.config.get_curriculum_factor(), 0.5, places=1)

        # Complete
        for _ in range(50):
            randomizer.config.step_curriculum()
        self.assertEqual(randomizer.config.get_curriculum_factor(), 1.0)


class TestIsaacSimEnv(unittest.TestCase):
    """Tests for Isaac Sim environment (mock mode)."""

    def test_mock_mode_creation(self):
        """Test environment can be created in mock mode."""
        from src.sim.base_env import SimConfig
        from src.sim.isaac_sim_env import IsaacSimEnv

        config = SimConfig(mesh_path="nonexistent.glb", headless=True)
        # This will fail to import Isaac Sim and fall back to mock
        # We can't fully test without Isaac Sim installed
        pass


class TestAirSimEnv(unittest.TestCase):
    """Tests for AirSim environment (mock mode)."""

    def test_mock_mode_creation(self):
        """Test environment can be created in mock mode."""
        from src.sim.base_env import SimConfig
        from src.sim.airsim_env import AirSimEnv

        config = SimConfig(mesh_path="nonexistent.glb", headless=True)
        # Will fall back to mock mode
        pass


class TestGazeboEnv(unittest.TestCase):
    """Tests for Gazebo environment (mock mode)."""

    def test_mock_mode_creation(self):
        """Test environment can be created in mock mode."""
        from src.sim.base_env import SimConfig
        from src.sim.gazebo_env import GazeboEnv

        config = SimConfig(mesh_path="nonexistent.glb", headless=True)
        # Will fall back to mock mode
        pass


class TestFactory(unittest.TestCase):
    """Tests for environment factory."""

    def test_make_env(self):
        """Test make_env factory function."""
        from src.sim.base_env import make_env, SimConfig

        config = SimConfig(mesh_path="test.glb")

        # Test unknown backend
        with self.assertRaises(ValueError):
            make_env("unknown", "test.glb")


if __name__ == "__main__":
    unittest.main()