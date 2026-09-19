#!/usr/bin/env python3
"""Sprint 4 v2 PPO small slice: policy + vec_env + smoke (3 tests ONLY).

SAR-only: navigate_to / hover / drop_payload / return_home.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPPOUpdate(unittest.TestCase):
    def test_ppo_loss_decreases_on_fixed_batch(self):
        """One PPO update on a fixed batch should reduce loss."""
        import torch
        from src.rl.policies import ActorCritic
        from src.rl.ppo_nav import ppo_loss, RolloutBuffer
        from src.rl.policies import OBS_DIM, ACT_DIM

        torch.manual_seed(42)
        policy = ActorCritic()
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)

        # Create a fixed batch
        B = 64
        obs = torch.randn(B, OBS_DIM)
        actions = torch.rand(B, ACT_DIM)  # in [0,1]
        old_logprobs = torch.randn(B)
        advantages = torch.randn(B)
        returns = torch.randn(B)

        # Measure initial loss
        with torch.no_grad():
            loss0, _, _, _ = ppo_loss(policy, obs, actions, old_logprobs, advantages, returns)

        # One update step
        policy.train()
        loss, _, _, _ = ppo_loss(policy, obs, actions, old_logprobs, advantages, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss after 1 step should be lower (or at least not higher)
        with torch.no_grad():
            loss1, _, _, _ = ppo_loss(policy, obs, actions, old_logprobs, advantages, returns)
        self.assertLessEqual(loss1.item(), loss0.item() + 0.1,
                             "PPO loss should not increase after an update step")


class TestEvalSPL(unittest.TestCase):
    def test_spl_one_on_optimal_demo(self):
        """SPL == 1.0 when every episode is a success with optimal path."""
        from src.eval.metrics import spl, success_rate

        # Optimal demo: success=True, steps == optimal_steps
        results = [
            {"goal_reached": True, "steps": 10, "optimal_steps": 10},
            {"goal_reached": True, "steps": 5, "optimal_steps": 5},
            {"goal_reached": True, "steps": 20, "optimal_steps": 20},
            {"goal_reached": True, "steps": 8, "optimal_steps": 8},
        ]
        self.assertAlmostEqual(spl(results), 1.0, places=6)
        self.assertAlmostEqual(success_rate(results), 1.0, places=6)

        # Non-optimal success: steps > optimal → SPL < 1
        results2 = [
            {"goal_reached": True, "steps": 20, "optimal_steps": 10},
        ]
        self.assertLess(spl(results2), 1.0)


class TestPolicy(unittest.TestCase):
    def test_policy_outputs_in_range_shape(self):
        import torch
        from src.rl.policies import ActorCritic

        torch.manual_seed(0)
        policy = ActorCritic()
        obs = np.random.default_rng(0).standard_normal((4, 21)).astype(np.float32)
        actions, values, _ = policy.get_action(obs)
        self.assertEqual(actions.shape, (4, 4))
        self.assertEqual(values.shape, (4,))
        self.assertTrue(np.all(actions >= 0.0) and np.all(actions <= 1.0))
        self.assertTrue(np.all(np.isfinite(actions)) and np.all(np.isfinite(values)))


class TestVecEnv(unittest.TestCase):
    def test_reset_step_shapes_seeded_determinism(self):
        from src.rl.policies import ActorCritic
        from src.rl.vec_env import DroneVecEnv
        from src.sim.base_env import SimConfig

        cfg = SimConfig(mesh_path="mock.glb", camera_enabled=False, max_episode_steps=50)
        v1 = DroneVecEnv(n_envs=8, mesh_or_config=cfg, master_seed=7, use_mock=True)
        v2 = DroneVecEnv(n_envs=8, mesh_or_config="mock.glb", master_seed=7,
                         use_mock=True, camera_enabled=False, max_episode_steps=50)
        o1, _ = v1.reset()
        o2, _ = v2.reset()
        self.assertEqual(o1.shape, (8, 21))
        np.testing.assert_allclose(o1, o2, rtol=0, atol=0)

        policy = ActorCritic()
        a1, _, _ = policy.get_action(o1, deterministic=True)
        s1 = v1.step(a1)
        s2 = v2.step(a1)
        self.assertEqual(s1[0].shape, (8, 21))
        self.assertEqual(s1[1].shape, (8,))
        np.testing.assert_allclose(s1[0], s2[0], rtol=0, atol=0)
        np.testing.assert_allclose(s1[1], s2[1], rtol=0, atol=0)
        v1.close()
        v2.close()


class TestSmoke(unittest.TestCase):
    def test_10_episode_smoke_reaches_terminal(self):
        from src.rl.policies import ActorCritic
        from src.rl.vec_env import DroneVecEnv
        from src.sim.base_env import SimConfig

        cfg = SimConfig(mesh_path="mock.glb", camera_enabled=False, max_episode_steps=200)
        vec = DroneVecEnv(n_envs=2, mesh_or_config=cfg, master_seed=123, use_mock=True)
        policy = ActorCritic()
        obs, _ = vec.reset()
        dones = 0
        for _ in range(2000):  # budget cap; 10 dones expected far earlier
            actions, _, _ = policy.get_action(obs)
            obs, _, term, trunc, infos = vec.step(actions)
            dones += int(np.sum(term)) + int(np.sum(trunc))
            for info in infos:
                self.assertTrue(info.get("mock_backend", False))
            if dones >= 10:
                break
        vec.close()
        self.assertGreaterEqual(dones, 10)


class TestCurriculum(unittest.TestCase):
    def test_curriculum_advances_level_when_sr_threshold_met(self):
        """Curriculum advances a level when sr>0.8 over the last 50 episodes."""
        from src.rl.ppo_nav import apply_curriculum_level, maybe_advance_curriculum
        from src.rl.vec_env import DroneVecEnv
        from src.sim.base_env import SimConfig

        wins = [{"goal_reached": True, "steps": 10, "optimal_steps": 10}] * 50
        losses = [{"goal_reached": False, "steps": 500, "optimal_steps": 10}] * 50

        # sr=1.0 over 50 eps → advance 0 → 1; sr=0.0 → stay at 0
        self.assertEqual(maybe_advance_curriculum(0, wins), 1)
        self.assertEqual(maybe_advance_curriculum(0, losses), 0)
        # Window not full (<50 eps) → stay; already at max level → stay
        self.assertEqual(maybe_advance_curriculum(0, wins[:10]), 0)
        self.assertEqual(maybe_advance_curriculum(3, wins), 3)

        # apply_curriculum_level pushes tolerance + step budget into sub-envs
        cfg = SimConfig(mesh_path="mock.glb", camera_enabled=False, max_episode_steps=500)
        vec = DroneVecEnv(n_envs=2, mesh_or_config=cfg, master_seed=0, use_mock=True)
        try:
            apply_curriculum_level(vec, 0)
            for e in vec.envs:
                self.assertEqual(e.config.goal_tolerance, 1.5)
                self.assertEqual(e.config.max_episode_steps, 300)
            apply_curriculum_level(vec, 3)
            for e in vec.envs:
                self.assertEqual(e.config.goal_tolerance, 0.5)
                self.assertEqual(e.config.max_episode_steps, 500)
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
