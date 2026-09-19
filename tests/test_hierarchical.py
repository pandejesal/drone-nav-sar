#!/usr/bin/env python3
"""Sprint 8 hierarchical navigation tests (4 tests).

Covers: GlobalPlanner pathfinding, LocalPolicy forward pass,
HierarchicalPolicy wrapper, integration with mock backend.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestGlobalPlanner(unittest.TestCase):
    def test_global_planner_pathfinding(self):
        """A* over the room graph yields waypoints from home to room 3."""
        from src.rl.global_planner import create_global_planner

        planner = create_global_planner()
        start = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        goal = np.array([6.0, 6.0, 1.5], dtype=np.float32)
        waypoints = planner.plan(start, goal)
        self.assertGreaterEqual(len(waypoints), 1)
        # Final waypoint must be at/near the goal node (room_3 center).
        np.testing.assert_allclose(waypoints[-1][:2], goal[:2], atol=1e-3)
        # Same-node query returns the goal directly.
        direct = planner.plan(goal, goal)
        self.assertEqual(len(direct), 1)
        np.testing.assert_allclose(direct[0], goal, atol=1e-6)


class TestLocalPolicy(unittest.TestCase):
    def test_local_policy_forward_pass(self):
        """LocalPolicy forward + get_action produce valid shapes/ranges."""
        import torch
        from src.rl.local_policy import LocalPolicy
        from src.rl.policies import OBS_DIM, ACT_DIM

        torch.manual_seed(0)
        policy = LocalPolicy()
        policy.eval()
        B = 8
        obs = torch.randn(B, OBS_DIM)
        subgoal = torch.randn(B, 3)
        task_id = torch.randint(0, 5, (B,))
        mean_raw, value = policy.forward(obs, subgoal, task_id)
        self.assertEqual(tuple(mean_raw.shape), (B, ACT_DIM))
        self.assertEqual(tuple(value.shape), (B,))

        action, v, logp = policy.get_action(
            np.zeros(OBS_DIM, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            0,
            deterministic=True,
        )
        self.assertEqual(action.shape, (ACT_DIM,))
        self.assertTrue(np.all(action >= 0.0) and np.all(action <= 1.0))
        self.assertTrue(np.isfinite(v) and np.isfinite(logp))


class TestHierarchicalPolicy(unittest.TestCase):
    def test_hierarchical_policy_wrapper(self):
        """HierarchicalPolicy plans, tracks, advances, and acts on subgoals."""
        from src.rl.hierarchical import HierarchicalPolicy
        from src.rl.global_planner import create_global_planner
        from src.rl.policies import OBS_DIM, ACT_DIM

        policy = HierarchicalPolicy(global_planner=create_global_planner())
        start = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        goal = np.array([6.0, 6.0, 1.5], dtype=np.float32)
        waypoints = policy.plan_from_positions(start, goal)
        self.assertGreaterEqual(len(waypoints), 1)
        self.assertFalse(policy.is_plan_complete())

        sg = policy.get_current_subgoal()
        self.assertIsNotNone(sg)
        # Standing on the subgoal advances the plan.
        self.assertTrue(policy.update_subgoal_progress(np.asarray(sg, dtype=np.float32)))

        action, v, logp = policy.get_action(
            np.zeros(OBS_DIM, dtype=np.float32), deterministic=True
        )
        self.assertEqual(np.asarray(action).shape, (ACT_DIM,))
        self.assertTrue(np.all(np.asarray(action) >= 0.0))
        self.assertTrue(np.all(np.asarray(action) <= 1.0))


class TestHierarchicalMockBackend(unittest.TestCase):
    def test_integration_with_mock_backend(self):
        """MissionPlanner subgoals drive policy actions stepped in mock backend."""
        from src.control.mission_planner import MissionPlanner, create_default_mission
        from src.rl.global_planner import create_global_planner
        from src.rl.hierarchical import HierarchicalPolicy
        from src.rl.policies import OBS_DIM
        from src.sim.base_env import SimConfig
        from src.sim.mock_backend import MockDroneBackend

        mission = create_default_mission()
        planner = create_global_planner()
        mission_planner = MissionPlanner(
            mission, global_planner=planner, use_global_planner=True
        )
        policy = HierarchicalPolicy(global_planner=planner)

        backend = MockDroneBackend(SimConfig(mesh_path=""), sar_mission=True)
        state = backend.reset()

        drone_pos = np.asarray(state.position, dtype=np.float32)
        cmd = mission_planner.update(
            drone_pos,
            np.asarray(state.orientation, dtype=np.float32),
            dt=0.02,
        )
        self.assertEqual(cmd["skill"], "navigate_to")
        self.assertIn("subgoals", cmd)
        self.assertGreaterEqual(len(cmd["subgoals"]), 1)

        # Feed the first subgoal to the hierarchical policy and step physics.
        subgoal = np.asarray(cmd["subgoals"][0], dtype=np.float32)
        policy.set_global_plan([np.asarray(s, dtype=np.float32) for s in cmd["subgoals"]])
        action, _, _ = policy.get_action(
            np.zeros(OBS_DIM, dtype=np.float32), subgoal, 0, deterministic=True
        )
        state2, reward, done, info = backend.step(
            np.asarray(action, dtype=np.float32), 0.02, subgoal
        )
        self.assertEqual(np.asarray(state2.position).shape, (3,))
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
