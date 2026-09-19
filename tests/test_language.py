#!/usr/bin/env python3
"""Sprint 11 language interface tests (SAR-only, 4 tests)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTextEncoder(unittest.TestCase):
    def test_encoder_output_shape_normalized_deterministic(self):
        from src.control.language_interface import get_encoder
        enc = get_encoder()
        e1 = enc.encode("deliver medkit to room 2")
        self.assertEqual(tuple(e1.shape), (512,))
        self.assertAlmostEqual(float(e1.norm()), 1.0, places=5)
        e2 = enc.encode("deliver medkit to room 2")
        np.testing.assert_allclose(e1.numpy(), e2.numpy(), rtol=0, atol=0)
        batch = enc.encode(["go to room 1", "return home"])
        self.assertEqual(tuple(batch.shape), (2, 512))


class TestProjection(unittest.TestCase):
    def test_projection_to_4dim_task_embedding(self):
        import torch
        from src.control.language_interface import (
            encode_text, project_to_task_embedding, text_to_task_embedding,
        )
        emb = text_to_task_embedding("deliver medkit to room 2")
        self.assertEqual(tuple(emb.shape), (4,))
        self.assertTrue(bool(torch.isfinite(emb).all()))
        batch = project_to_task_embedding(
            torch.stack([encode_text("go to room 1"), encode_text("return home")]))
        self.assertEqual(tuple(batch.shape), (2, 4))


class TestPlannerIntegration(unittest.TestCase):
    def test_planner_parses_deliver_to_room2(self):
        from src.control.mission_planner import MissionPlanner, create_default_mission
        planner = MissionPlanner(create_default_mission(), use_global_planner=False)
        parsed = planner.parse_language_command("deliver medkit to room 2")
        self.assertEqual(parsed["task_id"], 3)
        self.assertEqual(parsed["params"].get("room"), 2)
        cmd = planner.plan_from_language("deliver medkit to room 2",
                                         np.array([0.0, 0.0, 1.5], dtype=np.float32))
        self.assertIn(cmd["skill"], ("navigate_to", "drop_payload", "return_home", "hover"))
        # Unknown command falls back to nav-home.
        fb = planner.parse_language_command("blarg zzz wobble")
        self.assertEqual(fb["task_id"], 0)


class TestFullPipeline(unittest.TestCase):
    def test_text_to_action_and_clipscore(self):
        import torch
        from src.control.language_interface import text_to_task_embedding
        from src.control.language_validator import clip_score, validate_command
        from src.control.mission_planner import MissionPlanner, create_default_mission
        from src.rl.local_policy import LocalPolicy

        torch.manual_seed(11)
        policy = LocalPolicy()
        text = "deliver medkit to room 2"
        task_emb = text_to_task_embedding(text)
        self.assertEqual(tuple(task_emb.shape), (4,))

        obs = np.zeros(21, dtype=np.float32)
        subgoal = np.zeros(3, dtype=np.float32)
        action, value, logp, task_id, params = policy.get_action_for_text(
            obs, subgoal, text, deterministic=True)
        self.assertEqual(task_id, 3)
        self.assertEqual(params.get("room"), 2)
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= 0.0) and np.all(action <= 1.0))

        planner = MissionPlanner(create_default_mission(), use_global_planner=False)
        cmd = planner.plan_from_language(text, np.zeros(3, dtype=np.float32))
        self.assertEqual(cmd["task_embedding_id"], 3)

        res = validate_command(text)
        self.assertGreaterEqual(res["clip_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
