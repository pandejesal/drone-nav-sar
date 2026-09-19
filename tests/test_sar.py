#!/usr/bin/env python3
"""SAR Mission tests for DroneNav-SAR (Sprint 7).

4 tests:
1. Victim detection (multi-room layout + YOLO mock + backend range)
2. Med-kit drop mechanics (drop_payload skill)
3. Mission state machine transitions (IDLE -> ... -> COMPLETE)
4. Full SAR mission integration (backend + detector + planner)
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestVictimDetection(unittest.TestCase):
    """1. Victim detection: layout spawn, YOLO mock FOV, backend range."""

    def test_victim_detection(self):
        from src.sim.mock_backend import MockDroneBackend
        from src.sim.base_env import SimConfig
        from src.control.yolo_detector import MockYOLODetector
        import numpy as np

        # 4 rooms, one victim spot each (from mock backend default)
        config = SimConfig(mesh_path="mock.glb", camera_enabled=False,
                           max_episode_steps=500)
        backend = MockDroneBackend(config, curriculum_level=3, sar_mission=True)
        backend.reset()

        self.assertTrue(backend.sar_mission)
        self.assertEqual(len(backend.victims), 4)
        self.assertEqual(backend.total_victims, 4)
        self.assertTrue(backend.medkit_carried)
        self.assertEqual(backend.victims_served, 0)

        # Test victim detection range
        victim = backend.victims[0]
        drone_pos = np.array([victim.position[0], victim.position[1], 1.5], dtype=np.float32)
        dist = float(np.linalg.norm(drone_pos[:2] - victim.position[:2]))
        self.assertLess(dist, 5.0)  # within detection range

        # Test YOLO detector mock
        detector = MockYOLODetector()
        detections = detector.detect(drone_pos, np.array([1,0,0,0], dtype=np.float32))
        self.assertIsInstance(detections, list)
        self.assertEqual(len(backend.victims), 4)
        self.assertTrue(backend.medkit_carried)

        # YOLO mock: detects victim in FOV, empty otherwise.
        # Drone hovers 2m beside the victim facing it (+x forward).
        detector = MockYOLODetector(rng=np.random.default_rng(0))
        victim_pos = np.asarray(backend.victims[0].position, dtype=np.float32)
        detector.set_victims([victim_pos])
        near_pos = victim_pos + np.array([-2.0, 0.0, 1.0], dtype=np.float32)
        facing_x = np.array([0.7071, 0.0, 0.7071, 0.0])  # forward = +x
        dets = detector.detect(near_pos, facing_x)
        self.assertTrue(any(d.class_name == "victim" for d in dets))
        far_pos = np.array([100.0, 100.0, 1.5], dtype=np.float32)
        self.assertEqual(detector.detect(far_pos, facing_x), [])


class TestMedkitDrop(unittest.TestCase):
    """2. Med-kit drop mechanics (auto-drop on proximity)."""

    def test_drop_mechanics(self):
        from src.sim.mock_backend import MockDroneBackend, Victim
        from src.sim.base_env import SimConfig
        import numpy as np

        config = SimConfig(mesh_path="mock.glb", camera_enabled=False,
                           max_episode_steps=500)
        backend = MockDroneBackend(config, curriculum_level=3, sar_mission=True)
        backend.reset()
        victim = Victim(np.array([2.5, 2.5, 0.0], dtype=np.float32))
        backend.victims = [victim]
        backend.total_victims = 1
        backend.medkit_carried = True

        # Step at victim location triggers auto-drop
        # Must update internal state position before stepping
        backend._state_vec[0:3] = np.array([2.5, 2.5, 1.0], dtype=np.float32)
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        state, reward, done, info = backend.step(action, 0.02, None)

        # Check med-kit was dropped
        self.assertFalse(backend.medkit_carried)
        self.assertTrue(victim.served)
        self.assertEqual(backend.victims_served, 1)
        self.assertTrue(info.get("drop_medkit", False))
        self.assertTrue(info.get("victim_served", False))

        # Second step WITHOUT medkit doesn't drop
        backend.medkit_carried = False
        victim2 = Victim(np.array([5.0, 5.0, 0.0], dtype=np.float32))
        backend.victims = [victim2]
        backend.total_victims = 1
        backend._state_vec[0:3] = np.array([5.0, 5.0, 1.0], dtype=np.float32)
        state, reward, done, info = backend.step(action, 0.02, None)
        self.assertFalse(info.get("drop_medkit", False))
        self.assertFalse(victim2.served)

        # With medkit but far from victim doesn't trigger
        victim3 = Victim(np.array([100.0, 100.0, 0.0], dtype=np.float32))
        backend.victims = [victim3]
        backend.total_victims = 1
        backend.medkit_carried = True
        backend._state_vec[0:3] = np.array([10.0, 10.0, 1.0], dtype=np.float32)
        state, reward, done, info = backend.step(action, 0.02, None)
        self.assertFalse(info.get("drop_medkit", False))
        self.assertFalse(victim3.served)


class TestMissionStateMachine(unittest.TestCase):
    """3. Mission planner state machine transitions."""

    def test_mission_state_machine(self):
        from src.control.mission_planner import (
            MissionPlanner, MissionState, create_default_mission)

        config = create_default_mission()
        planner = MissionPlanner(config, use_global_planner=False)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertEqual(planner.state, MissionState.IDLE)

        cmd = planner.update(np.array([0.0, 0.0, 1.5]), quat, 0.02)
        self.assertEqual(planner.state, MissionState.NAV_TO_ROOM)
        self.assertEqual(cmd["skill"], "navigate_to")

        cmd = planner.update(np.array([2.0, 2.0, 1.5]), quat, 0.02,
                             goal_reached=True)
        self.assertEqual(planner.state, MissionState.SEARCH_VICTIM)
        self.assertEqual(cmd["skill"], "hover")

        cmd = planner.update(np.array([2.0, 2.0, 1.5]), quat, 0.02,
                             victim_detected=True,
                             victim_position=np.array([2.5, 2.5, 0.0]))
        self.assertEqual(planner.state, MissionState.HOVER_OVER_VICTIM)

        cmd = planner.update(np.array([2.5, 2.5, 1.5]), quat, 0.02)
        self.assertEqual(planner.state, MissionState.DROP_MEDKIT)
        self.assertEqual(cmd["skill"], "drop_payload")

        cmd = planner.update(np.array([2.5, 2.5, 1.5]), quat, 0.02,
                             drop_complete=True)
        self.assertEqual(planner.state, MissionState.NAV_TO_ROOM)

        # Crash anywhere funnels to FAILED -> return_home
        planner2 = MissionPlanner(config, use_global_planner=False)
        planner2.update(np.array([0.0, 0.0, 1.5]), quat, 0.02)
        cmd = planner2.update(np.array([0.0, 0.0, 1.5]), quat, 0.02,
                              crashed=True)
        self.assertEqual(planner2.state, MissionState.FAILED)
        self.assertEqual(cmd["skill"], "return_home")


class TestFullMission(unittest.TestCase):
    """4. Full mission: backend + planner + drop through last room to COMPLETE."""

    def test_full_mission(self):
        from src.sim.mock_backend import MockDroneBackend
        from src.sim.base_env import SimConfig
        from src.control.mission_planner import (
            MissionPlanner, MissionState, create_default_mission)

        config = SimConfig(mesh_path="mock.glb", camera_enabled=False,
                           max_episode_steps=500)
        backend = MockDroneBackend(config, curriculum_level=3, sar_mission=True)
        backend.reset()

        mission = create_default_mission()
        planner = MissionPlanner(mission, use_global_planner=False)
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Drive every room: nav -> search -> detect -> hover -> drop
        for room_idx in range(len(mission.rooms)):
            planner.update(np.array([0.0, 0.0, 1.5]), quat, 0.02)
            room = mission.rooms[room_idx]
            planner.update(room.center, quat, 0.02, goal_reached=True)
            victim_pos = room.victims[0]
            planner.update(room.center, quat, 0.02, victim_detected=True,
                           victim_position=victim_pos)
            planner.update(np.array([victim_pos[0], victim_pos[1], 1.5]),
                           quat, 0.02)
            self.assertEqual(planner.state, MissionState.DROP_MEDKIT)
            planner.update(np.array([victim_pos[0], victim_pos[1], 1.5]),
                           quat, 0.02, drop_complete=True)

        self.assertEqual(planner.state, MissionState.RETURN_HOME)
        cmd = planner.update(mission.home_position, quat, 0.02,
                             home_reached=True)
        self.assertEqual(planner.state, MissionState.COMPLETE)
        self.assertEqual(cmd["skill"], "hover")

        # Backend SAR info keys present on step
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        _, _, _, info = backend.step(action, 0.02, None)
        for key in ("sar_mission", "victim_detected", "victims_served",
                    "total_victims", "medkit_carried", "drop_medkit"):
            self.assertIn(key, info)
        self.assertTrue(info["sar_mission"])


if __name__ == "__main__":
    unittest.main()
