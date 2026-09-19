#!/usr/bin/env python3
"""Sprint 20: Sensor Fusion COP tests (SAR-only). 4 tests."""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFusionAccuracy(unittest.TestCase):
    def test_fusion_accuracy(self):
        from src.perception.sensor_fusion import SensorFusion, make_detection
        rng = np.random.default_rng(20)
        gt = [np.array([3.0, 2.0, 0.5]), np.array([7.0, 6.0, 0.5])]
        fusion = SensorFusion()
        for step in range(15):
            dets = []
            for g in gt:
                for sensor in ("rgb", "thermal", "lidar"):
                    noisy = g + rng.normal(0, 0.2, 3)
                    dets.append(make_detection(noisy, sensor, "victim", 0.9,
                                              timestamp=step * 0.1))
            fusion.step(dets, dt=0.1)
        purity = fusion.track_purity(gt, tol_m=1.0)
        false_rate = fusion.false_track_rate(gt, tol_m=1.0)
        self.assertGreater(purity, 0.95)
        self.assertLess(false_rate, 0.05)


class TestTrackContinuity(unittest.TestCase):
    def test_track_continuity(self):
        from src.perception.sensor_fusion import SensorFusion, make_detection
        rng = np.random.default_rng(21)
        fusion = SensorFusion()
        start_ids = None
        persist = 0
        total = 0
        for step in range(60):
            g = np.array([step * 0.1, 2.0, 0.5])
            dets = [make_detection(g + rng.normal(0, 0.15, 3), "lidar",
                                  "victim", 0.95, timestamp=step * 0.1)]
            tracks = fusion.step(dets, dt=0.1)
            ids = sorted(tracks.keys())
            total += 1
            if start_ids is None and ids:
                start_ids = ids
            if start_ids and ids == start_ids:
                persist += 1
        continuity = persist / max(total, 1)
        self.assertGreater(continuity, 0.99)


class TestCOPLatency(unittest.TestCase):
    def test_cop_latency(self):
        from src.perception.sensor_fusion import SensorFusion, make_detection
        from src.perception.cop import COPEngine
        fusion = SensorFusion()
        cop = COPEngine()
        dets = [make_detection(np.array([2.0, 2.0, 0.5]), s, "victim", 0.9)
                for s in ("rgb", "thermal", "lidar")]
        t0 = time.perf_counter()
        tracks = fusion.step(dets, dt=0.1)
        cop.update_from_tracks(tracks)
        snap = cop.snapshot()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(latency_ms, 50.0)
        self.assertGreater(len(snap["entities"]), 0)
        e = snap["entities"][0]
        for key in ("entity_id", "type", "state", "classification",
                    "timestamp", "source_sensors", "uncertainty"):
            self.assertIn(key, e)


class TestCOAQuality(unittest.TestCase):
    def test_coa_quality(self):
        from src.perception.sensor_fusion import SensorFusion, make_detection
        from src.perception.cop import COPEngine
        from src.control.mission_planner import create_default_mission, MissionPlanner
        fusion = SensorFusion()
        cop = COPEngine()
        for i in range(10):
            g = np.array([float(i), float(i % 3), 0.5])
            dets = [make_detection(g, "lidar", "victim", 0.9, timestamp=i * 0.01)]
            fusion.step(dets, dt=0.01)
        entities = cop.update_from_tracks(fusion.tracks)
        planner = MissionPlanner(create_default_mission())
        t0 = time.perf_counter()
        coas = planner.generate_coas(entities, np.array([0.0, 0.0, 1.5]))
        best = planner.select_coa(coas)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 500.0)
        self.assertIn("sequence", best)
        self.assertTrue(all(s in ("navigate_to", "hover", "drop_payload", "return_home")
                            for s in best["sequence"]))


if __name__ == "__main__":
    unittest.main()
