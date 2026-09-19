#!/usr/bin/env python3
"""Sprint 22 tests: state sync, HIL loop, reachability, safety invariants."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDigitalTwinSync(unittest.TestCase):
    def test_state_sync_converges_below_5cm(self):
        from src.sim.digital_twin import DigitalTwin, TwinState
        twin = DigitalTwin()
        hw = TwinState(position=np.array([1.0, 2.0, 10.0]),
                       velocity=np.array([0.5, -0.2, 0.1]),
                       timestamp=1.0)
        for _ in range(20):
            twin.step(np.zeros(3))
            twin.sync_from_hardware(hw)
        pos_err, vel_err = twin.sync_error(hw)
        self.assertLess(pos_err, 0.05)   # 5 cm
        self.assertLess(vel_err, 0.1)    # 0.1 m/s


class TestHILLoop(unittest.TestCase):
    def test_hil_loop_latency_within_budget(self):
        from src.sim.digital_twin_bridge import DigitalTwinBridge
        bridge = DigitalTwinBridge()
        bridge.connect()

        def hw_step(u):
            s = np.zeros(13)
            s[6] = 1.0
            s[0:3] = np.array([0.1, 0.0, 10.0])
            s[3:6] = np.array([0.2, 0.0, 0.0])
            return s
        bridge.hil.hardware_step = hw_step
        summary = bridge.run([np.zeros(3) for _ in range(5)])
        self.assertLess(summary["mean_roundtrip_ms"], 20.0)
        self.assertLess(summary["max_roundtrip_ms"], 20.0)
        self.assertTrue(summary["all_within_budget"])


class TestReachability(unittest.TestCase):
    def test_reachability_full_coverage(self):
        from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig
        val = AutonomyValidator(ValidatorConfig(n_reach_bins=2, goal_radius=5.0))
        goal = np.array([0.0, 0.0, 10.0])
        trajs = []
        for x in (-40.0, 40.0):
            for y in (-40.0, 40.0):
                for z in (5.0, 15.0):
                    T = np.zeros((4, 13))
                    T[:, 2] = z
                    T[0, 0:3] = [x, y, z]
                    T[1, 0:3] = [x / 2, y / 2, z]
                    T[2, 0:3] = [0.0, 0.0, 10.0]
                    T[3, 0:3] = [0.0, 0.0, 10.0]
                    T[:, 3:6] = 0.0
                    trajs.append(T)
        rep = val.check_reachability(trajs, goal)
        self.assertEqual(rep["hit_rate"], 1.0)
        self.assertGreater(rep["coverage"], 0.0)


class TestSafetyInvariants(unittest.TestCase):
    def test_safety_zero_violations(self):
        from src.sim.autonomy_validator import AutonomyValidator
        val = AutonomyValidator()
        S = np.zeros((50, 13))
        S[:, 2] = 10.0                       # safe altitude
        S[:, 0] = np.linspace(-5, 5, 50)     # inside geofence
        S[:, 3] = 1.0                        # 1 m/s, under limit
        episodes = [(S, np.array([5.0, 0.0, 10.0])) for _ in range(4)]
        agg = val.validate_batch(episodes)
        self.assertEqual(agg["safety_violations"], 0)
        self.assertEqual(agg["violation_free_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
