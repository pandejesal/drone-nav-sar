#!/usr/bin/env python3
"""Sprint 24 tests: HJ reachability, barrier safety, LTL liveness, CBF/SOS."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _straight_line(start, goal, n=11, speed_ok=True):
    P = np.linspace(np.asarray(start, float), np.asarray(goal, float), n)
    S = np.zeros((n, 6))
    S[:, 0:3] = P
    S[:-1, 3:6] = (P[1:] - P[:-1]) / 0.5  # dt=0.5 -> ~2 m/s, within limits
    S[-1, 3:6] = S[-2, 3:6]
    if not speed_ok:
        S[:, 3:6] *= 100.0
    return S


class TestHJReachability(unittest.TestCase):
    def test_reachability_target_in_backward_set(self):
        from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig
        val = AutonomyValidator(ValidatorConfig(goal_radius=2.0))
        goal = np.array([0.0, 0.0, 10.0])
        brs = val.hj_backward_reachable_set(goal, horizon=2.0, n_bins=8)
        self.assertTrue(brs["reachable"])
        self.assertGreater(brs["reachable_fraction"], 0.0)
        # Trajectory coverage: 100% of scenarios hit the goal ball.
        trajs = [_straight_line([-20.0, 0.0, 10.0], goal),
                 _straight_line([20.0, 0.0, 10.0], goal)]
        reach = val.check_reachability(trajs, goal)
        self.assertEqual(reach["hit_rate"], 1.0)


class TestBarrierSafety(unittest.TestCase):
    def test_safety_zero_violations_in_episode(self):
        from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig
        obs = [np.array([30.0, 30.0, 10.0])]
        val = AutonomyValidator(ValidatorConfig(), obstacles=obs)
        S = _straight_line([-20.0, 0.0, 10.0], [0.0, 0.0, 10.0])
        res = val.validate_episode(S, np.array([0.0, 0.0, 10.0]))
        self.assertEqual(res.safety_violations, [])
        cert = val.verify_barrier_certificate(
            S[:, 0:3], np.array([[30.0, 30.0, 10.0]]))
        self.assertTrue(cert["certified"])
        self.assertTrue(cert["sos_feasible"])


class TestLTLLiveness(unittest.TestCase):
    def test_liveness_eventually_reaches_goal(self):
        from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig
        val = AutonomyValidator(ValidatorConfig(goal_radius=1.0))
        goal = np.array([5.0, 0.0, 10.0])
        P = np.linspace(np.array([-5.0, 0.0, 10.0]), goal, 11)
        for formula, expected in (("F goal", True),
                                  ("G safe", True),
                                  ("G (request -> F response)", True)):
            res = val.check_ltl(P, goal, formula)
            self.assertTrue(res["satisfied"], msg=formula)
            self.assertEqual(res["formula"], formula)
        bad = val.check_ltl(P, np.array([45.0, 45.0, 10.0]), "F goal")
        self.assertFalse(bad["satisfied"])
        self.assertIsNotNone(bad["counterexample_step"])


class TestCBFVerification(unittest.TestCase):
    def test_cbf_sos_feasible_and_qp_safe(self):
        from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig
        obs = [np.array([30.0, 30.0, 10.0])]
        val = AutonomyValidator(ValidatorConfig(), obstacles=obs)
        S = _straight_line([-20.0, 0.0, 10.0], [0.0, 0.0, 10.0])
        cbf = val.verify_cbf_sos(S[:, 0:3])
        self.assertTrue(cbf["holds"])
        self.assertEqual(cbf["relative_degree"], 1)
        self.assertTrue(cbf["sos_feasible"])
        qp = val.synthesize_safe_control(np.array([0.0, 0.0, 10.0]),
                                         np.array([1.0, 0.0, 0.0]))
        self.assertTrue(qp["satisfies_cbf"])


if __name__ == "__main__":
    unittest.main()
