#!/usr/bin/env python3
"""Control-stack tests: safety filter default-deny + bridges (5 tests).

SAR-only: navigate_to / hover / drop_payload / return_home.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _nav(x=2.1, y=1.0, z=1.5, speed=0.8):
    return {
        "skill": "navigate_to",
        "params": {"x": x, "y": y, "z": z, "yaw_deg": 90.0, "speed_ms": speed},
        "constraints": {"timeout_s": 20.0},
        "task": "medkit_AB",
        "task_embedding_id": 0,
    }


class TestSafetyFilter(unittest.TestCase):
    def test_valid_skill_passes(self):
        from src.control.safety_filter import allow

        verdict = allow(_nav())
        self.assertTrue(verdict["ok"], verdict)

    def test_out_of_geofence_rejects(self):
        from src.control.safety_filter import allow

        verdict = allow(_nav(x=99.0))
        self.assertFalse(verdict["ok"])
        self.assertIn("geofence", verdict["reason"])

    def test_non_allowlist_rejects(self):
        from src.control.safety_filter import allow

        cmd = _nav()
        cmd["skill"] = "target_lock"
        verdict = allow(cmd)
        self.assertFalse(verdict["ok"])
        self.assertIn("allowlist", verdict["reason"])

    def test_ceiling_violation_rejects(self):
        from src.control.safety_filter import allow

        verdict = allow(_nav(z=9.9))
        self.assertFalse(verdict["ok"])
        self.assertTrue(
            "ceiling" in verdict["reason"] or "geofence" in verdict["reason"],
            verdict,
        )


class TestBridges(unittest.TestCase):
    def test_bridges_record_and_heartbeat(self):
        from src.control.mavlink_bridge import MavlinkBridge
        from src.control.ros2_bridge import Ros2Bridge
        from src.control.safety_filter import allow

        cmd = _nav()
        self.assertTrue(allow(cmd)["ok"])
        mav = MavlinkBridge()
        ros = Ros2Bridge()
        sp = mav.execute(cmd)
        msg = ros.execute(cmd)
        self.assertEqual(sp["skill"], "navigate_to")
        self.assertEqual(msg["skill"], "navigate_to")
        self.assertEqual(len(mav.transport.sent), 1)
        self.assertEqual(len(ros.transport.sent), 1)
        self.assertTrue(mav.heartbeat())
        self.assertTrue(ros.heartbeat())


if __name__ == "__main__":
    unittest.main()
