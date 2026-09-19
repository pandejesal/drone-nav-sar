#!/usr/bin/env python3
"""Drones importer tests: valid + 3 rejections (4 tests).

SAR-only geometry/dynamics; no weapons.
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEMO = Path(__file__).resolve().parent.parent / "assets" / "drones" / \
    "crazyflie-like-custom" / "drone.yaml"


class TestImporter(unittest.TestCase):
    def test_valid_loads_hash_stable(self):
        from src.drones.importer import load_drone, parse_subset_yaml

        spec = load_drone(DEMO)
        self.assertEqual(spec.api_version, 2)
        self.assertTrue(spec.spec_hash)
        again = load_drone(DEMO)
        self.assertEqual(spec.spec_hash, again.spec_hash)
        self.assertAlmostEqual(spec.params.mass, 0.027)
        # thrust margin: 4*0.62N >> weight
        self.assertGreater(4.0 * spec.params.max_thrust, 0.027 * 9.81 * 1.2)
        # dict path also works
        d = parse_subset_yaml(DEMO.read_text(encoding="utf-8"))
        via_dict = load_drone(d)
        self.assertEqual(via_dict.name, spec.name)

    def test_raw_rpm_rejected(self):
        from src.drones.importer import load_drone, parse_subset_yaml

        d = parse_subset_yaml(DEMO.read_text(encoding="utf-8"))
        d = copy.deepcopy(d)
        d["propulsion"]["thrust_unit"] = "rpm"
        with self.assertRaises(ValueError):
            load_drone(d)

    def test_bad_mass_rejected(self):
        from src.drones.importer import load_drone, parse_subset_yaml

        d = parse_subset_yaml(DEMO.read_text(encoding="utf-8"))
        d = copy.deepcopy(d)
        d["geometry"]["mass_kg"] = -1.0
        with self.assertRaises(ValueError):
            load_drone(d)

    def test_weak_thrust_rejected(self):
        from src.drones.importer import load_drone, parse_subset_yaml

        d = parse_subset_yaml(DEMO.read_text(encoding="utf-8"))
        d = copy.deepcopy(d)
        d["propulsion"]["max_thrust_N"] = 0.01
        with self.assertRaises(ValueError):
            load_drone(d)


if __name__ == "__main__":
    unittest.main()
