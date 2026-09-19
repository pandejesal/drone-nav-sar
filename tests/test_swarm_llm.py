#!/usr/bin/env python3
"""Sprint 21 Human-Swarm Teaming tests (SAR-only).

4 tests:
1. Parser accuracy (>0.9 over SAR command battery)
2. Intent → task graph (floors map to nodes, roles fit drone cap)
3. Execution (role election <5 s for 10 drones, allocation >90% optimal)
4. Latency (text → intent <500 ms, text → drone action <2 s)
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestParserAccuracy(unittest.TestCase):
    """1. SwarmLLM parses SAR commands with >90% intent accuracy."""

    def test_parser_accuracy(self):
        from src.control.swarm_llm import estimate_parser_accuracy

        commands = [
            "search building 3 for victims",
            "scan floor 2 of building 1",
            "search building 1 floors 1 to 2 for survivors",
            "find victims in building 2",
            "deliver medkit to room 2 in building 3",
            "drop payload at victim site building 1",
            "survey building 2 rooftops",
            "map building 1 for damage",
            "patrol building 1 floors 1 to 2",
            "monitor building 3 perimeter",
            "all drones return home",
            "recall all units to base",
            "search all floors of building 2",
            "urgent search building 3 for victims",
            "survey building 1 with 4 drones",
            "search building 2 in 20 min",
            "inspect building 3 rooftop",
            "patrol building 2",
            "deliver medicine to room 1",
            "land all drones now",
        ]
        expected = [
            "search_and_rescue", "search_and_rescue", "search_and_rescue",
            "search_and_rescue", "deliver_payload", "deliver_payload",
            "survey_area", "survey_area", "patrol", "patrol",
            "return_home", "return_home", "search_and_rescue",
            "search_and_rescue", "survey_area", "search_and_rescue",
            "survey_area", "patrol", "deliver_payload", "return_home",
        ]
        acc = estimate_parser_accuracy(commands, expected)
        self.assertGreaterEqual(acc, 0.9, f"parser accuracy {acc:.2f} < 0.9")
        # Schema validation holds on the headline command.
        from src.control.swarm_llm import parse_swarm_command
        intent = parse_swarm_command("search building 3 for victims")
        d = intent.to_dict()
        self.assertEqual(d["intent"], "search_and_rescue")
        self.assertEqual(d["area"]["building"], "building_3")
        self.assertEqual([o["area"] for o in d["objectives"]],
                         ["floor_1", "floor_2", "floor_3"])
        self.assertLessEqual(sum(d["roles"].values()),
                             d["constraints"]["max_drones"])


class TestIntentToTaskGraph(unittest.TestCase):
    """2. Intent → ordered task graph with role assignment."""

    def test_intent_to_task_graph(self):
        from src.control.swarm_llm import intent_to_task_graph, parse_swarm_command

        intent = parse_swarm_command("search building 3 for victims")
        graph = intent_to_task_graph(intent)
        self.assertEqual(len(graph), 3)
        self.assertEqual([n["node_id"] for n in graph], ["t0", "t1", "t2"])
        self.assertEqual(graph[1]["depends_on"], ["t0"])
        for n in graph:
            self.assertIn(n["skill"], ("navigate_to", "hover",
                                       "drop_payload", "return_home"))
        # Coordinator converts the intent into SAR tasks + roles.
        from src.control.swarm_coordinator import SwarmCoordinator
        coord = SwarmCoordinator(drone_ids=[0, 1, 2, 3, 4, 5])
        coord.heartbeat_all(t=0.0)
        tasks = coord.apply_swarm_intent(intent)
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(t.kind == "search_room" for t in tasks))
        roles = coord.elect_swarm_roles()
        self.assertEqual(len(roles), 6)
        self.assertLessEqual(sum(intent.roles.values()), 6)


class TestSwarmExecution(unittest.TestCase):
    """3. Role election converges <5 s; allocation >90% vs optimal."""

    def test_swarm_execution(self):
        from src.control.swarm_coordinator import SwarmCoordinator
        from src.control.swarm_llm import parse_swarm_command

        coord = SwarmCoordinator(drone_ids=list(range(10)))
        coord.heartbeat_all(t=0.0)
        intent = parse_swarm_command("search building 3 for victims",
                                     max_drones=10)
        t0 = time.perf_counter()
        roles = coord.elect_swarm_roles()
        elect_s = time.perf_counter() - t0
        self.assertEqual(len(roles), 10)
        self.assertLess(elect_s, 5.0)
        self.assertLess(getattr(coord, "swarm_election_time_s", elect_s), 5.0)
        # Allocation quality: greedy nearest vs brute-force optimal distance.
        import itertools
        tasks = coord.apply_swarm_intent(intent)[:4]
        rng = np.random.default_rng(21)
        positions = {d: rng.uniform(0, 20, 3).astype(np.float32)
                     for d in coord.drone_ids}
        graph = coord.allocate_from_task_graph(tasks, positions)
        assigned = [t for v in graph.values() for t in v]
        self.assertEqual(len(assigned), len(tasks))
        alloc = coord.assignment
        greedy_cost = 0.0
        for d, tid in alloc.items():
            if tid is None:
                continue
            goal = next(t.goal for t in tasks if t.task_id == tid)
            greedy_cost += float(np.linalg.norm(positions[d] - goal))
        best = float("inf")
        for combo in itertools.permutations(coord.drone_ids, len(tasks)):
            c = sum(float(np.linalg.norm(positions[d] - tasks[i].goal))
                    for i, d in enumerate(combo))
            best = min(best, c)
        self.assertGreaterEqual(best / max(greedy_cost, 1e-9), 0.9,
                                f"allocation {greedy_cost:.1f} vs opt {best:.1f}")


class TestSwarmLatency(unittest.TestCase):
    """4. text → intent <500 ms; text → drone action <2 s."""

    def test_swarm_latency(self):
        from src.control.mission_planner import MissionPlanner, create_default_mission
        from src.control.swarm_llm import parse_swarm_command

        intent = parse_swarm_command("search building 3 for victims")
        self.assertLess(intent.latency_ms, 500.0,
                        f"intent latency {intent.latency_ms:.1f} ms >= 500")
        cfg = create_default_mission()
        planner = MissionPlanner(config=cfg, use_global_planner=False)
        t0 = time.perf_counter()
        cmd = planner.plan_from_swarm_text(
            "search building 3 for victims",
            np.array([0.0, 0.0, 1.5], dtype=np.float32))
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(e2e_ms, 2000.0, f"e2e latency {e2e_ms:.1f} ms >= 2 s")
        self.assertLess(cmd["latency_ms"], 2000.0)
        self.assertIn(cmd["skill"], ("navigate_to", "hover", "drop_payload",
                                     "return_home"))
        self.assertEqual(cmd["swarm_intent"], "search_and_rescue")
        self.assertTrue(cmd["swarm_tasks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
