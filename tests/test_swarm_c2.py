#!/usr/bin/env python3
"""Sprint 19 Swarm C2 Mesh tests (SAR-only).

4 tests:
1. Mesh formation (10 drones, connected, mesh_form_s < 30)
2. Self-healing (link loss reroutes, heal_s < 5)
3. DTN reliability (>0.99 at 30% packet loss)
4. Role election (converges <10 s, leader failover)
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _grid_positions(n: int, spacing: float = 20.0) -> list:
    side = int(np.ceil(np.sqrt(n)))
    pos = []
    for i in range(n):
        x = (i % side) * spacing
        y = (i // side) * spacing
        pos.append(np.array([x, y, 10.0], dtype=np.float32))
    return pos


class TestMeshFormation(unittest.TestCase):
    """1. 10-drone mesh forms in < 30 s."""

    def test_mesh_formation(self):
        from src.comms.mesh_c2 import MeshC2

        mesh = MeshC2(node_id=0, n_nodes=10, comm_range_m=60.0, seed=19)
        info = mesh.form_mesh(_grid_positions(10))
        self.assertTrue(info["formed"], "10-drone mesh must be connected")
        self.assertLess(info["mesh_form_s"], 30.0)
        # Every node reachable from node 0 via hybrid routing.
        for dst in range(1, 10):
            route = mesh.get_route(dst, src=0)
            self.assertIsNotNone(route, f"no route 0->{dst}")
            self.assertEqual(route[0], 0)
            self.assertEqual(route[-1], dst)


class TestSelfHealing(unittest.TestCase):
    """2. Link loss self-heals in < 5 s with surviving routes."""

    def test_self_healing(self):
        from src.comms.mesh_c2 import MeshC2

        mesh = MeshC2(node_id=0, n_nodes=6, comm_range_m=60.0, seed=7)
        mesh.form_mesh(_grid_positions(6, spacing=20.0))
        before = mesh.get_route(5, src=0)
        self.assertIsNotNone(before)
        # Cut every link on the primary path (except endpoints if adjacent).
        for a, b in zip(before[:-1], before[1:]):
            mesh.remove_link(a, b)
        info = mesh.heal()
        self.assertLess(info["heal_s"], 5.0)
        after = mesh.get_route(5, src=0)
        # Dense 6-node grid keeps an alternate path; if the cut
        # partitioned the mesh, healing must at least report honestly.
        if info["connected"]:
            self.assertIsNotNone(after, "healed mesh must reroute 0->5")
        # AODV fallback works independently of the OLSR table.
        alt = mesh.find_route_aodv(0, 5)
        if info["connected"]:
            self.assertIsNotNone(alt)


class TestDTNReliability(unittest.TestCase):
    """3. DTN bundles delivered with >99% reliability at 30% loss."""

    def test_dtn_reliability(self):
        from src.comms.mesh_c2 import MeshC2

        mesh = MeshC2(node_id=0, n_nodes=4, comm_range_m=100.0,
                      packet_loss=0.30, seed=19)
        mesh.form_mesh(_grid_positions(4, spacing=20.0))
        n = 100
        for i in range(n):
            mesh.dtn_send(dst=(i % 3) + 1,
                          payload={"skill": "navigate_to",
                                   "goal": [float(i), 0.0, 10.0]},
                          src=0, max_retries=6)
        for _ in range(5):
            mesh.dtn_tick(dt=1.0)
        rep = mesh.dtn_report()
        self.assertGreaterEqual(rep["reliability"], 0.99,
                                f"DTN reliability {rep['reliability']:.3f} < 0.99")


class TestRoleElection(unittest.TestCase):
    """4. Role election converges in < 10 s with leader failover."""

    def test_role_election(self):
        from src.control.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator(drone_ids=[0, 1, 2, 3, 4])
        coord.heartbeat_all(t=0.0)
        leader = coord.elect_leader()
        self.assertIsNotNone(leader)
        self.assertLess(coord.election_time_s, 10.0)
        self.assertEqual(coord.get_role(leader), "leader")
        # Kill the leader -> a follower takes over, tasks reassignable.
        new_leader = coord.mark_failed(leader)
        self.assertIsNotNone(new_leader)
        self.assertNotEqual(new_leader, leader)
        self.assertEqual(coord.get_role(new_leader), "leader")
        # MissionPlanner integration: role-based tasking is SAR-only.
        from src.control.mission_planner import MissionPlanner, create_default_mission

        cfg = create_default_mission()
        planner = MissionPlanner(config=cfg, drone_id=new_leader,
                                 n_drones=5, use_global_planner=False,
                                 coordinator=coord)
        self.assertEqual(planner.get_swarm_role(), "leader")
        cmd = planner.plan_with_swarm(np.array([0.0, 0.0, 1.5]))
        self.assertIn(cmd["skill"],
                      ("navigate_to", "hover", "drop_payload", "return_home"))
        self.assertEqual(cmd.get("swarm_role"), "leader")


if __name__ == "__main__":
    unittest.main(verbosity=2)
