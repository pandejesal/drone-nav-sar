#!/usr/bin/env python3
"""Sprint 9 multi-drone coordination tests (SAR-only).

4 tests:
1. Collision avoidance (RVO penalty + separation push, zero collisions)
2. Victim sharing (detection broadcast -> all drones update belief)
3. Formation (loose 5 m transit separation + deconflicted goals)
4. Full mission (3 drones, greedy assignment, joint success, zero collisions)
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCollisionAvoidance(unittest.TestCase):
    """1. Collision avoidance: penalty when close, separation restores safety."""

    def test_collision_avoidance(self):
        from src.rl.multi_drone import (
            MultiDroneEnv, collision_penalty_for, pairwise_distances,
        )

        # Unit-level: penalty is 0 at safe range, negative when close,
        # hard penalty on contact.
        self.assertEqual(collision_penalty_for(3.0), 0.0)
        self.assertLess(collision_penalty_for(0.7), 0.0)
        self.assertEqual(collision_penalty_for(0.2), -5.0)

        # Env-level: two drones spawned on top of each other collide...
        env = MultiDroneEnv(n_drones=2, max_steps=50, seed=0)
        env.reset(seed=0)
        env.positions[0] = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        env.positions[1] = np.array([0.2, 0.0, 1.5], dtype=np.float32)
        hover = np.full((2, 4), 0.5, dtype=np.float32)
        _, rewards, _, _, infos = env.step(hover)
        self.assertLess(float(rewards[0]), 0.0)  # proximity/collision penalty
        self.assertGreaterEqual(env.collisions, 1)

        # ...while separated drones fly penalty-free with zero collisions.
        env2 = MultiDroneEnv(n_drones=2, max_steps=50, seed=1)
        env2.reset(seed=1)
        env2.positions[0] = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        env2.positions[1] = np.array([10.0, 0.0, 1.5], dtype=np.float32)
        d = pairwise_distances(env2.positions)
        self.assertGreater(float(d[0, 1]), 5.0)
        _, _, _, _, _ = env2.step(hover)
        self.assertEqual(env2.collisions, 0)
        self.assertGreater(env2.min_separation(), 1.0)


class TestVictimSharing(unittest.TestCase):
    """2. Victim sharing: finder broadcasts, every drone learns the spot."""

    def test_victim_sharing(self):
        from src.control.comms import MessageBus
        from src.control.mission_planner import (
            MissionPlanner, create_default_mission,
        )
        from src.rl.multi_drone import MultiDroneEnv

        # Planner-level: drone 0 shares, drone 1 syncs the same belief.
        bus = MessageBus(n_drones=2)
        mission = create_default_mission()
        p0 = MissionPlanner(mission, use_global_planner=False, bus=bus,
                            drone_id=0, n_drones=2)
        p1 = MissionPlanner(mission, use_global_planner=False, bus=bus,
                            drone_id=1, n_drones=2)
        spot = np.array([2.5, 2.5, 0.0], dtype=np.float32)
        p0.share_victim_detection(spot)
        self.assertEqual(len(bus.get_messages(msg_type="victim_found")), 1)
        synced = p1.sync_shared_victims()
        self.assertEqual(len(synced), 1)
        self.assertTrue(np.allclose(synced[0], spot))
        self.assertEqual(len(p1.victims_found), 1)

        # Env-level: one drone in range -> ALL drones hold the belief.
        victims = [np.array([2.5, 2.5, 0.0], dtype=np.float32)]
        env = MultiDroneEnv(n_drones=3, victims=victims, max_steps=50, seed=0)
        env.reset(seed=0)
        env.positions[0] = np.array([2.5, 2.5, 1.5], dtype=np.float32)  # on victim
        env.positions[1] = np.array([50.0, 50.0, 1.5], dtype=np.float32)  # far
        env.positions[2] = np.array([-50.0, -50.0, 1.5], dtype=np.float32)  # far
        hover = np.full((3, 4), 0.5, dtype=np.float32)
        _, _, _, _, infos = env.step(hover)
        self.assertTrue(env.victim_found[0])
        for d in range(3):
            self.assertIn(0, infos[d]["shared_beliefs"])
        self.assertEqual(len(env.bus.get_shared_victims()), 1)


class TestFormation(unittest.TestCase):
    """3. Formation: staggered starts, loose 5 m transit, deconflicted goals."""

    def test_formation(self):
        from src.control.comms import MessageBus
        from src.control.mission_planner import (
            MissionPlanner, MultiDroneMissionCoordinator,
            create_default_mission,
        )

        mission = create_default_mission()
        bus = MessageBus(n_drones=3)
        planners = [MissionPlanner(mission, use_global_planner=False, bus=bus,
                                   drone_id=i, n_drones=3) for i in range(3)]
        coord = MultiDroneMissionCoordinator(planners, bus=bus, desired_sep=5.0)

        # Reset-equivalent staggered starts are 5 m apart (loose formation).
        positions = [np.array([float(i) * 5.0, 0.0, 1.5]) for i in range(3)]
        report = coord.check_formation(positions)
        self.assertGreaterEqual(report["min_separation"], 5.0 - 1e-6)
        self.assertTrue(report["loose_formation"])
        self.assertTrue(report["in_formation"])

        # Deconfliction pushes overlapping goals apart beyond min_sep.
        goals = [np.array([2.0, 2.0, 1.5]), np.array([2.1, 2.0, 1.5]),
                 np.array([10.0, 10.0, 1.5])]
        fixed = coord.deconflict_goals(goals)
        d01 = float(np.linalg.norm(fixed[0] - fixed[1]))
        self.assertGreaterEqual(d01, 1.0)

        # Coordination tick emits SAR-only skills only.
        cmds = coord.coordinate_step(positions)
        self.assertEqual(len(cmds), 3)
        for c in cmds:
            self.assertIn(c["skill"], ("navigate_to", "hover",
                                       "drop_payload", "return_home"))


class TestFullMission(unittest.TestCase):
    """4. Full mission: 3 drones, greedy assignment, joint success, no collisions."""

    def test_full_mission(self):
        from src.control.comms import MessageBus
        from src.control.mission_planner import (
            MissionPlanner, MultiDroneMissionCoordinator,
            create_default_mission,
        )
        from src.rl.multi_drone import MultiDroneEnv, greedy_assign

        mission = create_default_mission()
        bus = MessageBus(n_drones=3)
        planners = [MissionPlanner(mission, use_global_planner=False, bus=bus,
                                   drone_id=i, n_drones=3) for i in range(3)]
        coord = MultiDroneMissionCoordinator(planners, bus=bus)

        victims = [np.array([2.5, 2.5, 0.0], dtype=np.float32),
                   np.array([6.0, 2.0, 0.0], dtype=np.float32),
                   np.array([2.0, 6.0, 0.0], dtype=np.float32)]
        env = MultiDroneEnv(n_drones=3, victims=victims, max_steps=500, seed=0,
                            bus=bus)
        obs, _ = env.reset(seed=0)

        # Greedy auction covers every victim exactly once.
        assignment = greedy_assign(victims, env.positions)
        assigned = [v for v in assignment.values() if v is not None]
        self.assertEqual(sorted(assigned), [0, 1, 2])

        # Fly each drone toward its assigned victim (proportional controller
        # through the env's velocity-mapped action space), sharing detections.
        rng = np.random.default_rng(0)
        for _ in range(400):
            actions = np.full((3, 4), 0.5, dtype=np.float32)
            for d in range(3):
                goal = env.goals[d]
                diff = goal - env.positions[d]
                # Map desired velocity direction onto thrust pairs.
                actions[d, 0] = float(np.clip(0.5 + diff[0], 0.0, 1.0))
                actions[d, 1] = float(np.clip(0.5 - diff[0], 0.0, 1.0))
                actions[d, 2] = float(np.clip(0.5 + diff[1], 0.0, 1.0))
                actions[d, 3] = float(np.clip(0.5 - diff[1], 0.0, 1.0))
            actions += rng.normal(0, 0.01, size=actions.shape).astype(np.float32)
            _, _, dones, _, infos = env.step(np.clip(actions, 0.0, 1.0))
            # Coordinator observes team positions each tick (intent/position bus).
            coord.coordinate_step([info["position"] for info in infos])
            if all(env.victim_served):
                break

        # Victim sharing: every drone knows every found victim.
        for d in range(3):
            for v, found in enumerate(env.victim_found):
                if found:
                    self.assertIn(v, infos[d]["shared_beliefs"])
        # Joint mission succeeds with zero collisions.
        self.assertGreater(sum(env.victim_found), 0)
        self.assertGreater(env.joint_success(), 0.0)
        self.assertEqual(env.collisions, 0)

        # Planner-level full sweep still reaches served counts via sharing.
        for vpos in victims:
            planners[0].share_victim_detection(vpos)
        for p in planners[1:]:
            p.sync_shared_victims()
        for p in planners:
            self.assertEqual(len(p.shared_victims), 3)

        # Agent-id conditioning present: obs dim = base + n_drones, one-hot tail.
        self.assertEqual(obs.shape, (3, 21 + 3))
        self.assertTrue(np.allclose(obs[1, -3:], np.array([0, 1, 0])))


if __name__ == "__main__":
    unittest.main()
