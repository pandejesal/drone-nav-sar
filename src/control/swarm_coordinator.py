#!/usr/bin/env python3
"""Sprint 19: Swarm coordinator — role election, task allocation, health.

SAR-only: tasks are search-room / assist-victim / return-home intents.
Roles: leader (tasking authority + C2 relay) vs follower.

  - Role election: bully-style over (capability, battery, id); lowest
    (score, id) wins deterministically -> converges in one round (<10 s).
  - Task allocation: greedy auction (nearest capable drone per SAR task).
  - Health monitoring: heartbeat deadline -> suspect/dead -> re-elect +
    reassign orphaned tasks.

Usage:
    coord = SwarmCoordinator(drone_ids=[0, 1, 2, 3])
    coord.heartbeat_all(t=0.0)
    leader = coord.elect_leader()
    alloc = coord.allocate_tasks(tasks, positions)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# SAR-only task kinds.
SAR_TASK_KINDS = ("search_room", "assist_victim", "return_home", "hover")
SAR_SKILL_FOR_TASK = {
    "search_room": "navigate_to",
    "assist_victim": "drop_payload",
    "return_home": "return_home",
    "hover": "hover",
}


@dataclass
class DroneHealth:
    drone_id: int
    alive: bool = True
    battery: float = 1.0          # 0..1
    capability: float = 1.0       # 0..1 (sensors / payload)
    last_heartbeat: float = 0.0
    role: str = "follower"        # leader | follower
    current_task: Optional[str] = None


@dataclass
class SwarmTask:
    task_id: str
    kind: str                     # one of SAR_TASK_KINDS
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    priority: int = 1


class SwarmCoordinator:
    """Leader election + auction allocation + heartbeat health."""

    def __init__(
        self,
        drone_ids: List[int],
        heartbeat_timeout_s: float = 3.0,
        mesh=None,  # optional MeshC2 handle for C2-aware election
    ):
        self.drone_ids = [int(d) for d in drone_ids]
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.mesh = mesh
        self.health: Dict[int, DroneHealth] = {
            d: DroneHealth(drone_id=d) for d in self.drone_ids
        }
        self.leader_id: Optional[int] = None
        self.election_time_s = 0.0
        self.election_rounds = 0
        self.assignment: Dict[int, Optional[str]] = {d: None for d in self.drone_ids}

    # -- health ---------------------------------------------------------
    def heartbeat(self, drone_id: int, t: float, battery: Optional[float] = None,
                  capability: Optional[float] = None) -> None:
        h = self.health[int(drone_id)]
        h.last_heartbeat = float(t)
        h.alive = True
        if battery is not None:
            h.battery = float(np.clip(battery, 0.0, 1.0))
        if capability is not None:
            h.capability = float(np.clip(capability, 0.0, 1.0))

    def heartbeat_all(self, t: float = 0.0) -> None:
        for d in self.drone_ids:
            self.heartbeat(d, t)

    def check_health(self, t: float) -> List[int]:
        """Mark drones silent past the deadline dead; returns newly-dead ids."""
        newly_dead = []
        for d, h in self.health.items():
            if h.alive and (float(t) - h.last_heartbeat) > self.heartbeat_timeout_s:
                h.alive = False
                h.role = "follower"
                newly_dead.append(d)
        if newly_dead and self.leader_id in newly_dead:
            self.elect_leader()
            self._reassign_orphans()
        return newly_dead

    def mark_failed(self, drone_id: int) -> Optional[int]:
        """Hard-fail a drone now; triggers re-election if it was leader."""
        d = int(drone_id)
        self.health[d].alive = False
        self.health[d].role = "follower"
        if self.leader_id == d:
            self.elect_leader()
            self._reassign_orphans()
        return self.leader_id

    def alive_ids(self) -> List[int]:
        return [d for d in self.drone_ids if self.health[d].alive]

    # -- role election ----------------------------------------------------
    def _election_key(self, d: int):
        """Bully key: most capable, best battery, then lowest id wins."""
        h = self.health[d]
        return (-h.capability, -h.battery, d)

    def elect_leader(self) -> Optional[int]:
        """One-round deterministic election over alive drones (<10 s)."""
        t0 = time.perf_counter()
        alive = self.alive_ids()
        if not alive:
            self.leader_id = None
            return None
        winner = min(alive, key=self._election_key)
        for d in self.drone_ids:
            self.health[d].role = "leader" if d == winner else "follower"
        self.leader_id = winner
        self.election_rounds += 1
        self.election_time_s = time.perf_counter() - t0
        return self.leader_id

    def get_role(self, drone_id: int) -> str:
        return self.health[int(drone_id)].role

    # -- task allocation ---------------------------------------------------
    def allocate_tasks(
        self,
        tasks: List[SwarmTask],
        positions: Dict[int, np.ndarray] | List[np.ndarray],
    ) -> Dict[int, Optional[str]]:
        """Greedy auction: highest priority first, nearest alive drone wins.

        SAR-only kinds enforced; low-battery drones (<0.2) only get
        return_home. Returns {drone_id: task_id or None}.
        """
        if isinstance(positions, list):
            pos = {i: np.asarray(p, dtype=np.float32).reshape(3)
                   for i, p in enumerate(positions)}
        else:
            pos = {int(k): np.asarray(v, dtype=np.float32).reshape(3)
                   for k, v in positions.items()}
        for t in tasks:
            if t.kind not in SAR_TASK_KINDS:
                raise ValueError(f"SAR-only task required, got {t.kind!r}")
        assign: Dict[int, Optional[str]] = {d: None for d in self.drone_ids}
        taken: set = set()
        ordered = sorted(tasks, key=lambda t: (-t.priority, t.task_id))
        for task in ordered:
            best, best_d = None, float("inf")
            for d in self.alive_ids():
                if d in taken:
                    continue
                h = self.health[d]
                if h.battery < 0.2 and task.kind != "return_home":
                    continue
                dist = float(np.linalg.norm(pos[d] - task.goal))
                score = dist - 5.0 * h.capability  # capable drones preferred
                if score < best_d:
                    best, best_d = d, score
            if best is not None:
                assign[best] = task.task_id
                taken.add(best)
                self.health[best].current_task = task.task_id
        self.assignment = assign
        # Idle alive drones hold hover.
        return dict(assign)

    def _reassign_orphans(self) -> None:
        for d in self.drone_ids:
            if not self.health[d].alive:
                self.assignment[d] = None
                self.health[d].current_task = None

    def role_task_command(self, drone_id: int) -> Dict:
        """SAR-only skill command for a drone's role + assigned task."""
        d = int(drone_id)
        h = self.health[d]
        skill = "hover"
        if h.role == "leader" and h.current_task is None:
            skill = "navigate_to"  # leader holds C2 relay station
        elif h.current_task is not None:
            # Map is informational; planner resolves the goal.
            skill = "navigate_to"
        return {"skill": skill, "role": h.role, "task": h.current_task,
                "leader": self.leader_id}

    def report(self) -> Dict:
        return {
            "leader": self.leader_id,
            "election_time_s": self.election_time_s,
            "election_rounds": self.election_rounds,
            "alive": self.alive_ids(),
            "assignment": dict(self.assignment),
        }

    # -- Sprint 21: intent → task graph → role assignment ------------------
    def apply_swarm_intent(self, intent) -> List[SwarmTask]:
        """Convert a SwarmIntent (or intent dict) → SAR SwarmTask list.

        Each task-graph node becomes one SwarmTask; SAR skill mapping:
        search/survey/patrol → search_room, deliver/assist → assist_victim,
        return → return_home.
        """
        d = intent.to_dict() if hasattr(intent, "to_dict") else dict(intent)
        graph = list(d.get("task_graph") or d.get("objectives") or [])
        kind_map = {"search": "search_room", "survey": "search_room",
                    "patrol": "search_room", "assist": "assist_victim",
                    "deliver": "assist_victim", "return": "return_home"}
        tasks: List[SwarmTask] = []
        for i, node in enumerate(graph):
            ntype = str(node.get("type", node.get("kind", "search")))
            kind = kind_map.get(ntype, "search_room")
            area = str(node.get("area", ""))
            m = None
            import re as _re
            m = _re.search(r"(\d+)", area)
            floor = int(m.group(1)) if m else (i + 1)
            goal = np.array([float(floor * 4.0), 4.0, 1.5], dtype=np.float32)
            tasks.append(SwarmTask(
                task_id=str(node.get("node_id", f"t{i}")),
                kind=kind,
                goal=goal,
                priority=int(node.get("priority", i + 1))))
        if not tasks:  # return_home / empty intent → one home task
            tasks.append(SwarmTask(task_id="t0", kind="return_home",
                                   goal=np.zeros(3, dtype=np.float32),
                                   priority=1))
        self._intent_tasks = tasks
        self._intent_roles = dict(d.get("roles", {}))
        return list(tasks)

    def elect_swarm_roles(self) -> Dict[int, str]:
        """Assign scout/searcher/relay roles over alive drones (<5 s).

        Most capable alive drones become scouts, next best searchers,
        remainder relays; counts come from the applied intent roles
        (default scout=2/searcher=3/relay=1 scaled to team size).
        """
        t0 = time.perf_counter()
        alive = self.alive_ids()
        if not alive:
            return {}
        want = dict(getattr(self, "_intent_roles", None) or
                    {"scout": 2, "searcher": 3, "relay": 1})
        ranked = sorted(alive,
                        key=lambda d: (-self.health[d].capability,
                                       -self.health[d].battery, d))
        roles: Dict[int, str] = {}
        idx = 0
        for role in ("scout", "searcher", "relay"):
            for _ in range(int(want.get(role, 0))):
                if idx >= len(ranked):
                    break
                roles[ranked[idx]] = role
                idx += 1
        # Any leftover drones become searchers (SAR default).
        for d in ranked[idx:]:
            roles[d] = "searcher"
        for d, r in roles.items():
            self.health[d].current_task = self.health[d].current_task
            setattr(self.health[d], "swarm_role", r)
        self.swarm_roles = roles
        self.swarm_election_time_s = time.perf_counter() - t0
        return dict(roles)

    def allocate_from_task_graph(
        self, tasks: Optional[List[SwarmTask]] = None,
        positions: Optional[Dict[int, np.ndarray] | List[np.ndarray]] = None,
    ) -> Dict[int, List[str]]:
        """Allocate intent tasks → per-drone task-graph slices.

        Uses the greedy auction (:meth:`allocate_tasks`) then expands the
        winner's single task into a one-node graph slice. Returns
        {drone_id: [task_id, ...]}.
        """
        task_list = list(tasks if tasks is not None
                         else getattr(self, "_intent_tasks", []))
        if positions is None:
            positions = {d: np.zeros(3, dtype=np.float32)
                         for d in self.drone_ids}
        alloc = self.allocate_tasks(task_list, positions)
        graph: Dict[int, List[str]] = {d: [] for d in self.drone_ids}
        for d, tid in alloc.items():
            if tid is not None:
                graph[d] = [tid]
        self.task_graph = graph
        return {d: list(v) for d, v in graph.items()}

    def execute_swarm_intent(
        self, intent,
        positions: Optional[Dict[int, np.ndarray] | List[np.ndarray]] = None,
    ) -> Dict:
        """End-to-end: intent → tasks → roles → per-drone graphs (<2 s)."""
        t0 = time.perf_counter()
        tasks = self.apply_swarm_intent(intent)
        self.elect_leader()
        roles = self.elect_swarm_roles()
        graph = self.allocate_from_task_graph(tasks, positions)
        return {"tasks": [t.task_id for t in tasks], "roles": roles,
                "task_graph": graph, "leader": self.leader_id,
                "latency_ms": (time.perf_counter() - t0) * 1000.0}
