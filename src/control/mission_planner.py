#!/usr/bin/env python3
"""SAR Mission Planner — high-level state machine for med-kit delivery (Sprint 7).

SAR-only: navigate_to / hover / drop_payload / return_home.
State machine: IDLE → NAV_TO_ROOM → SEARCH_VICTIM → HOVER_OVER_VICTIM → DROP_MEDKIT → RETURN_HOME → COMPLETE
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class MissionState(Enum):
    IDLE = "idle"
    NAV_TO_ROOM = "nav_to_room"
    SEARCH_VICTIM = "search_victim"
    HOVER_OVER_VICTIM = "hover_over_victim"
    DROP_MEDKIT = "drop_medkit"
    RETURN_HOME = "return_home"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Room:
    """Room definition in multi-room building."""
    name: str
    center: np.ndarray  # (3,) world position
    bounds: Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    victims: List[np.ndarray]  # list of victim positions in this room


@dataclass
class MissionConfig:
    """Configuration for SAR mission."""
    rooms: List[Room]
    home_position: np.ndarray  # (3,) home/base position
    search_pattern: str = "spiral"  # "spiral" or "grid"
    search_speed: float = 0.8  # m/s
    hover_height: float = 1.5  # m
    drop_height: float = 1.0  # m
    victim_detect_range: float = 5.0  # m
    max_search_time: float = 60.0  # s per room
    mission_timeout: float = 300.0  # s total


class MissionPlanner:
    """High-level SAR mission state machine (Sprint 8: global-planner subgoals)."""

    def __init__(
        self,
        config: MissionConfig,
        detector=None,  # YOLO detector instance
        rng: Optional[np.random.Generator] = None,
        global_planner=None,  # GlobalPlanner instance (lazy-built on demand)
        use_global_planner: bool = True,
        bus=None,  # Sprint 9: shared MessageBus for victim/intent/position
        drone_id: int = 0,  # Sprint 9: this drone's id within the team
        n_drones: int = 1,  # Sprint 9: team size
        coordinator=None,  # Sprint 19: SwarmCoordinator (role election + tasking)
    ):
        self.config = config
        self.detector = detector
        self._rng = rng or np.random.default_rng()
        self.global_planner = global_planner
        self.use_global_planner = bool(use_global_planner)
        self.bus = bus
        self.drone_id = int(drone_id)
        self.n_drones = max(1, int(n_drones))
        self.coordinator = coordinator  # Sprint 19: SwarmCoordinator or None
        self.active_subgoals: List[np.ndarray] = []
        self.state = MissionState.IDLE
        self.current_room_idx = 0
        self.target_victim: Optional[np.ndarray] = None
        self.search_start_time = 0.0
        self.mission_start_time = 0.0
        self.medkit_dropped = False
        self.victims_found: List[np.ndarray] = []
        self.victims_served: List[np.ndarray] = []
        self.shared_victims: List[np.ndarray] = []  # Sprint 9 team belief

    def reset(self) -> None:
        """Reset mission to initial state."""
        self.state = MissionState.IDLE
        self.current_room_idx = 0
        self.target_victim = None
        self.search_start_time = 0.0
        self.mission_start_time = 0.0
        self.medkit_dropped = False
        self.victims_found = []
        self.victims_served = []
        self.shared_victims = []
        self.active_subgoals = []

    # -- Sprint 9: multi-drone coordination (assignment + deconfliction) --
    def share_victim_detection(self, victim_position: np.ndarray) -> None:
        """Record a local detection and broadcast it team-wide (victim sharing)."""
        pos = np.asarray(victim_position, dtype=np.float32).reshape(3)
        if not any(np.allclose(pos, v) for v in self.victims_found):
            self.victims_found.append(pos)
        if not any(np.allclose(pos, v) for v in self.shared_victims):
            self.shared_victims.append(pos)
        if self.bus is not None:
            try:
                self.bus.broadcast_victim(sender_id=self.drone_id, position=pos)
            except Exception:
                pass

    def sync_shared_victims(self) -> List[np.ndarray]:
        """Pull team-wide victim spots from the bus into local belief."""
        if self.bus is None:
            return list(self.shared_victims)
        try:
            team = self.bus.get_shared_victims()
        except Exception:
            team = []
        for pos in team:
            if not any(np.allclose(pos, v) for v in self.shared_victims):
                self.shared_victims.append(np.asarray(pos, dtype=np.float32))
            if not any(np.allclose(pos, v) for v in self.victims_found):
                self.victims_found.append(np.asarray(pos, dtype=np.float32))
        return list(self.shared_victims)

    def broadcast_position(self, drone_pos: np.ndarray) -> None:
        """Publish current position for collision-avoidance deconfliction."""
        if self.bus is not None:
            try:
                self.bus.broadcast_position(sender_id=self.drone_id,
                                            position=np.asarray(drone_pos, dtype=np.float32))
            except Exception:
                pass

    def broadcast_intent(self, skill: str, goal: np.ndarray) -> None:
        """Publish planned intent (SAR-only skill + goal) for deconfliction."""
        if self.bus is not None:
            try:
                self.bus.broadcast_intent(sender_id=self.drone_id, skill=skill,
                                          goal=np.asarray(goal, dtype=np.float32))
            except Exception:
                pass

    # -- Sprint 19: SwarmCoordinator integration (role-based tasking) ----
    def attach_swarm_coordinator(self, coordinator) -> None:
        """Attach a SwarmCoordinator (role election + task allocation)."""
        self.coordinator = coordinator

    def get_swarm_role(self) -> str:
        """This drone's swarm role (leader/follower); follower when detached."""
        if self.coordinator is None:
            return "follower"
        try:
            return self.coordinator.get_role(self.drone_id)
        except Exception:
            return "follower"

    def plan_with_swarm(self, drone_pos: np.ndarray) -> Dict:
        """Role-based tasking via the attached SwarmCoordinator.

        Leader holds the C2 relay station (navigate_to relay point);
        followers run the standard SAR state machine. Detached planners
        fall back to the standard update path via update().
        SAR-only outputs.
        """
        if self.coordinator is None:
            return self.update(np.asarray(drone_pos, dtype=np.float32),
                               np.array([0.0, 0.0, 0.0, 1.0]), dt=0.1)
        try:
            cmd = self.coordinator.role_task_command(self.drone_id)
        except Exception:
            return self.update(np.asarray(drone_pos, dtype=np.float32),
                               np.array([0.0, 0.0, 0.0, 1.0]), dt=0.1)
        role = cmd.get("role", "follower")
        if role == "leader":
            # Leader: station-keep at relay point above home for C2 coverage.
            home = np.asarray(self.config.home_position, dtype=np.float32)
            relay = np.array([home[0], home[1], home[2] + 3.0], dtype=np.float32)
            return {
                "skill": "navigate_to",
                "params": {"x": float(relay[0]), "y": float(relay[1]),
                           "z": float(relay[2]), "yaw_deg": 0.0, "speed_ms": 0.8},
                "constraints": {"timeout_s": 30.0},
                "task": "sar_mission",
                "task_embedding_id": 0,
                "swarm_role": "leader",
            }
        task = cmd.get("task")
        if task is None:
            out = self.update(np.asarray(drone_pos, dtype=np.float32),
                              np.array([0.0, 0.0, 0.0, 1.0]), dt=0.1)
            out["swarm_role"] = "follower"
            return out
        # Assigned a swarm task: route to its goal via subgoals.
        goal = np.asarray(self.config.rooms[self.current_room_idx].center
                          if self.config.rooms else self.config.home_position,
                          dtype=np.float32)
        self.get_subgoals(np.asarray(drone_pos, dtype=np.float32), goal)
        out = self._skill_nav_to_room(np.asarray(drone_pos, dtype=np.float32))
        out["swarm_role"] = "follower"
        out["swarm_task"] = task
        return out

    # -- Sprint 8: global-planner subgoal generation -------------------
    def _ensure_planner(self):
        """Lazily build the default global planner (import here: no cycle)."""
        if self.global_planner is None and self.use_global_planner:
            try:
                from src.rl.global_planner import create_global_planner
                self.global_planner = create_global_planner()
            except Exception:
                self.global_planner = None
        return self.global_planner

    def get_subgoals(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Plan subgoal waypoints from `start` to `goal` via room graph.

        Falls back to [goal] when the planner is disabled/unavailable.
        The result is cached on ``self.active_subgoals`` for consumers.
        """
        goal_v = np.asarray(goal, dtype=np.float32).reshape(3)
        planner = self._ensure_planner()
        if planner is None:
            self.active_subgoals = [goal_v]
            return self.active_subgoals
        try:
            waypoints = planner.plan(np.asarray(start, dtype=np.float32), goal_v)
        except Exception:
            waypoints = [goal_v]
        self.active_subgoals = [np.asarray(w, dtype=np.float32).reshape(3) for w in waypoints]
        return self.active_subgoals

    def plan_to_current_room(self, drone_pos: np.ndarray) -> List[np.ndarray]:
        """Plan subgoals from `drone_pos` to the current room center."""
        if self.current_room_idx >= len(self.config.rooms):
            return self.get_subgoals(drone_pos, self.config.home_position)
        return self.get_subgoals(drone_pos, self.config.rooms[self.current_room_idx].center)

    def replan_on_door_change(self, room_a: int, room_b: int, is_open: bool) -> None:
        """Forward a door-state change to the global planner (dynamic edge)."""
        planner = self._ensure_planner()
        if planner is not None:
            try:
                planner.replan_on_door_change(room_a, room_b, is_open)
            except Exception:
                pass

    def update(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        dt: float,
        goal_reached: bool = False,
        victim_detected: bool = False,
        victim_position: Optional[np.ndarray] = None,
        drop_complete: bool = False,
        home_reached: bool = False,
        crashed: bool = False,
    ) -> Dict:
        """Update mission state and return current skill command.

        Returns:
            Dict with keys: skill, params, constraints, task, task_embedding_id
        """
        if crashed:
            self.state = MissionState.FAILED
            return self._skill_return_home(drone_pos)

        if self.state == MissionState.IDLE:
            self.mission_start_time = 0.0
            self.current_room_idx = 0
            self.state = MissionState.NAV_TO_ROOM
            return self._skill_nav_to_room(drone_pos)

        elif self.state == MissionState.NAV_TO_ROOM:
            if goal_reached:
                self.search_start_time = 0.0
                self.state = MissionState.SEARCH_VICTIM
                return self._skill_search_victim()
            return self._skill_nav_to_room(drone_pos)

        elif self.state == MissionState.SEARCH_VICTIM:
            self.search_start_time += dt
            if victim_detected and victim_position is not None:
                self.target_victim = victim_position
                self.victims_found.append(victim_position)
                self.state = MissionState.HOVER_OVER_VICTIM
                return self._skill_hover_over_victim()
            elif self.search_start_time > self.config.max_search_time:
                # Timeout: move to next room or return home
                self.current_room_idx += 1
                if self.current_room_idx < len(self.config.rooms):
                    self.state = MissionState.NAV_TO_ROOM
                    return self._skill_nav_to_room(drone_pos)
                else:
                    self.state = MissionState.RETURN_HOME
                    self.get_subgoals(np.asarray(drone_pos, dtype=np.float32),
                                      self.config.home_position)
                    return self._skill_return_home(drone_pos)
            return self._skill_search_victim()

        elif self.state == MissionState.HOVER_OVER_VICTIM:
            if self.target_victim is not None:
                dist = float(np.linalg.norm(drone_pos[:2] - self.target_victim[:2]))
                if dist < 0.5:  # close enough to drop
                    self.state = MissionState.DROP_MEDKIT
                    return self._skill_drop_medkit()
            return self._skill_hover_over_victim()

        elif self.state == MissionState.DROP_MEDKIT:
            if self.medkit_dropped:
                self.victims_served.append(self.target_victim)
                self.target_victim = None
                self.medkit_dropped = False
                # Check if more victims in current room
                room = self.config.rooms[self.current_room_idx]
                remaining = [v for v in room.victims if not any(np.allclose(v, s) for s in self.victims_served)]
                if remaining:
                    self.state = MissionState.SEARCH_VICTIM
                    self.search_start_time = 0.0
                    return self._skill_search_victim()
                else:
                    self.current_room_idx += 1
                    if self.current_room_idx < len(self.config.rooms):
                        self.state = MissionState.NAV_TO_ROOM
                        return self._skill_nav_to_room(drone_pos)
                    else:
                        self.state = MissionState.RETURN_HOME
                        self.get_subgoals(np.asarray(drone_pos, dtype=np.float32),
                                          self.config.home_position)
                        return self._skill_return_home(drone_pos)
            return self._skill_drop_medkit()

        elif self.state == MissionState.RETURN_HOME:
            if home_reached:
                self.state = MissionState.COMPLETE
                return {"skill": "hover", "params": {"z": 1.0}, "constraints": {}}
            self.get_subgoals(np.asarray(drone_pos, dtype=np.float32),
                              self.config.home_position)
            return self._skill_return_home(drone_pos)

        elif self.state == MissionState.COMPLETE:
            return {"skill": "hover", "params": {"z": 1.0}, "constraints": {}}

        elif self.state == MissionState.FAILED:
            return self._skill_return_home(drone_pos)

        return self._skill_return_home(drone_pos)

    def _skill_nav_to_room(self, drone_pos: Optional[np.ndarray] = None) -> Dict:
        if self.current_room_idx >= len(self.config.rooms):
            return self._skill_return_home(drone_pos)
        room = self.config.rooms[self.current_room_idx]
        target = room.center
        # Sprint 8: attach global-planner subgoal sequence for the local policy.
        subgoals: List[List[float]] = []
        if self.use_global_planner:
            try:
                start = np.asarray(drone_pos if drone_pos is not None else self.config.home_position,
                                   dtype=np.float32)
                for w in self.get_subgoals(start, target):
                    subgoals.append([float(w[0]), float(w[1]), float(w[2])])
            except Exception:
                subgoals = []
        return {
            "skill": "navigate_to",
            "params": {
                "x": float(target[0]),
                "y": float(target[1]),
                "z": float(target[2]),
                "yaw_deg": 0.0,
                "speed_ms": self.config.search_speed,
            },
            "constraints": {"timeout_s": 30.0},
            "task": "sar_mission",
            "task_embedding_id": 0,
            "subgoals": subgoals,
        }

    def _skill_search_victim(self) -> Dict:
        """Spiral search pattern around room center."""
        room = self.config.rooms[self.current_room_idx]
        center = room.center
        # Simple: hover at center, detector scans
        return {
            "skill": "hover",
            "params": {
                "z": float(center[2]),
                "yaw_deg": 0.0,
                "timeout_s": self.config.max_search_time,
            },
            "constraints": {"timeout_s": self.config.max_search_time},
            "task": "sar_mission",
            "task_embedding_id": 1,
        }

    def _skill_hover_over_victim(self) -> Dict:
        if self.target_victim is None:
            return self._skill_return_home()
        return {
            "skill": "hover",
            "params": {
                "x": float(self.target_victim[0]),
                "y": float(self.target_victim[1]),
                "z": self.config.hover_height,
                "yaw_deg": 0.0,
                "timeout_s": 10.0,
            },
            "constraints": {"timeout_s": 10.0},
            "task": "sar_mission",
            "task_embedding_id": 2,
        }

    def _skill_drop_medkit(self) -> Dict:
        if self.target_victim is None:
            return self._skill_return_home()
        self.medkit_dropped = True
        return {
            "skill": "drop_payload",
            "params": {
                "x": float(self.target_victim[0]),
                "y": float(self.target_victim[1]),
                "z": self.config.drop_height,
                "yaw_deg": 0.0,
            },
            "constraints": {"timeout_s": 5.0},
            "task": "sar_mission",
            "task_embedding_id": 3,
        }

    def _skill_return_home(self, drone_pos: Optional[np.ndarray] = None) -> Dict:
        home = self.config.home_position
        # Sprint 8: attach global-planner subgoal sequence for the local policy.
        subgoals: List[List[float]] = []
        if self.use_global_planner and drone_pos is not None:
            try:
                start = np.asarray(drone_pos, dtype=np.float32)
                for w in self.get_subgoals(start, home):
                    subgoals.append([float(w[0]), float(w[1]), float(w[2])])
            except Exception:
                subgoals = []
        return {
            "skill": "return_home",
            "params": {
                "x": float(home[0]),
                "y": float(home[1]),
                "z": float(home[2]),
                "speed_ms": 1.0,
            },
            "constraints": {"timeout_s": 60.0},
            "task": "sar_mission",
            "task_embedding_id": 4,
            "subgoals": subgoals,
        }

    # -- Sprint 20: COP-driven mission planning (COA generation) ----
    def plan_from_cop(self, entities, drone_pos: np.ndarray) -> Dict:
        """Route to the highest-confidence COP victim via the COP bridge.

        SAR-only outputs. Falls back to the standard state machine when
        the COP is empty or the bridge is unavailable.
        """
        try:
            from src.control.cop_bridge import COPBridge
            bridge = COPBridge()
            tasks = bridge.entities_to_tasks(
                list(entities.values()) if isinstance(entities, dict) else list(entities),
                np.asarray(drone_pos, dtype=np.float32),
            )
        except Exception:
            tasks = []
        if not tasks:
            return self.update(np.asarray(drone_pos, dtype=np.float32),
                               np.array([0.0, 0.0, 0.0, 1.0]), dt=0.1)
        top = tasks[0]
        out = self.update(np.asarray(drone_pos, dtype=np.float32),
                          np.array([0.0, 0.0, 0.0, 1.0]), dt=0.1)
        out["cop_entity_id"] = top["entity_id"]
        out["cop_skill_hint"] = top["skill"]
        out["task_embedding_id"] = int(top["task_embedding_id"])
        return out

    def generate_coas(self, entities, drone_pos: np.ndarray,
                      max_coas: int = 3) -> List[Dict]:
        """Generate candidate courses of action (nearest-victim orderings).

        Each COA is a SAR-only skill sequence with an estimated cost.
        Pure-Python, < 500 ms for a 10-entity scenario.
        """
        ents = list(entities.values()) if isinstance(entities, dict) else list(entities)
        victims = [e for e in ents if getattr(e, "type", "") == "victim"]
        if not victims:
            return [{"name": "continue_search", "sequence": ["hover"],
                     "cost": 1.0, "entities": []}]
        pos = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        order = sorted(victims,
                       key=lambda e: float(np.linalg.norm(np.asarray(e.pos) - pos)))
        coas = []
        for k in range(1, min(max_coas, len(order)) + 1):
            seq = order[:k]
            cost = sum(float(np.linalg.norm(np.asarray(e.pos) - pos)) for e in seq)
            cost += 2.0 * k  # drop_payload overhead per victim
            coas.append({
                "name": f"coa_serve_{k}",
                "sequence": ["navigate_to", "hover", "drop_payload"] * 1,
                "cost": cost,
                "entities": [getattr(e, "entity_id", f"e{i}") for i, e in enumerate(seq)],
            })
        return coas

    def select_coa(self, coas: List[Dict]) -> Dict:
        """Select the minimum-cost COA."""
        if not coas:
            return {"name": "continue_search", "sequence": ["hover"],
                    "cost": 1.0, "entities": []}
        return min(coas, key=lambda c: float(c.get("cost", float("inf"))))

    # -- Sprint 21: language → swarm intent → COA -------------------------
    def plan_from_swarm_text(self, text: str,
                             drone_pos: Optional[np.ndarray] = None) -> Dict:
        """Operator text → SwarmIntent → COA → SAR skill command (<2 s).

        Parses via SwarmLLM, feeds the task graph through the attached
        (or a transient) SwarmCoordinator, then returns the best COA's
        first SAR skill. SAR-only outputs.
        """
        import time as _time
        from src.control.swarm_llm import parse_swarm_command

        t0 = _time.perf_counter()
        intent = parse_swarm_command(text)
        return self.plan_from_swarm_intent(intent, drone_pos, _t0=t0)

    def plan_from_swarm_intent(self, intent,
                               drone_pos: Optional[np.ndarray] = None,
                               _t0=None) -> Dict:
        """SwarmIntent → coordinator execution → COA → SAR skill (<2 s)."""
        import time as _time
        t0 = _t0 if _t0 is not None else _time.perf_counter()
        pos = np.asarray(drone_pos, dtype=np.float32).reshape(3) \
            if drone_pos is not None else np.asarray(
                self.config.home_position, dtype=np.float32)
        coord = self.coordinator
        if coord is None:
            from src.control.swarm_coordinator import SwarmCoordinator
            coord = SwarmCoordinator(
                drone_ids=list(range(max(1, intent.constraints.get(
                    "max_drones", 1)))))
            coord.heartbeat_all(t=0.0)
        positions = {d: pos + np.array([float(d), 0.0, 0.0],
                                       dtype=np.float32)
                     for d in coord.drone_ids}
        exec_info = coord.execute_swarm_intent(intent, positions)
        # COA: one candidate per task-graph node (nearest-first ordering).
        tasks = getattr(coord, "_intent_tasks", [])
        coas = self.generate_coas(
            [type("E", (), {"type": "victim", "pos": t.goal,
                            "entity_id": t.task_id})() for t in tasks
             if t.kind in ("search_room", "assist_victim")],
            pos, max_coas=max(1, len(tasks)))
        best = self.select_coa(coas)
        seq = best.get("sequence", ["hover"])
        skill = seq[0] if seq and seq[0] in (
            "navigate_to", "hover", "drop_payload",
            "return_home") else "navigate_to"
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        out = {
            "skill": skill,
            "params": {"x": float(pos[0]), "y": float(pos[1]),
                       "z": float(pos[2]), "yaw_deg": 0.0,
                       "speed_ms": 0.8},
            "constraints": {"timeout_s": 30.0},
            "task": "sar_mission",
            "task_embedding_id": 0,
            "swarm_intent": intent.intent if hasattr(intent, "intent")
            else intent.get("intent"),
            "swarm_coa": best.get("name"),
            "swarm_tasks": exec_info.get("tasks", []),
            "swarm_roles": exec_info.get("roles", {}),
            "latency_ms": float(latency_ms),
        }
        return out

    def get_status(self) -> Dict:
        """Get current mission status for logging."""
        return {
            "state": self.state.value,
            "current_room": self.current_room_idx,
            "victims_found": len(self.victims_found),
            "victims_served": len(self.victims_served),
            "medkit_dropped": self.medkit_dropped,
        }

    # -- Sprint 11: language → task_id + params (SAR-only) ------------------
    def parse_language_command(self, text: str) -> Dict:
        """Parse a language command → {task_id, task_name, params}."""
        from src.control.language_interface import TASK_NAMES, parse_language
        task_id, params = parse_language(text)
        return {
            "task_id": int(task_id),
            "task_name": TASK_NAMES[int(task_id)],
            "params": dict(params),
        }

    def plan_from_language(self, text: str, drone_pos: Optional[np.ndarray] = None) -> Dict:
        """Build a SAR-only skill command from a language command.

        task_id map: 0=nav, 1=hover, 2=detect, 3=drop, 4=return.
        Unknown commands fall back to nav-home via parse_language.
        """
        parsed = self.parse_language_command(text)
        task_id, params = parsed["task_id"], parsed["params"]
        pos = np.asarray(drone_pos, dtype=np.float32).reshape(3) \
            if drone_pos is not None else np.asarray(self.config.home_position, dtype=np.float32)
        if task_id == 3:
            room_idx = params.get("room", self.current_room_idx)
            if 0 <= int(room_idx) < len(self.config.rooms):
                self.current_room_idx = int(room_idx)
                self.target_victim = np.asarray(
                    self.config.rooms[int(room_idx)].center, dtype=np.float32)
            return self._skill_drop_medkit() if self.target_victim is not None \
                else self._skill_nav_to_room(pos)
        if task_id == 4:
            return self._skill_return_home(pos)
        if task_id == 1:
            return self._skill_search_victim()
        if task_id == 2:
            room_idx = params.get("room", None)
            if room_idx is not None and 0 <= int(room_idx) < len(self.config.rooms):
                self.current_room_idx = int(room_idx)
            return self._skill_search_victim()
        # task_id == 0 (nav, incl. unknown-command fallback)
        room_idx = params.get("room", None)
        if room_idx is not None and 0 <= int(room_idx) < len(self.config.rooms):
            self.current_room_idx = int(room_idx)
        return self._skill_nav_to_room(pos)


class MultiDroneMissionCoordinator:
    """Sprint 9 team coordinator: greedy-auction assignment + deconfliction.

    SAR-only outputs (navigate_to / hover / drop_payload / return_home).
    Owns one MissionPlanner per drone sharing a single MessageBus.
    """

    def __init__(
        self,
        planners: List[MissionPlanner],
        bus=None,
        desired_sep: float = 5.0,
        min_sep: float = 1.0,
    ):
        self.planners = list(planners)
        self.n_drones = len(self.planners)
        if bus is None:
            try:
                from src.control.comms import MessageBus
                bus = MessageBus(n_drones=self.n_drones)
            except Exception:
                bus = None
        self.bus = bus
        for i, p in enumerate(self.planners):
            p.bus = bus
            p.drone_id = i
            p.n_drones = self.n_drones
        self.desired_sep = float(desired_sep)
        self.min_sep = float(min_sep)

    def greedy_assign(
        self,
        victim_positions: List[np.ndarray],
        drone_positions: List[np.ndarray],
    ) -> Dict[int, Optional[int]]:
        """Greedy auction: nearest-drone-first victim assignment."""
        remaining = list(range(len(victim_positions)))
        assignment: Dict[int, Optional[int]] = {}
        for d, dpos in enumerate(drone_positions):
            best, best_dist = None, float("inf")
            for v in remaining:
                dist = float(np.linalg.norm(
                    np.asarray(dpos, dtype=np.float32)[:3]
                    - np.asarray(victim_positions[v], dtype=np.float32)[:3]))
                if dist < best_dist:
                    best, best_dist = v, dist
            assignment[d] = best
            if best is not None:
                remaining.remove(best)
        return assignment

    def deconflict_goals(
        self,
        goals: List[np.ndarray],
        drone_positions: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Push apart goals closer than min_sep (keeps SAR goals flyable)."""
        out = [np.asarray(g, dtype=np.float32).reshape(3).copy() for g in goals]
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                diff = out[i] - out[j]
                dist = float(np.linalg.norm(diff)) + 1e-6
                if dist < self.min_sep:
                    push = (self.min_sep - dist) / 2.0 + 0.05
                    direction = diff / dist
                    out[i] = out[i] + direction * push
                    out[j] = out[j] - direction * push
        return out

    def check_formation(
        self, drone_positions: List[np.ndarray], desired_sep: Optional[float] = None
    ) -> Dict:
        """Loose-transit formation check (default desired_sep 5 m)."""
        want = float(desired_sep if desired_sep is not None else self.desired_sep)
        pos = [np.asarray(p, dtype=np.float32)[:3] for p in drone_positions]
        min_dist = float("inf")
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                min_dist = min(min_dist, float(np.linalg.norm(pos[i] - pos[j])))
        return {
            "min_separation": min_dist,
            "desired_sep": want,
            "in_formation": bool(min_dist >= self.min_sep),
            "loose_formation": bool(min_dist >= want * 0.5),
        }

    def coordinate_step(
        self, drone_positions: List[np.ndarray]
    ) -> List[Dict]:
        """One coordination tick: sync beliefs, assign, deconflict, command.

        Returns one SAR-only skill dict per drone.
        """
        for p in self.planners:
            try:
                p.sync_shared_victims()
            except Exception:
                pass
        # Collect known victims team-wide.
        known: List[np.ndarray] = []
        for p in self.planners:
            for v in p.shared_victims:
                if not any(np.allclose(v, k) for k in known):
                    known.append(v)
        if not known:
            # No victims yet: spread drones across rooms (transit formation).
            cmds = []
            for i, p in enumerate(self.planners):
                room = p.config.rooms[i % len(p.config.rooms)]
                goal = np.asarray(room.center, dtype=np.float32)
                goals = self.deconflict_goals(
                    [goal] * 1, [np.asarray(drone_positions[i])]
                )
                p.broadcast_position(np.asarray(drone_positions[i]))
                p.broadcast_intent("navigate_to", goals[0])
                cmds.append({
                    "skill": "navigate_to",
                    "params": {"x": float(goals[0][0]), "y": float(goals[0][1]),
                               "z": float(goals[0][2]), "yaw_deg": 0.0, "speed_ms": 0.8},
                    "constraints": {"timeout_s": 30.0},
                })
            return cmds
        assignment = self.greedy_assign(known, drone_positions)
        cmds = []
        for d, p in enumerate(self.planners):
            v_idx = assignment.get(d)
            if v_idx is None:
                cmds.append({"skill": "hover",
                             "params": {"z": 1.5, "yaw_deg": 0.0, "timeout_s": 10.0},
                             "constraints": {"timeout_s": 10.0}})
                continue
            goal = np.asarray(known[v_idx], dtype=np.float32).copy()
            goal[2] = p.config.hover_height
            p.broadcast_position(np.asarray(drone_positions[d]))
            p.broadcast_intent("navigate_to", goal)
            cmds.append({
                "skill": "navigate_to",
                "params": {"x": float(goal[0]), "y": float(goal[1]),
                           "z": float(goal[2]), "yaw_deg": 0.0, "speed_ms": 0.8},
                "constraints": {"timeout_s": 30.0},
            })
        return cmds


def create_default_mission() -> MissionConfig:
    """Create default 4-room SAR mission."""
    rooms = [
        Room("room_0", np.array([2.0, 2.0, 1.5]), (-1, 5, -1, 5), [np.array([2.5, 2.5, 0.0])]),
        Room("room_1", np.array([6.0, 2.0, 1.5]), (3, 9, -1, 5), [np.array([5.5, 3.0, 0.0])]),
        Room("room_2", np.array([2.0, 6.0, 1.5]), (-1, 5, 3, 9), [np.array([3.0, 7.0, 0.0])]),
        Room("room_3", np.array([6.0, 6.0, 1.5]), (3, 9, 3, 9), [np.array([7.0, 5.5, 0.0])]),
    ]
    return MissionConfig(
        rooms=rooms,
        home_position=np.array([0.0, 0.0, 1.5]),
        search_pattern="spiral",
        search_speed=0.8,
        hover_height=1.5,
        drop_height=1.0,
        victim_detect_range=5.0,
        max_search_time=60.0,
        mission_timeout=300.0,
    )