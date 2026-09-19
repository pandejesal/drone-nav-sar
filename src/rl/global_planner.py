#!/usr/bin/env python3
"""Global topological planner for DroneNav-SAR (Sprint 8).

Room graph planner using A* on topological map.
Nodes: room centers, doorways, home base.
Edges: traversable corridors, doorways (dynamic: open/closed).
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


@dataclass
class GraphNode:
    """Node in the topological graph."""
    node_id: str
    position: np.ndarray  # (3,) world position
    node_type: str  # "room_center", "doorway", "home"
    room_id: Optional[int] = None
    connections: Dict[str, float] = field(default_factory=dict)  # neighbor_id -> cost


@dataclass
class RoomGraph:
    """Topological map of the building."""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    room_centers: Dict[int, str] = field(default_factory=dict)  # room_id -> node_id
    doorways: Dict[Tuple[int, int], str] = field(default_factory=dict)  # (room_a, room_b) -> node_id
    home_node: str = "home"

    def add_room(self, room_id: int, center: np.ndarray, bounds: Tuple[float, float, float, float]):
        """Add a room with center and bounds."""
        node_id = f"room_{room_id}"
        node = GraphNode(node_id=node_id, position=center, node_type="room_center", room_id=room_id)
        self.nodes[node_id] = node
        self.room_centers[room_id] = node_id

    def add_doorway(self, room_a: int, room_b: int, position: np.ndarray):
        """Add a doorway between two rooms."""
        node_id = f"door_{room_a}_{room_b}"
        node = GraphNode(node_id=node_id, position=position, node_type="doorway")
        self.nodes[node_id] = node
        self.doorways[(room_a, room_b)] = node_id
        self.doorways[(room_b, room_a)] = node_id

    def add_home(self, position: np.ndarray):
        """Add home base node."""
        node = GraphNode(node_id="home", position=position, node_type="home")
        self.nodes["home"] = node

    def connect(self, node_a: str, node_b: str, cost: float = 1.0, bidirectional: bool = True):
        """Add edge between nodes."""
        if node_a in self.nodes and node_b in self.nodes:
            self.nodes[node_a].connections[node_b] = cost
            if bidirectional:
                self.nodes[node_b].connections[node_a] = cost

    def set_door_state(self, room_a: int, room_b: int, is_open: bool):
        """Update doorway traversability."""
        key = (room_a, room_b)
        if key in self.doorways:
            node_id = self.doorways[key]
            # Remove or add connections based on door state
            for neighbor in list(self.nodes[node_id].connections.keys()):
                if self.nodes[neighbor].node_type == "room_center":
                    if is_open:
                        # Ensure connection exists
                        room_id = self.nodes[neighbor].room_id
                        if room_id in (room_a, room_b):
                            other_room = room_b if room_id == room_a else room_a
                            other_node = self.room_centers.get(other_room)
                            if other_node:
                                self.nodes[node_id].connections[other_node] = 1.0
                                self.nodes[other_node].connections[node_id] = 1.0
                    else:
                        # Remove connection
                        self.nodes[node_id].connections.pop(neighbor, None)
                        self.nodes[neighbor].connections.pop(node_id, None)


def create_default_building() -> RoomGraph:
    """Create default 4-room building from Sprint 7."""
    graph = RoomGraph()

    # Room centers (4 rooms in 2x2 grid)
    rooms = {
        0: np.array([2.0, 2.0, 1.5]),
        1: np.array([6.0, 2.0, 1.5]),
        2: np.array([2.0, 6.0, 1.5]),
        3: np.array([6.0, 6.0, 1.5]),
    }
    for room_id, center in rooms.items():
        graph.add_room(room_id, center, (-1, 5, -1, 5))

    # Doorways between adjacent rooms
    doorways = [
        (0, 1, np.array([4.0, 2.0, 1.5])),  # room 0 <-> 1
        (0, 2, np.array([2.0, 4.0, 1.5])),  # room 0 <-> 2
        (1, 3, np.array([6.0, 4.0, 1.5])),  # room 1 <-> 3
        (2, 3, np.array([4.0, 6.0, 1.5])),  # room 2 <-> 3
    ]
    for room_a, room_b, pos in doorways:
        graph.add_doorway(room_a, room_b, pos)

    # Home base
    graph.add_home(np.array([0.0, 0.0, 1.5]))

    # Connect rooms to doorways
    for room_id in range(4):
        room_node = graph.room_centers[room_id]
        # Connect to adjacent doorways
        for (r_a, r_b), door_node in graph.doorways.items():
            if room_id in (r_a, r_b):
                graph.connect(room_node, door_node, cost=1.0)

    # Connect home to room 0
    graph.connect("home", graph.room_centers[0], cost=2.0)

    # All doors initially open
    for (r_a, r_b) in graph.doorways:
        graph.set_door_state(r_a, r_b, True)

    return graph


class GlobalPlanner:
    """A* planner on room graph."""

    def __init__(self, graph: RoomGraph):
        self.graph = graph

    def plan(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        current_room: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Plan path from start to goal, returning subgoal waypoints."""
        # Find nearest graph nodes
        start_node = self._nearest_node(start_pos)
        goal_node = self._nearest_node(goal_pos)

        if start_node == goal_node:
            return [goal_pos]

        # A* search
        path = self._astar(start_node, goal_node)
        if not path:
            return [goal_pos]  # fallback: direct

        # Convert node path to waypoints (skip start node)
        waypoints = []
        for node_id in path[1:]:
            waypoints.append(self.graph.nodes[node_id].position.copy())
        return waypoints

    def _nearest_node(self, pos: np.ndarray) -> str:
        """Find nearest graph node to position."""
        best_node = None
        best_dist = float("inf")
        for node_id, node in self.graph.nodes.items():
            dist = float(np.linalg.norm(node.position[:2] - pos[:2]))
            if dist < best_dist:
                best_dist = dist
                best_node = node_id
        return best_node

    def _astar(self, start: str, goal: str) -> List[str]:
        """A* search on graph."""
        open_set = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor, cost in self.graph.nodes[current].connections.items():
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # no path

    def _heuristic(self, node_a: str, node_b: str) -> float:
        """Euclidean distance heuristic."""
        pos_a = self.graph.nodes[node_a].position
        pos_b = self.graph.nodes[node_b].position
        return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))

    def _reconstruct_path(self, came_from: Dict[str, str], current: str) -> List[str]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def replan_on_door_change(self, room_a: int, room_b: int, is_open: bool):
        """Update graph when door state changes."""
        self.graph.set_door_state(room_a, room_b, is_open)


def create_global_planner() -> GlobalPlanner:
    """Factory for default global planner."""
    return GlobalPlanner(create_default_building())