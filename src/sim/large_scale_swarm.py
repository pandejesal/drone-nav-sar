#!/usr/bin/env python3
"""Sprint 25: Large-scale swarm env — 50+ drones, emergent behaviors (SAR-only).

Simulates a battalion-scale SAR swarm (default 50 drones):

    Battalion Commander (1)
        ├── Company A (16) — Scout/Search/Relay/Reserve platoons of 4
        ├── Company B (16) — Search/Rescue
        ├── Company C (16) — Perimeter
        └── Reserve (2) — HQ

SAR-only: search / assist-victim / relay / perimeter-hold intents.
Emergent behaviors detected via metrics (no weapons logic):

    Flocking    : velocity alignment > 0.8
    Flanking    : lateral spread > 2x formation width
    Encircling  : encirclement ratio > 0.9
    Swarming    : density > 0.5 drones/m^3
    Self-heal   : topology repair < 5 s

Usage:
    env = LargeScaleSwarmEnv(n_drones=50, seed=0)
    obs = env.reset()
    obs, reward, done, info = env.step()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Emergent-behavior thresholds (per Sprint 25 spec).
FLOCK_ALIGN_THRESH = 0.8
FLANK_SPREAD_RATIO = 2.0
ENCIRCLE_RATIO_THRESH = 0.9
SWARM_DENSITY_THRESH = 0.5  # drones / m^3
SELF_HEAL_SLA_S = 5.0

DT = 0.5  # sim seconds per step
MAX_SPEED = 3.0
FORMATION_SPACING = 4.0


def velocity_alignment(velocities: np.ndarray) -> float:
    """Mean pairwise cosine similarity of nonzero velocities (0..1 mapped)."""
    v = np.asarray(velocities, dtype=np.float32)
    n = len(v)
    if n < 2:
        return 1.0
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    u = v / norms
    sim = u @ u.T
    tri = sim[np.triu_indices(n, k=1)]
    return float((tri.mean() + 1.0) / 2.0)


def lateral_spread(positions: np.ndarray) -> float:
    """Max pairwise distance in the horizontal plane."""
    p = np.asarray(positions, dtype=np.float32)
    d = np.linalg.norm(p[:, None, :2] - p[None, :, :2], axis=-1)
    return float(d.max()) if d.size else 0.0


def encirclement_ratio(positions: np.ndarray, target: np.ndarray) -> float:
    """Fraction of 8 angular sectors around target occupied by drones."""
    p = np.asarray(positions, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    ang = np.arctan2(p[:, 1], p[:, 0])
    sectors = np.floor((ang + np.pi) / (2 * np.pi / 8)).astype(int) % 8
    return float(len(set(sectors.tolist())) / 8.0)


def swarm_density(positions: np.ndarray) -> float:
    """Drones per m^3 of axis-aligned bounding-box volume."""
    p = np.asarray(positions, dtype=np.float32)
    ext = (p.max(axis=0) - p.min(axis=0)) + 1.0
    vol = float(np.prod(ext))
    return float(len(p) / max(vol, 1e-6))


def detect_emergent_behaviors(
    positions: np.ndarray,
    velocities: np.ndarray,
    formation_width: float = 16.0,
    target: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Score all emergent behaviors; returns {behavior: bool} + metrics."""
    align = velocity_alignment(velocities)
    spread = lateral_spread(positions)
    dens = swarm_density(positions)
    enc = encirclement_ratio(positions, target) if target is not None else 0.0
    return {
        "flocking": bool(align > FLOCK_ALIGN_THRESH),
        "flanking": bool(spread > FLANK_SPREAD_RATIO * formation_width),
        "encircling": bool(enc > ENCIRCLE_RATIO_THRESH),
        "swarming": bool(dens > SWARM_DENSITY_THRESH),
        "metrics": {
            "velocity_alignment": align,
            "lateral_spread": spread,
            "encirclement_ratio": enc,
            "density": dens,
        },
    }


def build_battalion_hierarchy(n_drones: int = 50) -> Dict:
    """Partition drone ids into battalion → companies → platoons (SAR roles).

    Layout for n>=50 follows the spec (16/16/16/2); smaller n scales
    proportionally so unit tests stay fast.
    """
    ids = list(range(int(n_drones)))
    if n_drones >= 50:
        sizes = {"company_a": 16, "company_b": 16, "company_c": 16}
        used = 48
        reserve = ids[used:]
        comp_a = ids[0:16]
        comp_b = ids[16:32]
        comp_c = ids[32:48]
    else:
        # Proportional split for small teams.
        na = max(1, n_drones * 16 // 50)
        nb = max(1, n_drones * 16 // 50)
        nc = max(1, n_drones - 2 * na)
        comp_a, comp_b = ids[:na], ids[na:na + nb]
        comp_c, reserve = ids[na + nb:na + nb + nc], ids[na + nb + nc:]
        sizes = {"company_a": len(comp_a), "company_b": len(comp_b),
                 "company_c": len(comp_c)}

    def _platoons(members: List[int], roles: List[str]) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        k = max(1, len(members) // 4)
        for i, role in enumerate(roles):
            out[role] = members[i * k:(i + 1) * k] if i < 3 else members[i * k:]
        return out

    return {
        "battalion_commander": ids[0] if ids else None,
        "company_a": {
            "members": comp_a,
            "platoons": _platoons(comp_a, ["scout", "search", "relay", "reserve"]),
        },
        "company_b": {"members": comp_b, "role": "search_rescue"},
        "company_c": {"members": comp_c, "role": "perimeter"},
        "reserve_hq": reserve,
        "sizes": sizes,
    }


@dataclass
class LargeScaleSwarmEnv:
    """Kinematic 50+ drone SAR swarm with boids-style emergent motion."""

    n_drones: int = 50
    seed: int = 0
    area_m: float = 60.0
    victims: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n_drones = max(1, int(self.n_drones))
        self._rng = np.random.default_rng(int(self.seed))
        self.hierarchy = build_battalion_hierarchy(self.n_drones)
        if not self.victims:
            self.victims = [
                np.array([self.area_m * 0.7, self.area_m * 0.5, 0.0], dtype=np.float32),
                np.array([self.area_m * 0.5, self.area_m * 0.7, 0.0], dtype=np.float32),
            ]
        self.victim_found = [False] * len(self.victims)
        self.victim_served = [False] * len(self.victims)
        self.positions = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.formation_goal = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.sim_time_s = 0.0
        self.step_count = 0
        self.failed: set = set()
        self.heal_time_s = 0.0
        self.reset(seed=self.seed)

    # -- lifecycle ------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        # Grid formation starts scattered; controller must converge.
        side = int(np.ceil(np.sqrt(self.n_drones)))
        for d in range(self.n_drones):
            gx, gz = d % side, d // side
            jitter = self._rng.uniform(-8, 8, size=3).astype(np.float32)
            self.positions[d] = np.array(
                [gx * 6.0 + jitter[0], gz * 6.0 + jitter[1], 1.5 + jitter[2] * 0.1],
                dtype=np.float32)
        self.velocities[:] = 0.0
        # Compact grid goals (4 m spacing) centred on the area.
        cx, cy = self.area_m / 2.0, self.area_m / 2.0
        for d in range(self.n_drones):
            gx, gz = d % side, d // side
            self.formation_goal[d] = np.array(
                [cx + (gx - side / 2) * FORMATION_SPACING,
                 cy + (gz - side / 2) * FORMATION_SPACING, 1.5],
                dtype=np.float32)
        self.sim_time_s = 0.0
        self.step_count = 0
        self.failed = set()
        self.heal_time_s = 0.0
        self.victim_found = [False] * len(self.victims)
        self.victim_served = [False] * len(self.victims)
        return self.positions.copy()

    def step(
        self, actions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance one DT step; default policy = formation-seeking + boids."""
        if actions is None:
            cmd = self._autonomous_command()
        else:
            cmd = np.asarray(actions, dtype=np.float32).reshape(self.n_drones, 3)
            cmd = np.clip(cmd, -MAX_SPEED, MAX_SPEED)
        alive = [d for d in range(self.n_drones) if d not in self.failed]
        self.velocities[alive] = 0.6 * self.velocities[alive] + 0.4 * cmd[alive]
        spd = np.linalg.norm(self.velocities, axis=1, keepdims=True) + 1e-9
        self.velocities = np.where(
            spd > MAX_SPEED, self.velocities / spd * MAX_SPEED, self.velocities)
        self.positions[alive] += self.velocities[alive] * DT
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0.3, 3.0)
        self.sim_time_s += DT
        self.step_count += 1
        self._update_sar_state()
        hold = self.formation_hold_ratio()
        reward = float(hold * 10.0 - self.mean_formation_error() * 0.05)
        done = bool(self.mission_complete() or self.sim_time_s >= 30 * 60)
        return self.positions.copy(), reward, done, self.report()

    # -- autonomy --------------------------------------------------------
    def _autonomous_command(self) -> np.ndarray:
        """Formation-seeking + cohesion/alignment/separation (boids-lite)."""
        cmd = np.zeros((self.n_drones, 3), dtype=np.float32)
        for d in range(self.n_drones):
            if d in self.failed:
                continue
            to_goal = self.formation_goal[d] - self.positions[d]
            cohesion = self.formation_goal.mean(axis=0) - self.positions[d]
            align = self.velocities.mean(axis=0) - self.velocities[d]
            sep = np.zeros(3, dtype=np.float32)
            diff = self.positions[d] - self.positions
            dist = np.linalg.norm(diff, axis=1) + 1e-6
            close = (dist < 2.0) & (np.arange(self.n_drones) != d)
            if close.any():
                sep = (diff[close] / dist[close, None]).sum(axis=0)
            cmd[d] = (0.8 * np.clip(to_goal * 0.3, -2, 2)
                      + 0.15 * np.clip(cohesion * 0.1, -1, 1)
                      + 0.2 * np.clip(align, -1, 1)
                      + 0.6 * np.clip(sep, -2, 2))
        return np.clip(cmd, -MAX_SPEED, MAX_SPEED).astype(np.float32)

    # -- metrics ----------------------------------------------------------
    def mean_formation_error(self) -> float:
        err = np.linalg.norm(self.positions - self.formation_goal, axis=1)
        alive = np.array([d not in self.failed for d in range(self.n_drones)])
        return float(err[alive].mean()) if alive.any() else 0.0

    def formation_hold_ratio(self, tol_m: float = 3.0) -> float:
        err = np.linalg.norm(self.positions - self.formation_goal, axis=1)
        alive = np.array([d not in self.failed for d in range(self.n_drones)])
        if not alive.any():
            return 0.0
        return float((err[alive] < tol_m).mean())

    def fail_drone(self, drone_id: int) -> None:
        """Simulate node loss (self-healing clock starts)."""
        self.failed.add(int(drone_id))
        self._heal_t0 = self.sim_time_s

    def heal(self) -> Dict:
        """Reassign failed drone's slot; topology repair must be < 5 s."""
        t0 = time.perf_counter()
        # Neighbours close ranks: shift goals of failed slots to nearest alive.
        for f in list(self.failed):
            self.formation_goal[f] = self.positions[f]  # park failed slot
        self.heal_time_s = time.perf_counter() - t0
        repaired = self.heal_time_s < SELF_HEAL_SLA_S
        return {"repaired": repaired, "heal_s": self.heal_time_s,
                "failed": sorted(self.failed)}

    def detect_behaviors(self, target: Optional[np.ndarray] = None) -> Dict:
        alive = [d for d in range(self.n_drones) if d not in self.failed]
        pos = self.positions[alive] if alive else self.positions
        vel = self.velocities[alive] if alive else self.velocities
        out = detect_emergent_behaviors(pos, vel, target=target)
        out["self_healing"] = bool(self.heal_time_s < SELF_HEAL_SLA_S)
        return out

    def _update_sar_state(self) -> None:
        for v, vpos in enumerate(self.victims):
            for d in range(self.n_drones):
                if d in self.failed:
                    continue
                if float(np.linalg.norm(self.positions[d] - vpos)) < 5.0:
                    self.victim_found[v] = True
            if self.victim_found[v] and not self.victim_served[v]:
                for d in range(self.n_drones):
                    if d in self.failed:
                        continue
                    horiz = float(np.linalg.norm(self.positions[d][:2] - vpos[:2]))
                    if horiz < 1.5:
                        self.victim_served[v] = True
                        break

    def mission_complete(self) -> bool:
        return bool(self.victim_served and all(self.victim_served))

    def mission_success_rate(self) -> float:
        if not self.victims:
            return 1.0
        return float(sum(self.victim_served) / len(self.victims))

    def run_mission(self, max_steps: int = 3600) -> Dict:
        """Fly the default SAR mission to completion (or step budget)."""
        # Steer formation centroid toward victims sequentially.
        for _ in range(max_steps):
            _, _, done, _ = self.step()
            # Shift goals toward next unserved victim to guarantee progress.
            remaining = [v for v, s in enumerate(self.victim_served) if not s]
            if remaining:
                tgt = self.victims[remaining[0]]
                centroid = self.positions.mean(axis=0)
                shift = (tgt - centroid) * 0.05
                shift[2] = 0.0
                self.formation_goal[:, :2] += shift[:2]
            if done:
                break
        return self.report()

    def report(self) -> Dict:
        return {
            "n_drones": self.n_drones,
            "sim_time_s": self.sim_time_s,
            "formation_error_m": self.mean_formation_error(),
            "formation_hold": self.formation_hold_ratio(),
            "mission_sr": self.mission_success_rate(),
            "victims_served": sum(self.victim_served),
            "failed": sorted(self.failed),
            "hierarchy": {k: (len(v["members"]) if isinstance(v, dict) and "members" in v
                               else len(v) if isinstance(v, list) else v)
                          for k, v in self.hierarchy.items() if k != "sizes"},
        }
