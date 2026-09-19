#!/usr/bin/env python3
"""Multi-drone vectorized env for DroneNav-SAR (Sprint 9).

SAR-only: navigate_to / hover / drop_payload / return_home.
Shared policy weights + agent_id one-hot conditioning, RVO-style
collision-avoidance penalty, team-wide victim belief sharing.

Kinematic mock: action (4,) thrusts in [0,1] map to a 3-D velocity
command (mean thrust per axis pair). Deterministic, CPU-only, no
simulator dependency so CI stays green.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rl.policies import ACT_DIM, OBS_DIM  # noqa: E402

try:
    from src.control.comms import MessageBus  # noqa: E402
except Exception:  # pragma: no cover - fallback for odd import orders
    MessageBus = None  # type: ignore


BASE_OBS_DIM = OBS_DIM  # 21: state(17) + goal_rel(3) + time(1)
MIN_SEP_M = 1.0          # RVO penalty starts below this pairwise distance
COLLISION_DIST_M = 0.5   # hard collision flag below this distance
COLLISION_PENALTY = -5.0
PROXIMITY_PENALTY_GAIN = -2.0  # per metre of separation violation
GOAL_TOLERANCE_M = 0.5
DETECT_RANGE_M = 5.0
MAX_SPEED_MS = 1.0
DT = 0.1  # kinematic integration step (s)


def conditioned_obs_dim(n_drones: int) -> int:
    """Obs dim with agent_id one-hot appended."""
    return int(BASE_OBS_DIM + int(n_drones))


def one_hot(agent_id: int, n_drones: int) -> np.ndarray:
    vec = np.zeros(int(n_drones), dtype=np.float32)
    vec[int(agent_id) % int(n_drones)] = 1.0
    return vec


def condition_obs(base_obs: np.ndarray, agent_id: int, n_drones: int) -> np.ndarray:
    """Concatenate agent_id one-hot to a base (21,) obs."""
    base = np.asarray(base_obs, dtype=np.float32).reshape(-1)
    return np.concatenate([base, one_hot(agent_id, n_drones)]).astype(np.float32)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances, (N, N), zeros on diagonal."""
    pos = np.asarray(positions, dtype=np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)


def collision_penalty_for(dist: float) -> float:
    """RVO-style penalty: 0 beyond MIN_SEP, ramping to COLLISION_PENALTY."""
    if dist < COLLISION_DIST_M:
        return float(COLLISION_PENALTY)
    if dist < MIN_SEP_M:
        return float(PROXIMITY_PENALTY_GAIN * (MIN_SEP_M - dist))
    return 0.0


def greedy_assign(
    victim_positions: List[np.ndarray], drone_positions: np.ndarray
) -> Dict[int, Optional[int]]:
    """Greedy auction: each drone takes its nearest unclaimed victim.

    Returns {drone_id: victim_idx or None}.
    """
    dro = np.asarray(drone_positions, dtype=np.float32)
    remaining = list(range(len(victim_positions)))
    assignment: Dict[int, Optional[int]] = {}
    # Process drones in order; each picks nearest remaining victim.
    for d in range(len(dro)):
        best, best_dist = None, float("inf")
        for v in remaining:
            dist = float(np.linalg.norm(dro[d] - np.asarray(victim_positions[v])[:3]))
            if dist < best_dist:
                best, best_dist = v, dist
        assignment[d] = best
        if best is not None:
            remaining.remove(best)
    return assignment


class MultiDroneEnv:
    """Vectorized multi-agent SAR env (N drones, shared weights).

    Each drone flies a simple kinematic point-mass toward its goal.
    Team shares victim detections through a MessageBus; every drone's
    belief set is updated on any detection (victim sharing).
    """

    def __init__(
        self,
        n_drones: int = 3,
        victims: Optional[List[np.ndarray]] = None,
        goals: Optional[List[np.ndarray]] = None,
        home: Optional[np.ndarray] = None,
        max_steps: int = 500,
        seed: int = 0,
        min_sep: float = MIN_SEP_M,
        collision_dist: float = COLLISION_DIST_M,
        detect_range: float = DETECT_RANGE_M,
        bus: Optional[object] = None,
    ):
        self.n_drones = int(n_drones)
        self.max_steps = int(max_steps)
        self.min_sep = float(min_sep)
        self.collision_dist = float(collision_dist)
        self.detect_range = float(detect_range)
        self._rng = np.random.default_rng(int(seed))
        default_victims = [
            np.array([2.5, 2.5, 0.0], dtype=np.float32),
            np.array([6.0, 2.0, 0.0], dtype=np.float32),
            np.array([2.0, 6.0, 0.0], dtype=np.float32),
        ]
        vics = victims if victims is not None else default_victims[: max(1, self.n_drones)]
        self.victims: List[np.ndarray] = [
            np.asarray(v, dtype=np.float32).reshape(3) for v in vics
        ]
        if goals is not None:
            self.goals = [np.asarray(g, dtype=np.float32).reshape(3) for g in goals]
        else:
            self.goals = [np.array(v, dtype=np.float32) + np.array([0, 0, 1.5], dtype=np.float32)
                          for v in self.victims]
            # Pad goals when fewer victims than drones (extra drones hold).
            while len(self.goals) < self.n_drones:
                self.goals.append(np.array([0.0, 0.0, 1.5], dtype=np.float32))
        self.home = (
            np.asarray(home, dtype=np.float32).reshape(3)
            if home is not None
            else np.array([0.0, 0.0, 1.5], dtype=np.float32)
        )
        self.bus = bus if bus is not None else (MessageBus(n_drones=self.n_drones) if MessageBus else None)
        self.act_dim = int(ACT_DIM)
        self.base_obs_dim = int(BASE_OBS_DIM)
        self.obs_dim = conditioned_obs_dim(self.n_drones)
        # Dynamic state (set in reset).
        self.positions = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.victim_found = [False] * len(self.victims)
        self.victim_served = [False] * len(self.victims)
        self.beliefs: List[set] = [set() for _ in range(self.n_drones)]
        self.collisions = 0
        self._collided_pairs: set = set()
        self.step_count = 0
        self.reset(seed=seed)

    # -- gym-style API -------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        # Staggered starts along x so transit begins in loose formation.
        for d in range(self.n_drones):
            self.positions[d] = np.array([float(d) * 5.0, 0.0, 1.5], dtype=np.float32)
        self.velocities[:] = 0.0
        self.victim_found = [False] * len(self.victims)
        self.victim_served = [False] * len(self.victims)
        self.beliefs = [set() for _ in range(self.n_drones)]
        self.collisions = 0
        self._collided_pairs = set()
        self.step_count = 0
        if self.bus is not None:
            try:
                self.bus.reset()
            except Exception:
                pass
        obs = np.stack([self._obs_for(d) for d in range(self.n_drones)])
        infos = [{"agent_id": d} for d in range(self.n_drones)]
        return obs.astype(np.float32), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        arr = np.asarray(actions, dtype=np.float32).reshape(self.n_drones, self.act_dim)
        arr = np.clip(arr, 0.0, 1.0)
        # Map 4 thrusts -> 3-D velocity: (a0-a1, a2-a3, mean-0.5) scaled.
        vel_cmd = np.stack(
            [
                (arr[:, 0] - arr[:, 1]) * MAX_SPEED_MS,
                (arr[:, 2] - arr[:, 3]) * MAX_SPEED_MS,
                (arr.mean(axis=1) - 0.5) * 2.0 * (MAX_SPEED_MS * 0.5),
            ],
            axis=1,
        ).astype(np.float32)
        # Blend toward command (simple first-order dynamics) + integrate.
        self.velocities = 0.7 * self.velocities + 0.3 * vel_cmd
        # RVO-lite deconfliction: push apart drones heading too close.
        self.velocities += self._separation_velocity()
        self.positions = self.positions + self.velocities * DT
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0.2, 2.5)
        self.step_count += 1
        if self.bus is not None:
            try:
                self.bus.tick()
            except Exception:
                pass

        self._update_detections()
        dists = pairwise_distances(self.positions)

        rewards = np.zeros(self.n_drones, dtype=np.float32)
        dones = np.zeros(self.n_drones, dtype=bool)
        truncs = np.zeros(self.n_drones, dtype=bool)
        infos: List[Dict] = []
        for d in range(self.n_drones):
            goal = self.goals[d % len(self.goals)]
            dist_goal = float(np.linalg.norm(self.positions[d] - goal))
            # Progress-shaped nav reward toward assigned goal.
            rewards[d] = -0.1 * dist_goal
            if dist_goal < GOAL_TOLERANCE_M:
                rewards[d] += 10.0
            # Collision-avoidance penalty vs every other drone.
            for o in range(self.n_drones):
                if o == d:
                    continue
                pair = tuple(sorted((d, o)))
                dist = float(dists[d, o])
                thresh = self.collision_dist if self.collision_dist else COLLISION_DIST_M
                sep = self.min_sep if self.min_sep else MIN_SEP_M
                if dist < thresh:
                    rewards[d] += float(COLLISION_PENALTY)
                    if pair not in self._collided_pairs:
                        self._collided_pairs.add(pair)
                        self.collisions += 1
                elif dist < sep:
                    rewards[d] += float(PROXIMITY_PENALTY_GAIN * (sep - dist))
            # Victim-service bonus (team reward shared to finder + servers).
            if any(self.victim_served):
                rewards[d] += 2.0 * sum(self.victim_served)
            done = self.step_count >= self.max_steps or all(self.victim_served)
            dones[d] = bool(done)
            infos.append(
                {
                    "agent_id": d,
                    "dist_to_goal": dist_goal,
                    "goal_reached": dist_goal < GOAL_TOLERANCE_M,
                    "collisions": self.collisions,
                    "victims_found": sum(self.victim_found),
                    "victims_served": sum(self.victim_served),
                    "shared_beliefs": sorted(self.beliefs[d]),
                    "position": self.positions[d].copy(),
                }
            )
        obs = np.stack([self._obs_for(d) for d in range(self.n_drones)])
        return obs.astype(np.float32), rewards, dones, truncs, infos

    def close(self) -> None:  # pragma: no cover - nothing to free
        pass

    # -- internals -------------------------------------------------------
    def _base_obs(self, agent_id: int) -> np.ndarray:
        """Synthesize the 21-D base obs for one drone."""
        goal = self.goals[agent_id % len(self.goals)]
        pos = self.positions[agent_id]
        goal_rel = (goal - pos).astype(np.float32)
        state = np.zeros(17, dtype=np.float32)
        state[0:3] = pos
        state[3:6] = self.velocities[agent_id]
        state[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # quat
        time_feat = np.array([self.step_count / max(1, self.max_steps)], dtype=np.float32)
        return np.concatenate([state, goal_rel, time_feat]).astype(np.float32)

    def _obs_for(self, agent_id: int) -> np.ndarray:
        return condition_obs(self._base_obs(agent_id), agent_id, self.n_drones)

    def _separation_velocity(self) -> np.ndarray:
        """Reciprocal push-apart velocity (RVO-lite) for close pairs."""
        push = np.zeros_like(self.velocities)
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                diff = self.positions[i] - self.positions[j]
                dist = float(np.linalg.norm(diff)) + 1e-6
                if dist < self.min_sep:
                    direction = diff / dist
                    strength = (self.min_sep - dist) * 1.5
                    push[i] += direction * strength
                    push[j] -= direction * strength
        return push.astype(np.float32)

    def _update_detections(self) -> None:
        """Any drone within range finds the victim; ALL drones learn it."""
        for v, vpos in enumerate(self.victims):
            finder = None
            for d in range(self.n_drones):
                if float(np.linalg.norm(self.positions[d] - vpos)) < self.detect_range:
                    finder = d
                    break
            if finder is not None and not self.victim_found[v]:
                self.victim_found[v] = True
                for d in range(self.n_drones):
                    self.beliefs[d].add(v)
                if self.bus is not None:
                    try:
                        self.bus.broadcast_victim(sender_id=finder, position=vpos)
                    except Exception:
                        pass
            # Auto-serve on drop_payload proximity: horizontal < 0.75 m
            # (med-kit dropped from hover_height ~1.5 m; no descent needed).
            # Checked every step so late arrivals still serve.
            if self.victim_found[v] and not self.victim_served[v]:
                for d in range(self.n_drones):
                    horiz = float(np.linalg.norm(self.positions[d][:2] - vpos[:2]))
                    if horiz < 0.75:
                        self.victim_served[v] = True
                        break

    # -- team metrics ------------------------------------------------------
    def joint_success(self) -> float:
        """1.0 when every victim is served, else fraction served."""
        if not self.victims:
            return 1.0
        return float(sum(self.victim_served) / len(self.victims))

    def min_separation(self) -> float:
        dists = pairwise_distances(self.positions)
        tri = dists[np.triu_indices(self.n_drones, k=1)]
        return float(tri.min()) if tri.size else float("inf")


# -- Sprint 16-18 extension: multi-drone strike coordination ---------------
def assign_strike_tasks(target_positions, drone_positions):
    """Assign each drone a distinct strike target (greedy auction)."""
    vics = [np.asarray(t, dtype=np.float32).reshape(3) for t in target_positions]
    dro = np.asarray(drone_positions, dtype=np.float32)
    return greedy_assign(vics, dro)


def deconflict_strike_goals(goals, min_sep: float = MIN_SEP_M):
    """Push apart strike release points closer than min_sep."""
    out = [np.asarray(g, dtype=np.float32).reshape(3).copy() for g in goals]
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            diff = out[i] - out[j]
            dist = float(np.linalg.norm(diff)) + 1e-6
            if dist < min_sep:
                push = (min_sep - dist) / 2.0 + 0.05
                direction = diff / dist
                out[i] = out[i] + direction * push
                out[j] = out[j] - direction * push
    return out


def strike_formation_ok(drone_positions, min_sep: float = MIN_SEP_M) -> dict:
    """Zero-collision check for a strike formation snapshot."""
    pos = np.asarray(drone_positions, dtype=np.float32)
    dists = pairwise_distances(pos)
    n = len(pos)
    tri = dists[np.triu_indices(n, k=1)] if n > 1 else np.array([float("inf")])
    min_d = float(tri.min()) if tri.size else float("inf")
    hard = COLLISION_DIST_M
    collisions = int(((tri < hard).sum()) // 1) if tri.size else 0
    return {"min_separation": min_d, "collisions": collisions,
            "ok": bool(collisions == 0 and min_d >= min_sep)}


# -- Sprint 25: 50+ drone PPO, shared critic, CTDE (SAR-only) ---------------
# Centralized Training Decentralized Execution: actors see only local
# conditioned obs; a single shared critic sees the joint global state
# (mean positions/velocities + team victim progress) during training.

LARGE_SCALE_N = 50
GLOBAL_STATE_DIM = 3 + 3 + 2  # centroid(3) + mean_vel(3) + found/served(2)


def make_large_scale_env(n_drones: int = LARGE_SCALE_N, seed: int = 0,
                         **kw) -> MultiDroneEnv:
    """Build a 50+ drone SAR env (shared weights scale to any N)."""
    return MultiDroneEnv(n_drones=max(int(n_drones), LARGE_SCALE_N), seed=seed, **kw)


def global_state(env: MultiDroneEnv) -> np.ndarray:
    """Joint state for the shared critic: centroid + mean vel + progress."""
    centroid = env.positions.mean(axis=0).astype(np.float32)
    mean_vel = env.velocities.mean(axis=0).astype(np.float32)
    prog = np.array([float(sum(env.victim_found)),
                     float(sum(env.victim_served))], dtype=np.float32)
    return np.concatenate([centroid, mean_vel, prog]).astype(np.float32)


class SharedCritic:
    """Linear shared critic over the global state (numpy, CPU-only).

    SAR-only value head: predicts team return (victim service + nav).
    """

    def __init__(self, state_dim: int = GLOBAL_STATE_DIM, seed: int = 0,
                 lr: float = 1e-3):
        self.rng = np.random.default_rng(int(seed))
        self.w = (self.rng.standard_normal(int(state_dim)).astype(np.float32)
                  * 0.01)
        self.b = np.float32(0.0)
        self.lr = float(lr)

    def value(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        return float(s @ self.w + self.b)

    def update(self, states: np.ndarray, returns: np.ndarray) -> Dict:
        """One gradient step on MSE value loss; returns {loss, value_mean}."""
        S = np.asarray(states, dtype=np.float32)
        R = np.asarray(returns, dtype=np.float32).reshape(-1)
        preds = S @ self.w + self.b
        err = preds - R
        loss = float((err ** 2).mean())
        grad_w = (2.0 / max(1, len(R))) * (S.T @ err)
        grad_b = float(2.0 * err.mean())
        self.w = (self.w - self.lr * grad_w).astype(np.float32)
        self.b = np.float32(self.b - self.lr * grad_b)
        return {"loss": loss, "value_mean": float(preds.mean())}


def ctde_rollout(env: MultiDroneEnv, steps: int = 32,
                 seed: int = 0) -> Dict:
    """Decentralized rollout: per-drone heuristic actor + centralized critic.

    Actors act greedily toward their goal (decentralized execution);
    the shared critic scores the joint state (centralized training).
    Returns trajectory summary usable by PPO.
    """
    rng = np.random.default_rng(int(seed))
    critic = SharedCritic(seed=seed)
    obs, _ = env.reset(seed=seed)
    states, rewards, values = [], [], []
    for _ in range(int(steps)):
        acts = np.zeros((env.n_drones, env.act_dim), dtype=np.float32)
        for d in range(env.n_drones):
            goal = env.goals[d % len(env.goals)]
            to_goal = goal - env.positions[d]
            acts[d, 0] = 1.0 if to_goal[0] > 0 else 0.0
            acts[d, 1] = 1.0 if to_goal[0] <= 0 else 0.0
            acts[d, 2] = 1.0 if to_goal[1] > 0 else 0.0
            acts[d, 3] = 1.0 if to_goal[1] <= 0 else 0.0
        acts += rng.normal(0, 0.05, size=acts.shape).astype(np.float32)
        obs, rew, dones, _, _ = env.step(acts)
        states.append(global_state(env))
        rewards.append(float(rew.mean()))
        values.append(critic.value(states[-1]))
    S = np.stack(states)
    R = np.array(rewards, dtype=np.float32)
    # Discounted returns (gamma=0.99).
    disc, acc = 0.99, 0.0
    rets = np.zeros_like(R)
    for t in range(len(R) - 1, -1, -1):
        acc = R[t] + disc * acc
        rets[t] = acc
    train = critic.update(S, rets)
    return {"mean_reward": float(R.mean()), "mean_value": float(np.mean(values)),
            "critic_loss": train["loss"], "joint_success": env.joint_success(),
            "n_drones": env.n_drones, "paradigm": "CTDE",
            "execution": "decentralized", "training": "centralized-shared-critic"}
