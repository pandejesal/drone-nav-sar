#!/usr/bin/env python3
"""CleanRL-style PPO trainer for DroneNav-SAR (Sprint 4 v3).

CPU-only, tanh-Gaussian policy, N=8 seeded DroneVecEnv rollouts,
GAE lambda returns, clipped surrogate loss, value loss, entropy bonus,
gradient clipping, saves policy.pt.

SAR-only: navigate_to / hover / drop_payload / return_home.
No weapons, no targeting, no kinetic.

Usage:
    python -m src.rl.ppo_nav --episodes 200 --seed 0 --mesh mock.glb
"""

import argparse
import dataclasses
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rl.policies import ACT_DIM, OBS_DIM, DEVICE, ActorCritic
from src.rl.vec_env import DroneVecEnv
from src.sim.domain_randomization import get_dynamic_config
from src.rl.local_policy import LocalPolicy
from src.rl.hierarchical import HierarchicalPolicy
from src.rl.global_planner import create_global_planner
from src.eval.metrics import (
    EvalReport,
    build_report,
    energy_proxy,
    spl,
    success_rate,
)


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> np.ndarray:
    """Compute GAE-lambda advantages.

    Parameters
    ----------
    rewards : (T,) float32
    values  : (T+1,) float32  — includes bootstrap value
    dones   : (T,) bool

    Returns
    -------
    advantages : (T,) float32
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1]
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_nonterminal * last_gae
    return advantages


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RolloutBuffer:
    """Fixed-size buffer for one PPO update epoch."""
    obs: np.ndarray       # (T, n_envs, obs_dim)
    actions: np.ndarray   # (T, n_envs, act_dim)
    logprobs: np.ndarray  # (T, n_envs)
    rewards: np.ndarray   # (T, n_envs)
    dones: np.ndarray     # (T, n_envs)
    values: np.ndarray    # (T, n_envs)

    def __init__(self, T: int, n_envs: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((T, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, n_envs, act_dim), dtype=np.float32)
        self.logprobs = np.zeros((T, n_envs), dtype=np.float32)
        self.rewards = np.zeros((T, n_envs), dtype=np.float32)
        self.dones = np.zeros((T, n_envs), dtype=np.float32)
        self.values = np.zeros((T, n_envs), dtype=np.float32)
        self.advantages = np.zeros((T, n_envs), dtype=np.float32)
        self.returns = np.zeros((T, n_envs), dtype=np.float32)

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        """Compute GAE advantages and returns in-place."""
        T = self.obs.shape[0]
        n_envs = self.obs.shape[1]
        for e in range(n_envs):
            adv = compute_gae(
                self.rewards[:, e],
                np.append(self.values[:, e], last_values[e]),
                self.dones[:, e],
                gamma=gamma,
                lam=lam,
            )
            self.advantages[:, e] = adv
            self.returns[:, e] = adv + self.values[:, e]

    def minibatch_generator(self, n_minibatches: int):
        """Yield (obs, actions, logprobs, advantages, returns) shuffled."""
        T, n_envs = self.obs.shape[:2]
        total = T * n_envs
        obs_flat = self.obs.reshape(total, -1)
        act_flat = self.actions.reshape(total, -1)
        logp_flat = self.logprobs.reshape(total)
        adv_flat = self.advantages.reshape(total)
        ret_flat = self.returns.reshape(total)

        indices = np.arange(total)
        np.random.shuffle(indices)
        mb_size = total // n_minibatches

        for start in range(0, total, mb_size):
            end = start + mb_size
            mb_idx = indices[start:end]
            yield (
                obs_flat[mb_idx],
                act_flat[mb_idx],
                logp_flat[mb_idx],
                adv_flat[mb_idx],
                ret_flat[mb_idx],
            )


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def ppo_loss(
    policy,  # ActorCritic | HierarchicalPolicy (both expose forward/log_std/act_dim)
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
):
    """Compute PPO clipped surrogate loss + value loss + entropy bonus.

    Returns (total_loss, policy_loss, value_loss, entropy).
    """
    mean_raw, values = policy.forward(obs)
    std = torch.exp(torch.clamp(policy.log_std, -5.0, 2.0))

    # Invert the squash to recover the pre-squash sample u for the STORED
    # actions: a in [0,1] -> tanh-space [-1,1] -> u = atanh. Sampling fresh
    # noise here (old bug) compares the wrong logp and breaks the PPO ratio.
    a_clipped = torch.clamp(actions * 2.0 - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)
    u = 0.5 * (torch.log1p(a_clipped) - torch.log1p(-a_clipped))
    logprobs = policy._squashed_logprob(u, mean_raw, std)

    # Ratio
    ratio = torch.exp(logprobs - old_logprobs)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss with clipping
    value_loss = 0.5 * ((values - returns) ** 2).mean()

    # Entropy bonus (approximate from log_std)
    entropy = 0.5 * (1.0 + torch.log(2.0 * np.pi * torch.ones(policy.act_dim).to(DEVICE))) + policy.log_std
    entropy = entropy.mean()

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return total, policy_loss.item(), value_loss.item(), entropy.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Sprint 7 SAR: multi-task heads (nav, hover, detect, drop, return).
# One shared trunk + one value head per sub-task so a single --sar-mission
# policy can serve the mission-planner skills.
SAR_TASKS = ("nav", "hover", "detect", "drop", "return")
SAR_TASK_IDS = {name: i for i, name in enumerate(SAR_TASKS)}


class SARMultiTaskHeads(nn.Module):
    """Multi-task value heads over a shared trunk (Sprint 7 SAR).

    Wraps an ActorCritic: the actor trunk is shared, and each SAR
    sub-task (nav/hover/detect/drop/return) gets its own value head.
    forward(obs, task_id) returns (mean_raw, task_value).
    """

    def __init__(self, base_policy: ActorCritic, n_tasks: int = 5):
        super().__init__()
        self.base = base_policy
        self.n_tasks = n_tasks
        hidden = 64
        self.task_value_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            for _ in range(n_tasks)
        ])
        self.obs_dim = base_policy.obs_dim
        self.act_dim = base_policy.act_dim
        self.log_std = base_policy.log_std  # shared actor noise
        self.to(DEVICE)

    def forward(self, obs: torch.Tensor, task_id: int = 0):
        mean_raw, shared_value = self.base.forward(obs)
        head = self.task_value_heads[int(task_id) % self.n_tasks]
        task_value = head(shared_value.unsqueeze(-1)).squeeze(-1)
        return mean_raw, task_value

    def _squashed_logprob(self, u, mean_raw, std):
        return self.base._squashed_logprob(u, mean_raw, std)

    @torch.no_grad()
    def get_action(self, obs, task_id: int = 0, deterministic: bool = False):
        single = False
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            single = True
            arr = arr[None, :]
        t = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        mean_raw, value = self.forward(t, task_id=int(task_id))
        std = torch.exp(torch.clamp(self.log_std, -5.0, 2.0))
        u = mean_raw if deterministic else mean_raw + std * torch.randn_like(mean_raw)
        action = self.base._squash(u)
        logp = self._squashed_logprob(u, mean_raw, std)
        action_np = np.clip(action.cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if single:
            return action_np[0], float(value.cpu().numpy()[0]), float(logp.cpu().numpy()[0])
        return action_np, value.cpu().numpy(), logp.cpu().numpy()


def apply_sar_mission(env: DroneVecEnv, enabled: bool = True) -> Dict:
    """Enable SAR mission mode (victims + med-kit) on every sub-env backend."""
    for sub in getattr(env, "envs", []):
        backend = getattr(sub, "backend", None)
        if backend is not None and hasattr(backend, "sar_mission"):
            backend.sar_mission = bool(enabled)
            try:
                backend._spawn_victims()
            except Exception:
                pass
    return {"sar_mission": bool(enabled)}


# FIX-4A: fixed in-room goal per spec (Sprint 4 trains fixed A->B pairs).
# Base _sample_random_goal draws radius 3-15m — outside the 4m room and
# undiscoverable (2003 eps, sr=0.000). Random sampler in base_env is
# untouched (Sprint-3-tested); the trainer pins a reachable fixed goal.
# Default B sits inside the 4x4x2.5m geofence, ~2.9m from start [0,0,2.0].
FIXED_GOAL = [2.0, 1.5, 1.5]
NOMINAL_SPEED_MS = 1.0  # reference speed for SPL optimal_steps
CTRL_DT = 0.02  # 50 Hz control timestep


# Sprint-4 curriculum: start easy (level 0), auto-advance to target (level 3).
# Mirrors MockDroneBackend.CURRICULUM goal/step schedule (reward weights live
# in the backend); the trainer applies goal tolerance + step budget per level.
CURRICULUM_WINDOW = 50
CURRICULUM_SR_THRESHOLD = 0.8
CURRICULUM_MAX_LEVEL = 3
CURRICULUM_GOAL_TOLERANCE = [1.5, 1.0, 0.7, 0.5]
CURRICULUM_MAX_STEPS = [300, 400, 500, 500]


def maybe_advance_curriculum(
    current_level: int,
    episode_results: List[Dict],
    window: int = CURRICULUM_WINDOW,
    threshold: float = CURRICULUM_SR_THRESHOLD,
) -> int:
    """Advance one curriculum level when rolling success rate beats threshold.

    Returns current_level + 1 when the last `window` episodes have
    success_rate > threshold, else the unchanged level (also unchanged when
    the window is not full yet or already at max level).
    """
    if current_level >= CURRICULUM_MAX_LEVEL:
        return current_level
    recent = episode_results[-window:]
    if len(recent) < window:
        return current_level
    if success_rate(recent) > threshold:
        return current_level + 1
    return current_level


def apply_curriculum_level(env: DroneVecEnv, level: int) -> None:
    """Push a curriculum level into every sub-env (tolerance + step budget)."""
    level = max(0, min(int(level), CURRICULUM_MAX_LEVEL))
    for sub in getattr(env, "envs", []):
        sub.config.goal_tolerance = CURRICULUM_GOAL_TOLERANCE[level]
        sub.config.max_episode_steps = CURRICULUM_MAX_STEPS[level]
        backend = getattr(sub, "backend", None)
        if backend is not None and hasattr(backend, "curriculum_level"):
            backend.curriculum_level = level


def apply_dynamic_obstacles(
    env: DroneVecEnv, level: int, enabled: bool = True
) -> Dict:
    """Pass the Sprint 5 obstacle config for `level` into every sub-env.

    Level map (spec): L0 static; L1 1 person, no wind; L2 2 people + light
    wind; L3 3 people + doors + gusts. Sub-envs backed by MockDroneBackend
    get live obstacle lists; other mock envs keep a marker attribute.
    Returns the applied config dict.
    """
    level = max(0, min(int(level), CURRICULUM_MAX_LEVEL))
    cfg = get_dynamic_config(level) if enabled else {
        "num_people": 0, "use_doors": False, "wind": False, "gusts": False,
    }
    for sub in getattr(env, "envs", []):
        sub.dynamic_obstacle_config = dict(cfg)
        backend = getattr(sub, "backend", None)
        if backend is not None and hasattr(backend, "spawn_obstacles"):
            backend.dynamic_obstacles = bool(enabled)
            backend.num_people = int(cfg["num_people"])
            backend.use_doors = bool(cfg["use_doors"])
            try:
                backend.spawn_obstacles()
            except Exception:
                pass
    return cfg


def train_multi_drone(
    n_drones: int = 3,
    episodes: int = 200,
    seed: int = 0,
    rollout_steps: int = 128,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    lr: float = 3e-4,
    grad_clip: float = 0.5,
    max_steps: int = 500,
    save_path: str = "policy_sprint9.pt",
    log_interval: int = 10,
) -> EvalReport:
    """Sprint 9: shared-weights PPO over a MultiDroneEnv team.

    Shared ActorCritic with agent_id one-hot conditioning
    (obs_dim = OBS_DIM + n_drones). Each drone is one rollout stream;
    collision-avoidance + victim-sharing rewards come from the env.
    SAR-only skills downstream: navigate_to/hover/drop_payload/return_home.
    """
    from src.rl.multi_drone import MultiDroneEnv, conditioned_obs_dim

    np.random.seed(seed)
    torch.manual_seed(seed)
    n_drones = max(2, int(n_drones))
    cond_dim = conditioned_obs_dim(n_drones)
    env = MultiDroneEnv(n_drones=n_drones, max_steps=max_steps, seed=seed)
    print(f"[multi-drone] team={n_drones} cond_obs_dim={cond_dim} "
          f"(base {OBS_DIM} + agent_id one-hot {n_drones})")
    policy = ActorCritic(obs_dim=cond_dim, act_dim=ACT_DIM)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer(T=rollout_steps, n_envs=n_drones,
                           obs_dim=cond_dim, act_dim=ACT_DIM)
    episodes_target, episodes_collected, update_count = episodes, 0, 0
    t_start = time.time()
    loss = torch.zeros((), device=DEVICE)
    episode_results: List[Dict] = []
    all_actions: List[np.ndarray] = []
    joint_sr, team_collisions = 0.0, 0
    obs, _ = env.reset(seed=seed)
    while episodes_collected < episodes_target:
        policy.eval()
        for t in range(rollout_steps):
            with torch.no_grad():
                action, value, logprob = policy.get_action(obs)
            obs_next, rewards, dones, truncs, infos = env.step(action)
            buffer.obs[t] = obs
            buffer.actions[t] = action
            buffer.logprobs[t] = logprob
            buffer.rewards[t] = rewards
            buffer.dones[t] = dones | truncs
            buffer.values[t] = np.asarray(value, dtype=np.float32).reshape(-1)
            all_actions.append(action.copy())
            obs = obs_next
            if bool(dones.any()) or env.joint_success() >= 1.0:
                team_collisions = int(env.collisions)
                joint_sr = float(env.joint_success())
                for i, info in enumerate(infos):
                    episode_results.append({
                        "goal_reached": bool(info.get("goal_reached", False)),
                        "steps": int(env.step_count),
                        "optimal_steps": max(1, int(max_steps // 2)),
                        "position": info.get("position"),
                        "dist_to_goal": float(info.get("dist_to_goal", 0.0)),
                    })
                    episodes_collected += 1
                obs, _ = env.reset(seed=seed + episodes_collected)
        with torch.no_grad():
            _, last_values_raw, _ = policy.get_action(obs)
        last_values = np.asarray(last_values_raw, dtype=np.float32).reshape(-1)
        buffer.compute_returns_and_advantages(last_values, gamma=gamma, lam=lam)
        policy.train()
        for _ in range(n_epochs):
            for mb_obs, mb_act, mb_logp, mb_adv, mb_ret in buffer.minibatch_generator(
                n_minibatches
            ):
                t_obs = torch.as_tensor(mb_obs, dtype=torch.float32, device=DEVICE)
                t_act = torch.as_tensor(mb_act, dtype=torch.float32, device=DEVICE)
                t_logp = torch.as_tensor(mb_logp, dtype=torch.float32, device=DEVICE)
                t_adv = torch.as_tensor(mb_adv, dtype=torch.float32, device=DEVICE)
                t_ret = torch.as_tensor(mb_ret, dtype=torch.float32, device=DEVICE)
                t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
                loss, _, _, _ = ppo_loss(
                    policy, t_obs, t_act, t_logp, t_adv, t_ret,
                    clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optimizer.step()
        update_count += 1
        if update_count % log_interval == 0:
            elapsed = time.time() - t_start
            print(f"[multi-drone update {update_count:4d}] "
                  f"episodes={episodes_collected}/{episodes_target} "
                  f"joint_sr={joint_sr:.3f} collisions={team_collisions} "
                  f"loss={loss.item():.4f} elapsed={elapsed:.1f}s")
    torch.save(policy.state_dict(), save_path)
    print(f"Policy saved to {save_path}")
    elapsed = time.time() - t_start
    actions_arr = np.stack(all_actions) if all_actions else np.zeros((1, n_drones, ACT_DIM))
    report = build_report(
        episode_results=episode_results if episode_results else [
            {"goal_reached": False, "steps": max_steps,
             "optimal_steps": max(1, max_steps // 2)}],
        all_actions=actions_arr,
        dt=0.02,
        spec_hash=hashlib.sha256(b"multi-drone-sprint9").hexdigest()[:16],
        seed=seed,
        elapsed_s=elapsed,
    )
    print(report.summary())
    print(f"[multi-drone] joint_sr={joint_sr:.3f} collisions={team_collisions}")
    env.close()
    return report


def train(
    episodes: int = 200,
    seed: int = 0,
    mesh: str = "mock.glb",
    n_envs: int = 8,
    rollout_steps: int = 128,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    lr: float = 3e-4,
    grad_clip: float = 0.5,
    goal: Optional[List[float]] = None,
    max_steps: int = 500,
    save_path: str = "policy.pt",
    log_interval: int = 10,
    curriculum: bool = False,
    dynamic_obstacles: bool = False,
    sar_mission: bool = False,
    hierarchical: bool = False,
    multi_drone: int = 0,
    language: bool = False,
) -> EvalReport:
    """Run PPO training loop and return final EvalReport.

    Parameters
    ----------
    episodes : int
        Total episodes to collect across all envs.
    seed : int
        Random seed.
    mesh : str
        Mesh file path for the mock env.
    n_envs : int
        Number of parallel environments.
    rollout_steps : int
        Steps per rollout before each PPO update.
    n_epochs : int
        PPO epochs per update.
    n_minibatches : int
        Minibatches per epoch.
    gamma : float
        Discount factor.
    lam : float
        GAE lambda.
    clip_eps : float
        PPO clipping epsilon.
    vf_coef : float
        Value loss coefficient.
    ent_coef : float
        Entropy bonus coefficient.
    lr : float
        Learning rate.
    grad_clip : float
        Max gradient norm.
    save_path : str
        Path to save trained policy.
    log_interval : int
        Print stats every N updates.
    goal : list of float or None
        Fixed goal [x, y, z] (default in-room [2.0, 1.5, 1.5]). None falls
        back to per-env random sampling (not recommended: radius 3-15m).
    max_steps : int
        Max steps per episode (default 500; room crossing needs ~150-300).
    hierarchical : bool
        Enable hierarchical policy (global planner + local policy).
    sar_mission : bool
        Enable SAR mission (multi-room victims + med-kit drop) with
        multi-task value heads (nav/hover/detect/drop/return).

    Returns
    -------
    EvalReport
    """
    # Sprint 9: shared-weights multi-drone team (agent_id conditioning).
    if int(multi_drone or 0) >= 2:
        return train_multi_drone(
            n_drones=int(multi_drone),
            episodes=episodes,
            seed=seed,
            rollout_steps=rollout_steps,
            n_epochs=n_epochs,
            n_minibatches=n_minibatches,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            lr=lr,
            grad_clip=grad_clip,
            max_steps=max_steps,
            save_path=save_path,
            log_interval=log_interval,
        )
    np.random.seed(seed)
    torch.manual_seed(seed)

    # FIX-4A: pin a fixed in-room goal (start is [0,0,2.0] in mock).
    # Default B=[2.0, 1.5, 1.5] sits inside the 4x4x2.5m geofence (~2.9m away).
    goal_vec = np.array(
        goal if goal is not None else list(FIXED_GOAL), dtype=np.float32
    )

    # Build env and policy
    env = DroneVecEnv(
        n_envs=n_envs,
        mesh_or_config=mesh,
        master_seed=seed,
        goal_position=goal_vec,
        max_episode_steps=max_steps,
    )
    # Sprint-4 curriculum: start easy, auto-advance on rolling success rate.
    curriculum_level = 0 if curriculum else CURRICULUM_MAX_LEVEL
    if curriculum:
        apply_curriculum_level(env, curriculum_level)
        print(
            f"[curriculum] start level {curriculum_level} "
            f"(tol={CURRICULUM_GOAL_TOLERANCE[0]}m, "
            f"max_steps={CURRICULUM_MAX_STEPS[0]})"
        )
    # Sprint 5: dynamic obstacles (people/doors) + wind per curriculum level.
    if dynamic_obstacles:
        dyn_cfg = apply_dynamic_obstacles(env, curriculum_level, enabled=True)
        print(f"[dynamic-obstacles] level {curriculum_level}: {dyn_cfg}")

    # Sprint 7: SAR mission (multi-room victims + med-kit drop).
    multitask_policy = None
    if sar_mission:
        sar_cfg = apply_sar_mission(env, enabled=True)
        print(f"[sar-mission] enabled: {sar_cfg}")

    # Sprint 11: language-conditioned tasks (SAR-only).
    if language:
        try:
            from src.control.language_to_task import text_to_task
            _demo = ["deliver medkit to room 2", "go to room 1", "return home"]
            for _cmd in _demo:
                _res = text_to_task(_cmd)
                print(f"[language] '{_cmd}' -> task={_res['task_embedding_id']} "
                      f"skill={_res['skill']} latency_ms={_res['latency_ms']:.1f}")
            print("[language] enabled: text commands → task_embedding_id + params")
        except Exception as _e:
            print(f"[language] enabled (validation skipped: {_e})")

    # Sprint 8: hierarchical policy (global planner + local policy)
    if hierarchical:
        import importlib
        import src.rl.local_policy
        importlib.reload(src.rl.local_policy)
        global_planner = create_global_planner()
        local_policy = src.rl.local_policy.LocalPolicy()
        policy = HierarchicalPolicy(local_policy, global_planner)
        print("[hierarchical] enabled: global planner + local policy")
    else:
        base_policy = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        if sar_mission:
            # Sprint 7: wrap with multi-task heads (nav/hover/detect/drop/return).
            multitask_policy = SARMultiTaskHeads(base_policy)
            policy = multitask_policy
            print(f"[sar-mission] multi-task heads: {list(SAR_TASKS)}")
        else:
            policy = base_policy
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    buffer = RolloutBuffer(
        T=rollout_steps,
        n_envs=n_envs,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
    )

    episodes_target = episodes
    episodes_collected = 0
    update_count = 0
    t_start = time.time()
    loss = torch.zeros((), device=DEVICE)  # bound before first update for logging

    # Episode tracking for eval
    episode_results: List[Dict] = []
    all_actions: List[np.ndarray] = []

    obs, reset_infos = env.reset(seed=seed)
    # Start-of-episode distances for honest SPL optimal_steps: straight-line
    # distance at 1 m/s nominal speed, dt=0.02 (50 Hz).
    start_dists = np.array(
        [float(info.get("dist_to_goal", 0.0)) for info in reset_infos],
        dtype=np.float32,
    )

    while episodes_collected < episodes_target:
        # ---- Rollout ----
        policy.eval()
        for t in range(rollout_steps):
            with torch.no_grad():
                if hierarchical:
                    # Hierarchical policy: get subgoal from global planner
                    subgoal = policy.get_current_subgoal()
                    if subgoal is None:
                        # No global plan, use goal as subgoal
                        subgoal = goal_vec
                    # Determine task_id from info (simplified: nav=0)
                    task_id = 0
                    action, value, logprob = policy.get_action(obs, subgoal, task_id)
                else:
                    action, value, logprob = policy.get_action(obs)
            obs_next, rewards, dones, truncs, infos = env.step(action)

            buffer.obs[t] = obs
            buffer.actions[t] = action
            buffer.logprobs[t] = logprob
            buffer.rewards[t] = rewards
            buffer.dones[t] = dones | truncs
            buffer.values[t] = value

            all_actions.append(action.copy())
            obs = obs_next

            # Track episode completions. On auto-reset the vec env preserves
            # terminal outcome as flat keys (goal_reached/terminal_steps/...)
            # while dist_to_goal/position describe the FRESH episode.
            for i, info in enumerate(infos):
                if info.get("episode_done", False) or dones[i] or truncs[i]:
                    final_obs = info.get("final_obs", obs[i])
                    optimal = max(1, int(np.ceil(start_dists[i] / (1.0 * 0.02))))
                    ep_info = {
                        "goal_reached": info.get("goal_reached", False),
                        "steps": info.get("terminal_steps", info.get("step", 0)),
                        "optimal_steps": optimal,
                        "position": info.get("terminal_position"),
                        "dist_to_goal": info.get("terminal_dist", float("inf")),
                    }
                    episode_results.append(ep_info)
                    episodes_collected += 1
                    # Fresh dist_to_goal is always present after reset.
                    if "dist_to_goal" in info:
                        start_dists[i] = float(info["dist_to_goal"])

        # Bootstrap value (hierarchical policy accepts plain obs too)
        with torch.no_grad():
            if hierarchical:
                _, last_values_raw, _ = policy.get_action(obs)
            else:
                _, last_values_raw, _ = policy.get_action(obs)
        last_values: np.ndarray = np.asarray(last_values_raw, dtype=np.float32).reshape(-1)
        buffer.compute_returns_and_advantages(last_values, gamma=gamma, lam=lam)

        # ---- PPO update ----
        policy.train()
        for epoch in range(n_epochs):
            for mb_obs, mb_act, mb_logp, mb_adv, mb_ret in buffer.minibatch_generator(
                n_minibatches
            ):
                t_obs = torch.as_tensor(mb_obs, dtype=torch.float32, device=DEVICE)
                t_act = torch.as_tensor(mb_act, dtype=torch.float32, device=DEVICE)
                t_logp = torch.as_tensor(mb_logp, dtype=torch.float32, device=DEVICE)
                t_adv = torch.as_tensor(mb_adv, dtype=torch.float32, device=DEVICE)
                t_ret = torch.as_tensor(mb_ret, dtype=torch.float32, device=DEVICE)

                # Normalize advantages
                t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

                loss, pol_loss, val_loss, ent = ppo_loss(
                    policy, t_obs, t_act, t_logp, t_adv, t_ret,
                    clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optimizer.step()

        update_count += 1

        # ---- Curriculum auto-advance: sr>0.8 over last 50 eps → next level.
        if curriculum:
            new_level = maybe_advance_curriculum(curriculum_level, episode_results)
            if new_level != curriculum_level:
                print(
                    f"[curriculum] advanced level {curriculum_level} -> {new_level} "
                    f"(tol={CURRICULUM_GOAL_TOLERANCE[new_level]}m, "
                    f"max_steps={CURRICULUM_MAX_STEPS[new_level]})"
                )
                curriculum_level = new_level
                apply_curriculum_level(env, curriculum_level)
                if dynamic_obstacles:
                    dyn_cfg = apply_dynamic_obstacles(
                        env, curriculum_level, enabled=True
                    )
                    print(
                        f"[dynamic-obstacles] level {curriculum_level}: {dyn_cfg}"
                    )

        # ---- Logging ----
        if update_count % log_interval == 0 and len(episode_results) > 0:
            recent = episode_results[-min(50, len(episode_results)):]
            sr = success_rate(recent)
            sp = spl(recent)
            elapsed = time.time() - t_start
            lvl = f" lvl={curriculum_level}" if curriculum else ""
            print(
                f"[update {update_count:4d}] "
                f"episodes={episodes_collected}/{episodes_target} "
                f"sr={sr:.3f} spl={sp:.3f}{lvl} "
                f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
            )

    # ---- Save policy ----
    torch.save(policy.state_dict(), save_path)
    print(f"Policy saved to {save_path}")

    # ---- Final eval ----
    elapsed = time.time() - t_start
    actions_arr = np.stack(all_actions) if all_actions else np.zeros((1, n_envs, ACT_DIM))
    spec_hash = hashlib.sha256(mesh.encode()).hexdigest()[:16]

    report = build_report(
        episode_results=episode_results,
        all_actions=actions_arr,
        dt=0.02,
        spec_hash=spec_hash,
        seed=seed,
        elapsed_s=elapsed,
    )
    print(report.summary())
    env.close()
    return report


# ---------------------------------------------------------------------------
# Sprint 10: edge export hook (SAR-only policies -> ONNX/TensorRT/INT8)
# ---------------------------------------------------------------------------

def maybe_export_policy(save_path: str, export: Optional[str] = None,
                        device: str = "jetson",
                        precision: str = "fp16") -> Optional[Dict]:
    """Export a saved policy to ONNX (+ device artifact) after training.

    device: jetson -> TensorRT manifest; rpi -> dynamic INT8 artifact.
    Returns the export report dict, or None when export is falsy.
    """
    if not export:
        return None
    from src.control.edge_export import (
        export_onnx,
        optimize_tensorrt,
        quantize_for_device,
    )
    report = export_onnx(save_path, export, validate=True)
    print(f"[edge-export] onnx={export} valid={report.get('valid')} "
          f"err={report.get('max_abs_err')}")
    if device == "jetson":
        trt = optimize_tensorrt(export, export.replace(".onnx", "_trt.engine"),
                                fp16=(precision == "fp16"),
                                precision=precision, device=device)
        print(f"[edge-export] TensorRT: {trt}")
        report["tensorrt"] = trt
    elif device == "rpi":
        qrep = quantize_for_device(export, export.replace(".onnx", "_quant.onnx"),
                                   precision=precision, device=device)
        print(f"[edge-export] Quant: {qrep}")
        report["quant"] = qrep
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PPO trainer for DroneNav-SAR (SAR-only, CPU)"
    )
    parser.add_argument("--episodes", type=int, default=200, help="Total episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--mesh", type=str, default="mock.glb", help="Mesh file")
    parser.add_argument("--n-envs", type=int, default=8, help="Num parallel envs")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="policy.pt", help="Save path")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--goal", type=float, nargs=3, default=[2.0, 1.5, 1.5],
                        help="Fixed goal x y z (default in-room B)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps/episode")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--dynamic-obstacles", action="store_true", help="Enable dynamic obstacles (people, doors, wind gusts)")
    parser.add_argument("--sar-mission", action="store_true", help="Enable SAR mission (multi-room, victim detection, med-kit drop)")
    parser.add_argument("--hierarchical", action="store_true", help="Enable hierarchical policy (global planner + local subgoal policy)")
    parser.add_argument("--multi-drone", type=int, default=0, help="Sprint 9: team size (>=2 enables shared-policy multi-drone + agent_id conditioning)")
    parser.add_argument("--language", action="store_true", help="Sprint 11: enable language-conditioned tasks (text → task_embedding_id + params, SAR-only)")
    parser.add_argument("--export", type=str, default=None, help="Sprint 10: export trained policy to this ONNX path")
    parser.add_argument("--device", type=str, default="jetson", choices=["jetson", "rpi", "cpu"],
                        help="Sprint 10: edge target device")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Sprint 10: export precision")
    args = parser.parse_args()

    report = train(
        episodes=args.episodes,
        seed=args.seed,
        mesh=args.mesh,
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        save_path=args.save_path,
        log_interval=args.log_interval,
        goal=list(args.goal),
        max_steps=args.max_steps,
        curriculum=args.curriculum,
        dynamic_obstacles=args.dynamic_obstacles,
        sar_mission=args.sar_mission,
        hierarchical=args.hierarchical,
        multi_drone=args.multi_drone,
        language=args.language,
    )
    print(f"Final: {report.summary()}")
    if args.export:
        maybe_export_policy(args.save_path, export=args.export,
                            device=args.device, precision=args.precision)


if __name__ == "__main__":
    main()
