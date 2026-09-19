#!/usr/bin/env python3
"""Hierarchical PPO wrapper for DroneNav-SAR (Sprint 8).

High level: GlobalPlanner (room-graph A*) emits subgoal waypoints
(room centers, doorways). Low level: LocalPolicy (subgoal-conditioned
PPO with task embeddings) executes them.

SAR-only: navigate_to / hover / drop_payload / return_home.

This module is the canonical home of HierarchicalPolicy. The copy in
src.rl.local_policy is kept as a thin backwards-compatible alias.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rl.local_policy import LocalPolicy, SUBGOAL_DIM, NUM_TASKS  # noqa: E402
from src.rl.policies import ACT_DIM, DEVICE, OBS_DIM  # noqa: E402

SUBGOAL_TOLERANCE_M = 0.6


class HierarchicalPolicy(nn.Module):
    """Global planner + local subgoal-conditioned policy.

    Training-compatible: ``forward(obs)`` with a single 21-D obs tensor
    falls back to a zero subgoal + nav task so the standard PPO loss
    (ActorCritic signature) works unchanged. ``get_action`` accepts both
    the hierarchical ``(obs, subgoal, task_id)`` form and the plain
    ``(obs)`` form, for single obs or (B, 21) batches.
    """

    def __init__(
        self,
        local_policy: Optional[LocalPolicy] = None,
        global_planner=None,
    ):
        super().__init__()
        self.local_policy = local_policy or LocalPolicy()
        self.global_planner = global_planner
        self.current_subgoals: List[np.ndarray] = []
        self.current_subgoal_idx = 0
        # Passthrough attributes so PPO helpers treat this like ActorCritic.
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the current local_policy module.

        Ensures _squashed_logprob, _squash, log_std, act_dim (and any
        future LocalPolicy attributes) are always resolved from the
        current local_policy even after reload/replacement.
        Called only when normal attribute lookup fails. Note: under
        nn.Module, submodule access itself routes through __getattr__,
        so the local_policy name must resolve via super().__getattr__
        (which checks _modules) rather than raising.
        """
        try:
            local_policy = super().__getattr__("local_policy")
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )
        if name == "local_policy":
            return local_policy
        return getattr(local_policy, name)

    # -- nn.Module passthroughs ----------------------------------------
    @property
    def log_std(self):  # type: ignore[override]
        return self.local_policy.log_std

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        """Delegate squash to local_policy (explicit for PPO helpers)."""
        return self.local_policy._squash(u)

    def _squashed_logprob(
        self, u: torch.Tensor, mean_raw: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Delegate squashed logprob to local_policy (explicit for ppo_loss)."""
        return self.local_policy._squashed_logprob(u, mean_raw, std)

    # -- global-plan bookkeeping ---------------------------------------
    def set_global_plan(self, waypoints: List[np.ndarray]) -> None:
        self.current_subgoals = [np.asarray(w, dtype=np.float32).reshape(3) for w in waypoints]
        self.current_subgoal_idx = 0

    def plan_from_positions(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Ask the global planner for waypoints and store them."""
        if self.global_planner is None:
            waypoints = [np.asarray(goal, dtype=np.float32).reshape(3)]
        else:
            waypoints = self.global_planner.plan(
                np.asarray(start, dtype=np.float32),
                np.asarray(goal, dtype=np.float32),
            )
        self.set_global_plan(waypoints)
        return self.current_subgoals

    def get_current_subgoal(self) -> Optional[np.ndarray]:
        if self.current_subgoal_idx < len(self.current_subgoals):
            return self.current_subgoals[self.current_subgoal_idx]
        return None

    def advance_subgoal(self) -> None:
        self.current_subgoal_idx += 1

    def is_plan_complete(self) -> bool:
        return self.current_subgoal_idx >= len(self.current_subgoals)

    def reset_plan(self) -> None:
        self.current_subgoals = []
        self.current_subgoal_idx = 0

    def update_subgoal_progress(self, drone_pos: np.ndarray, tolerance: float = SUBGOAL_TOLERANCE_M) -> bool:
        """Advance when within `tolerance` of the active subgoal. Returns True if advanced."""
        sg = self.get_current_subgoal()
        if sg is None:
            return False
        if float(np.linalg.norm(np.asarray(drone_pos, dtype=np.float32)[:3] - sg[:3])) <= tolerance:
            self.advance_subgoal()
            return True
        return False

    # -- torch-compatible forward --------------------------------------
    def forward(self, obs: torch.Tensor, subgoal: Optional[torch.Tensor] = None,
                task_id: Optional[torch.Tensor] = None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        single_extra = False
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            single_extra = True
        batch = obs_t.shape[0]
        if subgoal is None:
            subgoal_t = torch.zeros((batch, SUBGOAL_DIM), dtype=torch.float32, device=DEVICE)
        else:
            subgoal_t = torch.as_tensor(subgoal, dtype=torch.float32, device=DEVICE)
            if subgoal_t.dim() == 1:
                subgoal_t = subgoal_t.unsqueeze(0).expand(batch, -1)
        if task_id is None:
            task_t = torch.zeros((batch,), dtype=torch.long, device=DEVICE)
        else:
            task_t = torch.as_tensor(task_id, dtype=torch.long, device=DEVICE)
            if task_t.dim() == 0:
                task_t = task_t.unsqueeze(0).expand(batch)
            elif task_t.numel() == 1 and batch > 1:
                task_t = task_t.expand(batch)
        mean_raw, value = self.local_policy.forward(obs_t, subgoal_t, task_t)
        if single_extra:
            return mean_raw, value
        return mean_raw, value

    # -- action sampling (both signatures) ------------------------------
    @torch.no_grad()
    def get_action(self, obs, subgoal=None, task_id=0, deterministic: bool = False):
        arr = np.asarray(obs, dtype=np.float32)
        if subgoal is None:
            sg = self.get_current_subgoal()
            subgoal = sg if sg is not None else np.zeros(3, dtype=np.float32)
        sg_arr = np.asarray(subgoal, dtype=np.float32)
        if arr.ndim == 1:
            return self.local_policy.get_action(arr, sg_arr.reshape(3), int(task_id) if np.ndim(task_id) == 0 else int(np.asarray(task_id).flat[0]),
                                                deterministic=deterministic)
        # Batch path: vectorise through get_action_batch.
        batch = arr.shape[0]
        if sg_arr.ndim == 1:
            sg_batch = np.tile(sg_arr.reshape(1, 3), (batch, 1))
        else:
            sg_batch = sg_arr.reshape(batch, 3)
        task_arr = np.asarray(task_id)
        if task_arr.ndim == 0:
            task_batch = torch.full((batch,), int(task_arr), dtype=torch.long, device=DEVICE)
        else:
            task_batch = torch.as_tensor(task_arr.reshape(batch), dtype=torch.long, device=DEVICE)
        obs_t = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        sg_t = torch.as_tensor(sg_batch, dtype=torch.float32, device=DEVICE)
        action_t, value_t, logp_t = self.local_policy.get_action_batch(obs_t, sg_t, task_batch, deterministic=deterministic)
        return (action_t.cpu().numpy().astype(np.float32), value_t.cpu().numpy().astype(np.float32),
                logp_t.cpu().numpy().astype(np.float32))

    def get_action_batch(self, obs, subgoal, task_id, deterministic: bool = False):
        return self.local_policy.get_action_batch(obs, subgoal, task_id, deterministic=deterministic)


# Alias expected by the sprint DONE line wording.
HierarchicalPPO = HierarchicalPolicy


def create_hierarchical_policy(global_planner=None) -> HierarchicalPolicy:
    """Factory: fresh LocalPolicy + given (or no) global planner."""
    return HierarchicalPolicy(LocalPolicy(), global_planner)
