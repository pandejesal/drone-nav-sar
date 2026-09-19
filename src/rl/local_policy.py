#!/usr/bin/env python3
"""Local subgoal-conditioned PPO policy for DroneNav-SAR (Sprint 8).

Local policy: obs (21) + subgoal (3) + task_embedding (4) → action (4)
Shared backbone + task-specific heads.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple

from src.rl.policies import ActorCritic, OBS_DIM, ACT_DIM, DEVICE


SUBGOAL_DIM = 3
TASK_EMBED_DIM = 4
NUM_TASKS = 5  # nav, hover, detect, drop, return
LOCAL_OBS_DIM = OBS_DIM + SUBGOAL_DIM + TASK_EMBED_DIM  # 21 + 3 + 4 = 28


class TaskEmbedding(nn.Module):
    """Learnable task embeddings for multi-task heads."""

    def __init__(self, num_tasks: int = NUM_TASKS, embed_dim: int = TASK_EMBED_DIM):
        super().__init__()
        self.embeddings = nn.Embedding(num_tasks, embed_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.1)

    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        """task_id: (B,) or () → (B, embed_dim)"""
        return self.embeddings(task_id)


class LocalPolicy(nn.Module):
    """Subgoal-conditioned local policy with task-specific heads."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        subgoal_dim: int = SUBGOAL_DIM,
        task_embed_dim: int = TASK_EMBED_DIM,
        act_dim: int = ACT_DIM,
        hidden_dims: Tuple[int, ...] = (128, 128),
        num_tasks: int = NUM_TASKS,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.subgoal_dim = subgoal_dim
        self.task_embed_dim = task_embed_dim
        self.act_dim = act_dim
        self.num_tasks = num_tasks

        input_dim = obs_dim + subgoal_dim + task_embed_dim

        # Task embeddings
        self.task_embedding = TaskEmbedding(num_tasks, task_embed_dim)

        # Shared backbone
        self.backbone = self._make_mlp(input_dim, hidden_dims)

        # Task-specific heads (actor + critic per task)
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], act_dim) for _ in range(num_tasks)
        ])
        self.critic_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1) for _ in range(num_tasks)
        ])

        # Log std (shared across tasks)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5, dtype=torch.float32))

        # Initialize
        self._init_weights()
        self.to(DEVICE)

    def _make_mlp(self, in_dim: int, hidden_dims: Tuple[int, ...]) -> nn.Sequential:
        layers = []
        prev = in_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            prev = h
        return nn.Sequential(*layers)

    def _init_weights(self):
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        for head in self.critic_heads:
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        subgoal: torch.Tensor,
        task_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns (mean_raw, value) for given task."""
        # obs: (B, 21), subgoal: (B, 3), task_id: (B,) or ()
        task_embed = self.task_embedding(task_id)  # (B, 4)
        x = torch.cat([obs, subgoal, task_embed], dim=-1)  # (B, 28)
        features = self.backbone(x)  # (B, hidden)

        # Select task-specific heads
        task_idx = task_id.long()
        if task_idx.dim() == 0:
            task_idx = task_idx.unsqueeze(0)

        # Batch-wise head selection
        B = obs.shape[0]
        mean_raw = torch.zeros(B, self.act_dim, device=DEVICE)
        value = torch.zeros(B, device=DEVICE)

        for t in range(self.num_tasks):
            mask = (task_idx == t)
            if mask.any():
                mean_raw[mask] = self.actor_heads[t](features[mask])
                value[mask] = self.critic_heads[t](features[mask]).squeeze(-1)

        return mean_raw, value

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(u) + 1.0) / 2.0

    def _squashed_logprob(
        self, u: torch.Tensor, mean_raw: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        logp = (-0.5 * ((u - mean_raw) / std) ** 2 - torch.log(std)).sum(-1)
        logp -= torch.log(1.0 - torch.tanh(u) ** 2 + 1e-6).sum(-1)
        logp += float(self.act_dim) * np.log(2.0)
        return logp

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        subgoal: np.ndarray,
        task_id: int,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Get action for single obs + subgoal + task."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        subgoal_t = torch.as_tensor(subgoal, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        task_t = torch.tensor([task_id], dtype=torch.long, device=DEVICE)

        mean_raw, value = self.forward(obs_t, subgoal_t, task_t)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 2.0))

        if deterministic:
            u = mean_raw
        else:
            u = mean_raw + std * torch.randn_like(mean_raw)

        action = self._squash(u)
        logp = self._squashed_logprob(u, mean_raw, std)

        return (
            np.clip(action.cpu().numpy()[0], 0.0, 1.0).astype(np.float32),
            float(value.cpu().numpy()[0]),
            float(logp.cpu().numpy()[0]),
        )

    def get_action_batch(
        self,
        obs: torch.Tensor,
        subgoal: torch.Tensor,
        task_id: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch version for training."""
        mean_raw, value = self.forward(obs, subgoal, task_id)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 2.0))

        if deterministic:
            u = mean_raw
        else:
            u = mean_raw + std * torch.randn_like(mean_raw)

        action = self._squash(u)
        logp = self._squashed_logprob(u, mean_raw, std)

        return action, value, logp

    # -- Sprint 11: language interface (SAR-only) ---------------------------
    def encode_text_command(self, text: str) -> torch.Tensor:
        """Project a text command → 4-dim task_embedding (lazy import)."""
        from src.control.language_interface import text_to_task_embedding
        return text_to_task_embedding(text)

    def parse_text_command(self, text: str) -> Tuple[int, dict]:
        """Parse text → (task_id, params) (lazy import, no cycle)."""
        from src.control.language_interface import parse_language
        return parse_language(text)

    def forward_with_text(
        self,
        obs: torch.Tensor,
        subgoal: torch.Tensor,
        texts,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass driven by text command(s) instead of task_id.

        Single string → shared task for the batch; list[str] → per-row task.
        """
        from src.control.language_interface import parse_language
        if isinstance(texts, str):
            texts = [texts] * obs.shape[0]
        task_ids = [parse_language(t)[0] for t in texts]
        task_t = torch.as_tensor(task_ids, dtype=torch.long, device=obs.device)
        return self.forward(obs, subgoal, task_t)

    @torch.no_grad()
    def get_action_for_text(
        self,
        obs: np.ndarray,
        subgoal: np.ndarray,
        text: str,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, int, dict]:
        """Get action from a text command. Returns (action, value, logp, task_id, params)."""
        task_id, params = self.parse_text_command(text)
        action, value, logp = self.get_action(obs, subgoal, int(task_id),
                                              deterministic=deterministic)
        return action, value, logp, int(task_id), dict(params)

    def forward_with_embedding(
        self,
        obs: torch.Tensor,
        subgoal: torch.Tensor,
        task_embedding: torch.Tensor,
        task_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with an externally supplied task_embedding (e.g. from text).

        The embedding conditions the shared backbone input; the task-specific
        head is still selected by task_id for SAR-only compatibility.
        """
        x = torch.cat([obs, subgoal, task_embedding], dim=-1)
        features = self.backbone(x)
        task_idx = task_id.long()
        if task_idx.dim() == 0:
            task_idx = task_idx.unsqueeze(0)
        B = obs.shape[0]
        mean_raw = torch.zeros(B, self.act_dim, device=obs.device)
        value = torch.zeros(B, device=obs.device)
        for t in range(self.num_tasks):
            mask = (task_idx == t)
            if mask.any():
                mean_raw[mask] = self.actor_heads[t](features[mask])
                value[mask] = self.critic_heads[t](features[mask]).squeeze(-1)
        return mean_raw, value


class HierarchicalPolicy(nn.Module):
    """Full hierarchical policy: GlobalPlanner + LocalPolicy.

    ActorCritic-compatible: forward(obs) with a single 21-D obs tensor
    falls back to a zero subgoal + nav task so the standard PPO loss
    works unchanged.
    """

    def __init__(
        self,
        local_policy: Optional['LocalPolicy'] = None,
        global_planner=None,
    ):
        super().__init__()
        self.local_policy = local_policy or LocalPolicy()
        self.global_planner = global_planner
        self.current_subgoals: List[np.ndarray] = []
        self.current_subgoal_idx = 0
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the current local_policy module.

        Ensures _squashed_logprob, _squash, log_std, act_dim (and any
        future LocalPolicy attributes) are always resolved from the
        current local_policy even after reload/replacement.
        Called only when normal attribute lookup fails.
        """
        if name == "local_policy":
            return super().__getattr__(name)
        try:
            local_policy = super().__getattr__("local_policy")
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )
        return getattr(local_policy, name)

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

    def set_global_plan(self, waypoints: List[np.ndarray]):
        """Set new global plan (list of waypoints)."""
        self.current_subgoals = waypoints
        self.current_subgoal_idx = 0

    def get_current_subgoal(self) -> Optional[np.ndarray]:
        if self.current_subgoal_idx < len(self.current_subgoals):
            return self.current_subgoals[self.current_subgoal_idx]
        return None

    def advance_subgoal(self):
        self.current_subgoal_idx += 1

    def is_plan_complete(self) -> bool:
        return self.current_subgoal_idx >= len(self.current_subgoals)

    def forward(self, obs, subgoal=None, task_id=None):
        """ActorCritic-compatible forward; defaults: zero subgoal, nav task."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        batch = obs_t.shape[0] if obs_t.dim() > 1 else 1
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
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
        return self.local_policy.forward(obs_t, subgoal_t, task_t)

    def get_action(self, obs, subgoal=None, task_id=0, deterministic=False):
        if subgoal is None:
            sg = self.get_current_subgoal()
            subgoal = sg if sg is not None else np.zeros(3, dtype=np.float32)
        arr = np.asarray(obs, dtype=np.float32)
        sg_arr = np.asarray(subgoal, dtype=np.float32)
        if arr.ndim == 1:
            tid = int(np.asarray(task_id).flat[0]) if np.ndim(task_id) != 0 else int(task_id)
            return self.local_policy.get_action(arr, sg_arr.reshape(3), tid, deterministic=deterministic)
        batch = arr.shape[0]
        sg_batch = np.tile(sg_arr.reshape(1, 3), (batch, 1)) if sg_arr.ndim == 1 else sg_arr.reshape(batch, 3)
        task_arr = np.asarray(task_id)
        if task_arr.ndim == 0:
            task_batch = torch.full((batch,), int(task_arr), dtype=torch.long, device=DEVICE)
        else:
            task_batch = torch.as_tensor(task_arr.reshape(batch), dtype=torch.long, device=DEVICE)
        obs_t = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        sg_t = torch.as_tensor(sg_batch, dtype=torch.float32, device=DEVICE)
        action_t, value_t, logp_t = self.local_policy.get_action_batch(
            obs_t, sg_t, task_batch, deterministic=deterministic)
        return (action_t.cpu().numpy().astype(np.float32),
                value_t.cpu().numpy().astype(np.float32),
                logp_t.cpu().numpy().astype(np.float32))

    def get_action_batch(self, obs, subgoal=None, task_id=0, deterministic=False):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=DEVICE)
        batch = obs_t.shape[0]
        if subgoal is None:
            sg_t = torch.zeros((batch, SUBGOAL_DIM), dtype=torch.float32, device=DEVICE)
        else:
            sg_t = torch.as_tensor(np.asarray(subgoal), dtype=torch.float32, device=DEVICE)
            if sg_t.dim() == 1:
                sg_t = sg_t.unsqueeze(0).expand(batch, -1)
        task_arr = np.asarray(task_id)
        if task_arr.ndim == 0:
            task_t = torch.full((batch,), int(task_arr), dtype=torch.long, device=DEVICE)
        else:
            task_t = torch.as_tensor(task_arr.reshape(batch), dtype=torch.long, device=DEVICE)
        return self.local_policy.get_action_batch(obs_t, sg_t, task_t, deterministic=deterministic)


def create_local_policy() -> 'LocalPolicy':
    """Factory for default local policy."""
    return LocalPolicy()


def create_hierarchical_policy(global_planner=None) -> 'HierarchicalPolicy':
    """Factory for hierarchical policy."""
    return HierarchicalPolicy(create_local_policy(), global_planner)