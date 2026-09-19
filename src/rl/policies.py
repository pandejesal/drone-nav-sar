#!/usr/bin/env python3
"""MLP actor-critic policy for DroneNav-SAR (Sprint 4 small slice).

SAR-only: navigate_to / hover / drop_payload / return_home.
No targeting, no kinetic, no missile guidance.

Contract:
- obs: (..., 21) = state(17) + goal_rel(3) + time(1), float32.
- action: (..., 4) normalized thrusts in [0, 1] (never raw RPM).
- torch CPU-only: no CUDA hard dependency so CI stays green.
- tanh-squashed Gaussian: u ~ N(mean, std), a = tanh(u),
  action = (a + 1) / 2 in [0, 1].
"""

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 21
ACT_DIM = 4

DEVICE = torch.device("cpu")
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden_dims, out_dim: int) -> nn.Sequential:
    """Orthogonally-initialized MLP with tanh activations."""
    layers: list = []
    prev = in_dim
    for h in hidden_dims:
        lin = nn.Linear(prev, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        prev = h
    head = nn.Linear(prev, out_dim)
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Shared-shape MLP actor (tanh-Gaussian) + critic (value). CPU-only."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden_dims=(64, 64),
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor_trunk = _mlp(obs_dim, hidden_dims, act_dim)
        self.log_std = nn.Parameter(
            torch.full((act_dim,), float(log_std_init), dtype=torch.float32)
        )
        self.critic_trunk = _mlp(obs_dim, hidden_dims, 1)
        self.to(DEVICE)

    def forward(self, obs: torch.Tensor):
        """Return (mean_raw, value) for a batch of obs on CPU."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        mean_raw = self.actor_trunk(obs)
        value = self.critic_trunk(obs).squeeze(-1)
        return mean_raw, value

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(u) + 1.0) / 2.0

    def _squashed_logprob(
        self, u: torch.Tensor, mean_raw: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """log p(action in [0,1]) with tanh + affine corrections."""
        logp = (-0.5 * ((u - mean_raw) / std) ** 2 - torch.log(std)).sum(-1)
        logp -= torch.log(1.0 - torch.tanh(u) ** 2 + 1e-6).sum(-1)
        logp += float(self.act_dim) * np.log(2.0)  # a01 = a/2 + 1/2 scale
        return logp

    @torch.no_grad()
    def get_action(self, obs, deterministic: bool = False):
        """Sample (or take the mean) action in [0, 1].

        Returns (action_np, value_np, logprob_np); single obs (21,)
        yields (4,), batch (B, 21) yields (B, 4).
        """
        single = False
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            single = True
            arr = arr[None, :]
        t = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        mean_raw, value = self.forward(t)
        std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        if deterministic:
            u = mean_raw
        else:
            u = mean_raw + std * torch.randn_like(mean_raw)
        action = self._squash(u)
        logp = self._squashed_logprob(u, mean_raw, std)
        action_np = np.clip(action.cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if single:
            return action_np[0], float(value.cpu().numpy()[0]), float(logp.cpu().numpy()[0])
        return action_np, value.cpu().numpy(), logp.cpu().numpy()
