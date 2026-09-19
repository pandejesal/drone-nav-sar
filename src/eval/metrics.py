#!/usr/bin/env python3
"""Evaluation metrics for DroneNav-SAR PPO trainer (Sprint 4 v3).

SAR-only: navigate_to / hover / drop_payload / return_home.
No weapons, no targeting.

Metrics:
- success_rate: fraction of episodes that reached goal within tolerance.
- spl: Success-weighted Path Length (Anderson/Habitat):
  SPL = success * optimal_len / actual_len  (per-episode, then averaged).
- energy_proxy: sum(thrust^2) * dt (proxy for energy cost).
- EvalReport: frozen dataclass aggregating all metrics + metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import time

import numpy as np


def success_rate(results: Sequence[Dict]) -> float:
    """Fraction of episodes where info['goal_reached'] is True.

    Parameters
    ----------
    results : list of dicts
        Each dict is an episode info at terminal step containing
        'goal_reached' (bool), 'steps' (int), and optionally
        'optimal_steps' (int).

    Returns
    -------
    float in [0.0, 1.0]
    """
    if len(results) == 0:
        return 0.0
    successes = sum(1 for r in results if r.get("goal_reached", False))
    return successes / len(results)


def spl(results: Sequence[Dict]) -> float:
    """Success-weighted Path Length (Anderson/Habitat).

    SPL = (1/N) * sum_i(success_i * optimal_i / actual_i)

    optimal_i: shortest-path steps (default: same as actual if not given).
    actual_i:  steps taken in the episode.

    Parameters
    ----------
    results : list of dicts
        Each dict must contain 'goal_reached' (bool) and 'steps' (int).
        Optionally 'optimal_steps' (int); defaults to 'steps' if absent
        (making SPL == success_rate for non-optimal episodes).

    Returns
    -------
    float in [0.0, 1.0]
    """
    if len(results) == 0:
        return 0.0
    total = 0.0
    for r in results:
        success = 1.0 if r.get("goal_reached", False) else 0.0
        actual = max(int(r.get("steps", 1)), 1)
        optimal = int(r.get("optimal_steps", actual))
        total += success * min(optimal / actual, 1.0)
    return total / len(results)


def energy_proxy(
    actions: np.ndarray, dt: float = 0.02
) -> float:
    """Proxy energy cost: sum(thrust^2) * dt.

    Parameters
    ----------
    actions : np.ndarray of shape (T, 4) or (N, T, 4)
        Normalized thrusts in [0, 1].
    dt : float
        Timestep (default 0.02 s for 50 Hz).

    Returns
    -------
    float : total energy proxy (lower is better).
    """
    arr = np.asarray(actions, dtype=np.float32)
    return float(np.sum(arr ** 2) * dt)


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation report for one training checkpoint.

    Fields mirror the architecture §5 interface:
    {success_rate, SPL, energy_proxy, time_s, spec_hash, seed}.
    """
    success_rate: float
    spl: float
    energy_proxy: float
    time_s: float
    spec_hash: str
    seed: int
    n_episodes: int
    details: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"EvalReport(n={self.n_episodes}, sr={self.success_rate:.3f}, "
            f"spl={self.spl:.3f}, energy={self.energy_proxy:.1f}, "
            f"time={self.time_s:.1f}s, seed={self.seed})"
        )


def build_report(
    episode_results: Sequence[Dict],
    all_actions: np.ndarray,
    dt: float = 0.02,
    spec_hash: str = "",
    seed: int = 0,
    elapsed_s: float = 0.0,
) -> EvalReport:
    """Build an EvalReport from raw episode results and actions.

    Parameters
    ----------
    episode_results : list of dicts
        One dict per episode (from vec_env terminal info).
    all_actions : np.ndarray (T, 4) or (N, T, 4)
        All actions taken across episodes.
    dt : float
        Control timestep.
    spec_hash : str
        Hash of the drone spec for traceability.
    seed : int
        Random seed used for evaluation.
    elapsed_s : float
        Wall-clock time for evaluation.

    Returns
    -------
    EvalReport
    """
    sr = success_rate(episode_results)
    s = spl(episode_results)
    e = energy_proxy(all_actions, dt=dt)
    return EvalReport(
        success_rate=sr,
        spl=s,
        energy_proxy=e,
        time_s=elapsed_s,
        spec_hash=spec_hash,
        seed=seed,
        n_episodes=len(episode_results),
    )
