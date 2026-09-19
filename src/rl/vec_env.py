#!/usr/bin/env python3
"""Vectorized mock envs for DroneNav-SAR PPO smoke slice (Sprint 4 v2).

SAR-only: navigate_to / hover / drop_payload / return_home.

- N mock envs built via src.sim.make_env("isaac", mesh, use_mock=True),
  seeded master_seed + i.
- Accepts a SimConfig OR a bare mesh-path string (auto-handled).
- Gymnasium API: reset -> (obs, infos); step -> 5-tuple, batched.
- Never breaks the make_env mock fallback: `use_mock` is consumed here
  (SimConfig has no such field) and each env is forced into mock mode so
  headless CI never touches a real simulator. info["mock_backend"] is set
  (setdefault: never overrides a real backend flag).
"""

import dataclasses
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.sim.base_env import BaseDroneEnv, SimConfig, make_env
from src.sim.domain_randomization import get_dynamic_config

DEFAULT_MESH = "mock.glb"  # bare string; isaac mock path ignores the file
_BACKEND = "isaac"

MeshOrConfig = Union[str, SimConfig]


def coerce_config(mesh_or_config: MeshOrConfig, **kwargs) -> SimConfig:
    """Build a SimConfig from a SimConfig (as-is) or a mesh-path string."""
    kwargs.pop("use_mock", None)  # consumed here; SimConfig has no such field
    dynamic_config = kwargs.pop("dynamic_config", None)
    if isinstance(mesh_or_config, SimConfig):
        return mesh_or_config
    if isinstance(mesh_or_config, dict):
        merged = dict(mesh_or_config)
        merged.update(kwargs)
        return SimConfig(**merged)
    return SimConfig(mesh_path=str(mesh_or_config), **kwargs)  # type: ignore[arg-type]


def _build_single(config: SimConfig, dynamic_config: Optional[Dict] = None) -> BaseDroneEnv:
    """Construct one mock env through make_env (never a real simulator)."""
    params = dataclasses.asdict(config)
    mesh_path = params.pop("mesh_path")
    headless = params.pop("headless", True)
    env = make_env(_BACKEND, mesh_path, headless=headless, **params)
    # Force headless mock fallback: IsaacSimEnv never sets _mock_mode when
    # its real backend is absent (its _initialize_sim is never auto-called),
    # so set the flag the mock guards already check.
    if not getattr(env, "_mock_mode", False):
        env._mock_mode = True
    if not hasattr(env, "use_mock"):
        env.use_mock = True
    # Sprint 5: pass dynamic_config to backend if it supports it
    if dynamic_config is not None and hasattr(env, "_backend"):
        env._backend.dynamic_config = dynamic_config
    return env


class DroneVecEnv:
    """Batched mock envs with reset-on-done and seeded determinism."""

    def __init__(
        self,
        n_envs: int = 8,
        mesh_or_config: MeshOrConfig = DEFAULT_MESH,
        master_seed: int = 0,
        dynamic_config: Optional[Dict] = None,
        **kwargs,
    ):
        kwargs.pop("use_mock", None)
        self.n_envs = int(n_envs)
        self.master_seed = int(master_seed)
        self.dynamic_config = dynamic_config
        base = coerce_config(mesh_or_config, **kwargs)
        self.envs: List[BaseDroneEnv] = [
            _build_single(base, dynamic_config=self.dynamic_config)
            for _ in range(self.n_envs)
        ]
        self._auto_seed = self.master_seed + self.n_envs
        self.episodes_done = 0

        ref = self.envs[0]
        self.observation_space = ref.observation_space
        self.action_space = ref.action_space
        self.obs_dim = int(np.asarray(ref.observation_space.shape).tolist()[0])
        self.act_dim = int(np.asarray(ref.action_space.shape).tolist()[0])

    # -- gym API ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        base = self.master_seed if seed is None else int(seed)
        obs, infos = [], []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=base + i)
            info.setdefault("mock_backend", True)
            obs.append(np.asarray(o, dtype=np.float32))
            infos.append(info)
        return np.stack(obs), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        arr = np.asarray(actions, dtype=np.float32)
        assert arr.shape == (self.n_envs, self.act_dim), (
            f"expected {(self.n_envs, self.act_dim)}, got {arr.shape}"
        )
        obs, rew, term, trunc, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, t, tr, info = env.step(np.clip(arr[i], 0.0, 1.0))
            info.setdefault("mock_backend", True)
            done = bool(t) or bool(tr)
            if done:  # reset-on-done: fresh obs, terminal stashed in info.
                # Preserve terminal outcome: reset() replaces info, which
                # would otherwise lose goal_reached/steps/dist (sr stuck 0).
                terminal = {
                    "goal_reached": bool(info.get("goal_reached", False)),
                    "crashed": bool(info.get("crashed", False)),
                    "out_of_bounds": bool(info.get("out_of_bounds", False)),
                    "terminal_steps": int(info.get("step", 0)),
                    "terminal_dist": float(info.get("dist_to_goal", float("inf"))),
                    "terminal_position": info.get("position"),
                }
                info["final_obs"] = np.asarray(o, dtype=np.float32)
                self.episodes_done += 1
                self._auto_seed += 1
                o, fresh = env.reset(seed=self._auto_seed)
                fresh.setdefault("mock_backend", True)
                fresh["episode_done"] = True
                fresh["final_obs"] = info["final_obs"]
                fresh.update(terminal)
                info = fresh
            obs.append(np.asarray(o, dtype=np.float32))
            rew.append(float(r))
            term.append(bool(t))
            trunc.append(bool(tr))
            infos.append(info)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(rew, dtype=np.float32),
            np.asarray(term, dtype=bool),
            np.asarray(trunc, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
