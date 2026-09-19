#!/usr/bin/env python3
"""Sim2Real evaluation harness for DroneNav-SAR (Sprint 6).

Compares policy performance in mock sim vs real hardware (or high-fidelity sim).
Generates comparison report with SR, SPL, energy, trajectory deviation.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rl.policies import ActorCritic, OBS_DIM, ACT_DIM, DEVICE
from src.rl.vec_env import DroneVecEnv
from src.eval.metrics import EvalReport, build_report, success_rate, spl, energy_proxy


class Sim2RealEvaluator:
    """Evaluates a policy in mock sim and optionally real hardware."""

    def __init__(
        self,
        policy_path: str,
        seed: int = 0,
        n_envs: int = 8,
        max_steps: int = 500,
        goal: Optional[List[float]] = None,
    ):
        self.policy_path = policy_path
        self.seed = seed
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.goal = np.array(goal if goal is not None else [2.0, 1.5, 1.5], dtype=np.float32)

        # Load policy
        self.policy = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        state_dict = torch.load(policy_path, map_location=DEVICE)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.policy.to(DEVICE)

    def evaluate_mock(
        self,
        episodes: int = 200,
        curriculum: bool = False,
        dynamic_obstacles: bool = False,
    ) -> EvalReport:
        """Evaluate in mock backend (current sim)."""
        from src.sim.domain_randomization import get_dynamic_config

        dynamic_config = get_dynamic_config(3) if dynamic_obstacles else None

        env = DroneVecEnv(
            n_envs=self.n_envs,
            mesh_or_config="mock.glb",
            master_seed=self.seed,
            goal_position=self.goal,
            max_episode_steps=self.max_steps,
            dynamic_config=dynamic_config,
        )

        # Run evaluation episodes
        episode_results: List[Dict] = []
        all_actions: List[np.ndarray] = []
        start_dists = None

        obs, reset_infos = env.reset(seed=self.seed)
        start_dists = np.array(
            [float(info.get("dist_to_goal", 0.0)) for info in reset_infos],
            dtype=np.float32,
        )

        episodes_collected = 0
        t_start = time.time()

        while episodes_collected < episodes:
            with torch.no_grad():
                action, value, logprob = self.policy.get_action(obs)
            obs_next, rewards, dones, truncs, infos = env.step(action)

            all_actions.append(action.copy())
            obs = obs_next

            for i, info in enumerate(infos):
                if info.get("episode_done", False) or dones[i] or truncs[i]:
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
                    if "dist_to_goal" in info:
                        start_dists[i] = float(info["dist_to_goal"])

        elapsed = time.time() - t_start
        actions_arr = np.stack(all_actions) if all_actions else np.zeros((1, self.n_envs, ACT_DIM))
        spec_hash = "mock_eval"

        report = build_report(
            episode_results=episode_results,
            all_actions=actions_arr,
            dt=0.02,
            spec_hash=spec_hash,
            seed=self.seed,
            elapsed_s=elapsed,
        )
        env.close()
        return report

    def evaluate_real(
        self,
        episodes: int = 20,
        mavlink_uri: str = "udp://:14540",
    ) -> EvalReport:
        """Evaluate on real hardware via MAVLink (placeholder for Sprint 6 hardware).
        Requires MAVSDK connection to Crazyflie/PX4.
        """
        # This is a stub - real implementation connects to MAVSDK
        # For now, returns a mock report indicating not available
        return EvalReport(
            success_rate=0.0,
            spl=0.0,
            energy_proxy=0.0,
            time_s=0.0,
            spec_hash="real_hardware_unavailable",
            seed=self.seed,
            n_episodes=0,
            details={"error": "Real hardware evaluation not implemented - requires MAVSDK connection"},
        )

    def compare(self, mock_report: EvalReport, real_report: EvalReport) -> Dict:
        """Generate comparison report between mock and real."""
        return {
            "mock": {
                "success_rate": mock_report.success_rate,
                "spl": mock_report.spl,
                "energy_proxy": mock_report.energy_proxy,
                "time_s": mock_report.time_s,
                "n_episodes": mock_report.n_episodes,
            },
            "real": {
                "success_rate": real_report.success_rate,
                "spl": real_report.spl,
                "energy_proxy": real_report.energy_proxy,
                "time_s": real_report.time_s,
                "n_episodes": real_report.n_episodes,
            },
            "gap": {
                "sr_delta": real_report.success_rate - mock_report.success_rate,
                "spl_delta": real_report.spl - mock_report.spl,
                "energy_delta": real_report.energy_proxy - mock_report.energy_proxy,
            },
            "timestamp": time.time(),
        }


def main():
    parser = argparse.ArgumentParser(description="Sim2Real evaluation for DroneNav-SAR")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy.pt")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes for mock eval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel envs")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps/episode")
    parser.add_argument("--goal", type=float, nargs=3, default=[2.0, 1.5, 1.5], help="Goal x y z")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum")
    parser.add_argument("--dynamic-obstacles", action="store_true", help="Enable dynamic obstacles")
    parser.add_argument("--real", action="store_true", help="Also evaluate on real hardware (stub)")
    parser.add_argument("--output", type=str, default="sim2real_report.json", help="Output report path")
    args = parser.parse_args()

    evaluator = Sim2RealEvaluator(
        policy_path=args.policy,
        seed=args.seed,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        goal=list(args.goal),
    )

    print(f"Evaluating mock sim ({args.episodes} episodes)...")
    mock_report = evaluator.evaluate_mock(
        episodes=args.episodes,
        curriculum=args.curriculum,
        dynamic_obstacles=args.dynamic_obstacles,
    )
    print(f"Mock: {mock_report.summary()}")

    real_report = None
    if args.real:
        print("Evaluating real hardware...")
        real_report = evaluator.evaluate_real(episodes=min(20, args.episodes))
        print(f"Real: {real_report.summary()}")

    comparison = evaluator.compare(mock_report, real_report or EvalReport(
        success_rate=0.0, spl=0.0, energy_proxy=0.0, time_s=0.0,
        spec_hash="none", seed=args.seed, n_episodes=0
    ))

    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Report saved to {args.output}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()