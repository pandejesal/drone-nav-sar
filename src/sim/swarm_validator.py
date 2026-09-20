#!/usr/bin/env python3
"""Sprint 33: Swarm Validator — Emergent behavior detection, safety monitoring.

Validates swarm behaviors (flocking, flanking, encircling, swarming, self-healing),
safety invariants, and mission success for SAR swarms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.sim.large_scale_swarm import (
    detect_emergent_behaviors,
    velocity_alignment,
    lateral_spread,
    encirclement_ratio,
    swarm_density,
    FLOCK_ALIGN_THRESH,
    FLANK_SPREAD_RATIO,
    ENCIRCLE_RATIO_THRESH,
    SWARM_DENSITY_THRESH,
    SELF_HEAL_SLA_S,
    FORMATION_SPACING,
)
from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig


@dataclass
class SwarmMetrics:
    """Aggregated swarm behavior metrics."""
    velocity_alignment: float
    lateral_spread: float
    encirclement_ratio: float
    density: float
    formation_error_m: float
    formation_hold_ratio: float
    mission_success_rate: float


@dataclass
class EmergentBehaviorReport:
    """Report on detected emergent behaviors."""
    flocking: bool
    flanking: bool
    encircling: bool
    swarming: bool
    self_healing: bool
    confidence: Dict[str, float]  # 0-1 confidence per behavior
    raw_metrics: Dict[str, float]


@dataclass
class SafetyReport:
    """Safety validation report."""
    violations: List[str]
    min_obstacle_distance: float
    min_geofence_margin: float
    min_altitude_margin: float
    max_speed_violation: float
    cbf_margin: float
    safe: bool


@dataclass
class SwarmValidationResult:
    """Complete swarm validation result."""
    scenario_id: str
    n_drones: int
    duration_s: float
    swarm_metrics: SwarmMetrics
    emergent_behaviors: EmergentBehaviorReport
    safety: SafetyReport
    formal_verification: Dict[str, float]
    passed: bool


class SwarmValidator:
    """Validates swarm behaviors, safety, and formal properties."""

    def __init__(
        self,
        n_drones: int,
        area_m: float = 60.0,
        geofence_xy: float = 50.0,
        min_altitude: float = 0.3,
        max_altitude: float = 120.0,
        max_speed: float = 15.0,
        obstacle_margin: float = 1.0,
        formation_spacing: float = FORMATION_SPACING,
        enable_formal: bool = True,
    ):
        self.n_drones = n_drones
        self.area_m = area_m
        self.formation_spacing = formation_spacing

        # Formal validator
        self.formal_validator: Optional[AutonomyValidator] = None
        if enable_formal:
            self.formal_validator = AutonomyValidator(ValidatorConfig(
                geofence_xy=geofence_xy,
                min_altitude=min_altitude,
                max_altitude=max_altitude,
                max_speed=max_speed,
                obstacle_margin=obstacle_margin,
            ))

        # History for trend analysis
        self.history: List[Dict[str, Any]] = []

    def validate_step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        formation_goals: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        obstacles: Optional[List[np.ndarray]] = None,
        failed_drones: Optional[set] = None,
        heal_time_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Validate a single simulation step."""
        # Emergent behaviors
        emergent = detect_emergent_behaviors(
            positions, velocities,
            formation_width=self.formation_spacing * np.sqrt(self.n_drones),
            target=target,
        )
        emergent["self_healing"] = heal_time_s < SELF_HEAL_SLA_S

        # Swarm metrics
        align = velocity_alignment(velocities)
        spread = lateral_spread(positions)
        enc = encirclement_ratio(positions, target) if target is not None else 0.0
        dens = swarm_density(positions)

        # Formation metrics
        formation_error = 0.0
        formation_hold = 0.0
        if formation_goals is not None:
            err = np.linalg.norm(positions - formation_goals, axis=1)
            alive = np.ones(len(positions), dtype=bool)
            if failed_drones:
                alive[list(failed_drones)] = False
            if alive.any():
                formation_error = float(err[alive].mean())
                formation_hold = float((err[alive] < 3.0).mean())

        # Safety check
        safety = self._check_safety(positions, velocities, obstacles)

        # Formal verification (periodic)
        formal = {}
        if self.formal_validator and obstacles:
            self.formal_validator.obstacles = [
                np.asarray(o, dtype=np.float64).reshape(3) for o in obstacles
            ]
            states = np.hstack([positions, velocities])
            violations = self.formal_validator.verify_safety_invariants(states)
            cbf_result = self.formal_validator.verify_cbf_sos(positions)
            formal = {
                "safety_violations": len(violations),
                "cbf_holds": cbf_result.get("holds", False),
                "cbf_sos_feasible": cbf_result.get("sos_feasible", False),
            }

        result = {
            "emergent": emergent,
            "swarm_metrics": {
                "velocity_alignment": align,
                "lateral_spread": spread,
                "encirclement_ratio": enc,
                "density": dens,
                "formation_error_m": formation_error,
                "formation_hold_ratio": formation_hold,
            },
            "safety": safety,
            "formal": formal,
        }

        self.history.append(result)
        return result

    def _check_safety(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        obstacles: Optional[List[np.ndarray]] = None,
    ) -> SafetyReport:
        """Check safety invariants."""
        violations = []
        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)

        # Geofence
        geofence_margin = min(
            self.area_m / 2 - np.max(np.abs(pos[:, 0])),
            self.area_m / 2 - np.max(np.abs(pos[:, 1])),
        )
        if geofence_margin < 0:
            violations.append("geofence")

        # Altitude
        min_alt_margin = float(np.min(pos[:, 2]) - 0.3)
        max_alt_margin = float(120.0 - np.max(pos[:, 2]))
        if min_alt_margin < 0:
            violations.append("min_altitude")
        if max_alt_margin < 0:
            violations.append("max_altitude")

        # Speed
        speeds = np.linalg.norm(vel, axis=1)
        max_speed_viol = float(np.max(speeds) - 15.0)
        if max_speed_viol > 0:
            violations.append("max_speed")

        # Obstacles
        min_obs_dist = float("inf")
        if obstacles:
            for obs in obstacles:
                obs = np.asarray(obs, dtype=np.float64).reshape(3)
                d = np.linalg.norm(pos - obs, axis=1)
                min_obs_dist = min(min_obs_dist, float(np.min(d)))
                if np.any(d < 2.0):  # obstacle_radius + margin
                    violations.append("obstacle_cbf")

        # CBF margin
        cbf_margin = float("inf")
        if self.formal_validator:
            cbf_ok, cbf_margin = self.formal_validator.verify_cbf(positions)

        return SafetyReport(
            violations=violations,
            min_obstacle_distance=min_obs_dist,
            min_geofence_margin=geofence_margin,
            min_altitude_margin=min_alt_margin,
            max_speed_violation=max_speed_viol,
            cbf_margin=cbf_margin,
            safe=len(violations) == 0,
        )

    def validate_episode(
        self,
        positions_history: List[np.ndarray],
        velocities_history: List[np.ndarray],
        formation_goals_history: Optional[List[np.ndarray]] = None,
        targets_history: Optional[List[np.ndarray]] = None,
        obstacles: Optional[List[np.ndarray]] = None,
        failed_history: Optional[List[set]] = None,
        heal_times: Optional[List[float]] = None,
    ) -> SwarmValidationResult:
        """Validate a complete episode."""
        if not positions_history:
            raise ValueError("Empty episode")

        n_steps = len(positions_history)
        scenario_id = f"val_{np.random.randint(1000000):06d}"

        # Aggregate metrics
        all_emergent = []
        all_swarm_metrics = []
        all_safety = []
        all_formal = []

        for i in range(n_steps):
            pos = positions_history[i]
            vel = velocities_history[i]
            goals = formation_goals_history[i] if formation_goals_history else None
            target = targets_history[i] if targets_history else None
            failed = failed_history[i] if failed_history else set()
            heal_t = heal_times[i] if heal_times else 0.0

            step_result = self.validate_step(pos, vel, goals, target, obstacles, failed, heal_t)
            all_emergent.append(step_result["emergent"])
            all_swarm_metrics.append(step_result["swarm_metrics"])
            all_safety.append(step_result["safety"])
            all_formal.append(step_result["formal"])

        # Aggregate emergent behaviors (majority vote)
        behaviors = ["flocking", "flanking", "encircling", "swarming", "self_healing"]
        emergent_agg = {}
        confidence = {}
        for b in behaviors:
            count = sum(1 for e in all_emergent if e.get(b, False))
            emergent_agg[b] = count > n_steps / 2
            confidence[b] = count / n_steps

        # Aggregate swarm metrics
        swarm_agg = {}
        for key in all_swarm_metrics[0].keys():
            vals = [m[key] for m in all_swarm_metrics]
            swarm_agg[key] = float(np.mean(vals))

        # Aggregate safety
        all_violations = []
        for s in all_safety:
            all_violations.extend(s.violations)
        unique_violations = sorted(set(all_violations))

        min_obs_dist = min(s.min_obstacle_distance for s in all_safety)
        min_geofence = min(s.min_geofence_margin for s in all_safety)
        min_alt = min(s.min_altitude_margin for s in all_safety)
        max_speed_v = max(s.max_speed_violation for s in all_safety)
        min_cbf = min(s.cbf_margin for s in all_safety)

        # Aggregate formal
        formal_agg = {}
        if all_formal and all_formal[0]:
            for key in all_formal[0].keys():
                vals = [f.get(key, 0) for f in all_formal]
                if isinstance(vals[0], bool):
                    formal_agg[key] = float(sum(vals) / len(vals))
                else:
                    formal_agg[key] = float(np.mean(vals))

        # Mission success rate (from last step)
        mission_sr = swarm_agg.get("formation_hold_ratio", 0.0)

        swarm_metrics = SwarmMetrics(
            velocity_alignment=swarm_agg.get("velocity_alignment", 0.0),
            lateral_spread=swarm_agg.get("lateral_spread", 0.0),
            encirclement_ratio=swarm_agg.get("encirclement_ratio", 0.0),
            density=swarm_agg.get("density", 0.0),
            formation_error_m=swarm_agg.get("formation_error_m", 0.0),
            formation_hold_ratio=swarm_agg.get("formation_hold_ratio", 0.0),
            mission_success_rate=mission_sr,
        )

        emergent_report = EmergentBehaviorReport(
            flocking=emergent_agg.get("flocking", False),
            flanking=emergent_agg.get("flanking", False),
            encircling=emergent_agg.get("encircling", False),
            swarming=emergent_agg.get("swarming", False),
            self_healing=emergent_agg.get("self_healing", False),
            confidence=confidence,
            raw_metrics={
                "velocity_alignment": swarm_agg.get("velocity_alignment", 0.0),
                "lateral_spread": swarm_agg.get("lateral_spread", 0.0),
                "encirclement_ratio": swarm_agg.get("encirclement_ratio", 0.0),
                "density": swarm_agg.get("density", 0.0),
            },
        )

        safety_report = SafetyReport(
            violations=unique_violations,
            min_obstacle_distance=min_obs_dist,
            min_geofence_margin=min_geofence,
            min_altitude_margin=min_alt,
            max_speed_violation=max_speed_v,
            cbf_margin=min_cbf,
            safe=len(unique_violations) == 0,
        )

        # Determine pass/fail based on acceptance criteria
        passed = (
            safety_report.safe and
            swarm_metrics.formation_hold_ratio > 0.95 and
            emergent_report.confidence.get("flocking", 0) > 0.5 and
            formal_agg.get("cbf_sos_feasible", 0) > 0.5
        )

        return SwarmValidationResult(
            scenario_id=scenario_id,
            n_drones=self.n_drones,
            duration_s=n_steps * 0.5,  # DT = 0.5s
            swarm_metrics=swarm_metrics,
            emergent_behaviors=emergent_report,
            safety=safety_report,
            formal_verification=formal_agg,
            passed=passed,
        )

    def detect_emergent_behavior_accuracy(
        self,
        ground_truth: Dict[str, bool],
        predicted: Dict[str, bool],
    ) -> float:
        """Compute detection accuracy against ground truth."""
        if not ground_truth:
            return 1.0
        correct = sum(1 for k, v in ground_truth.items() if predicted.get(k) == v)
        return correct / len(ground_truth)

    def get_behavior_timeline(self) -> Dict[str, List[bool]]:
        """Get timeline of each emergent behavior."""
        if not self.history:
            return {}

        behaviors = ["flocking", "flanking", "encircling", "swarming", "self_healing"]
        timeline = {b: [] for b in behaviors}
        for h in self.history:
            e = h.get("emergent", {})
            for b in behaviors:
                timeline[b].append(e.get(b, False))
        return timeline

    def get_safety_timeline(self) -> List[List[str]]:
        """Get timeline of safety violations."""
        return [h.get("safety", SafetyReport([], 0, 0, 0, 0, 0, True)).violations for h in self.history]

    def reset(self) -> None:
        """Reset validator history."""
        self.history.clear()