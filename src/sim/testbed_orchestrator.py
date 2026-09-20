#!/usr/bin/env python3
"""Sprint 33: TestBed Orchestrator — Scenario runner, result aggregation, report generation.

Orchestrates test bed scenarios, runs parameter sweeps, aggregates results,
and generates formal verification reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import json
import uuid
from pathlib import Path

import numpy as np

from src.sim.large_scale_testbed import (
    LargeScaleTestBed,
    ScenarioConfig,
    ScenarioType,
    ScenarioResult,
    create_scenario,
)
from src.sim.swarm_validator import SwarmValidator, SwarmValidationResult
from src.sim.hil_orchestrator import HILOrchestrator, DroneHILConfig
from src.sim.autonomy_validator import AutonomyValidator, ValidatorConfig


@dataclass
class Campaign:
    """A campaign of related test scenarios."""
    campaign_id: str
    name: str
    scenarios: List[ScenarioConfig]
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class CampaignResult:
    """Aggregated results for a test campaign."""
    campaign_id: str
    name: str
    scenario_results: List[ScenarioResult]
    swarm_validation_results: List[SwarmValidationResult]
    summary: Dict[str, Any]
    passed: bool
    duration_s: float


class BedOrchestrator:
    """Orchestrates test bed campaigns, scenarios, and reporting."""

    def __init__(
        self,
        default_n_drones: int = 50,
        default_area_m: float = 60.0,
        enable_hil: bool = False,
        enable_formal: bool = True,
        output_dir: str = "testbed_results",
    ):
        self.default_n_drones = default_n_drones
        self.default_area_m = default_area_m
        self.enable_hil = enable_hil
        self.enable_formal = enable_formal
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test bed instance
        self.testbed = LargeScaleTestBed(
            n_drones=default_n_drones,
            area_m=default_area_m,
            enable_hil=enable_hil,
            enable_formal=enable_formal,
        )

        # Swarm validator
        self.swarm_validator = SwarmValidator(
            n_drones=default_n_drones,
            area_m=default_area_m,
            enable_formal=enable_formal,
        )

        # HIL orchestrator (if enabled)
        self.hil_orchestrator: Optional[HILOrchestrator] = None
        if enable_hil:
            self._init_hil()

        # Campaign history
        self.campaigns: List[TestCampaign] = []
        self.campaign_results: List[CampaignResult] = []

    def _init_hil(self) -> None:
        """Initialize HIL orchestrator with default configs."""
        drone_configs = [
            DroneHILConfig(drone_id=d, transport_type="mock")
            for d in range(min(self.default_n_drones, 10))  # HIL limited to 10
        ]
        self.hil_orchestrator = HILOrchestrator(drone_configs)
        self.hil_orchestrator.connect_all()

    def add_campaign(self, campaign: TestCampaign) -> None:
        """Add a test campaign."""
        self.campaigns.append(campaign)

    def run_campaign(self, campaign: TestCampaign) -> CampaignResult:
        """Run a complete test campaign."""
        start_time = time.perf_counter()
        scenario_results = []
        swarm_results = []

        for scenario in campaign.scenarios:
            # Run scenario on test bed
            result = self.testbed.run_scenario(scenario)
            scenario_results.append(result)

            # Validate with swarm validator
            # Reconstruct history from testbed
            positions_hist = [self.testbed.env.positions.copy()]  # simplified
            velocities_hist = [self.testbed.env.velocities.copy()]
            goals_hist = [self.testbed.env.formation_goal.copy()]

            val_result = self.swarm_validator.validate_episode(
                positions_hist, velocities_hist, goals_hist,
                targets_history=[np.array([0., 0., 10.])] if not self.testbed.env.victims else self.testbed.env.victims,
            )
            swarm_results.append(val_result)

        duration_s = time.perf_counter() - start_time

        # Aggregate summary
        summary = self._aggregate_summary(scenario_results, swarm_results)

        # Determine overall pass
        passed = all(r.success for r in scenario_results) and all(r.passed for r in swarm_results)

        campaign_result = CampaignResult(
            campaign_id=campaign.campaign_id,
            name=campaign.name,
            scenario_results=scenario_results,
            swarm_validation_results=swarm_results,
            summary=summary,
            passed=passed,
            duration_s=duration_s,
        )

        self.campaign_results.append(campaign_result)
        self._save_campaign_result(campaign_result)

        return campaign_result

    def run_scenario(self, scenario: ScenarioConfig) -> ScenarioResult:
        """Run a single scenario."""
        return self.testbed.run_scenario(scenario)

    def run_parameter_sweep(
        self,
        base_scenario: ScenarioConfig,
        param_name: str,
        param_values: List[Any],
        n_runs: int = 3,
    ) -> List[ScenarioResult]:
        """Run parameter sweep."""
        return self.testbed.run_parameter_sweep(base_scenario, param_name, param_values, n_runs)

    def run_hil_scenario(
        self,
        scenario: ScenarioConfig,
        control_provider: Callable[[int, int], np.ndarray],
        rate_hz: float = 50.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Run HIL scenario."""
        if not self.hil_orchestrator:
            raise RuntimeError("HIL not enabled")
        return self.hil_orchestrator.run_scenario(
            scenario.max_steps, control_provider, rate_hz
        )

    def _aggregate_summary(
        self,
        scenario_results: List[ScenarioResult],
        swarm_results: List[SwarmValidationResult],
    ) -> Dict[str, Any]:
        """Aggregate campaign summary statistics."""
        if not scenario_results:
            return {}

        n_scenarios = len(scenario_results)
        n_drones = scenario_results[0].n_drones

        # Mission metrics
        mission_srs = [r.metrics.get("mission_sr", 0.0) for r in scenario_results]
        formation_errors = [r.metrics.get("formation_error_m", 0.0) for r in scenario_results]
        formation_holds = [r.metrics.get("formation_hold", 0.0) for r in scenario_results]

        # Emergent behaviors
        emergent_counts = {}
        for r in scenario_results:
            for behavior, detected in r.emergent_behaviors.items():
                if behavior not in emergent_counts:
                    emergent_counts[behavior] = 0
                if detected:
                    emergent_counts[behavior] += 1

        # Safety
        total_violations = sum(len(r.safety_violations) for r in scenario_results)
        safety_clean = sum(1 for r in scenario_results if not r.safety_violations)

        # Formal verification
        formal_reach = np.mean([r.formal_verification.get("reachability", 0.0) for r in scenario_results])
        formal_safety = np.mean([r.formal_verification.get("safety", 0.0) for r in scenario_results])
        formal_cbf = np.mean([r.formal_verification.get("cbf_feasible", 0.0) for r in scenario_results])
        formal_liveness = np.mean([r.formal_verification.get("liveness", 0.0) for r in scenario_results])

        # Swarm validation
        swarm_passed = sum(1 for r in swarm_results if r.passed)
        swarm_formation_hold = np.mean([r.swarm_metrics.formation_hold_ratio for r in swarm_results])

        return {
            "n_scenarios": n_scenarios,
            "n_drones": n_drones,
            "mission_sr_mean": float(np.mean(mission_srs)),
            "mission_sr_min": float(np.min(mission_srs)),
            "formation_error_mean": float(np.mean(formation_errors)),
            "formation_hold_mean": float(np.mean(formation_holds)),
            "emergent_behavior_rates": {k: v / n_scenarios for k, v in emergent_counts.items()},
            "safety_violations_total": total_violations,
            "safety_clean_rate": safety_clean / n_scenarios,
            "formal_reachability": float(formal_reach),
            "formal_safety": float(formal_safety),
            "formal_cbf_feasible": float(formal_cbf),
            "formal_liveness": float(formal_liveness),
            "swarm_validation_passed": swarm_passed,
            "swarm_validation_rate": swarm_passed / len(swarm_results) if swarm_results else 0.0,
            "swarm_formation_hold": float(swarm_formation_hold),
        }

    def _save_campaign_result(self, result: CampaignResult) -> None:
        """Save campaign result to JSON."""
        output = {
            "campaign_id": result.campaign_id,
            "name": result.name,
            "passed": result.passed,
            "duration_s": result.duration_s,
            "summary": result.summary,
            "scenarios": [
                {
                    "scenario_id": r.scenario_id,
                    "type": r.scenario_type.value,
                    "n_drones": r.n_drones,
                    "success": r.success,
                    "metrics": r.metrics,
                    "emergent_behaviors": r.emergent_behaviors,
                    "safety_violations": r.safety_violations,
                    "formal_verification": r.formal_verification,
                    "duration_s": r.duration_s,
                }
                for r in result.scenario_results
            ],
            "swarm_validation": [
                {
                    "scenario_id": r.scenario_id,
                    "passed": r.passed,
                    "swarm_metrics": {
                        "velocity_alignment": r.swarm_metrics.velocity_alignment,
                        "lateral_spread": r.swarm_metrics.lateral_spread,
                        "encirclement_ratio": r.swarm_metrics.encirclement_ratio,
                        "density": r.swarm_metrics.density,
                        "formation_error_m": r.swarm_metrics.formation_error_m,
                        "formation_hold_ratio": r.swarm_metrics.formation_hold_ratio,
                        "mission_success_rate": r.swarm_metrics.mission_success_rate,
                    },
                    "emergent_behaviors": {
                        "flocking": r.emergent_behaviors.flocking,
                        "flanking": r.emergent_behaviors.flanking,
                        "encircling": r.emergent_behaviors.encircling,
                        "swarming": r.emergent_behaviors.swarming,
                        "self_healing": r.emergent_behaviors.self_healing,
                        "confidence": r.emergent_behaviors.confidence,
                    },
                    "safety": {
                        "violations": r.safety.violations,
                        "safe": r.safety.safe,
                        "cbf_margin": r.safety.cbf_margin,
                    },
                    "formal_verification": r.formal_verification,
                }
                for r in result.swarm_validation_results
            ],
        }

        path = self.output_dir / f"campaign_{result.campaign_id}_{result.name}.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def generate_report(self, campaign_result: CampaignResult) -> str:
        """Generate human-readable report."""
        lines = [
            f"# Test Campaign Report: {campaign_result.name}",
            f"**Campaign ID:** {campaign_result.campaign_id}",
            f"**Duration:** {campaign_result.duration_s:.1f}s",
            f"**Overall:** {'PASSED' if campaign_result.passed else 'FAILED'}",
            "",
            "## Summary",
            f"- Scenarios: {campaign_result.summary.get('n_scenarios', 0)}",
            f"- Drones per scenario: {campaign_result.summary.get('n_drones', 0)}",
            f"- Mission SR (mean): {campaign_result.summary.get('mission_sr_mean', 0):.3f}",
            f"- Formation Hold (mean): {campaign_result.summary.get('formation_hold_mean', 0):.3f}",
            f"- Formation Error (mean): {campaign_result.summary.get('formation_error_mean', 0):.3f}m",
            f"- Safety Clean Rate: {campaign_result.summary.get('safety_clean_rate', 0):.3f}",
            f"- Formal Reachability: {campaign_result.summary.get('formal_reachability', 0):.3f}",
            f"- Formal Safety: {campaign_result.summary.get('formal_safety', 0):.3f}",
            f"- Formal CBF Feasible: {campaign_result.summary.get('formal_cbf_feasible', 0):.3f}",
            f"- Formal Liveness: {campaign_result.summary.get('formal_liveness', 0):.3f}",
            f"- Swarm Validation Rate: {campaign_result.summary.get('swarm_validation_rate', 0):.3f}",
            "",
            "## Emergent Behaviors",
        ]

        for behavior, rate in campaign_result.summary.get("emergent_behavior_rates", {}).items():
            lines.append(f"- {behavior}: {rate:.1%}")

        lines.extend([
            "",
            "## Scenario Details",
        ])

        for r in campaign_result.scenario_results:
            lines.extend([
                f"### {r.scenario_type.value} (ID: {r.scenario_id})",
                f"- Success: {r.success}",
                f"- Mission SR: {r.metrics.get('mission_sr', 0):.3f}",
                f"- Formation Hold: {r.metrics.get('formation_hold', 0):.3f}",
                f"- Duration: {r.duration_s:.1f}s",
                f"- Emergent: {', '.join([k for k, v in r.emergent_behaviors.items() if v]) or 'none'}",
                f"- Safety: {'clean' if not r.safety_violations else 'violations: ' + ', '.join(r.safety_violations)}",
                "",
            ])

        return "\n".join(lines)

    def run_standard_campaign(self, name: str = "standard") -> CampaignResult:
        """Run the standard Sprint 33 campaign with all scenario types."""
        scenarios = [
            create_scenario(ScenarioType.FORMATION_FLIGHT, n_drones=50, seed=42),
            create_scenario(ScenarioType.SWARM_SEARCH, n_drones=50, seed=43),
            create_scenario(ScenarioType.SEARCH_RESCUE, n_drones=24, seed=44),
            create_scenario(ScenarioType.SWARM_PATROL, n_drones=12, seed=45),
            create_scenario(ScenarioType.CONTESTED_COMMS, n_drones=24, seed=46),
        ]

        if self.enable_hil:
            scenarios.append(create_scenario(ScenarioType.HIL_VALIDATION, n_drones=10, seed=47))

        campaign = TestCampaign(
            campaign_id=str(uuid.uuid4())[:8],
            name=name,
            scenarios=scenarios,
            description="Standard Sprint 33 validation campaign",
            tags=["sprint33", "standard"],
        )

        return self.run_campaign(campaign)

    def shutdown(self) -> None:
        """Clean shutdown."""
        self.testbed.shutdown()
        if self.hil_orchestrator:
            self.hil_orchestrator.disconnect_all()