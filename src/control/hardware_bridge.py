#!/usr/bin/env python3
"""Unified sim <-> hardware bridge for DroneNav-SAR (Sprint 14).

Routes SAR skill commands to either a sim backend (MockDroneBackend or any
object with execute/step) or a hardware backend (CrazyflieBridge /
PX4Interface). Supports policy hot-swap without reconnecting.

SAR-only: navigate_to / hover / drop_payload / return_home. No weapons.
Every command passes safety_filter.allow + SafetyMonitor before dispatch.
"""

from typing import Any, Callable, Dict, List, Optional

from src.control.safety_filter import allow as _safety_allow
from src.hardware.safety_monitor import SafetyMonitor

SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")


class _SimRecorder:
    """Minimal sim backend used when none is supplied (records commands)."""

    def __init__(self) -> None:
        self.executed: List[Dict[str, Any]] = []

    def execute(self, skill_cmd: Dict[str, Any]) -> Dict[str, Any]:
        self.executed.append(dict(skill_cmd))
        return {"ok": True, "backend": "sim", "skill": skill_cmd.get("skill")}


class HardwareBridge:
    """Unified interface: mode in {"sim", "hardware"} + hot-swappable policy."""

    def __init__(self, mode: str = "sim", sim_backend=None,
                 hw_backend=None, safety_monitor: Optional[SafetyMonitor] = None,
                 policy: Any = None) -> None:
        if mode not in ("sim", "hardware"):
            raise ValueError("mode must be 'sim' or 'hardware'")
        self.mode = mode
        self.sim_backend = sim_backend or _SimRecorder()
        self.hw_backend = hw_backend
        self.safety = safety_monitor or SafetyMonitor()
        self.policy = policy
        self.policy_version = 0 if policy is None else 1
        self.history: List[Dict[str, Any]] = []

    # -- mode / policy ------------------------------------------------------
    def set_mode(self, mode: str) -> Dict[str, Any]:
        if mode not in ("sim", "hardware"):
            return {"ok": False, "reason": f"bad_mode: {mode!r}"}
        if mode == "hardware" and self.hw_backend is None:
            return {"ok": False, "reason": "no_hw_backend"}
        self.mode = mode
        return {"ok": True, "mode": self.mode}

    def load_policy(self, policy: Any) -> Dict[str, Any]:
        """Attach a policy object (must expose predict/step/__call__ or be a dict)."""
        self.policy = policy
        self.policy_version += 1
        return {"ok": True, "policy_version": self.policy_version}

    def swap_policy(self, new_policy: Any) -> Dict[str, Any]:
        """Hot-swap policy at runtime; backend connections are untouched."""
        old = self.policy_version
        res = self.load_policy(new_policy)
        res["old_version"] = old
        res["hot_swapped"] = True
        return res

    # -- dispatch ---------------------------------------------------------------
    def _check(self, skill_cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        verdict = _safety_allow(skill_cmd)
        if not verdict["ok"]:
            return {"ok": False, "reason": verdict["reason"], "fallback": "hover"}
        params = skill_cmd.get("params", {}) or {}
        sv = self.safety.evaluate({
            "position": [float(params.get("x", 2.0)), float(params.get("y", 2.0)),
                         float(params.get("z", 1.0))],
        })
        if not sv["safe"]:
            return {"ok": False, "reason": f"safety: {sv['violations']}",
                    "fallback": sv["action"]}
        return None

    def execute_skill(self, skill_cmd: Dict[str, Any]) -> Dict[str, Any]:
        blocked = self._check(skill_cmd)
        if blocked is not None:
            self.history.append({"cmd": dict(skill_cmd), "res": blocked})
            return blocked
        backend = self.hw_backend if self.mode == "hardware" else self.sim_backend
        if backend is None:
            res = {"ok": False, "reason": "no_backend", "fallback": "hover"}
        else:
            try:
                out = backend.execute(dict(skill_cmd))
                res = {"ok": bool(out.get("ok", True)), "mode": self.mode,
                       "skill": skill_cmd.get("skill"), "backend_res": out}
                if not res["ok"] and "reason" in out:
                    res["reason"] = out["reason"]
            except Exception as exc:
                res = {"ok": False, "reason": f"backend_error: {exc}",
                       "fallback": "hover"}
        self.history.append({"cmd": dict(skill_cmd), "res": res})
        return res

    def step(self, obs: Any = None) -> Dict[str, Any]:
        """One policy step: policy(obs) -> skill_cmd -> execute_skill.

        Policy may be: callable(obs)->cmd, object with .predict(obs),
        object with .step(obs), or a fixed skill-cmd dict.
        """
        if self.policy is None:
            return {"ok": False, "reason": "no_policy"}
        if isinstance(self.policy, dict):
            cmd = dict(self.policy)
        elif callable(getattr(self.policy, "predict", None)):
            cmd = self.policy.predict(obs)
        elif callable(getattr(self.policy, "step", None)):
            cmd = self.policy.step(obs)
        elif callable(self.policy):
            cmd = self.policy(obs)
        else:
            return {"ok": False, "reason": "policy_not_callable"}
        if not isinstance(cmd, dict) or "skill" not in cmd:
            return {"ok": False, "reason": "policy_bad_cmd"}
        cmd.setdefault("params", {})
        cmd.setdefault("constraints", {"timeout_s": 20.0})
        return self.execute_skill(cmd)

    def get_status(self) -> Dict[str, Any]:
        return {"mode": self.mode,
                "policy_version": self.policy_version,
                "has_hw_backend": self.hw_backend is not None,
                "estop_latched": self.safety.estop_latched,
                "history_len": len(self.history)}
