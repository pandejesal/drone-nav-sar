#!/usr/bin/env python3
"""SLM/LLM adapter stub for DroneNav-SAR (Sprint C1).

The policy emits structured skill JSON; the SLM only fills params within
skill_schema.json. This stub validates offline (stdlib+json, no new deps):
- parses JSON str or dict,
- enforces allowlist + required params per skill,
- fills safe defaults (speed_ms, timeout_s, constraints),
- raises ValueError on any violation (no silent fix-ups).

Real LLM decoding (JSON-mode/grammar via llguidance, R16) plugs in behind
ADAPTER_MODEL env flag later; default stays offline for CI stability.
"""

import copy
import json
from pathlib import Path
from typing import Dict, Union

_SCHEMA_PATH = Path(__file__).resolve().parent / "skill_schema.json"

_DEFAULTS = {
    "navigate_to": {"yaw_deg": 0.0, "speed_ms": 0.8},
    "hover": {"yaw_deg": 0.0, "timeout_s": 10.0},
    "drop_payload": {"yaw_deg": 0.0},
    "return_home": {"speed_ms": 0.8},
}

_DEFAULT_CONSTRAINTS = {
    "ceiling_m": 2.5,
    "geofence_xyz": [[0.0, 4.0], [0.0, 4.0], [0.2, 2.5]],
    "timeout_s": 20.0,
}


def _load_schema() -> Dict:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def adapt(raw: Union[str, Dict], task: str = "medkit_AB") -> Dict:
    """Validate + complete a raw skill command. Raises ValueError if invalid."""
    schema = _load_schema()
    cmd = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    if not isinstance(cmd, dict):
        raise ValueError("adapter: command must be a JSON object")
    skill = cmd.get("skill")
    if skill not in schema["allowlist"]:
        raise ValueError(f"adapter: skill not allowlisted: {skill!r}")
    if not isinstance(skill, str):
        raise ValueError(f"adapter: skill must be a string: {skill!r}")
    required_params = schema["skills"][skill]["params"]
    params = cmd.get("params")
    if not isinstance(params, dict):
        raise ValueError("adapter: params must be an object")
    completed = dict(_DEFAULTS.get(skill, {}))
    completed.update(params)
    missing = [k for k in required_params if k not in completed]
    if missing:
        raise ValueError(f"adapter: missing params for {skill}: {missing}")
    constraints = cmd.get("constraints")
    if constraints is None:
        constraints = {}
    if not isinstance(constraints, dict):
        raise ValueError("adapter: constraints must be an object")
    merged_constraints = dict(_DEFAULT_CONSTRAINTS)
    merged_constraints.update(constraints)
    out = {
        "skill": skill,
        "params": {k: completed[k] for k in required_params if k in completed},
        "constraints": merged_constraints,
        "task": cmd.get("task", task),
        "task_embedding_id": int(cmd.get("task_embedding_id", 0)),
    }
    # keep any extra known-good defaults that callers rely on (e.g. timeout)
    for k, v in completed.items():
        out["params"].setdefault(k, v)
    return out
