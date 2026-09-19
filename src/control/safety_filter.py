#!/usr/bin/env python3
"""Default-deny safety filter for DroneNav-SAR skill commands (Sprint C1).

SAR-only: navigate_to / hover / drop_payload / return_home.
Reject = hover in place. No bypass path. stdlib only.

allow(cmd) -> dict(ok=bool, reason=str).
"""

import json
from pathlib import Path
from typing import Dict, Tuple

_SCHEMA_PATH = Path(__file__).resolve().parent / "skill_schema.json"


def _load_schema() -> Dict:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def allow(cmd: Dict) -> Dict:
    """Validate a skill command. Default-deny: any violation -> ok False."""
    try:
        schema = _load_schema()
    except Exception as exc:  # fail closed if schema unreadable
        return {"ok": False, "reason": f"schema_unavailable: {exc}"}
    if not isinstance(cmd, dict):
        return {"ok": False, "reason": "not_a_dict"}
    skill = cmd.get("skill")
    if skill not in schema.get("allowlist", []):
        return {"ok": False, "reason": f"skill_not_allowlisted: {skill!r}"}
    params = cmd.get("params")
    constraints = cmd.get("constraints")
    if not isinstance(params, dict):
        return {"ok": False, "reason": "params_not_a_dict"}
    if not isinstance(constraints, dict):
        return {"ok": False, "reason": "constraints_not_a_dict"}

    limits = schema.get("limits", {})
    ceiling = float(limits.get("ceiling_m", 2.5))
    geofence = limits.get("geofence_xyz", [[0, 4], [0, 4], [0.2, 2.5]])
    max_speed = float(limits.get("max_speed_ms", 1.5))
    max_timeout = float(limits.get("max_timeout_s", 60.0))

    def in_range(v: float, lo: float, hi: float) -> bool:
        return lo <= float(v) <= hi

    # Skill-specific target checks
    if skill == "navigate_to":
        for axis, key in (("x", "x"), ("y", "y"), ("z", "z")):
            if key not in params:
                return {"ok": False, "reason": f"missing_param: {key}"}
        x, y, z = float(params["x"]), float(params["y"]), float(params["z"])
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = geofence
        if not (in_range(x, xlo, xhi) and in_range(y, ylo, yhi) and in_range(z, zlo, zhi)):
            return {"ok": False, "reason": f"out_of_geofence: {[x, y, z]}"}
        if z > ceiling:
            return {"ok": False, "reason": f"above_ceiling: {z} > {ceiling}"}
        speed = float(params.get("speed_ms", 0.8))
        if not 0 < speed <= max_speed:
            return {"ok": False, "reason": f"speed_violation: {speed}"}
    elif skill == "hover":
        z = float(params.get("z", 1.0))
        if z > ceiling:
            return {"ok": False, "reason": f"above_ceiling: {z} > {ceiling}"}
    elif skill == "drop_payload":
        z = float(params.get("z", 1.0))
        if z > ceiling:
            return {"ok": False, "reason": f"above_ceiling: {z} > {ceiling}"}
    elif skill == "return_home":
        speed = float(params.get("speed_ms", 0.8))
        if not 0 < speed <= max_speed:
            return {"ok": False, "reason": f"speed_violation: {speed}"}

    timeout = float(constraints.get("timeout_s", 20.0))
    if not 0 < timeout <= max_timeout:
        return {"ok": False, "reason": f"timeout_violation: {timeout}"}
    return {"ok": True, "reason": "allow"}


def check(cmd: Dict) -> Tuple[bool, str]:
    """Tuple form: (ok, reason)."""
    verdict = allow(cmd)
    return bool(verdict["ok"]), str(verdict["reason"])


# -- Sprint 16-18 extension: drop zones, no-drop zones, altitude floor ----
_DROP_ZONES: list = []      # list of (xmin,xmax,ymin,ymax)
_NO_DROP_ZONES: list = []   # list of (xmin,xmax,ymin,ymax)
_ALTITUDE_FLOOR_M: float = 0.0


def set_drop_zones(drop_zones=None, no_drop_zones=None,
                   altitude_floor_m: float = 0.0) -> Dict:
    """Configure strike drop-zone policy. Empty drop_zones = anywhere allowed."""
    global _DROP_ZONES, _NO_DROP_ZONES, _ALTITUDE_FLOOR_M
    _DROP_ZONES = [tuple(z) for z in (drop_zones or [])]
    _NO_DROP_ZONES = [tuple(z) for z in (no_drop_zones or [])]
    _ALTITUDE_FLOOR_M = float(altitude_floor_m)
    return {"drop_zones": len(_DROP_ZONES), "no_drop_zones": len(_NO_DROP_ZONES),
            "altitude_floor_m": _ALTITUDE_FLOOR_M}


def clear_drop_zones() -> Dict:
    return set_drop_zones([], [], 0.0)


def _in_zone(x: float, y: float, zone) -> bool:
    xmin, xmax, ymin, ymax = zone
    return bool(xmin <= x <= xmax and ymin <= y <= ymax)


def check_drop_allowed(x: float, y: float, z: float) -> Dict:
    """Strike drop-zone gate: floor + no-drop exclusion + drop-zone inclusion."""
    if float(z) < _ALTITUDE_FLOOR_M:
        return {"ok": False, "reason": f"below_altitude_floor: {z}"}
    for zone in _NO_DROP_ZONES:
        if _in_zone(float(x), float(y), zone):
            return {"ok": False, "reason": f"in_no_drop_zone: {[x, y]}"}
    if _DROP_ZONES:
        for zone in _DROP_ZONES:
            if _in_zone(float(x), float(y), zone):
                return {"ok": True, "reason": "allow"}
        return {"ok": False, "reason": f"outside_drop_zones: {[x, y]}"}
    return {"ok": True, "reason": "allow"}
