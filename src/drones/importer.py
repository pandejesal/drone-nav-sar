#!/usr/bin/env python3
"""Minimal drone.yaml importer for DroneNav-SAR (Sprint D2).

stdlib only — CI venv has no pyyaml. Strict 2-space `key: value` subset
parser (nested maps only, no lists/anchors). Anything outside the subset
raises ValueError (fail closed, never silently misread a datasheet).

Validation (taboo guards):
- api_version must be 2 (unknown majors rejected).
- mass_kg > 0, arm_length_m > 0, motor_tau_s > 0.
- thrust_unit must be "normalized" — raw-RPM configs rejected loudly,
  never silently mixed (Sprint-3 motor-units lesson).
- 4 * max_thrust_N must exceed weight * 1.2 (thrust margin).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from src.sim.drone_dynamics import QuadrotorParams

GRAVITY = 9.81
THRUST_MARGIN = 1.2
SUPPORTED_API = 2


def _parse_scalar(token: str) -> Any:
    t = token.strip()
    if t == "":
        return None
    low = t.lower()
    if low in ("null", "~", "none"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if (len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"')):
        return t[1:-1]
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def parse_subset_yaml(text: str) -> Dict[str, Any]:
    """Parse the strict subset used by drone.yaml. Raises ValueError."""
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if "\t" in raw:
            raise ValueError(f"yaml:{lineno}: tabs not allowed")
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"yaml:{lineno}: indent must be 2-space multiples")
        stripped = raw.strip()
        if ":" not in stripped:
            raise ValueError(f"yaml:{lineno}: expected 'key: value'")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        if not key or " " in key:
            raise ValueError(f"yaml:{lineno}: bad key {key!r}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if key in parent:
            raise ValueError(f"yaml:{lineno}: duplicate key {key!r}")
        value = rest.strip()
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if "#" in value:  # no inline comments in subset
                raise ValueError(f"yaml:{lineno}: inline comments not allowed")
            parent[key] = _parse_scalar(value)
    return root


@dataclass(frozen=True)
class DroneSpec:
    """Validated drone + traceability hash."""

    api_version: int
    name: str
    params: QuadrotorParams
    spec_hash: str
    model_file: str = ""


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ValueError(f"drone.yaml: missing '{key}' in {where}")
    return d[key]


def spec_from_dict(d: Dict[str, Any], canonical_text: str = "") -> DroneSpec:
    api = _require(d, "api_version", "root")
    if int(api) != SUPPORTED_API:
        raise ValueError(f"drone.yaml: unsupported api_version {api!r} (want 2)")
    name = str(_require(d, "name", "root"))
    geo = _require(d, "geometry", "root")
    prop = _require(d, "propulsion", "root")
    if not isinstance(geo, dict) or not isinstance(prop, dict):
        raise ValueError("drone.yaml: geometry/propulsion must be maps")

    arm = float(_require(geo, "arm_length_m", "geometry"))
    mass = float(_require(geo, "mass_kg", "geometry"))
    thrust = float(_require(prop, "max_thrust_N", "propulsion"))
    tau = float(prop.get("motor_tau_s", 0.02))
    unit = str(prop.get("thrust_unit", "normalized")).strip().lower()
    rpm = float(prop.get("max_rpm", 25000.0))

    if arm <= 0:
        raise ValueError(f"drone.yaml: arm_length_m must be > 0 (got {arm})")
    if mass <= 0:
        raise ValueError(f"drone.yaml: mass_kg must be > 0 (got {mass})")
    if tau <= 0:
        raise ValueError(f"drone.yaml: motor_tau_s must be > 0 (got {tau})")
    if unit != "normalized":
        raise ValueError(
            f"drone.yaml: thrust_unit must be 'normalized' (got {unit!r}); "
            "raw-RPM configs are rejected, never silently mixed"
        )
    weight = mass * GRAVITY
    if 4.0 * thrust < weight * THRUST_MARGIN:
        raise ValueError(
            f"drone.yaml: thrust margin failed: 4*{thrust:.3f}N < "
            f"{weight:.3f}N*{THRUST_MARGIN}"
        )

    inertia = None
    raw_in = d.get("inertia")
    if isinstance(raw_in, dict) and {"ixx", "iyy", "izz"} <= set(raw_in):
        inertia = np.diag(
            [float(raw_in["ixx"]), float(raw_in["iyy"]), float(raw_in["izz"])]
        ).astype(np.float32)

    from typing import Any as _Any

    kwargs: Dict[str, _Any] = dict(
        mass=mass,
        arm_length=arm,
        max_thrust=thrust,
        max_rpm=rpm,
        motor_time_constant=tau,
    )
    if inertia is not None:
        kwargs["inertia"] = inertia
    params = QuadrotorParams(**kwargs)  # type: ignore[arg-type]
    model_file = ""
    raw_mf = d.get("model_files")
    if isinstance(raw_mf, dict):
        model_file = str(raw_mf.get("urdf", raw_mf.get("sdf", "")))
    digest = hashlib.sha1(canonical_text.encode("utf-8")).hexdigest() if canonical_text else ""
    return DroneSpec(api_version=2, name=name, params=params, spec_hash=digest,
                     model_file=model_file)


def load_drone(path_or_dict: Union[str, Path, Dict[str, Any]]) -> DroneSpec:
    """Load drone.yaml (path) or an already-parsed dict."""
    if isinstance(path_or_dict, dict):
        return spec_from_dict(path_or_dict, canonical_text="")
    path = Path(path_or_dict)
    text = path.read_text(encoding="utf-8")
    return spec_from_dict(parse_subset_yaml(text), canonical_text=text)
