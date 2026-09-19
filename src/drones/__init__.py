"""DroneNav-SAR custom drone import (Sprint D2 minimal slice).

Anyone-brings-own-drone: drone.yaml -> validated QuadrotorParams + spec_hash.
stdlib only (no pyyaml in CI venv): strict 2-space subset parser below.
SAR-only geometry/dynamics; no weapons.
"""

from src.drones.importer import DroneSpec, load_drone

__all__ = ["DroneSpec", "load_drone"]
