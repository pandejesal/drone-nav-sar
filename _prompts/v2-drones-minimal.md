# Drones Importer v2 — Minimal Slice (SAR-only)

## Why v2
- v1 20260916_235027_ae179c HONEST_ABANDON: zero files, 29m stall on nemotron-lightning. Smaller slice: yaml + params only, URDF as opaque path (no parser deps).

## Scope (create ONLY)
- src/drones/__init__.py (re-export DroneSpec, load_drone)
- src/drones/importer.py (stdlib+yaml only: parse drone.yaml api_version 2, validate name/mass_kg>0/max_thrust_N>weight*1.2 margin/thrust_unit==normalized else ValueError with clear msg, inertia defaults if missing, model_files urdf/sdf stored as path string NO parsing (defer yourdfpy to v2b), emit DroneSpec dataclass {api_version,name,params:QuadrotorParams,spec_hash:sha1(canonical yaml)}, load_drone(path)->DroneSpec)
- assets/drones/crazyflie-like-custom/drone.yaml (api_version 2, arm 0.046, mass 0.027, max_thrust 0.62, tau 0.02, thrust_unit normalized, inertia ixx/iyy/izz, model_files urdf model.urdf)
- tests/test_drones.py (4 tests: valid loads + hash stable across reload; raw-RPM rejected ValueError; mass<=0 rejected; thrust<weight rejected)

## Ground (read)
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/drone_dynamics.py (QuadrotorParams fields — match names exactly)

## TABOO
- No new pip deps, stdlib + yaml (yaml already used?) else手工 parse fallback; normalized only; forward-slash paths

## Safety
SAR-ONLY geometry/dynamics, no weapons.

## Acceptance
C:/Users/DELL/Desktop/3d_environment_drone_neural_network/.venv/Scripts/python -m pytest tests/test_drones.py -q (4 passed)

## DONE
DONE-D2 | files: src/drones/__init__.py,src/drones/importer.py,assets/drones/crazyflie-like-custom/drone.yaml,tests/test_drones.py | tests: 4/4 green | metrics: {import:ok}
