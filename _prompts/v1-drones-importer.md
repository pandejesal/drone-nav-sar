# Sprint Drones v1 — Custom Drone Importer (SAR-only)

## Goal
Anyone-brings-own-drone: drone.yaml + URDF/SDF -> validated QuadrotorParams + spec_hash. Per docs/ARCHITECTURE-v2.md section 2 + docs/RESEARCH-v2.md R13-R15,R19-R20.

## Scope (create ONLY)
- src/drones/__init__.py
- src/drones/importer.py (parse drone.yaml api_version 2, validate mass>0 thrust>weight margin thrust_unit==normalized reject raw-RPM with clear error, load URDF via yourdfpy/urdfpy if present else geometry-only fallback from yaml, SDF via sdformat_urdf if present else warn, emit QuadrotorParams + DroneSpec(spec_hash sha1 of canonical yaml), make_env-compatible)
- assets/drones/crazyflie-like-custom/drone.yaml + model.urdf (minimal valid example from hector template, arm 0.046 mass 0.027 max_thrust 0.62)
- tests/test_drones.py (4 tests: valid yaml imports + spec_hash stable; raw-RPM rejected; mass<=0 rejected; bare URDF geometry loads or graceful fallback)

## Ground
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/drone_dynamics.py (QuadrotorParams type)
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/docs/ARCHITECTURE-v2.md section 2
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/docs/RESEARCH-v2.md R13-R15

## TABOO
- Normalized thrust only, dt/tau<=1 enforced downstream, no bpy, forward-slash paths, optional deps (yourdfpy/urdfpy) try/except never hard-require for CI

## Safety
SAR-ONLY. Geometry/dynamics only, no payload weapon logic.

## Acceptance
C:/Users/DELL/Desktop/3d_environment_drone_neural_network/.venv/Scripts/python -m pytest tests/test_drones.py -q (4 passed)

## DONE
DONE-D1 | files: src/drones/__init__.py,src/drones/importer.py,assets/drones/crazyflie-like-custom/drone.yaml,assets/drones/crazyflie-like-custom/model.urdf,tests/test_drones.py | tests: x/y green | metrics: {import:ok}
