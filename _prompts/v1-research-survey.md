# Research Survey — Customizable Drone SAR Stack (SAR-ONLY) — v1

## Goal
Survey papers + repos for: photos -> 3D env -> import custom drone -> multi/single-task tuning -> deployable onboard software where NN outputs prompts/commands that SLM/LLM turns into drone actions. Target task: medkit A->B with obstacles in scanned room. Output must be customizable + stable for anyone with own drone + own photos.

## Scope (read THESE first, inside workdir)
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/prompt.md
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/__init__.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/mock_backend.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/reconstruction/reconstruct.py

## Tasks
1. Papers (6-10, with why-useful 1-line + link):
   - 3D from photos: NeRF / 3DGS / COLMAP / OpenMVS for sim meshes
   - Sim2Real drone nav: domain randomization, system ID
   - RL/IL for point-nav A->B with obstacles (PPO, SAC, DreamerV3)
   - Language-conditioned control: LLM/SLM -> waypoint/skill prompts -> low-level controller (say-and-execute, RT-2 style, MAVLink skill primitives)
   - Eval: SPL, success rate, energy
2. Repos (8-12, license + what to copy):
   - reconstruction: COLMAP, OpenMVS, Nerfstudio, BlenderProc
   - sim: Isaac Sim / AirSim / Gazebo Harmonic + PX4 SITL
   - RL: CleanRL (PPO), Sample-Factory, DreamerV3-JAX
   - control/deploy: MAVSDK-Python, ROS2, PX4, ONNX/TensorRT, Jetson/RPi examples
   - For each: exact files/modules to reuse, license, integration cost (low/med/high)
3. Architecture proposal for THIS repo (customizable + stable):
   - config-driven drone import (URDF/SDF + data sheet: mass, thrust, inertia) -> validated Stable API
   - NN outputs structured prompts/commands (JSON: skill, params, constraints) -> SLM/LLM -> safety filter (geofence, ceiling) -> MAVLink/ROS2
   - single-task now (medkit A->B), multi-task heads later (task embeddings)
   - stability: versioned configs, seeded vec_env N=8, regression tests, ONNX export
4. Write to: docs/RESEARCH.md (papers+repos table) + docs/ARCHITECTURE-v2.md (modules, interfaces, stability contract)

## TABOO (from .harness-memory/evolution-lessons.json)
- No RPM vs normalized thrust mixups, dt/tau clamp <=1
- No bpy import outside Blender, guard it
- No case-sensitive glob double-count on Windows
- Do not break make_env mock fallback + info[mock_backend]

## Safety (non-negotiable)
SAR-ONLY. Medkit delivery, navigate_to/hover/drop_payload/return_home. No targeting, no kinetic effectors, no missile/projectile weapon guidance. If source material covers weapons, cite navigation-only reuse and refuse weapon part.

## Acceptance
- docs/RESEARCH.md + docs/ARCHITECTURE-v2.md exist, tables complete, links live
- No code changes outside docs/
- Final line ONLY:
DONE-R1 | files: docs/RESEARCH.md,docs/ARCHITECTURE-v2.md | tests: n/a | metrics: {papers:N,repos:N}
