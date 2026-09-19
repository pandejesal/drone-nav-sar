# Deep Research v2 — SLM Prompt Control + Custom Drone Import + Edge Deploy (SAR-ONLY)

## Goal
Go deeper than docs/RESEARCH.md (v1: 10 papers, 12 repos). Focus on 3 gaps for customizable stable system: (1) NN->JSON skill -> SLM paraphrase -> safety filter -> MAVLink, (2) anyone-brings-own-drone URDF/SDF + datasheet importer, (3) ONNX edge deploy Jetson/RPi for medkit A->B.

## Scope (read first)
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/docs/RESEARCH.md
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/docs/ARCHITECTURE-v2.md
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/drone_dynamics.py

## Tasks
1. Papers/addenda (5-8 NEW, not repeating v1, 1-line why + link):
   - JSON-mode / grammar-constrained LLM output for robot skills
   - Small LM onboard (Phi, Gemma, TinyLlama) for paraphrase/parameterize within schema
   - URDF/SDF system ID + thrust-curve calibration for custom quads
   - Vision-language nav with safety filter / shield (formal or runtime)
   - ONNX/TensorRT edge inference for policies
2. Repos/code to copy (5-8 NEW, license + exact files):
   - skill schema validation (JSON schema, pydantic) examples
   - URDF parsers (urdfpy, yourdfpy, sdformat) + thrust-stand calibration scripts
   - MAVSDK offboard + geofence examples, ROS2 safety filter nodes
   - torch.onnx.export + onnxruntime Jetson/RPi samples
3. Write docs/RESEARCH-v2.md with tables + integration notes into THIS repo (which file in src/drones/, src/control/, src/eval/ gets what, cost low/med/high)
4. Do NOT touch src/, only docs/

## TABOO
- Normalized thrust only, dt/tau <=1, bpy guarded, Windows glob case-insensitive, keep make_env mock fallback

## Safety
SAR-ONLY medkit. If source covers weapons, reuse nav-only, refuse weapon part.

## Acceptance
- docs/RESEARCH-v2.md exists, 5+ papers + 5+ repos, links live
- Final line ONLY:
DONE-R2 | files: docs/RESEARCH-v2.md | tests: n/a | metrics: {papers:N,repos:N}
