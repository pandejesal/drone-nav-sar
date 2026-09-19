# Sprint 13 — Hardening, CI/CD, Tutorials, Community Launch (SAR-only)

## Goal
Production hardening: CI/CD pipeline, flake suppression, tutorial series, community launch. Target: zero-flake CI, tutorial completion rate > 80%, community ready.

## Scope (modify ONLY these)
- .github/workflows/ci.yml: NEW — matrix (win/linux), pytest + flake8 + mypy, artifact upload, flake suppression
- .github/workflows/release.yml: NEW — tag push → build wheels, ONNX/TensorRT artifacts, Hugging Face Hub push
- scripts/precommit.sh: NEW — pre-commit hooks (black, isort, flake8, mypy, pytest -q)
- docs/tutorial_1_reconstruction.md: NEW — photos → 3D mesh (COLMAP + Nerfstudio + BlenderProc)
- docs/tutorial_2_simulation.md: NEW — Isaac Sim / AirSim / Gazebo + mock backend
- docs/tutorial_3_rl_training.md: NEW — PPO + curriculum + hierarchical + multi-drone
- docs/tutorial_4_language.md: NEW — SAR commands → task embeddings
- docs/tutorial_5_edge.md: NEW — ONNX → TensorRT (Jetson) / ONNX Runtime INT8 (RPi CM4)
- docs/tutorial_6_sar_mission.md: NEW — multi-room SAR with language commands
- docs/api_reference.md: NEW — full API docs (Sphinx + autodoc)
- docs/architecture.md: NEW — system diagram, data flows, safety invariants
- README.md: UPDATE — badges, quickstart, links to tutorials, community links

## CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e .[dev]
      - run: scripts/precommit.sh
      - run: pytest -q --tb=short
      - run: mypy --strict src/
```

## Tutorial Series (6 parts, ~30 min each)
1. **Reconstruction**: phone photos → COLMAP/Nerfstudio → BlenderProc → textured .glb
2. **Simulation**: mock backend + Isaac/AirSim/Gazebo + domain randomization
3. **RL Training**: PPO + curriculum + hierarchical + multi-drone + SAR mission
4. **Language**: SAR commands → task embeddings → hierarchical policy
5. **Edge**: ONNX → TensorRT (Jetson FP16) / ONNX Runtime INT8 (RPi CM4)
6. **SAR Mission**: 4-room building, language commands, med-kit drop, victim detection

## Community Launch
- Hugging Face Hub: dataset (100+ spaces), policies (10+ checkpoints), spaces (demo)
- Discord server: #drone-nav-sar (support, PR reviews, issue triage)
- GitHub Discussions: FAQ, show-and-tell, feature requests
- Monthly office hours (first Tuesday, 19:00 UTC)

## Acceptance
- CI: zero flakes for 30 days, < 15 min runtime, artifact upload works
- Tutorials: all 6 render without errors, runnable in < 30 min each
- Community: Discord invite link in README, GH Discussions active
- Release: v1.0.0 tag pushes wheels + ONNX + TensorRT manifest to HF Hub

## DONE
DONE-13 | files: .github/workflows/*,scripts/precommit.sh,docs/*.md,README.md | CI: green 30 days | tutorials: 6/6 | community: launched