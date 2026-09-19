# DroneNav-SAR — 3D Indoor Search-and-Rescue Drone Navigation (SAR-only)

![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/OWNER/REPO/actions/workflows/release.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License CC-BY-4.0](https://img.shields.io/badge/license-CC--BY--4.0-green)

Phone photos -> 3D mesh -> simulator -> PPO policy -> language commands ->
edge deployment -> SAR mission. SAR-only: `navigate_to / hover /
drop_payload (aid kits) / return_home`. No weapons, targeting, kinetic, or
surveillance payloads — PRs violating this are closed.

## Quickstart

```bash
pip install -e .[dev]
bash scripts/precommit.sh            # black, isort, flake8, mypy, pytest -q

python scripts/dataset_build.py --spaces 5 --quick   # smoke spaces
python -m src.rl.ppo_nav --quick --spaces 2 --eps 5  # smoke training
python scripts/eval_all.py --policy checkpoints/policy.pt --spaces 2
pytest -q --tb=short
```

## Tutorials (~30 min each)

1. [Reconstruction](docs/tutorial_1_reconstruction.md) — photos -> textured `.glb`
2. [Simulation](docs/tutorial_2_simulation.md) — mock / Isaac / AirSim / Gazebo
3. [RL Training](docs/tutorial_3_rl_training.md) — PPO + curriculum + hierarchical + multi-drone
4. [Language](docs/tutorial_4_language.md) — SAR commands -> task embeddings
5. [Edge](docs/tutorial_5_edge.md) — ONNX -> TensorRT (Jetson) / ORT INT8 (RPi CM4)
6. [SAR Mission](docs/tutorial_6_sar_mission.md) — 4-room building capstone

Plus [API reference](docs/api_reference.md) and
[architecture](docs/architecture.md) (system diagram, data flows, safety
invariants). Contributing: [docs/contributing.md](docs/contributing.md).

## CI / Release

- CI (`.github/workflows/ci.yml`): matrix win/linux x py3.11/3.12,
  precommit + pytest (1x rerun flake suppression) + `mypy --strict src/`,
  artifact upload. Target: zero flakes, < 15 min.
- Release (`.github/workflows/release.yml`): `v*` tag -> wheels + ONNX +
  TensorRT manifest -> Hugging Face Hub (dataset 100+ spaces, 10+ policy
  checkpoints, demo Space).

## Community

- Discord: `#drone-nav-sar` (support, PR reviews, issue triage)
- GitHub Discussions: FAQ, show-and-tell, feature requests
- Office hours: first Tuesday monthly, 19:00 UTC
- Hugging Face Hub: dataset + policies + demo Space (see Release above)
