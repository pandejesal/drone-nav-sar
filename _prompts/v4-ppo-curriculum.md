# Sprint 4 PPO — Curriculum + Stronger Shaping (SAR-only)

## Why
- Fixed goal + shaped reward (vel_toward) gave sr=0.002 at 500 eps — policy explores but doesn't converge.
- Hand-crafted policy reaches goal in ~30 steps (proven). PPO needs curriculum: start easy, add difficulty.

## Scope (modify ONLY these)
- src/sim/mock_backend.py: ADD `curriculum_level` param (0..3), scale reward terms + goal tolerance
- src/rl/ppo_nav.py: ADD `--curriculum` flag, auto-advance when sr>0.8 over 50 eps, log level
- tests/test_rl.py: ADD 1 test — curriculum advances level when sr threshold met

## Curriculum levels
| Level | Goal tolerance | Dist weight | Vel weight | Effort weight | Alive | Max steps | Note |
|-------|----------------|-------------|------------|---------------|-------|-----------|------|
| 0 (easy) | 1.5m | -0.3 | +0.5 | -0.01 | +0.1 | 300 | Hover near goal counts |
| 1 | 1.0m | -0.4 | +0.4 | -0.015 | +0.07 | 400 | Must approach |
| 2 | 0.7m | -0.5 | +0.3 | -0.02 | +0.05 | 500 | Precision |
| 3 (target) | 0.5m | -0.5 | +0.3 | -0.02 | +0.05 | 500 | Sprint-4 spec |

## Acceptance
.venv/Scripts/python -m pytest tests/test_rl.py -q (6 passed)
.venv/Scripts/python -m src.rl.ppo_nav --episodes 2000 --seed 0 --curriculum --save-path policy_sprint4_curriculum.pt → sr>0.5 by level 3

## DONE
DONE-4C | files: src/sim/mock_backend.py,src/rl/ppo_nav.py,tests/test_rl.py | tests: 6/6 green | metrics: {curriculum:advances}