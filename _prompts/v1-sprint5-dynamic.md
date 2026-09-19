# Sprint 5 — Dynamic Obstacles (SAR-only)

## Goal
Add moving obstacles (people, doors) + wind to the mock backend. Train PPO with curriculum to handle dynamic environments. Target: maintain >50% SR with 3 moving obstacles + wind.

## Scope (modify ONLY these)
- src/sim/mock_backend.py: ADD `DynamicObstacle` class + `spawn_obstacles()` + `step_obstacles()` + collision check in step()
- src/sim/domain_randomization.py: ADD obstacle randomization (count, speed, size) + wind gusts
- src/rl/ppo_nav.py: ADD `--dynamic-obstacles` flag, pass obstacle config to env
- tests/test_sim.py: ADD 2 tests — obstacle collision detection, wind affects trajectory

## Dynamic obstacles spec
- People: cylinders (radius 0.3m, height 1.8m), random walk at 0.5-1.0 m/s, bounce off walls
- Doors: rectangles (1.0x2.1m), open/close cycle (period 10-30s), block passage when closed
- Wind: OU process already exists; add gust events (spike 3-5 m/s for 1-2s)

## Curriculum integration
- Level 0: static only (current)
- Level 1: 1 person, no wind
- Level 2: 2 people + light wind
- Level 3: 3 people + doors + gusts (target)

## Acceptance
.venv/Scripts/python -m pytest tests/test_sim.py -q (17 passed)
.venv/Scripts/python -m src.rl.ppo_nav --episodes 1000 --seed 0 --curriculum --dynamic-obstacles --save-path policy_sprint5.pt → sr>0.5 at level 3

## DONE
DONE-5 | files: src/sim/mock_backend.py,src/sim/domain_randomization.py,src/rl/ppo_nav.py,tests/test_sim.py | tests: 17/17 green | metrics: {sr_level3:...}