# Sprint 4 v3 — PPO Trainer + Eval (SAR-only)

## Why v3
- v2 ACCEPT: policies.py + vec_env.py + test_rl.py, 3 passed in 45.91s. Next: trainer + eval to close Sprint 4 single-task loop.

## Scope (create ONLY)
- src/rl/ppo_nav.py (CleanRL-style PPO CPU-only: rollout DroneVecEnv N=8, GAE, clipped loss, value loss, entropy, grad clip, save policy.pt, CLI --episodes 200 --seed 0 --mesh mock.glb, logs success_rate + SPL via src/eval)
- src/eval/__init__.py + src/eval/metrics.py (success_rate, SPL per Anderson/Habitat: SPL = success * optimal_len/actual_len, energy proxy sum(thrust^2)*dt, EvalReport {success_rate,SPL,energy,time,spec_hash,seed})
- Extend tests/test_rl.py ADD 2 tests (keep existing 3 green): ppo loss decreases on fixed batch (1 update step); eval SPL==1.0 on straight-line optimal demo
- Do NOT modify policies.py/vec_env.py except imports if needed

## Ground
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/rl/policies.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/rl/vec_env.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/docs/ARCHITECTURE-v2.md sections 4-6

## TABOO
- Normalized thrust [0,1], dt/tau<=1, torch CPU, keep make_env mock fallback, Windows paths forward-slash

## Safety
SAR-ONLY navigate_to/hover/drop_payload/return_home. No weapons.

## Acceptance
C:/Users/DELL/Desktop/3d_environment_drone_neural_network/.venv/Scripts/python -m pytest tests/test_rl.py -q (5 passed)
C:/Users/DELL/Desktop/3d_environment_drone_neural_network/.venv/Scripts/python -m src.rl.ppo_nav --episodes 10 --seed 0 (smoke, prints success_rate)

## DONE
DONE-4v3 | files: src/rl/ppo_nav.py,src/eval/__init__.py,src/eval/metrics.py,tests/test_rl.py | tests: x/y green | metrics: {success_rate:...,spl:...}
