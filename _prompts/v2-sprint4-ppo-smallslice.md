# Sprint 4 PPO Small Slice — v2 (SAR-only)

## Why v2 (evolution from v1)
- v1 20260916_205047_abfc58 stalled: backslash dir mangled (C:UsersDELL...), forward-slash retry never returned, zero files. Lesson: use forward-slash dir ONLY, small slice to fit free-tier timeout.
- This v2 does policies + vec_env + smoke test ONLY. No full 200-ep PPO run yet. v3 will do ppo_nav + eval.

## Scope (create ONLY these 3 files)
- src/rl/__init__.py
- src/rl/policies.py (MLP actor-critic obs 21 -> 4 thrusts, orthogonal init, tanh Gaussian, outputs [0,1], torch CPU, no GPU deps)
- src/rl/vec_env.py (N=8 mock envs via src.sim.make_env("isaac", mesh, use_mock=True) seeded master seed+i, gym API reset/step, auto-handle SimConfig OR bare mesh-path string)
- tests/test_rl.py (3 tests ONLY: policy outputs in [0,1] shape (B,4); vec_env reset/step shapes seeded determinism; 10-episode smoke run reaches terminal without crash)

## Ground copies (read THESE first, inside workdir)
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/__init__.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/mock_backend.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/base_env.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/src/sim/drone_dynamics.py
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/tests/test_sim.py

## TABOO (from .harness-memory/evolution-lessons.json)
- Normalized thrust [0,1] throughout, never raw RPM mixup, dt/tau clamp <=1
- No bpy import, guard if needed
- No case-sensitive glob double-count
- Do not break make_env mock fallback + info[mock_backend]
- Keep torch CPU-only, no CUDA hard dep for CI

## Safety
SAR-ONLY. navigate_to/hover/drop_payload/return_home. No targeting, no kinetic, no missile guidance.

## Acceptance (run in workdir .venv)
C:/Users/DELL/Desktop/3d_environment_drone_neural_network/.venv/Scripts/python -m pytest tests/test_rl.py -q

## DONE line (final line ONLY)
DONE-4v2 | files: src/rl/__init__.py,src/rl/policies.py,src/rl/vec_env.py,tests/test_rl.py | tests: x/y green | metrics: {smoke:pass}
