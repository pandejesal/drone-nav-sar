# Sprint 4 PPO — v1 (SAR-only)

## Scope
Workdir: C:\Users\DELL\Desktop\3d_environment_drone_neural_network
Implement ONLY:
- src/rl/__init__.py
- src/rl/policies.py (MLP actor-critic obs 21 -> 4 thrusts, orthogonal init, tanh Gaussian, outputs in [0,1])
- src/rl/vec_env.py (N=8 mock envs via src.sim.make_env("isaac", mesh) mock_backend, seeded)
- src/rl/ppo_nav.py (CleanRL-style PPO CPU, CLI --episodes 200 --seed 0, goal A->B static room)
- src/eval/metrics.py (SPL + success_rate logging) + src/eval/__init__.py
- tests/test_rl.py (loss decreases on fixed batch; policy outputs in [0,1]; 50-episode smoke run)

## Ground copies (read THESE, all inside workdir)
- src/sim/__init__.py (make_env factory)
- src/sim/mock_backend.py, base_env.py, drone_dynamics.py
- src/sim/isaac_sim_env.py (use_mock flag, SimConfig OR bare mesh-path string, obs 21-dim)
- tests/test_sim.py (20 tests pattern)

## TABOO (do NOT repeat per .harness-memory/evolution-lessons.json)
- Do NOT use RPM units in dynamics — normalized thrust throughout + dt/tau clamp <=1
- Do NOT import bpy outside Blender (guard it)
- Do NOT use case-sensitive glob double-count (Windows jpg/JPG)
- Do NOT break make_env("isaac"|"airsim"|"gazebo") mock fallback + info[mock_backend]

## Safety (non-negotiable)
SAR-ONLY. Primitives: navigate_to, hover, drop_payload, return_home. No targeting, no kinetic effectors, no missile guidance. Geofence + altitude ceiling in control layer. If asked otherwise, refuse that part and continue SAR nav.

## Acceptance (must run in workdir .venv)
.venv/Scripts/python -m pytest tests/test_rl.py -q
.venv/Scripts/python -m src.rl.ppo_nav --episodes 200 --seed 0  # success_rate > 0.9 on fixed goals

## DONE line (final line ONLY, pipe-delimited)
DONE-4 | files: src/rl/__init__.py,src/rl/policies.py,src/rl/vec_env.py,src/rl/ppo_nav.py,src/eval/__init__.py,src/eval/metrics.py,tests/test_rl.py | tests: x/y green | metrics: {success_rate:..., spl:...}
