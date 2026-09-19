# Tutorial 2 — Simulation: Mock, Isaac Sim, AirSim, Gazebo (~30 min, SAR-only)

Run the SAR navigation env on any backend. The mock backend is the default in
CI (headless, seeded, no GPU); Isaac/AirSim/Gazebo add fidelity when available.

## Backend matrix

| Backend | Module | Needs |
|---|---|---|
| mock (default) | `src/sim/mock_backend.py` | nothing |
| isaac | `src/sim/isaac_sim_env.py` | Isaac Sim + GPU |
| airsim | `src/sim/airsim_env.py` | AirSim build |
| gazebo | `src/sim/gazebo_env.py` | Gazebo + ROS 2 |

## Steps

### 1. Mock backend smoke test

```bash
python scripts/test_sim_envs.py --backend mock
```

### 2. Make any backend explicitly

```python
from src.sim import make_env

env = make_env(backend="mock", mesh="data/dataset/spaces/space_000/mesh/space.glb")
print(env.info.get("mock_backend"))  # True when on the mock path — must stay explicit
obs = env.reset()
```

`make_env(backend, mesh)` returns a `BaseDroneEnv`; unknown backends raise
instead of silently falling back (the `info["mock_backend"]` flag tells you
which physics you actually got).

### 3. High-fidelity backend (example: AirSim)

```python
env = make_env(backend="airsim", mesh="data/dataset/spaces/space_000/mesh/space.glb")
```

Same `BaseDroneEnv` API — policies trained on mock transfer without code changes.

### 4. Domain randomization (sim2real)

```python
from src.sim.domain_randomization import DomainRandomizer
dr = DomainRandomizer(seed=0)
```

Randomizes lighting, textures, mass/inertia, wind, and sensor noise. The eval
gate (`src/eval/sim2real.py`) measures the mock-to-real gap; see Tutorial 3
for training with it on.

## Safety invariants (all backends)

- Geofence + altitude ceiling enforced in the control layer, not the policy
- MAVLink heartbeat + RC override always active on real hardware
- SAR skills only: `navigate_to / hover / drop_payload (aid kits) / return_home`

## Next

Tutorial 3 — train a PPO policy on these envs.
