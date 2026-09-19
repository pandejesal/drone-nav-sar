# API Reference — DroneNav-SAR (SAR-only)

SAR skills closed vocabulary: `navigate_to / hover / drop_payload (aid kits) /
return_home`. Sphinx + autodoc source below; this page is the human-readable
index with identical content.

## `src.sim` — environments

```python
from src.sim import make_env
env = make_env(backend="mock", mesh="path/to/space.glb")
# backend: "mock" | "isaac" | "airsim" | "gazebo"
# info["mock_backend"] is True iff the mock physics path is active.
```

- `src/sim/base_env.py` — `BaseDroneEnv` (reset/step/render, `DroneState`)
- `src/sim/mock_backend.py` — headless seeded physics (CI default)
- `src/sim/isaac_sim_env.py`, `airsim_env.py`, `gazebo_env.py` — HiFi backends
- `src/sim/drone_dynamics.py` — `QuadrotorParams`, `linearize`
- `src/sim/domain_randomization.py` — `DomainRandomizer(seed)`

## `src.drones` — custom drone import

```python
from src.drones.importer import import_drone
spec = import_drone("assets/drones/<name>/drone.yaml")  # -> DroneSpec.spec_hash
```

`assets/drones/<name>/drone.yaml` (`api_version: 2`): geometry, propulsion
(`thrust_unit: normalized` only — raw-RPM rejected), inertia, URDF/SDF files.

## `src.reconstruction` — photos to mesh

- `reconstruct.py` — `reconstruct(space_dir)`, `verify_mesh(path)`
- `colmap_pipeline.py`, `nerfstudio_pipeline.py`, `blender_cleanup.py`

## `src.rl` — training

- `ppo_nav.py` — PPO entry point (`--quick`, `--curriculum`)
- `policies.py` — actor-critic nets; `vec_env.py` — N=8 seeded vec envs
- `hierarchical.py` — `HierarchicalPolicy` (global + local)
- `global_planner.py`, `local_policy.py`, `multi_drone.py` (`MultiDroneTeam`)

## `src.control` — language to motors

```text
command -> language_to_task.parse_command -> language_validator (SAR-only gate)
        -> language_interface.embed_task -> slm_adapter -> safety_filter
        -> mavlink_bridge / ros2_bridge
```

- `language_to_task.py` — `parse_command(str) -> Task`
- `language_validator.py` — rejects non-SAR skills (weapons/targeting/kinetic/surveillance)
- `mission_planner.py` — `MissionPlanner(policy_path).run_mission(env, commands)`
- `safety_filter.py` — geofence, altitude ceiling, payload, separation
- `edge_export.py` — `export_onnx(pt, onnx)`, `quantize_int8(...)`
- `onnx_validator.py`, `yolo_detector.py`, `comms.py`

## `src.eval` — metrics and gates

- `metrics.py` — success rate, SPL, energy proxy, episode time
- `sim2real.py` — mock-to-real gap; `edge_benchmark.py` — on-device latency

## `scripts`

| Script | Purpose |
|---|---|
| `dataset_build.py` | photos -> space bundle (`--quick` for CI stubs) |
| `dataset_download.py` | fetch Hub dataset splits |
| `eval_all.py` | full eval gate over policy + spaces |
| `test_sim_envs.py` | backend smoke test |
| `precommit.sh` | black + isort + flake8 + mypy + pytest -q |

## Sphinx stub

```python
# docs/conf.py (when publishing to ReadTheDocs)
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
```

`automodule` pages for `src.sim`, `src.drones`, `src.reconstruction`,
`src.rl`, `src.control`, `src.eval` mirror the sections above.
