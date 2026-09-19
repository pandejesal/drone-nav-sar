# Architecture v2 — Customizable + Stable Drone SAR Stack (SAR-ONLY)

> Status: proposal for Sprint 4+ implementation. Single-task now (medkit A→B),
> multi-task heads later. SAR-ONLY primitives: `navigate_to / hover /
> drop_payload / return_home`. Geofence + altitude ceiling hardcoded in the
> control layer; MAVLink heartbeat + RC override always active.
>
> Grounding: builds on `src/sim/make_env` (isaac/airsim/gazebo + mock
> fallback, `info["mock_backend"]` explicit), `src/sim/mock_backend.py`
> (shared headless physics), `src/reconstruction/reconstruct.py`
> (photos → `.glb`/`.usd`), and the R1–R12 reuse map in `docs/RESEARCH.md`.

## 1. Modules

```text
src/
  reconstruction/   photos → mesh.glb/mesh.usd + collision hulls (R1–R4, CLI boundary)
  sim/              make_env(backend, mesh) → BaseDroneEnv; mock_backend default in CI (existing)
  drones/           NEW: config-driven drone import (URDF/SDF + datasheet) → QuadrotorParams
  rl/               NEW (Sprint 4): ppo_nav.py, policies.py, vec_env.py (N=8 seeded) on make_env
  control/          NEW: skill schema → slm_adapter → safety_filter → mavlink_bridge / ros2_bridge
  eval/             NEW: SPL, success rate, energy proxy, time; regression gates
```

`drones/`, `rl/`, `control/`, `eval/` are new packages; `sim/` and
`reconstruction/` keep their current public APIs unchanged (stability).

## 2. Custom drone import → validated Stable API

`assets/drones/<name>/drone.yaml` (versioned, see §6) is the single source of
truth — anyone with their own drone fills this in, no code changes:

```yaml
api_version: 2
name: crazyflie-like-custom
geometry: {arm_length_m: 0.046, mass_kg: 0.027}
propulsion: {max_thrust_N: 0.62, max_rpm: 25000, motor_tau_s: 0.02, thrust_unit: normalized}
inertia: {ixx: 1.4e-5, iyy: 1.4e-5, izz: 2.2e-5}
model_files: {urdf: model.urdf}   # or {sdf: model.sdf} — collision meshes for SDF/Gazebo path
```

`src/drones/importer.py`:

1. Parse YAML → validate ranges (mass > 0, thrust > weight margin,
   `thrust_unit == normalized` — **taboo**: reject raw-RPM configs with a
   clear error, never silently mix units).
2. Load URDF/SDF for geometry/collision only; dynamics come from the
   datasheet (avoids mesh-unit surprises).
3. Emit `QuadrotorParams` (existing `src/sim/drone_dynamics.py` type) with
   `motor_time_constant` and a `dt/tau` rate clamp `<= 1` enforced in `step`.
4. Return a `DroneSpec` with `spec_hash` recorded in every training run and
   eval report (traceability across custom drones).

Stable API guarantee: `make_env(backend, mesh, drone_spec)` keeps accepting
`SimConfig` or bare mesh-path string (current shorthand preserved); unknown
YAML keys warn, never break.

## 3. NN → prompts/commands → drone actions

The policy never emits raw text to motors. It emits a **structured skill
command**; a small SLM/LLM only paraphrases/parameterizes within schema:

```jsonc
// src/control/skill_schema.json (versioned, api_version: 2)
{ "skill": "navigate_to",
  "params": { "x": 2.1, "y": 1.0, "z": 1.5, "yaw_deg": 90, "speed_ms": 0.8 },
  "constraints": { "ceiling_m": 2.5, "geofence_xyz": [[0,4],[0,4],[0.2,2.5]], "timeout_s": 20 },
  "task": "medkit_AB", "task_embedding_id": 0 }
```

Pipeline (`src/control/`):

1. `policy_head.py` — NN outputs skill JSON (single-task now: always
   `navigate_to`→`hover`→`drop_payload`→`return_home` sequence for medkit A→B;
   later: `task_embedding_id` selects among multi-task heads).
2. `slm_adapter.py` — SLM/LLM fills `params` from observation + mission
   prompt (e.g. "deliver medkit to B, avoid obstacle"), constrained to schema
   (JSON-mode / grammar; schema validation rejects anything else).
3. `safety_filter.py` — hard checks: skill in SAR allowlist, target inside
   geofence, below ceiling, speed capped, RC override live. **Reject =
   hover in place.** No bypass path.
4. `mavlink_bridge.py` (R10) / `ros2_bridge.py` (R11) — translate the
   *filtered* skill to MAVLink/ROS 2 setpoints at fixed rate with heartbeat.

## 4. Single-task now, multi-task later

- **Now (Sprint 4–5):** one head, `task_embedding_id: 0` (`medkit_AB`).
  PPO (CleanRL pattern, R7) on 21-dim obs → 4 normalized thrusts, N=8 seeded
  mock vec-env, static room then obstacle curriculum.
- **Later (no re-architecture):** shared backbone + per-task heads selected
  by `task_embedding_id` (`hover_hold`, `inspect`, `deliver`, …); DreamerV3
  (R9) or Sample-Factory rollouts (R8) swap behind the same env + eval
  contract. Skill schema gains new `skill` values only via `api_version`
  bump with migration note.

## 5. Interfaces (frozen for v2)

| Interface | Signature / artifact | Owner |
|-----------|----------------------|-------|
| Env factory | `make_env(backend: str, mesh, drone_spec=None, seed) -> BaseDroneEnv`, `info["mock_backend"]` always set | `src/sim/` (existing) |
| Drone spec | `DroneSpec(api_version, params: QuadrotorParams, spec_hash)` from `drone.yaml` | `src/drones/` (new) |
| Skill command | `skill_schema.json` (schema above); skills ⊆ SAR allowlist | `src/control/` (new) |
| Safety verdict | `allow(skill) -> {ok: bool, reason: str}`; default-deny, hover on reject | `src/control/safety_filter.py` |
| Eval report | `{success_rate, SPL, energy_proxy, time_s, spec_hash, seed, git_sha}` | `src/eval/` (new) |
| Edge artifact | `policy.onnx` + `edge_report.json` (runtime check passed) | `src/control/edge_export.py` (R12) |

## 6. Stability contract

1. **Versioned configs:** `drone.yaml` and `skill_schema.json` carry
   `api_version`; loaders reject unknown majors, warn on unknown minor keys.
2. **Seeded vec-env:** `vec_env.py` N=8, per-env seeds derived from master
   seed (`seed+i`); determinism smoke test in CI.
3. **Regression tests (CI, mock backend only):** hover holds position,
   A→B reaches `goal_tolerance`, safety filter rejects out-of-geofence and
   non-allowlist skills, ONNX export round-trips one inference.
4. **ONNX export:** every promoted checkpoint ships `policy.onnx`
   (opset pinned) + `edge_report.json`; deploy targets Jetson/RPi run the
   same file the eval gate passed.
5. **Taboo guards (from evolution lessons):** normalized thrust only,
   `dt/tau <= 1`, `bpy` inside Blender only, case-insensitive image counting
   on Windows, `make_env` mock fallback + `info[mock_backend]` never broken —
   each with a named regression test.
6. **Docs-only rule for R1:** this survey touches `docs/` only; no `src/`
   changes.
