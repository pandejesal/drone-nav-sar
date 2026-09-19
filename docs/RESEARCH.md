# Research Survey — Customizable Drone SAR Stack (SAR-ONLY) — v1

> Scope: photos → 3D env → custom drone import → single/multi-task tuning →
> deployable onboard software where the NN emits structured prompts/commands
> that an SLM/LLM turns into drone actions. Target task: medkit A→B with
> obstacles in a scanned room. Must stay customizable (any drone + any photos)
> and stable (versioned configs, seeded envs, regression tests, ONNX export).
>
> **Safety (non-negotiable, SAR-ONLY):** medkit delivery with primitives
> `navigate_to / hover / drop_payload / return_home` only. No targeting, no
> kinetic effectors, no missile/projectile weapon guidance. Where a cited
> source covers weapons or dual-use control, this survey reuses the
> **navigation-only** component and refuses the weapon part.
>
> **Repo grounding (read first):** `prompt.md` (Sprints 1–3 done, Sprint 4 PPO
> next), `src/sim/__init__.py` (`make_env` factory over 3 backends),
> `src/sim/mock_backend.py` (shared headless physics + synthetic camera,
> `info["mock_backend"]` explicit), `src/reconstruction/reconstruct.py`
> (COLMAP/Nerfstudio auto-select + BlenderProc cleanup).
>
> **Taboo constraints honored:** normalized thrust throughout (never raw RPM
> mixups), `dt/tau` rate clamp `<= 1`, `bpy` import only inside Blender
> (guarded), case-insensitive image counting on Windows (no glob
> double-count), never break `make_env` mock fallback or `info[mock_backend]`.

## 1. Papers (10)

### 1a. 3D from photos → sim meshes

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P1 | Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields" (ECCV 2020) | Baseline for photo→3D appearance; justifies the Nerfstudio branch of `reconstruct.py`. | https://arxiv.org/abs/2003.08934 |
| P2 | Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023) | Fast splatfacto-style capture→mesh path for rooms scanned with few phone photos. | https://arxiv.org/abs/2308.04079 |
| P3 | Schönberger & Frahm, "Structure-from-Motion Revisited" / COLMAP (CVPR 2016) | Sparse/dense SfM that produces the textured mesh our Blender cleanup consumes. | https://arxiv.org/abs/1606.03947 |

### 1b. Sim2Real drone nav

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P4 | Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (IROS 2017) | Theory behind `src/sim/domain_randomization.py`: randomize lighting/texture/physics so mock-trained policies transfer. | https://arxiv.org/abs/1703.06907 |
| P5 | Kaufmann et al., "Deep Drone Racing: Learning Agile Flight in Dynamic Environments" (CoRL 2018) | Shows learned quadrotor point-nav with system ID closing the sim2real gap — template for our Crazyflie demo (Sprint 6). | https://arxiv.org/abs/1808.05000 |

### 1c. RL / IL for point-nav A→B with obstacles

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P6 | Schulman et al., "Proximal Policy Optimization Algorithms" (2017) | PPO is the Sprint 4 trainer (`src/rl/ppo_nav.py`): stable on the 21-dim obs → 4-thrust action space. | https://arxiv.org/abs/1707.06347 |
| P7 | Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023) | Sample-efficient world-model alternative for later sprints when mock steps get expensive. | https://arxiv.org/abs/2301.04104 |

### 1d. Language-conditioned control (NN → prompts → actions)

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P8 | Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances" (SayCan, 2022) | Say-and-execute pattern: LLM proposes skills, affordance/safety filter grounds them — our JSON skill → safety filter design. | https://arxiv.org/abs/2204.01691 |
| P9 | Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023) | Vision-language → action-token precedent for mapping NN outputs + SLM paraphrase into waypoint/skill commands. | https://arxiv.org/abs/2307.15818 |

### 1e. Eval

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P10 | Anderson et al., "On Evaluation of Embodied Navigation Agents" (2018) + Savva et al., Habitat (2019) | Defines SPL / success rate (our `src/eval/` contract) and the PointNav A→B benchmark we clone for medkit delivery. | https://arxiv.org/abs/1807.06757 |

Energy (supplemental, no extra paper row): log cumulative `sum(thrust²)·dt`
proxy in `src/eval/` per flight — P10's SPL + success rate stay primary.

## 2. Repos (12)

Conventions: **Reuse** = exact files/modules to copy or wrap. **Cost** =
integration effort into THIS repo (low = wrap in a day, med = adapter +
tests, high = new service + SITL/hardware loop).

| # | Repo | License | What to copy (exact files/modules) | Integration cost |
|---|------|---------|------------------------------------|------------------|
| R1 | COLMAP — https://github.com/colmap/colmap | BSD-3-Clause | CLI binaries (`colmap automatic_reconstructor`); wrap via `src/reconstruction/colmap_pipeline.py`. Already the `run_colmap()` path. | Low (already wired; keep subprocess wrapper, never link internals) |
| R2 | OpenMVS — https://github.com/cdcseacave/openMVS | AGPL-3.0 | `DensifyPointCloud` / `ReconstructMesh` / `TextureMesh` binaries for dense→mesh when COLMAP sparse is thin. | Med (adds AGPL binaries to docker `colmap` stage only; keep behind CLI boundary so repo code stays MIT) |
| R3 | Nerfstudio — https://github.com/nerfstudio-project/nerfstudio | Apache-2.0 | `ns-train splatfacto` + `ns-export poisson/tsdf` for appearance-first rooms; wrap via `src/reconstruction/nerfstudio_pipeline.py`. | Med (heavy deps; isolate in optional docker profile) |
| R4 | BlenderProc — https://github.com/DLR-RM/BlenderProc | GPL-3.0 | Procedural cleanup/randomization recipes; port pattern into `src/reconstruction/blender_cleanup.py` (decimate, UV unwrap, PBR bake, collision hulls, .glb/.usd export). | Med (keep `bpy` import inside Blender only — taboo guard) |
| R5 | Gazebo Harmonic (gz-sim) — https://github.com/gazebosim/gz-sim | Apache-2.0 | SDF worlds + physics plugin pattern; real-backend hook in `src/sim/gazebo_env.py` + PX4 SITL bridge. | High (needs ROS 2 + PX4 SITL running; mock fallback stays default in CI) |
| R6 | PX4-Autopilot — https://github.com/PX4/PX4-Autopilot | BSD-3-Clause | SITL target (`make px4_sitl gz_x500`) as the real-flight reference our mock dynamics must match via system ID. | High (SITL loop; navigation-only reuse — refuse any weaponized airframe/mission reuse) |
| R7 | CleanRL (PPO) — https://github.com/vwxyzjn/cleanrl | MIT | `cleanrl/ppo.py` + `ppo_atari.py` vec-env pattern → `src/rl/ppo_nav.py`, `src/rl/vec_env.py` (N=8 seeded), `src/rl/policies.py` (MLP actor-critic, orthogonal init). | Low (single-file PPO ports directly onto `make_env`) |
| R8 | Sample-Factory — https://github.com/alex-petrenko/sample-factory | MIT | High-throughput async rollout (` Enjoy / train` split) for Sprint 5 dynamic-obstacle curriculum when N=8 saturates. | Med (adopt rollout API only; keep our env interface) |
| R9 | DreamerV3 — https://github.com/danijar/dreamerv3 | MIT | `dreamerv3/agent.py` world-model trainer as Sprint 8 long-horizon option behind the same `vec_env` + eval contract. | Med (JAX dep; gate behind optional extras) |
| R10 | MAVSDK-Python — https://github.com/mavlink/MAVSDK-Python | BSD-3-Clause | `System`, `action.goto_location()`, `telemetry` plugins → `src/control/mavlink_bridge.py` executing **only** SAR skills (`navigate_to/hover/drop_payload/return_home`). | Low (thin bridge + geofence/ceiling filter in front) |
| R11 | ROS 2 rclpy — https://github.com/ros2/rclpy | Apache-2.0 | Node/executor + `sensor_msgs`/`geometry_msgs` subscription pattern → `src/control/ros2_bridge.py` alternative transport to the same skill schema. | Med (needs ROS 2 install; mock transport in CI) |
| R12 | ONNX Runtime — https://github.com/microsoft/onnxruntime | MIT | `torch.onnx.export` + `onnxruntime.InferenceSession` for Jetson/RPi edge deploy (`src/control/edge_export.py`); TensorRT EP optional on Jetson. | Low (export + runtime check in regression tests) |

Notes on dropped-but-considered: AirSim upstream
(https://github.com/microsoft/AirSim, MIT, archived) — keep only as the
`airsim_env.py` client-hook pattern, recommend the community Colosseum fork
for new work; Isaac Sim (proprietary, free-of-charge) — keep as optional
`isaac_sim_env.py` hook, never a CI dependency.

## 3. How this maps to THIS repo (customizable + stable)

- **Custom photos in:** `reconstruct.py` auto-select stays (`auto` → COLMAP for
  >30 images/high quality, else Nerfstudio), then Blender cleanup → `.glb` +
  `.usd` + collision hulls. R1–R4 above are the only reconstruction
  dependencies; all behind subprocess/CLI boundaries.
- **Custom drone in:** config-driven import (URDF/SDF + datasheet: mass,
  thrust, inertia) validated against `QuadrotorParams`; see
  `docs/ARCHITECTURE-v2.md` §2. Normalized thrust `[0,1]` end-to-end (taboo).
- **Single-task now:** PPO point-nav medkit A→B on `make_env(..., use_mock)`
  N=8 seeded vec-env (R7). **Multi-task later:** shared backbone + task
  embeddings; DreamerV3 (R9) / Sample-Factory (R8) plug in behind the same
  env + eval contract.
- **Deployable onboard path:** NN → JSON skill → SLM paraphrase/parameterize
  → safety filter (geofence, ceiling, RC override) → MAVLink (R10) / ROS 2
  (R11); policy exported to ONNX (R12). Full contract in
  `docs/ARCHITECTURE-v2.md` §3–§5.
- **Eval:** SPL + success rate primary (P10), energy proxy + time secondary;
  regression gates in `docs/ARCHITECTURE-v2.md` §6.
