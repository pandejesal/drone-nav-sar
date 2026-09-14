# DroneNav-SAR: Autonomous Drone Navigation for Search & Rescue
## Year-Long AI-Orchestrated Project | Hermes /loop + OpenCode + Kilo Code

---

## VISION
Build a **fully open-source, free-to-run** pipeline:
1. **Scan** any indoor/outdoor space (phone LiDAR, photogrammetry, RGB-D) → 3D mesh
2. **Import** into Blender → clean, optimize, add physics/collision
3. **Simulate** in Isaac Sim / AirSim / Gazebo with drone dynamics
4. **Train** RL/IL policies (PPO, SAC, DreamerV3) for multi-task navigation
5. **Deploy** to real drone (PX4/ArduPilot) via MAVLink — zero-cost APIs only

**Core constraint:** $0 runtime cost. Free-tier LLMs (OpenCode Zen), free sims, open datasets.

---

## TECH STACK (all free/opensource)
| Layer | Choice | Why |
|-------|--------|-----|
| 3D Reconstruction | COLMAP + OpenMVS, or Nerfstudio (Instant-NGP) | Photogrammetry/NeRF from photos |
| Blender Pipeline | bpy headless + BlenderProc | Proc-gen, physics, domain randomization |
| Simulator | **Isaac Sim** (free) + **AirSim** (free) + **Gazebo Harmonic** | GPU-accel, ROS2, PX4 SITL |
| RL Framework | **CleanRL** + **Sample-Factory** + **DreamerV3 (JAX)** | Fast, scalable, free |
| Drone Control | **MAVSDK-Python** + **ROS2** | Hardware-agnostic |
| Orchestration | **Hermes /loop** → OpenCode / Kilo Code | 24×7 autonomous evolution |
| Models | OpenCode Zen free (nemotron-3-ultra-free, muse-spark-1.3-contributor-free) | $0 inference |

---

## REPO STRUCTURE
```
drone-nav-sar/
├── src/
│   ├── reconstruction/     # COLMAP/Nerfstudio wrappers
│   ├── blender_pipeline/   # bpy scripts: clean, randomize, export USD/glTF
│   ├── sim/                # Isaac Sim / AirSim / Gazebo envs + drone dynamics
│   ├── rl/                 # CleanRL/DreamerV3 trainers, multi-task heads
│   ├── control/            # MAVSDK/ROS2 bridge, safety filters
│   └── eval/               # Metrics: SPL, success rate, energy, time
├── assets/
│   ├── scans/              # Raw input (photos, point clouds)
│   ├── meshes/             # Cleaned .glb/.usd
│   ├── drones/             # SDF/URDF models (Crazyflie, DJI, custom)
│   └── textures/           # Domain randomization sets
├── models/                 # Checkpoints (git-lfs or DVC)
├── prompts/                # Hermes /loop prompt.md evolves here
├── docs/                   # Architecture, API, tutorials
├── tests/                  # Unit + integration (CI)
├── .harness-memory/        # evolution-lessons.json
├── 04-Prompt-Queues/Ecosystem/InterHarness/  # Inter-agent bus
└── docker/                 # Reproducible dev container
```

---

## YEAR-1 MILESTONES (13 x 4-week sprints)

| Sprint | Focus | Deliverable |
|--------|-------|-------------|
| 1 | **Reconstruction pipeline** | `photos → textured .glb` CLI, BlenderProc cleanup |
| 2 | **Sim integration** | Isaac Sim + AirSim envs, PX4 SITL, domain rand |
| 3 | **Single-task RL** | PPO nav A→B in static room, >90% SPL |
| 4 | **Multi-task heads** | Shared backbone + task embeddings (nav, hover, inspect, deliver) |
| 5 | **Dynamic obstacles** | Moving people, doors, wind — curriculum |
| 6 | **Sim2Real gap** | Domain rand + system ID → real Crazyflie demo |
| 7 | **SAR scenario** | Multi-room building, victim detection (YOLO), med-kit drop |
| 8 | **Long-horizon** | Hierarchical policy: global planner + local RL |
| 9 | **Multi-drone** | Decentralized comms, collision avoidance |
| 10 | **Edge deployment** | ONNX/TensorRT → Jetson Orin / RPi CM4 |
| 11 | **Human-in-loop** | Language commands → task embeddings ("go to kitchen") |
| 12 | **Open dataset** | Release 100+ scanned spaces + policies |
| 13 | **Hardening + docs** | Tutorial series, CI/CD, community launch |

---

## HERMES /LOOP CONFIGURATION

```bash
# In project root:
/loop 30m "Read prompt.md. Check evolution-lessons.json. Run ONE sprint task from current milestone. Delegate to OpenCode/Kilo Code via ecosystem orchestrator. Write results to sprint-N/ folder. Update prompt.md with next task. --times 520 --until 'all 13 sprints complete'"
```

**Prompt evolution protocol:**
- Each loop iteration: read `prompt.md` → execute ONE atomic task → append learnings → bump `prompt_version`
- Failed approaches → `evolution-lessons.json` (failed:true) → never retry
- Successful patterns → `evolution-lessons.json` (failed:false) → reuse

---

## INTER-HARNESS DELEGATION (via ecosystem-orchestrator skill)

| Harness | Role (THIS project) | Trigger |
|---------|---------------------|---------|
| **Prime** | Long RL evolution (hours): curriculum search, hyperparam sweeps, DreamerV3/JAX multi-seed runs. Use for Sprints 4-6 RL core. | `prime-agent --cwd . --provider opencode --goal "Evolve PPO hyperparams Sprint 4 (CleanRL, mock_backend, N=8 vec_env)" --autonomous` |
| **OpenCode** | Code gen (free-tier): recon, sim envs, RL trainers, tests, CI. Primary coder for Sprints 7-9 (friend's lane). | `opencode run "Implement X" -f prompts/sprintN.md -m muse-spark-1.3-contributor-free` |
| **Kilo Code** | Inline edits, bug fixes, test patches during review | IDE-level patches |
| **Antigravity** | Deep refactors: sim abstraction layer, control stack, USD/glTF pipeline rewrite | Major architecture shifts (needs human approve) |

Jules is NOT used in this project.

**Bus:** `04-Prompt-Queues/Ecosystem/InterHarness/<from>-to-<to>-YYYY-MM-DD.md` — both machines read/write here; append-only.

---

## FREE-TIER MODEL ROTATION (OpenCode Zen)
```json
// opencode/settings.json
{
  "defaultProvider": "opencode",
  "defaultModel": "muse-spark-1.3-contributor-free",
  "alternatives": [
    "nemotron-3-ultra-free",
    "mimo-v2.5-free",
    "nemotron-3.5-lightning-free"
  ]
}
```
- 15-min NordVPN rotation on 429 (vpn-location-rotator skill)
- Pin ONE model per /loop cycle for consistency

---

## SAFETY & ETHICS (non-negotiable)
- **No weaponization code** — no targeting, no kinetic effectors
- **SAR-only primitives**: `navigate_to`, `hover`, `drop_payload`, `return_home`
- **Geofence + altitude ceiling** hardcoded in control layer
- **Human-on-the-loop**: MAVLink heartbeat + RC override always active
- **Open license**: MIT/Apache-2.0 — community audit welcome

---

## CURRENT STATE (Sprint 3 complete — 2026-09-14)
- [x] Sprints 1-3 done (docker → reconstruction → sim mock_backend + 3 envs, 20 tests green)
- [x] Repo: https://github.com/pandejesal/drone-nav-sar (private, master)
- [x] Isolation: .gitignore, project-local .venv/.opencode/.harness-memory; vault submodule at ./vault
- [x] Friend bootstrap: scripts/setup-friend-mac.sh + prompts/bootstrap-opencode.md
- [ ] Sprint 4 next (RL PPO) — see NEXT ACTION below

---

## COLLABORATION — OWNERSHIP MAP (no overlaps, both use /loop + InterHarness bus)

**Rule:** One sprint per person at a time. Claim by creating `prompts/sprintN_<owner>.md` and pushing. Never edit the other's active sprint files without a PR.

| Owner | Machine | Owns (code) | Owns (harness) | Branch naming |
|-------|---------|-------------|----------------|---------------|
| **YOU** | Windows (Pro) | Sprint 4: `src/rl/` PPO (`ppo_nav.py`, `policies.py`, `vec_env.py`) + `src/eval/` SPL; Sprint 5: dynamic obstacles curriculum; Sprint 6: Sim2Real + Crazyflie | **Prime** (long RL evolution), **Antigravity** (deep refactors) | `sprint4-ppo-<initials>`, `sprint5-dynamic-<initials>` |
| **FRIEND** | MacBook (free) | Sprint 7: SAR multi-room + YOLO victim detect + `drop_payload`; Sprint 8: hierarchical planner; Sprint 9: multi-drone comms | **OpenCode** (code gen), **Kilo Code** (inline fixes) | `sprint7-sar-<initials>`, `sprint8-hier-<initials>` |
| **SHARED** | both (PR required) | `prompt.md` (append-only, bump version), `.harness-memory/evolution-lessons.json` (append-only), `tests/` for your sprint, `docs/`, `docker/` | InterHarness bus `04-Prompt-Queues/Ecosystem/InterHarness/` | `main` protected — merge via PR only |

**Git workflow (both machines):**
```bash
git checkout -b sprint4-ppo-you
# work on your sprint files only
git add src/rl/ tests/test_rl.py
git commit -m "Sprint 4: PPO trainer (mock_backend, vec_env N=8)"
git push -u origin sprint4-ppo-you
# open PR on GitHub, other person reviews, squash-merge
git checkout master && git pull
```

**InterHarness bus (how AIs understand each other):**
- After each /loop iteration or manual run, write ONE file:
  `04-Prompt-Queues/Ecosystem/InterHarness/<from>-to-<to>-YYYY-MM-DD.md`
  e.g. `prime-to-opencode-2026-09-15.md` or `mac-bootstrap-2026-09-15.md`
- Format: `## From / To / Sprint / Done / Next / Blockers` — keep it short.
- Both machines `git pull` at start of day to read the bus.

**Free-tier model rotation (both machines, pinned per /loop cycle):**
- `muse-spark-1.3-contributor-free` (default), fallback `nemotron-3-ultra-free`, `mimo-v2.5-free`, `nemotron-3.5-lightning-free` on 429.
- Config: `.opencode/opencode.json` (project-local, committed). No global overwrite.

## ISOLATION GUARANTEE (your concern — other projects unaffected)
- This repo's configs are **project-local**: `.venv/`, `.opencode/`, `.harness-memory/`, `vault/` (submodule).
- Global vault stays at `C:/Users/DELL/Documents/Obsidian Vault` (Windows) / `~/Documents/Obsidian Vault` (Mac) — **not moved**. Submodule `vault/` is a second clone *inside* this repo only; other projects still import from the global path.
- Global Hermes (`%LOCALAPPDATA%/hermes` / `~/.config/hermes`), global opencode (`~/.opencode`), Mnemosyne global DB are **never overwritten** — friend's `scripts/setup-friend-mac.sh` writes only to `$PROJECT_DIR`.
- `.gitignore` excludes `.venv`, `__pycache__`, `models/`, `assets/scans/` so no cross-project pollution.
- To verify: `git status` should never show changes outside this repo's root.

---

## SPRINT 1 — COMPLETED (Docker Dev Container)
- [x] Multi-stage Dockerfile (base → python-deps → blender → colmap → final)
- [x] docker-compose.yml with profiles for isaac-sim, airsim, gazebo, tensorboard, wandb
- [x] .dockerignore optimized for build context
- [x] build_docker.sh with --no-cache, --target, --tag options
- [x] verify_install.py checks all critical imports + CUDA + system commands
- [x] entrypoint.sh sources ROS2, sets up X11, FastDDS
- [x] fastdds.xml for ROS2 local communication
- [x] evolution-lessons.json updated with lesson: OpenCode free tier unreliable → direct implementation fallback

---

## SPRINT 2 — COMPLETED (Reconstruction Pipeline)
- [x] `src/reconstruction/colmap_pipeline.py` — CLI: photos → sparse → dense → textured mesh (.glb + .usd)
- [x] `src/reconstruction/nerfstudio_pipeline.py` — CLI: photos → NeRF/3DGS (splatfacto) → mesh export
- [x] `src/reconstruction/blender_cleanup.py` — BlenderProc script: decimate, UV unwrap, PBR bake, collision hulls, export .glb/.usd
- [x] `src/reconstruction/reconstruct.py` — Unified entry point: auto-detect best method (COLMAP for structure, Nerfstudio for appearance)
- [x] `tests/test_reconstruction.py` — Unit tests for CLI parsing, auto-selection logic
- [x] evolution-lessons.json updated: OpenCode free tier consistently failing → direct implementation works

---

## SPRINT 3 — COMPLETED (Simulation Environments, 2026-09-14)
- [x] `src/sim/domain_randomization.py` — backend-agnostic lighting/texture/physics/wind/noise/pose randomization
- [x] `src/sim/mock_backend.py` — shared headless physics (QuadrotorDynamics) + synthetic RGB/depth camera
- [x] `src/sim/isaac_sim_env.py` — Isaac Sim hook with mock fallback (`use_mock` flag)
- [x] `src/sim/airsim_env.py` — AirSim/Unreal client hook with mock fallback
- [x] `src/sim/gazebo_env.py` — Gazebo Harmonic + ROS2/PX4 SITL hook with mock fallback
- [x] `src/sim/__init__.py` — full exports; `make_env()` factory verified for all 3 backends
- [x] `tests/test_sim.py` — 20 tests + 27 subtests GREEN (venv: gymnasium 1.3.0)
- [x] Bugfix: `drone_dynamics.py` motor-unit mismatch (RPM vs normalized) + dt/tau clamp
- [x] Project `.venv` created via uv (gymnasium/numpy/pytest) — use `.venv/Scripts/python -m pytest`
- [x] evolution-lessons.json: 3 new entries (sim_envs, motor_units_bugfix, reconstruction_tests_win)
- Note: constructors accept `SimConfig` OR bare mesh-path string; obs 21-dim; mock mode explicit in info[]

## NEXT ACTION (for /loop iteration 4 — Sprint 4: Single-Task RL)
**Implement `src/rl/` PPO point-nav trainer (CleanRL-style, numpy/PyTorch CPU):**
1. `ppo_nav.py` — PPO trainer on `make_env("isaac", mesh)` mock backend, goal A→B static room
2. `policies.py` — MLP actor-critic (obs 21 → 4 motor thrusts), orthogonal init, tanh Gaussian
3. `vec_env.py` — vectorized N-env runner for throughput (N=8 mock envs, seeded)
4. `tests/test_rl.py` — loss decreases on fixed batch; policy outputs in [0,1]; 50-episode smoke run
**Target:** >90% success (reach within goal_tolerance) on fixed A→B pairs, SPL logged in `src/eval/`
**ACCEPTANCE:**
```bash
.venv/Scripts/python -m pytest tests/test_rl.py -q   # green
.venv/Scripts/python -m src.rl.ppo_nav --episodes 200 --seed 0  # success_rate > 0.9 on fixed goals
```

## EVOLUTION LESSONS (append only)
```json
[
  {"timestamp": "2026-09-14T00:00:00Z", "sprint": 0, "task": "docker_setup", "method": "opencode_nemotron", "result": "pending", "failed": null, "notes": "First iteration"}
]
```