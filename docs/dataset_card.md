# Dataset Card — DroneNav-SAR Open Dataset (Sprint 12)

> **Datasheets for Datasets** (Gebru et al.) — SAR-only release.
> Hub: `drone-nav-sar/dataset` (spaces) + `drone-nav-sar/policies` (checkpoints).
> License: **CC-BY-4.0**. See `docs/contributing.md` for additions.

## 1. Motivation
- **Purpose:** reproducible Search-and-Rescue (SAR) drone navigation research:
  navigate_to / hover / drop_payload / return_home in 100+ scanned
  indoor/outdoor spaces with trained policies and benchmarks.
- **Creators:** DroneNav-SAR team, Sprint 12 dataset release.
- **Funding:** internal SAR research budget; no military funding.

## 2. Composition
| Split | Spaces | Type | Resolution |
|-------|--------|------|------------|
| train | 70 | Indoor (rooms, corridors) + Outdoor (yards) | 64×64 RGB-D |
| val   | 15 | Mixed | 64×64 RGB-D |
| test  | 15 | Unseen layouts | 64×64 RGB-D |

- Each space: photos (50–200) → COLMAP sparse/dense (or Nerfstudio) →
  mesh (`.glb` + `.usd`) → BlenderProc cleanup → domain randomization →
  policy training (2000 eps) → ONNX export.
- Per-space bundle: `photos/`, `colmap_sparse/`, `mesh/space.glb`,
  `domain_rand.json`, `checkpoints/policy.pt`, `checkpoints/policy.onnx`.
- `manifest.json` lists split/type/paths/SHA-256; `checksums.sha256`
  covers every mesh + checkpoint.
- No personally identifiable information; faces/license plates blurred
  before release. No human subjects data.

## 3. Collection & Processing
- Photos captured with consent on private property / public areas where
  photography is permitted; property releases on file.
- Reconstruction: `src/reconstruction/colmap_pipeline.py` or
  `nerfstudio_pipeline.py`, then `blender_cleanup.py`
  (decimate → target tris, 64×64 textures, watertight check).
- Domain randomization: `src/sim/domain_randomization.py`
  (lighting, texture, mass, wind, latency).
- Training: PPO (`src/rl/ppo_nav.py`), 2000 episodes/space, seeds logged
  in manifest. Quick/smoke mode uses `--quick` stubs with identical schema.

## 4. Uses
- **Intended:** SAR navigation research, sim-to-real transfer, benchmark
  comparisons (SR / SPL / energy / latency), education
  ("Train your own SAR drone in 30 minutes" tutorial).
- **Out of scope / prohibited:** any weaponization, targeting, kinetic
  payloads, surveillance of people, or law-enforcement targeting use.
  Payload action is `drop_payload` (medical/aid kits) only.

## 5. Distribution & License
- Hub repos: `drone-nav-sar/dataset`, `drone-nav-sar/policies`.
- License: **CC-BY-4.0** (attribution required, see below).
- Download: `.venv/Scripts/python scripts/dataset_download.py --split all`
  (verifies `checksums.sha256` by default).
- Attribution: "DroneNav-SAR Open Dataset (Sprint 12), CC-BY-4.0".

## 6. Maintenance
- Maintainers: DroneNav-SAR team. Issues/PRs via GitHub; see
  `docs/contributing.md`.
- Versioning: `manifest.json` carries `sprint` + seed; Hub revisions are
  immutable; errata published as new revisions.
- Retention: spaces with unresolved consent/privacy flags are removed in
  the next revision.

## 7. Ethics Statement (SAR-only)
- This dataset exists **solely for civilian search-and-rescue**.
- Prohibited uses: weapons integration, human targeting, autonomous harm,
  covert surveillance. The action space exposes normalized thrusts only;
  there is no targeting interface in `src/rl/policies.py`.
- Safety: policies ship with `src/control/safety_filter.py` geofence +
  kill-switch requirements; real-world flight requires human oversight.
- Bias/limits: 100 spaces cannot cover all buildings/terrain/weather;
  test-split layouts are unseen but still simulation-derived meshes —
  expect a sim-to-real gap (see `docs/benchmarks.md`).
- Contact for misuse reports / takedowns: maintainers via repo issues.
