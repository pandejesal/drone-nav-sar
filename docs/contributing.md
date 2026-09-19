# Contributing — DroneNav-SAR Open Dataset (Sprint 12, SAR-only)

## Ground rules (non-negotiable)

- **SAR-only.** Accepted tasks: `navigate_to`, `hover`, `drop_payload`
  (medical/aid kits), `return_home`. No weapons, no targeting, no kinetic
  payloads, no human surveillance. PRs violating this are closed.
- **License:** all space/policy contributions are **CC-BY-4.0**.
  Only submit photos/meshes you own or that are CC-BY-compatible with
  consent/property releases.
- **Privacy:** blur faces, license plates, and legible private documents
  before submitting. No PII in manifests or filenames.

## How to add a new space

1. Capture 50–200 photos (overlap ≥ 60%, varied heights/angles).
2. Place them in `data/new_space_<name>/photos/`.
3. Run the pipeline (single space):
   ```bash
   .venv/Scripts/python scripts/dataset_build.py --spaces 1 --output-dir data/new_space_<name>
   ```
   For a real reconstruction use the full mode (needs COLMAP or
   Nerfstudio + Blender); `--quick` only produces schema stubs for CI.
4. Check outputs: `mesh/space.glb`, `domain_rand.json`,
   `checkpoints/policy.pt` + `policy.onnx`, `manifest.json`.
5. Open a PR adding the space bundle + manifest entry; include:
   - location type (indoor_room / indoor_corridor / outdoor_yard),
   - photo count + capture device,
   - consent statement + blur confirmation,
   - train/val/test split suggestion (maintainers assign test).

## How to train a policy (30-minute tutorial)

```bash
# 1. Get data
.venv/Scripts/python scripts/dataset_download.py --split all
# or build a smoke copy:
.venv/Scripts/python scripts/dataset_build.py --spaces 5 --quick

# 2. Train (full: 2000 eps/space; smoke: --quick → 5 eps stubs)
.venv/Scripts/python scripts/dataset_build.py --spaces 5 --quick --output-dir data/my_run

# 3. Evaluate + leaderboard
.venv/Scripts/python scripts/eval_all.py --checkpoint policy_sprint4.pt --episodes 20
```

Training entry point: `src/rl/ppo_nav.py`; eval metrics:
`src/eval/metrics.py` (SR, SPL, energy_proxy).

## Submit benchmarks / policies

- Run `scripts/eval_all.py` and paste the leaderboard rows + `SR_avg`.
- Upload checkpoints to `drone-nav-sar/policies` naming:
  `policy_<author>_<space>_<date>.pt` (+ `.onnx` via
  `src/control/edge_export.py`).
- Update `docs/benchmarks.md` checkpoint table in the same PR.

## PR checklist

- [ ] SAR-only (no targeting/weaponization content)
- [ ] CC-BY-4.0 compatible + consent/blur statement
- [ ] `manifest.json` + `checksums.sha256` verify
     (`dataset_download.py --local-dir <dir>`)
- [ ] `eval_all.py` rows included for new/changed spaces
- [ ] No secrets, no absolute local paths, no PII
