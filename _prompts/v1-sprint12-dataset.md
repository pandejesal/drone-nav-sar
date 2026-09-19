# Sprint 12 — Open Dataset Release: 100+ Scanned Spaces + Policies (SAR-only)

## Goal
Release a public dataset of 100+ scanned indoor/outdoor spaces with trained policies, benchmarks, and documentation. Target: reproducible SAR research, community adoption.

## Scope (modify ONLY these)
- scripts/dataset_build.py: NEW — pipeline: photos → COLMAP/Nerfstudio → BlenderProc → mesh → domain randomization → policy training
- scripts/dataset_download.py: NEW — download from Hugging Face / AWS S3, verify checksums
- docs/dataset_card.md: NEW — dataset card (Datasheets for Datasets), license (CC-BY-4.0), ethics statement
- docs/benchmarks.md: NEW — per-space results (SR, SPL, energy), policy checkpoints
- scripts/eval_all.py: NEW — batch evaluation across dataset, leaderboard table
- docs/contributing.md: NEW — how to add new spaces, train policies, submit PRs

## Dataset Composition
| Split | Spaces | Type | Resolution |
|-------|--------|------|------------|
| train | 70 | Indoor (rooms, corridors) + Outdoor (yards) | 64x64 RGB-D |
| val | 15 | Mixed | 64x64 RGB-D |
| test | 15 | Unseen layouts | 64x64 RGB-D |

Each space: photos (50-200) → COLMAP sparse/dense → mesh → BlenderProc cleanup → domain randomization → policy training (2000 eps) → ONNX export.

## Deliverables
- Dataset on Hugging Face Hub (drone-nav-sar/dataset)
- Policy checkpoints on Hugging Face Hub (drone-nav-sar/policies)
- Leaderboard: per-space SR, SPL, energy, latency
- Tutorial: "Train your own SAR drone in 30 minutes"

## Acceptance
.venv/Scripts/python scripts/dataset_build.py --spaces 5 --quick (smoke test, < 30 min)
.venv/Scripts/python scripts/eval_all.py --checkpoint policy_sprint11_final.pt (leaderboard generation)
Dataset card published on Hugging Face, license CC-BY-4.0, ethics statement (SAR-only)

## DONE
DONE-12 | files: scripts/dataset_build.py,scripts/dataset_download.py,docs/dataset_card.md,docs/benchmarks.md,scripts/eval_all.py,docs/contributing.md | tests: dataset smoke pass | metrics: {spaces: 100+, SR_avg: >0.6}