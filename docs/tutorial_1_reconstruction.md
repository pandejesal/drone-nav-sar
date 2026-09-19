# Tutorial 1 — Reconstruction: Phone Photos to 3D Mesh (~30 min, SAR-only)

Turn phone photos of an indoor space into a textured `.glb` mesh the simulator
can fly in. SAR-only: only scan spaces you own or have consent to scan; blur
faces, plates, and private documents before submitting anything.

## Prerequisites

- 50-200 overlapping photos (60%+ overlap, varied heights/angles)
- COLMAP **or** Nerfstudio installed; Blender for cleanup (optional)
- `pip install -e .` in this repo

## Steps

### 1. Collect photos

```bash
mkdir -p data/new_space_lab/photos
# copy phone photos in; name them img_0000.jpg, img_0000.jpg, ...
```

### 2. Run the pipeline (single space)

```bash
python scripts/dataset_build.py --spaces 1 --output-dir data/new_space_lab
# Quick schema stub for CI (no COLMAP needed):
python scripts/dataset_build.py --spaces 1 --quick --output-dir data/new_space_lab
```

What it does (`src/reconstruction/reconstruct.py`):
1. `colmap_pipeline.py` — SfM sparse reconstruction (`colmap_sparse/`)
2. `nerfstudio_pipeline.py` — dense mesh (fallback when COLMAP sparse is thin)
3. `blender_cleanup.py` — decimate, close holes, bake texture, export `mesh/space.glb`

### 3. Verify outputs

```bash
ls data/new_space_lab/mesh/          # space.glb + collision hulls
python -c "from src.reconstruction.reconstruct import verify_mesh; verify_mesh('data/new_space_lab/mesh/space.glb')"
```

Checks: mesh loads, watertight-ish hull, scale in meters, texture present.

### 4. Fly it in the mock backend

```bash
python scripts/test_sim_envs.py --mesh data/new_space_lab/mesh/space.glb --backend mock
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| COLMAP finds < 100 points | More overlap, better lighting, avoid blank walls |
| Mesh scale wrong | Re-run with a known-size reference object in frame |
| Simulator falls through floor | Regenerate collision hulls (`blender_cleanup.py --hulls`) |

## Next

Tutorial 2 — load this mesh into Isaac Sim / AirSim / Gazebo (or keep the mock backend).
