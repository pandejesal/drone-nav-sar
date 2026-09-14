# Sprint 2 Task: 3D Reconstruction Pipeline (COLMAP + Nerfstudio + BlenderProc)

## CONTEXT
- Project: DroneNav-SAR (autonomous drone navigation for search & rescue)
- Sprint: 2/13 — Reconstruction pipeline
- Previous: Docker dev container complete (Sprint 1)
- This task: Implement `src/reconstruction/` module for `photos → 3D mesh` pipeline
- Free-tier only: OpenCode Zen models, no paid APIs

## REQUIREMENTS
Create `src/reconstruction/` with:

### 1. `colmap_pipeline.py` — Classical SfM/MVS Pipeline
```python
# CLI: python -m src.reconstruction.colmap_pipeline <input_dir> <output_dir> [--quality high|medium|low]
# Steps:
# 1. Feature extraction (SIFT) — colmap feature_extractor
# 2. Feature matching (exhaustive or vocab_tree) — colmap exhaustive_matcher
# 3. Sparse reconstruction (incremental mapper) — colmap mapper
# 4. Dense reconstruction (patch-match stereo) — colmap patch_match_stereo
# 5. Stereo fusion → dense point cloud — colmap stereo_fusion
# 6. Poisson meshing — colmap poisson_mesher
# 7. Texture mapping — colmap texture_mapper
# 8. Export to .glb (via trimesh) + .usd (via pyusd)
```
- All steps via `subprocess.run()` with proper error handling
- Configurable quality presets (affects image resolution, matcher, mesher params)
- Progress logging with timestamps
- Output: `output_dir/sparse/`, `output_dir/dense/`, `output_dir/mesh.glb`, `output_dir/mesh.usd`

### 2. `nerfstudio_pipeline.py` — NeRF/3DGS Pipeline
```python
# CLI: python -m src.reconstruction.nerfstudio_pipeline <input_dir> <output_dir> [--method nerfacto|splatfacto]
# Steps:
# 1. ns-process-data images --data <input_dir> --output-dir <output_dir>/nerfstudio
# 2. ns-train <method> --data <output_dir>/nerfstudio --output-dir <output_dir>/nerfstudio
# 3. ns-export mesh --load-config <output_dir>/nerfstudio/config.yml --output-dir <output_dir>/mesh
# 4. Convert exported mesh to .glb + .usd
```
- Use `nerfstudio` Python API directly (not subprocess) for better control
- Support both NeRF (nerfacto) and 3D Gaussian Splatting (splatfacto)
- 3DGS preferred for speed/quality on indoor scenes
- Output: same structure as COLMAP

### 3. `blender_cleanup.py` — BlenderProc Post-Processing
```python
# CLI: blender --background --python blender_cleanup.py -- <input.glb> <output.glb> [--target-tris 50000]
# Steps (via BlenderProc bpy):
# 1. Import mesh
# 2. Remove loose geometry, non-manifold edges
# 3. Decimate to target triangle count (preserve UV seams)
# 4. Smart UV unwrap (if UVs missing or overlapping)
# 5. Bake PBR materials (albedo, normal, roughness, metallic) to textures
# 6. Generate convex collision hulls (for physics sim)
# 7. Center pivot, apply scale, normalize to unit scale
# 8. Export .glb (with textures embedded) + .usd (for Isaac Sim)
```
- Headless Blender via `blenderproc` or raw `bpy`
- Texture atlas: 2048x2048 max, power-of-2
- Collision hull: convex decomposition (vhacd) for complex shapes

### 4. `reconstruct.py` — Unified Entry Point
```python
# CLI: python -m src.reconstruction.reconstruct <input_dir> <output_dir> [--method auto|colmap|nerfstudio] [--quality high|medium|low]
# Logic:
# - auto: try COLMAP first (fast, good structure), if fails/sparse → Nerfstudio
# - colmap: force COLMAP pipeline
# - nerfstudio: force Nerfstudio pipeline
# - Always runs blender_cleanup as final step
# - Returns: path to final .glb and .usd
```

### 5. `tests/test_reconstruction.py` — Unit Tests
- Mock tests for CLI argument parsing
- Integration test with tiny synthetic dataset (2-3 images of simple cube)
- Verify output files exist, valid mesh topology, UVs present

## INPUT DATA
- Place sample photos in `assets/scans/sample_room/` (20-50 photos, varied angles, good overlap)
- Can use phone photos of a real room, or generate synthetic via Blender
- For testing: create a simple synthetic dataset in `tests/fixtures/simple_room/`

## ACCEPTANCE CRITERIA
```bash
# Inside Docker container:
cd /workspace
python -m src.reconstruction.reconstruct assets/scans/sample_room assets/meshes/sample_room --method auto --quality medium

# Must produce:
# assets/meshes/sample_room.glb  (textured, <50k tris, valid UVs)
# assets/meshes/sample_room.usd  (for Isaac Sim)

# Verification:
python -c "
import trimesh
mesh = trimesh.load('assets/meshes/sample_room.glb')
print(f'Triangles: {len(mesh.faces)}')
print(f'Vertices: {len(mesh.vertices)}')
print(f'Has UVs: {mesh.visual.uv is not None}')
print(f'Bounds: {mesh.bounds}')
assert len(mesh.faces) < 50000
assert mesh.visual.uv is not None
print('PASS')
"
```

## CONSTRAINTS
- **Headless only** — no GUI, no display needed (BlenderProc, COLMAP, Nerfstudio all support this)
- **Docker compatible** — all deps in Dockerfile, no host-specific paths
- **Error handling** — each step logs, cleans up temp files on failure
- **Configurable** — quality presets, method selection, output formats
- **Performance** — COLMAP ~5-15 min for 50 photos; Nerfstudio ~30-60 min (3DGS faster)

## DELEGATION NOTES
- Use `opencode run` with model `muse-spark-1.3-contributor-free` or `nemotron-3-ultra-free`
- Workdir: `C:/Users/DELL/Desktop/3d_environment_drone_neural_network` (repo root)
- Stage this prompt at `prompts/sprint2_reconstruction.md` inside workdir
- Output DONE line: `DONE-S2-RECON | files: src/reconstruction/..., tests/... | verify: PASS`

## EVOLUTION LESSONS (from .harness-memory/evolution-lessons.json)
- OpenCode free tier unreliable (UnknownError) → direct implementation fallback works
- Docker multi-stage build successful → use same pattern for any heavy deps
- BlenderProc works headless in container → prefer over raw bpy for proc-gen
- Nerfstudio 3DGS (splatfacto) faster than NeRF for indoor → default to splatfacto