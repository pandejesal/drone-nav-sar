# Sprint 1 Task: Docker Dev Container for DroneNav-SAR

## CONTEXT
- Project: DroneNav-SAR (autonomous drone navigation for search & rescue)
- Sprint: 1/13 — Reconstruction pipeline foundation
- This task: Create reproducible Docker dev container with ALL sim/ML/deps
- Free-tier only: OpenCode Zen models, no paid APIs

## REQUIREMENTS
Create `docker/` folder with:
1. **Dockerfile** — multi-stage build:
   - Base: `nvidia/cuda:12.4-devel-ubuntu22.04`
   - Stage 1: System deps (ROS2 Humble, Gazebo Harmonic, Blender, COLMAP, Nerfstudio deps)
   - Stage 2: Isaac Sim 4.2 headless (from NVIDIA container or manual install)
   - Stage 3: AirSim 1.8 (build from source)
   - Stage 4: Python 3.11 env + all ML deps (CleanRL, Sample-Factory, DreamerV3, MAVSDK, etc.)
2. **docker-compose.yml** — GPU passthrough, volume mounts for `assets/`, `models/`, `src/`
3. **scripts/build_docker.sh** — one-command build with cache
4. **scripts/verify_install.py** — imports all critical modules, prints versions

## CRITICAL DEPS TO INCLUDE
```python
# Simulators
isaacsim==4.2.0          # headless, from NVIDIA
airsim==1.8.0            # build from source
gazebo==11.14.0          # Harmonic via ROS2

# 3D/Reconstruction
bpy==4.2.0               # Blender Python API (headless)
blenderproc==2.0.0       # Proc-gen + randomization
colmap==3.9.1            # SfM/MVS
nerfstudio==1.1.0        # NeRF/3DGS (instant-ngp)

# RL/ML
cleanrl==2.0.0
sample-factory==2.0.0
dreamerv3-jax==0.1.0     # JAX version
jax==0.4.28
jaxlib==0.4.28+cuda12
torch==2.4.0+cu124
torchvision==0.19.0+cu124

# Drone/Control
mavsdk==2.0.0
rclpy==3.2.0             # ROS2 Humble
px4_msgs==1.0.0

# Utils
opencv-python==4.10.0
open3d==0.18.0
trimesh==4.0.0
pyyaml==6.0.1
tqdm==4.66.0
```

## ACCEPTANCE CRITERIA (must pass for DONE)
```bash
# 1. Build succeeds
./scripts/build_docker.sh

# 2. Container runs with GPU
docker run --gpus all -it drone-nav-sar python scripts/verify_install.py
# Output must show: Isaac Sim OK, AirSim OK, Gazebo OK, Blender OK, COLMAP OK, Nerfstudio OK, JAX GPU OK, Torch CUDA OK, MAVSDK OK

# 3. Headless Blender works
docker run --gpus all drone-nav-sar blender --background --python -c "import bpy; print(bpy.app.version)"
```

## CONSTRAINTS
- **Total image size < 40GB** (use multi-stage, clean apt cache, remove docs)
- **Build time < 60 min** on free-tier cloud (GitHub Actions) or local
- **No paid dependencies** — everything from public repos
- **Windows host compatible** — paths in compose use forward slashes

## OUTPUT STRUCTURE
```
docker/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── scripts/
│   ├── build_docker.sh
│   └── verify_install.py
└── entrypoint.sh          # Optional: ROS2 source + env setup
```

## DELEGATION NOTES
- Use `opencode run` with model `nemotron-3-ultra-free` or `muse-spark-1.3-contributor-free`
- Workdir: `C:/Users/DELL/Desktop/3d_environment_drone_neural_network` (repo root)
- Stage any reference files (this prompt) inside workdir — sandbox rejects absolute outside paths
- Output DONE line: `DONE-S1-DOCKER | files: docker/Dockerfile,docker/docker-compose.yml,... | verify: PASS`

## EVOLUTION LESSONS (from .harness-memory/evolution-lessons.json)
- First iteration — no prior lessons
- If Docker build fails on Isaac Sim: try NVIDIA's prebuilt `nvcr.io/nvidia/isaac-sim:4.2.0` as base instead of manual install
- If AirSim build fails: use `microsoft/airsim:1.8.0` Docker Hub image as sidecar in compose instead