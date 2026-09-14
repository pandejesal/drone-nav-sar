# syntax=docker/dockerfile:1.4
# DroneNav-SAR Development Container
# Multi-stage build for minimal final image

################################################################################
# STAGE 1: Base system dependencies
################################################################################
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    wget \
    curl \
    ca-certificates \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # ROS2 Humble dependencies
    software-properties-common \
    gnupg2 \
    lsb-release \
    # Graphics/Vulkan for Isaac Sim, Gazebo
    libvulkan1 \
    vulkan-tools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    # COLMAP deps
    libsuitesparse-dev \
    libeigen3-dev \
    libboost-all-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    # Open3D deps
    libglfw3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # General
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 Humble repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-px4-msgs \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Source ROS2 in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

################################################################################
# STAGE 2: Python environment + ML dependencies
################################################################################
FROM base AS python-deps

WORKDIR /workspace

# Create venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build deps
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install JAX with CUDA 12
RUN pip install jax==0.4.28 jaxlib==0.4.28+cuda12 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Core ML/RL libraries
RUN pip install \
    cleanrl==2.0.0 \
    sample-factory==2.0.0 \
    dreamerv3-jax==0.1.0 \
    gymnasium==0.29.1 \
    stable-baselines3==2.4.1 \
    optuna==4.0.0 \
    wandb==0.17.6 \
    tensorboard==2.17.0

# Scientific stack
RUN pip install \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    opencv-python==4.10.0 \
    open3d==0.18.0 \
    trimesh==4.0.5 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    imageio==2.35.1 \
    imageio-ffmpeg==0.4.9

# Drone/ROS2 control
RUN pip install \
    mavsdk==2.0.0 \
    rclpy==3.2.7 \
    px4-msgs==1.0.0

# 3D/NeRF
RUN pip install \
    nerfstudio==1.1.9 \
    --no-deps

# Install nerfstudio deps separately to avoid conflicts
RUN pip install \
    torchmetrics==1.4.0 \
    torchvision==0.19.0 \
    kornia==0.7.2 \
    tinycudann==1.6 \
    --index-url https://pypi.org/simple

################################################################################
# STAGE 3: Blender + BlenderProc (headless)
################################################################################
FROM python-deps AS blender-stage

# Install Blender 4.2 headless
ARG BLENDER_VERSION=4.2.0
ARG BLENDER_URL=https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz

RUN wget -q ${BLENDER_URL} -O /tmp/blender.tar.xz \
    && tar -xf /tmp/blender.tar.xz -C /opt \
    && mv /opt/blender-${BLENDER_VERSION}-linux-x64 /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && rm /tmp/blender.tar.xz

# Install BlenderProc
RUN pip install blenderproc==2.0.0 -f https://github.com/DLR-RM/BlenderProc/releases/download/v2.0.0/blenderproc-2.0.0-cp311-cp311-linux_x86_64.whl

# Verify Blender Python API works
RUN blender --background --python -c "import bpy; print('Blender:', bpy.app.version_string)"

################################################################################
# STAGE 4: COLMAP + OpenMVS
################################################################################
FROM blender-stage AS colmap-stage

# Build COLMAP from source (Ubuntu 22.04 package is old)
WORKDIR /tmp
RUN git clone https://github.com/colmap/colmap.git --branch 3.9.1 --depth 1 \
    && cd colmap \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON \
    && make -j$(nproc) \
    && make install \
    && cd / && rm -rf /tmp/colmap

# Verify
RUN colmap -h

################################################################################
# STAGE 5: AirSim (build from source)
################################################################################
FROM colmap-stage AS airsim-stage

WORKDIR /tmp
# AirSim requires Unreal Engine - we'll use the prebuilt Docker approach instead
# For now, install AirSim Python client only (connects to external Unreal instance)
RUN pip install airsim==1.8.1

# Note: Full AirSim requires Unreal Engine which is heavy.
# Recommended: Run AirSim as sidecar container or on Windows host with Unreal.
# This container has the Python client for connecting to external AirSim.

################################################################################
# STAGE 6: Isaac Sim (use NVIDIA prebuilt as base alternative)
################################################################################
# Isaac Sim is large (~20GB). Two options:
# Option A: Use NVIDIA's prebuilt container as base (recommended)
# Option B: Install manually (very slow, complex)
# We'll document Option A in docker-compose and keep this stage minimal

# Install Isaac Sim Python packages (available on PyPI for headless)
RUN pip install \
    isaacsim==4.2.0.0 \
    --extra-index-url https://pypi.nvidia.com \
    || echo "Isaac Sim PyPI install failed - will use nvcr.io base in compose"

################################################################################
# FINAL STAGE: Application
################################################################################
FROM colmap-stage AS final

WORKDIR /workspace

# Copy Python environment
COPY --from=python-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Blender
COPY --from=blender-stage /opt/blender /opt/blender
ENV PATH="/opt/blender:$PATH"

# Copy COLMAP
COPY --from=colmap-stage /usr/local/bin/colmap /usr/local/bin/colmap

# Create non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace /opt/venv

# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER $USERNAME
ENV HOME=/home/$USERNAME

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]