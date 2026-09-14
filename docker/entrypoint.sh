#!/bin/bash
# DroneNav-SAR Container Entrypoint
# Sources ROS2, sets up environment, handles X11 forwarding

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DroneNav-SAR Development Container   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Source ROS2
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}✓${NC} ROS2 Humble sourced"
else
    echo -e "${YELLOW}⚠${NC} ROS2 not found at /opt/ros/humble/setup.bash"
fi

# Source local workspace if built
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
    echo -e "${GREEN}✓${NC} Local workspace sourced"
fi

# X11 forwarding setup
if [ -n "${DISPLAY:-}" ]; then
    # Allow X11 connections from container
    xhost +local:docker 2>/dev/null || true
    echo -e "${GREEN}✓${NC} X11 forwarding enabled (DISPLAY=${DISPLAY})"
else
    echo -e "${YELLOW}⚠${NC} No DISPLAY set - GUI apps may not work"
fi

# FastDDS configuration for ROS2
if [ -f /workspace/fastdds.xml ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds.xml
    echo -e "${GREEN}✓${NC} FastDDS config loaded"
fi

# Python path
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
echo -e "${GREEN}✓${NC} PYTHONPATH includes /workspace/src"

# CUDA visible devices
if [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
    echo -e "${GREEN}✓${NC} GPU access: ${NVIDIA_VISIBLE_DEVICES}"
fi

# Working directory
cd /workspace
echo -e "${GREEN}✓${NC} Working directory: $(pwd)"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Ready for development!               ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Available commands:"
echo "  blender --background --python script.py"
echo "  colmap <command>"
echo "  gz sim <world>"
echo "  ros2 <command>"
echo "  python -m src.reconstruction.colmap_pipeline --help"
echo "  python docker/scripts/verify_install.py"
echo ""

# Execute the command passed to docker run
exec "$@"