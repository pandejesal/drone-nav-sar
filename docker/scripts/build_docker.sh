#!/bin/bash
# DroneNav-SAR Docker Build Script
# Usage: ./scripts/build_docker.sh [--no-cache] [--target <stage>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"

# Default values
NO_CACHE=false
TARGET="final"
IMAGE_TAG="drone-nav-sar:latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--target <stage>] [--tag <tag>]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "DroneNav-SAR Docker Build"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Docker dir: ${DOCKER_DIR}"
echo "Target: ${TARGET}"
echo "Tag: ${IMAGE_TAG}"
echo "No cache: ${NO_CACHE}"
echo "=========================================="

cd "${PROJECT_ROOT}"

# Check for NVIDIA Docker support
if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit not detected. GPU access may not work."
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Build command
BUILD_ARGS=(
    "build"
    "--target" "${TARGET}"
    "--tag" "${IMAGE_TAG}"
    "--file" "${DOCKER_DIR}/Dockerfile"
    "--progress=plain"
)

if [[ "${NO_CACHE}" == "true" ]]; then
    BUILD_ARGS+=("--no-cache")
fi

BUILD_ARGS+=(".")

echo "Running: docker ${BUILD_ARGS[*]}"
echo ""

# Execute build
docker "${BUILD_ARGS[@]}"

BUILD_EXIT=$?

if [[ ${BUILD_EXIT} -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "BUILD SUCCESSFUL"
    echo "=========================================="
    echo "Image: ${IMAGE_TAG}"
    echo ""
    echo "Next steps:"
    echo "  1. Run verification: docker run --gpus all --rm ${IMAGE_TAG} python docker/scripts/verify_install.py"
    echo "  2. Start dev container: docker-compose -f ${DOCKER_DIR}/docker-compose.yml up -d drone-nav-sar"
    echo "  3. Enter container: docker exec -it drone-nav-sar-dev bash"
else
    echo ""
    echo "=========================================="
    echo "BUILD FAILED (exit code: ${BUILD_EXIT})"
    echo "=========================================="
    exit ${BUILD_EXIT}
fi