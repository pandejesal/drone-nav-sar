#!/usr/bin/env python3
"""
Unified Reconstruction Entry Point
Auto-detects best method (COLMAP or Nerfstudio) and runs full pipeline
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_command(cmd: list, description: str, cwd: Optional[Path] = None) -> bool:
    """Run a command and return success status."""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
        )
        if result.returncode != 0:
            log(f"FAILED: {description}", "ERROR")
            log(f"stderr: {result.stderr}", "ERROR")
            return False
        log(f"OK: {description}")
        return True
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {description}", "ERROR")
        return False
    except FileNotFoundError:
        log(f"NOT FOUND: {cmd[0]}", "ERROR")
        return False


def run_colmap(input_dir: Path, output_dir: Path, quality: str) -> bool:
    """Run COLMAP pipeline."""
    cmd = [
        sys.executable, "-m", "src.reconstruction.colmap_pipeline",
        str(input_dir), str(output_dir),
        "--quality", quality,
    ]
    return run_command(cmd, "COLMAP pipeline")


def run_nerfstudio(input_dir: Path, output_dir: Path, method: str, quality: str) -> bool:
    """Run Nerfstudio pipeline."""
    cmd = [
        sys.executable, "-m", "src.reconstruction.nerfstudio_pipeline",
        str(input_dir), str(output_dir),
        "--method", method,
        "--quality", quality,
    ]
    return run_command(cmd, f"Nerfstudio ({method}) pipeline")


def run_blender_cleanup(input_glb: Path, output_glb: Path, target_tris: int, texture_size: int) -> bool:
    """Run BlenderProc cleanup."""
    cmd = [
        "blender", "--background",
        "--python", str(Path(__file__).parent / "blender_cleanup.py"),
        "--",
        str(input_glb), str(output_glb),
        "--target-tris", str(target_tris),
        "--texture-size", str(texture_size),
        "--export-usd",
    ]
    return run_command(cmd, "BlenderProc cleanup")


def check_colmap_available() -> bool:
    """Check if COLMAP is installed."""
    try:
        subprocess.run(["colmap", "-h"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_nerfstudio_available() -> bool:
    """Check if Nerfstudio is installed."""
    try:
        import nerfstudio
        return True
    except ImportError:
        return False


def check_blender_available() -> bool:
    """Check if Blender is installed."""
    try:
        subprocess.run(["blender", "--version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def count_images(input_dir: Path) -> int:
    """Count image files in directory."""
    return len(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")) +
                list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG")))


def auto_select_method(input_dir: Path, quality: str) -> str:
    """Auto-select best reconstruction method based on input."""
    num_images = count_images(input_dir)

    colmap_ok = check_colmap_available()
    nerfstudio_ok = check_nerfstudio_available()

    log(f"Available methods: COLMAP={colmap_ok}, Nerfstudio={nerfstudio_ok}")
    log(f"Input images: {num_images}")

    if not colmap_ok and not nerfstudio_ok:
        log("ERROR: Neither COLMAP nor Nerfstudio available", "ERROR")
        return "none"

    # Heuristics:
    # - COLMAP: better for structured scenes, faster, needs good overlap
    # - Nerfstudio (splatfacto): better for appearance, handles sparse views, slower
    # - For many images (>50) with good overlap: COLMAP
    # - For fewer images or complex appearance: Nerfstudio
    # - If COLMAP fails, fallback to Nerfstudio

    if colmap_ok and (num_images > 30 or quality == "high"):
        return "colmap"
    elif nerfstudio_ok:
        return "nerfstudio"
    elif colmap_ok:
        return "colmap"
    else:
        return "none"


def run_reconstruction(
    input_dir: Path,
    output_dir: Path,
    method: str = "auto",
    quality: str = "medium",
) -> bool:
    """Run full reconstruction pipeline with auto method selection."""
    log("=" * 60)
    log("DroneNav-SAR Unified Reconstruction")
    log(f"  Input: {input_dir}")
    log(f"  Output: {output_dir}")
    log(f"  Method: {method}")
    log(f"  Quality: {quality}")
    log("=" * 60)

    if not input_dir.exists():
        log(f"Input directory does not exist: {input_dir}", "ERROR")
        return False

    num_images = count_images(input_dir)
    if num_images < 3:
        log(f"Need at least 3 images, found {num_images}", "ERROR")
        return False

    # Auto-select method
    if method == "auto":
        method = auto_select_method(input_dir, quality)
        log(f"Auto-selected method: {method}")

    if method == "none":
        log("No reconstruction method available", "ERROR")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run selected reconstruction method
    recon_success = False
    intermediate_dir = output_dir / "intermediate"

    if method == "colmap":
        recon_success = run_colmap(input_dir, intermediate_dir, quality)
        # COLMAP outputs mesh.glb in intermediate_dir
        mesh_glb = intermediate_dir / "mesh.glb"
    elif method == "nerfstudio":
        recon_success = run_nerfstudio(input_dir, intermediate_dir, "splatfacto", quality)
        mesh_glb = intermediate_dir / "mesh.glb"
    else:
        log(f"Unknown method: {method}", "ERROR")
        return False

    if not recon_success or not mesh_glb.exists():
        log("Reconstruction failed - no mesh produced", "ERROR")
        return False

    log(f"Reconstruction mesh: {mesh_glb}")

    # Run BlenderProc cleanup for final output
    final_glb = output_dir / "mesh.glb"
    target_tris = 50000 if quality != "high" else 100000
    texture_size = 2048 if quality != "high" else 4096

    log("--- BlenderProc Post-Processing ---")
    cleanup_success = run_blender_cleanup(
        mesh_glb, final_glb, target_tris, texture_size
    )

    if not cleanup_success or not final_glb.exists():
        log("BlenderProc cleanup failed", "ERROR")
        return False

    # Verify output
    try:
        import trimesh
        mesh = trimesh.load(str(final_glb))
        log(f"Final mesh verified:")
        log(f"  Triangles: {len(mesh.faces)}")
        log(f"  Vertices: {len(mesh.vertices)}")
        log(f"  Has UVs: {mesh.visual.uv is not None}")
        log(f"  Bounds: {mesh.bounds}")

        if len(mesh.faces) > target_tris:
            log(f"WARNING: Triangle count exceeds target ({len(mesh.faces)} > {target_tris})", "WARNING")
        if mesh.visual.uv is None:
            log("WARNING: No UV coordinates found", "WARNING")
    except Exception as e:
        log(f"Verification error: {e}", "WARNING")

    log("=" * 60)
    log("Reconstruction pipeline completed successfully!")
    log(f"  Final output: {final_glb}")
    log(f"  USD output: {output_dir / 'mesh.usd'}")
    log("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="DroneNav-SAR: Unified 3D Reconstruction from Photos"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input photos")
    parser.add_argument("output_dir", type=Path, help="Output directory for reconstruction")
    parser.add_argument(
        "--method",
        choices=["auto", "colmap", "nerfstudio"],
        default="auto",
        help="Reconstruction method (default: auto)",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality preset (default: medium)",
    )
    args = parser.parse_args()

    success = run_reconstruction(
        args.input_dir, args.output_dir, args.method, args.quality
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()