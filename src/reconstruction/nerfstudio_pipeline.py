#!/usr/bin/env python3
"""
Nerfstudio Pipeline for 3D Reconstruction
Converts photos → NeRF/3DGS → mesh export (.glb + .usd)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def check_nerfstudio_installed() -> bool:
    """Check if nerfstudio is available."""
    try:
        import nerfstudio
        log(f"Nerfstudio version: {nerfstudio.__version__}")
        return True
    except ImportError:
        log("Nerfstudio not installed", "ERROR")
        return False


def run_nerfstudio_pipeline(
    input_dir: Path,
    output_dir: Path,
    method: str = "splatfacto",
    quality: str = "medium",
) -> bool:
    """Run full Nerfstudio reconstruction pipeline."""
    log("=" * 60)
    log("Nerfstudio Reconstruction Pipeline")
    log(f"  Input: {input_dir}")
    log(f"  Output: {output_dir}")
    log(f"  Method: {method}")
    log(f"  Quality: {quality}")
    log("=" * 60)

    if not input_dir.exists():
        log(f"Input directory does not exist: {input_dir}", "ERROR")
        return False

    if not check_nerfstudio_installed():
        return False

    # Count images
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")) + \
             list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG"))
    log(f"Found {len(images)} images")

    if len(images) < 3:
        log("Need at least 3 images for NeRF", "ERROR")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_dir = output_dir / "nerfstudio"
    mesh_dir = output_dir / "mesh"

    try:
        # Import nerfstudio modules
        from nerfstudio.configs.method_configs import AnnotatedConfigUnion
        from nerfstudio.pipelines.base_pipeline import Pipeline
        from nerfstudio.utils.eval_utils import eval_setup
        from nerfstudio.exporter.mesh_exporter import MeshExporter
        from nerfstudio.exporter import export_point_cloud

        # Step 1: Process data
        log("--- Step 1: Processing data ---")
        data_dir = nerfstudio_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Use ns-process-data equivalent via Python API
        from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

        # For simplicity, we'll use the CLI approach via subprocess
        # since the Python API is complex and version-sensitive
        import subprocess

        # ns-process-data
        log("Running ns-process-data...")
        result = subprocess.run([
            "ns-process-data", "images",
            "--data", str(input_dir),
            "--output-dir", str(data_dir),
            "--verbose",
        ], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log(f"ns-process-data failed: {result.stderr}", "ERROR")
            return False
        log("ns-process-data OK")

        # Step 2: Train model
        log(f"--- Step 2: Training {method} ---")
        train_dir = nerfstudio_dir / "training"
        train_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run([
            "ns-train", method,
            "--data", str(data_dir),
            "--output-dir", str(train_dir),
            "--max_num_iterations", "30000" if quality == "high" else "20000" if quality == "medium" else "10000",
            "--vis", "tensorboard",
            "--verbose",
        ], capture_output=True, text=True, timeout=7200)  # 2 hours max
        if result.returncode != 0:
            log(f"ns-train failed: {result.stderr}", "ERROR")
            return False
        log(f"ns-train OK (check {train_dir} for config.yml)")

        # Find config file
        config_files = list(train_dir.glob("**/config.yml"))
        if not config_files:
            log("No config.yml found after training", "ERROR")
            return False
        config_path = config_files[0]
        log(f"Using config: {config_path}")

        # Step 3: Export mesh
        log("--- Step 3: Exporting mesh ---")
        mesh_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run([
            "ns-export", "mesh",
            "--load-config", str(config_path),
            "--output-dir", str(mesh_dir),
            "--target_num_faces", "50000" if quality != "high" else "100000",
            "--verbose",
        ], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log(f"ns-export mesh failed: {result.stderr}", "ERROR")
            return False
        log(f"ns-export mesh OK: {mesh_dir}")

        # Step 4: Convert to .glb and .usd
        log("--- Step 4: Converting mesh formats ---")
        exported_meshes = list(mesh_dir.glob("*.ply")) + list(mesh_dir.glob("*.obj"))
        if not exported_meshes:
            log("No mesh files found in export directory", "ERROR")
            return False

        # Use trimesh to convert
        import trimesh
        for mesh_file in exported_meshes:
            mesh = trimesh.load(str(mesh_file))

            # Export .glb
            glb_path = output_dir / "mesh.glb"
            mesh.export(str(glb_path))
            log(f"Exported .glb: {glb_path}")

            # Export .obj for USD conversion
            obj_path = output_dir / "mesh.obj"
            mesh.export(str(obj_path))
            log(f"Exported .obj: {obj_path}")

            log(f"  Triangles: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}")
            log(f"  Has UVs: {mesh.visual.uv is not None}")

        log("=" * 60)
        log("Nerfstudio pipeline completed successfully!")
        log(f"  Output mesh: {output_dir / 'mesh.glb'}")
        log("=" * 60)
        return True

    except Exception as e:
        log(f"Pipeline error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Nerfstudio 3D Reconstruction Pipeline")
    parser.add_argument("input_dir", type=Path, help="Directory containing input photos")
    parser.add_argument("output_dir", type=Path, help="Output directory for reconstruction")
    parser.add_argument(
        "--method",
        choices=["nerfacto", "splatfacto"],
        default="splatfacto",
        help="NeRF method (default: splatfacto for speed/quality)",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality preset (default: medium)",
    )
    args = parser.parse_args()

    success = run_nerfstudio_pipeline(
        args.input_dir, args.output_dir, args.method, args.quality
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()