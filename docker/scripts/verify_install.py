#!/usr/bin/env python3
"""
DroneNav-SAR Installation Verification Script
Run inside the Docker container to verify all critical dependencies.
"""

import sys
import subprocess
import importlib
from typing import Tuple, List

def check_import(module_name: str, attr: str = None) -> Tuple[bool, str]:
    """Try to import a module and optionally get version."""
    try:
        mod = importlib.import_module(module_name)
        if attr and hasattr(mod, attr):
            version = getattr(mod, attr)
        elif hasattr(mod, '__version__'):
            version = mod.__version__
        else:
            version = "OK (no version attr)"
        return True, f"{module_name}: {version}"
    except ImportError as e:
        return False, f"{module_name}: FAILED - {e}"
    except Exception as e:
        return False, f"{module_name}: ERROR - {e}"

def check_command(cmd: List[str]) -> Tuple[bool, str]:
    """Run a command and check it succeeds."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            output = result.stdout.strip().split('\n')[0]
            return True, f"{' '.join(cmd)}: {output}"
        else:
            return False, f"{' '.join(cmd)}: FAILED (exit {result.returncode}) - {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(cmd)}: TIMEOUT"
    except FileNotFoundError:
        return False, f"{' '.join(cmd)}: NOT FOUND"
    except Exception as e:
        return False, f"{' '.join(cmd)}: ERROR - {e}"

def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability for JAX and PyTorch."""
    results = []
    
    # PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            results.append(f"PyTorch CUDA: {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
        else:
            results.append("PyTorch CUDA: NOT AVAILABLE")
    except Exception as e:
        results.append(f"PyTorch CUDA: ERROR - {e}")
    
    # JAX CUDA
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            results.append(f"JAX GPU: {len(gpu_devices)} device(s) - {gpu_devices[0]}")
        else:
            results.append("JAX GPU: NOT AVAILABLE (CPU only)")
    except Exception as e:
        results.append(f"JAX GPU: ERROR - {e}")
    
    return all("NOT AVAILABLE" not in r and "ERROR" not in r for r in results), "; ".join(results)

def main():
    print("=" * 60)
    print("DroneNav-SAR Installation Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    results = []
    
    # Core Python packages
    print("--- Core Python Packages ---")
    packages = [
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("pandas", "__version__"),
        ("torch", "__version__"),
        ("torchvision", "__version__"),
        ("jax", "__version__"),
        ("jaxlib", "__version__"),
        ("gymnasium", "__version__"),
        ("stable_baselines3", "__version__"),
        ("cleanrl", "__version__"),
        ("sample_factory", "__version__"),
        ("dreamerv3", "__version__"),
        ("optuna", "__version__"),
        ("wandb", "__version__"),
        ("tensorboard", "__version__"),
        ("cv2", "__version__"),  # opencv-python
        ("open3d", "__version__"),
        ("trimesh", "__version__"),
        ("yaml", "__version__"),  # pyyaml
        ("tqdm", "__version__"),
        ("mavsdk", "__version__"),
        ("rclpy", "__version__"),
        ("px4_msgs", "__version__"),
    ]
    
    for pkg, attr in packages:
        ok, msg = check_import(pkg, attr)
        results.append((ok, msg))
        status = "✓" if ok else "✗"
        print(f"  {status} {msg}")
        if not ok:
            all_passed = False
    
    print()
    
    # CUDA/GPU
    print("--- GPU / CUDA ---")
    ok, msg = check_cuda()
    results.append((ok, msg))
    status = "✓" if ok else "✗"
    print(f"  {status} {msg}")
    if not ok:
        all_passed = False
    
    print()
    
    # System commands
    print("--- System Commands ---")
    commands = [
        (["blender", "--version"], "Blender"),
        (["colmap", "-h"], "COLMAP"),
        (["gz", "sim", "--version"], "Gazebo"),
        (["ros2", "--version"], "ROS2"),
        (["python3", "-c", "import isaacsim; print(isaacsim.__version__)"], "Isaac Sim (Python)"),
    ]
    
    for cmd, name in commands:
        ok, msg = check_command(cmd)
        results.append((ok, msg))
        status = "✓" if ok else "✗"
        print(f"  {status} {msg}")
        if not ok:
            all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 60)
        failed = [msg for ok, msg in results if not ok]
        for msg in failed:
            print(f"  FAIL: {msg}")
        return 1

if __name__ == "__main__":
    sys.exit(main())