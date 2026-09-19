#!/usr/bin/env python3
"""Edge latency/throughput/memory profiler (Sprint 10, SAR-only).

Profiles a PyTorch SAR policy and/or an exported ONNX artifact on the
current host as a proxy for Jetson Orin (FP16) and RPi CM4 (INT8) targets.
On-device runs use the same CLI; budgets below encode the sprint targets.

Budgets (per step): Jetson Orin FP16 < 5ms, RPi CM4 INT8 < 20ms.

SAR-only: navigate_to / hover / drop_payload / return_home policies.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.control.edge_export import (  # noqa: E402
    decode_actor_onnx,
    deterministic_action,
    load_policy,
    numpy_forward,
    validate_with_onnxruntime,
)
from src.rl.policies import OBS_DIM  # noqa: E402

BUDGETS_MS = {"jetson": 5.0, "rpi": 20.0, "cpu": 50.0}


def _rss_mb() -> float:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / 1e6
    except Exception:
        try:
            import resource
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e3
        except Exception:
            return -1.0


def profile_pytorch(policy_path: str, batch: int = 1,
                    n_warmup: int = 5, n_iters: int = 200) -> Dict[str, Any]:
    """Latency/throughput/memory for the PyTorch policy."""
    policy = load_policy(policy_path)
    probe = np.zeros((batch, OBS_DIM), dtype=np.float32)
    mem_before = _rss_mb()
    for _ in range(n_warmup):
        deterministic_action(policy, probe)
    t0 = time.time()
    for _ in range(n_iters):
        deterministic_action(policy, probe)
    dt = (time.time() - t0) / n_iters
    return {
        "backend": "pytorch",
        "batch": int(batch),
        "latency_ms_mean": round(dt * 1000.0, 3),
        "throughput_hz": round(batch / dt, 1) if dt > 0 else 0.0,
        "rss_mb": round(_rss_mb(), 2),
        "rss_delta_mb": round(_rss_mb() - mem_before, 2),
        "n_iters": int(n_iters),
    }


def profile_onnx(onnx_path: str, batch: int = 1,
                 n_warmup: int = 5, n_iters: int = 200) -> Dict[str, Any]:
    """Latency/throughput for the ONNX artifact (numpy graph + ORT if present)."""
    payload = Path(onnx_path).read_bytes()
    decoded = decode_actor_onnx(payload)
    probe = np.zeros((batch, OBS_DIM), dtype=np.float32)
    for _ in range(n_warmup):
        numpy_forward(decoded, probe)
    t0 = time.time()
    for _ in range(n_iters):
        numpy_forward(decoded, probe)
    dt = (time.time() - t0) / n_iters
    out: Dict[str, Any] = {
        "backend": "onnx_numpy",
        "batch": int(batch),
        "latency_ms_mean": round(dt * 1000.0, 3),
        "throughput_hz": round(batch / dt, 1) if dt > 0 else 0.0,
        "onnx_bytes": len(payload),
        "rss_mb": round(_rss_mb(), 2),
        "n_iters": int(n_iters),
    }
    ort = validate_with_onnxruntime(onnx_path)
    if ort.get("available"):
        out["onnxruntime_ms"] = ort["latency_ms_mean"]
    return out


def check_budgets(results: Dict[str, Dict[str, Any]],
                  device: str) -> Dict[str, Any]:
    """Compare measured latency against the device budget."""
    budget = BUDGETS_MS.get(device, BUDGETS_MS["cpu"])
    lat = None
    for key in ("onnx_numpy", "pytorch"):
        if key in results:
            lat = results[key].get("latency_ms_mean", results[key].get("onnxruntime_ms"))
            break
    passed = bool(lat is not None and lat < budget)
    return {"device": device, "budget_ms": budget,
            "measured_ms": lat, "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge benchmark (Sprint 10)")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--device", type=str, default="jetson",
                        choices=["jetson", "rpi", "cpu"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    if not args.policy and not args.onnx:
        parser.error("supply --policy and/or --onnx")

    results: Dict[str, Dict[str, Any]] = {}
    if args.policy:
        results["pytorch"] = profile_pytorch(args.policy, batch=args.batch,
                                             n_iters=args.iters)
        print(f"PyTorch: {results['pytorch']}")
    if args.onnx:
        results["onnx_numpy"] = profile_onnx(args.onnx, batch=args.batch,
                                             n_iters=args.iters)
        print(f"ONNX: {results['onnx_numpy']}")
    verdict = check_budgets(results, args.device)
    print(f"Budget [{args.device}]: {verdict}")
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
