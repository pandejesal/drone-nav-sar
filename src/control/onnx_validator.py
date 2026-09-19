#!/usr/bin/env python3
"""ONNX/TensorRT output validator (Sprint 10, SAR-only).

Compares PyTorch vs ONNX vs TensorRT policy outputs on shared probes and
reports max absolute difference plus a small latency benchmark.

SAR-only: navigate_to / hover / drop_payload / return_home policies.
No weapons, no targeting, no kinetic.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _probes(n: int = 8, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [np.zeros((1, OBS_DIM), np.float32),
           np.ones((1, OBS_DIM), np.float32)]
    while len(out) < max(2, n):
        out.append(rng.uniform(-2, 2, size=(2, OBS_DIM)).astype(np.float32))
    return out[:max(2, n)]


def compare_outputs(policy_path: str, onnx_path: str,
                    trt_engine_path: Optional[str] = None,
                    tolerance: float = 1e-5,
                    n_probes: int = 8) -> Dict[str, Any]:
    """Compare PyTorch vs ONNX (vs TensorRT manifest) outputs.

    TensorRT engines need a Jetson GPU; on CPU CI the engine artifact is a
    JSON build-manifest (see edge_export.optimize_tensorrt), so the TRT leg
    is reported as skipped with a reason instead of failing.
    """
    policy = load_policy(policy_path)
    payload = Path(onnx_path).read_bytes()
    decoded = decode_actor_onnx(payload)
    probes = _probes(n_probes)
    worst_pt_onnx = 0.0
    for obs in probes:
        ref = deterministic_action(policy, obs)
        got = numpy_forward(decoded, obs)
        worst_pt_onnx = max(worst_pt_onnx, float(np.max(np.abs(ref - got))))

    trt: Dict[str, Any] = {"checked": False,
                           "reason": "no TensorRT engine supplied"}
    if trt_engine_path is not None and Path(trt_engine_path).exists():
        # CPU CI manifests are JSON; a real engine would be binary.
        try:
            import json
            manifest = json.loads(Path(trt_engine_path).read_text())
            trt = {"checked": False, "simulated": True,
                   "reason": f"simulated engine manifest ({manifest.get('precision')})",
                   "max_diff": 0.0}
        except Exception:
            trt = {"checked": False,
                   "reason": "binary engine needs Jetson TensorRT runtime; skipped on CPU"}

    max_diff = worst_pt_onnx
    return {
        "pytorch_vs_onnx_max_diff": worst_pt_onnx,
        "trt": trt,
        "max_diff": max_diff,
        "tolerance": float(tolerance),
        "passed": bool(max_diff <= tolerance),
        "n_probes": len(probes),
    }


def benchmark_latency(policy_path: str, onnx_path: str,
                      n_warmup: int = 5, n_iters: int = 100) -> Dict[str, Any]:
    """Per-step latency (ms) for PyTorch vs ONNX-numpy vs ONNX Runtime."""
    policy = load_policy(policy_path)
    payload = Path(onnx_path).read_bytes()
    decoded = decode_actor_onnx(payload)
    probe = np.zeros((1, OBS_DIM), dtype=np.float32)

    for _ in range(n_warmup):
        deterministic_action(policy, probe)
        numpy_forward(decoded, probe)
    t0 = time.time()
    for _ in range(n_iters):
        deterministic_action(policy, probe)
    pt_ms = (time.time() - t0) / n_iters * 1000.0
    t0 = time.time()
    for _ in range(n_iters):
        numpy_forward(decoded, probe)
    onnx_ms = (time.time() - t0) / n_iters * 1000.0

    out: Dict[str, Any] = {
        "pytorch_ms": round(pt_ms, 3),
        "onnx_numpy_ms": round(onnx_ms, 3),
        "n_iters": int(n_iters),
    }
    ort = validate_with_onnxruntime(onnx_path)
    if ort.get("available"):
        out["onnxruntime_ms"] = ort["latency_ms_mean"]
    else:
        out["onnxruntime"] = ort.get("reason")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ONNX vs PyTorch (Sprint 10)")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--trt-engine", type=str, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--n-probes", type=int, default=8)
    args = parser.parse_args()

    rep = compare_outputs(args.policy, args.onnx,
                          trt_engine_path=args.trt_engine,
                          tolerance=args.tolerance, n_probes=args.n_probes)
    print(f"max_diff={rep['max_diff']:.2e} tolerance={rep['tolerance']:.0e} "
          f"passed={rep['passed']}")
    print(f"PyTorch vs ONNX: {rep['pytorch_vs_onnx_max_diff']:.2e}; TRT: {rep['trt']}")
    lat = benchmark_latency(args.policy, args.onnx)
    print(f"Latency: {lat}")
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
