# Tutorial 5 — Edge Deployment: ONNX to TensorRT / ONNX Runtime (~30 min, SAR-only)

Run the SAR policy onboard: Jetson via TensorRT (FP16) or Raspberry Pi CM4
via ONNX Runtime (INT8). The exported graph contains the navigation policy
only — safety filter and geofence stay in the control layer on the vehicle.

## Steps

### 1. Export to ONNX

```python
from src.control.edge_export import export_onnx
export_onnx("checkpoints/policy.pt", "dist/policy.onnx")
```

Validate the graph (`src/control/onnx_validator.py`): input/output shapes,
opset support, deterministic outputs vs the torch checkpoint.

```bash
pytest tests/test_edge_export.py -q
```

### 2a. Jetson path — TensorRT FP16

```bash
trtexec --onnx=dist/policy.onnx --saveEngine=dist/policy.plan --fp16
```

The release workflow ships a `tensorrt-manifest.json` (engine name, precision,
source ONNX); the `.plan` itself is built on the Jetson at deploy time since
engines are GPU-specific.

### 2b. Pi CM4 path — ONNX Runtime INT8

```python
from src.control.edge_export import quantize_int8
quantize_int8("dist/policy.onnx", "dist/policy_int8.onnx")
```

### 3. Benchmark on device

```bash
python -m src.eval.edge_benchmark --model dist/policy.plan --device jetson
python -m src.eval.edge_benchmark --model dist/policy_int8.onnx --device rpi_cm4
```

Target: policy inference < 20 ms/frame on both targets; full eval reports
latency p50/p99 plus power draw.

## Troubleshooting

| Symptom | Fix |
|---|---|
| trtexec op error | Check opset in `onnx_validator.py`; re-export pinned opset |
| INT8 accuracy drop | Recalibrate with SAR-space activations, re-run `eval_all.py` gate |
| OOM on CM4 | INT8 + reduce observation history window |

## Next

Tutorial 6 — full multi-room SAR mission with language commands.
