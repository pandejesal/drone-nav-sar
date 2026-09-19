# Sprint 10 — Edge Deployment: ONNX/TensorRT → Jetson Orin / RPi CM4 (SAR-only)

## Goal
Export trained policies to ONNX, optimize with TensorRT for Jetson Orin, quantize for RPi CM4. Validate inference latency < 10ms on Jetson, < 50ms on RPi. Target: deployable SAR policy on edge hardware.

## Scope (modify ONLY these)
- src/control/edge_export.py: ADD TensorRT optimization (FP16/INT8), dynamic quantization for RPi, ONNX Runtime validation
- src/control/onnx_validator.py: NEW — compare PyTorch vs ONNX vs TensorRT outputs, latency benchmark
- src/rl/ppo_nav.py: ADD --export flag, --device flag (jetson/rpi), --precision flag (fp32/fp16/int8)
- src/eval/edge_benchmark.py: NEW — latency, throughput, memory profiling on target hardware
- tests/test_edge_export.py: NEW — 3 tests (ONNX validity, TensorRT build, quantization accuracy)

## Export Pipeline
```
PyTorch policy.pt → ONNX (opset 17) → TensorRT engine (FP16) → Jetson Orin
                → ONNX Runtime (dynamic quant INT8) → RPi CM4
```

## Acceptance
.venv/Scripts/python -m pytest tests/test_edge_export.py -q (3 passed)
.venv/Scripts/python -m src.control.edge_export --policy policy_sprint9_final.pt --output policy.onnx --tensorrt --fp16 --device jetson → valid TensorRT engine
.venv/Scripts/python -m src.control.onnx_validator --policy policy_sprint9_final.pt --onnx policy.onnx --tolerance 1e-5 → max diff < 1e-5
Latency: Jetson Orin FP16 < 5ms, RPi CM4 INT8 < 20ms (per step)

## DONE
DONE-10 | files: src/control/edge_export.py,src/control/onnx_validator.py,src/rl/ppo_nav.py,src/eval/edge_benchmark.py,tests/test_edge_export.py | tests: 3/3 green | metrics: {jetson_latency_ms:..., rpi_latency_ms:..., max_diff:...}