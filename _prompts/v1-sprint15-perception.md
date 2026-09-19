# Sprint 15 — Advanced Perception: 3D Detection + SLAM (SAR-only)

## Goal
Onboard perception for victim detection, obstacle avoidance, and SLAM. Target: >90% victim detection AP, <10cm localization drift, runs on Jetson Orin.

## Scope (modify ONLY these)
- src/perception/yolo_detector.py: NEW — YOLOv8n/YOLOv10n ONNX, TensorRT FP16, victim/person detection
- src/perception/depth_estimator.py: NEW — MiDaS/DepthAnything ONNX, metric depth from mono
- src/perception/slam.py: NEW — ORB-SLAM3 / VIO fusion, loop closure, map serialization
- src/perception/fusion.py: NEW — multi-sensor fusion (RGB-D + IMU + optical flow)
- src/control/perception_bridge.py: NEW — perception → policy interface (detections → task embeddings)
- tests/test_perception.py: NEW — 4 tests (detection AP, depth RMSE, SLAM drift, fusion latency)

## Perception Stack (SAR-only)
```
Camera (640x480 @ 30Hz)
    ↓
YOLOv8n (victim detection) → bbox + conf → task_embedding (detect)
    ↓
DepthAnything (metric depth) → point cloud → obstacle map
    ↓
VIO (IMU + optical flow) → 6DOF pose → SLAM map
    ↓
Fusion → obstacle avoidance waypoints + victim positions → LocalPolicy
```

## Model Specs (edge-optimized)
| Model | Size | Latency (Jetson Orin) | Precision |
|-------|------|----------------------|-----------|
| YOLOv8n | 3.2 MB | 4.2 ms | FP16 TensorRT |
| DepthAnything-Small | 11 MB | 8.5 ms | FP16 TensorRT |
| ORB-SLAM3 | N/A | 12 ms | FP32 CPU |

## Acceptance
- Victim detection: AP@0.5 > 0.9 on SAR test set (1000 images)
- Depth: RMSE < 0.15m at 5m range
- SLAM: drift < 10cm over 50m trajectory
- Fusion latency: < 20ms end-to-end
- All models export to ONNX → TensorRT FP16

## DONE
DONE-15 | files: src/perception/*.py, src/control/perception_bridge.py, tests/test_perception.py | tests: 4/4 green | metrics: {AP@0.5: >0.9, depth_rmse: <0.15m, slam_drift: <10cm}