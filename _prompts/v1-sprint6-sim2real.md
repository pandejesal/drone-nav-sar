# Sprint 6 — Sim2Real Gap Closure (SAR-only)

## Goal
Close the sim2real gap: domain randomization + system identification → deploy on real Crazyflie 2.1. Target: policy trained in mock transfers to real hardware with >50% SR on same A→B task.

## Scope (modify ONLY these)
- src/sim/domain_randomization.py: ADD aggressive randomizer config + system ID data collection utilities
- src/control/edge_export.py: ADD ONNX export + TensorRT optimization + runtime validation
- src/control/mavlink_bridge.py: ADD real MAVSDK connection + safety filter integration
- src/control/ros2_bridge.py: ADD real ROS2 node + PX4 SITL bridge
- src/eval/sim2real.py: NEW — sim2real evaluation harness (mock vs real comparison)
- tests/test_sim2real.py: NEW — 3 tests (ONNX export valid, MAVSDK connect mock, system ID data format)

## System ID Protocol
1. Collect flight data: motor commands + IMU + position (motion capture or onboard)
2. Fit dynamics parameters: mass, thrust_coeff, drag_coeff, motor_tau, inertia
3. Update mock backend params to match real drone
4. Re-train/fine-tune policy with identified params

## Domain Randomization (Aggressive)
- Mass: ±20%, Inertia: ±30%, Thrust: ±15%, Motor tau: ±50%
- Wind: OU base 0-3 m/s + gusts 3-5 m/s
- Sensor noise: IMU accel 0.01-0.1, GPS 0.05-0.5
- Lighting/textures: full ranges for visual policies later

## Acceptance
.venv/Scripts/python -m pytest tests/test_sim2real.py -q (3 passed)
.venv/Scripts/python -m src.control.edge_export --policy policy_sprint5_final.pt --output policy.onnx → valid ONNX
.venv/Scripts/python -m src.eval.sim2real --policy policy_sprint5_final.pt --mock-only → SR comparison report

## DONE
DONE-6 | files: src/sim/domain_randomization.py,src/control/edge_export.py,src/control/mavlink_bridge.py,src/control/ros2_bridge.py,src/eval/sim2real.py,tests/test_sim2real.py | tests: 3/3 green | metrics: {onnx:valid,mavlink:connects,sysid:format}