# Sprint 7 — SAR Scenario: Multi-Room + Victim Detection + Med-Kit Drop (SAR-only)

## Goal
Full SAR mission in a multi-room building: navigate corridors, detect victims (YOLO), deliver med-kit, return home. Target: >60% mission success in mock with dynamic obstacles.

## Scope (modify ONLY these)
- src/sim/mock_backend.py: ADD multi-room layout generation + victim entities + med-kit drop mechanics
- src/control/yolo_detector.py: NEW — YOLOv8 victim detection (mock: synthetic bbox + confidence)
- src/control/mission_planner.py: NEW — high-level mission state machine (nav → detect → hover → drop → return)
- src/rl/ppo_nav.py: ADD `--sar-mission` flag, multi-task heads (nav, hover, detect, drop, return)
- tests/test_sar.py: NEW — 4 tests (victim detection, drop mechanics, mission state machine, full mission)

## Multi-Room Layout
- 4 rooms (4×4m each) connected by corridors (1m wide)
- Walls: collision boundaries, doors: cycling open/close (from Sprint 5)
- Victims: static cylinders (radius 0.3m, height 1.8m) at random room positions
- Med-kit: attached to drone, dropped via `drop_payload` skill at victim location

## YOLO Detector (Mock)
- Input: synthetic 64×64 RGB from mock camera
- Output: list of detections {class: "victim", bbox: [x,y,w,h], conf: 0.0-1.0}
- Mock: returns GT bbox + noise when victim in FOV, empty otherwise
- Real: YOLOv8n ONNX (6.2MB, 8.7ms on Jetson)

## Mission State Machine
```
IDLE → NAV_TO_ROOM → SEARCH_VICTIM → HOVER_OVER_VICTIM → DROP_MEDKIT → RETURN_HOME → COMPLETE
```
- Transitions: goal_reached, victim_detected, drop_complete, home_reached
- Failure: crash, timeout, lost_victim → RETURN_HOME

## Acceptance
.venv/Scripts/python -m pytest tests/test_sar.py -q (4 passed)
.venv/Scripts/python -m src.rl.ppo_nav --episodes 500 --seed 0 --sar-mission --curriculum --dynamic-obstacles --save-path policy_sprint7.pt → mission SR > 0.6

## DONE
DONE-7 | files: src/sim/mock_backend.py,src/control/yolo_detector.py,src/control/mission_planner.py,src/rl/ppo_nav.py,tests/test_sar.py | tests: 4/4 green | metrics: {mission_sr:...}