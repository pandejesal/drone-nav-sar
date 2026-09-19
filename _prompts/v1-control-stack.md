# Control Stack v1 — Skill Schema + Safety Filter + Bridges (SAR-only)

## Goal
Implement research (R10,R11,P13,P14,R16): NN->JSON skill -> slm_adapter (schema-constrained stub) -> safety_filter (default-deny) -> mavlink/ros2 bridges (mock transports in CI).

## Scope (create ONLY)
- src/control/__init__.py
- src/control/skill_schema.json (api_version 2, skills enum [navigate_to,hover,drop_payload,return_home], params x/y/z/yaw_deg/speed_ms, constraints ceiling_m/geofence_xyz/timeout_s, task/task_embedding_id)
- src/control/safety_filter.py (allow(skill)->{ok,reason}: skill in allowlist, target inside geofence, z<=ceiling, speed<=max, timeout>0 else reject=hover; pure function, no deps)
- src/control/slm_adapter.py (validate-against-schema stub: json.loads + jsonschema-less manual check against skill_schema.json (no new deps), fill defaults, raise on violation; LLM call deferred behind env flag, default offline)
- src/control/mavlink_bridge.py + ros2_bridge.py (MockTransport recording sent skills, heartbeat() true, translate filtered skill to setpoint dict; real MAVSDK/rclpy imports guarded try/except)
- tests/test_control.py (5 tests: valid skill passes; out-of-geofence rejects; non-allowlist rejects; ceiling violation rejects; bridge records + heartbeat)

## Ground
- C:/Users/DELL/Desktop/3d_environment_drone_neural_network/docs/ARCHITECTURE-v2.md sections 3+5

## TABOO
- No new pip deps, stdlib+json only; forward-slash; default-deny never default-allow

## Safety
SAR-ONLY. Reject=hover. No bypass path. No weapons.

## Acceptance
C:/Users/DELL/Desktop/3d_environment_drone_neural_network/.venv/Scripts/python -m pytest tests/test_control.py -q (5 passed)

## DONE
DONE-C1 | files: src/control/__init__.py,src/control/skill_schema.json,src/control/safety_filter.py,src/control/slm_adapter.py,src/control/mavlink_bridge.py,src/control/ros2_bridge.py,tests/test_control.py | tests: 5/5 green | metrics: {safety:deny-by-default}
