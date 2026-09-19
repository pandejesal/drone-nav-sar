# Sprint 22: Digital Twin / HIL (Dstl-like Test Bed)

## Goal
Real-time digital twin with hardware-in-the-loop (HIL) simulation for autonomy validation. Rivals UK Dstl Swarm Capability Test Bed.

## Scope (modify ONLY these)
- src/sim/digital_twin.py: NEW — DigitalTwin class, real-time sim ↔ hardware sync, state estimation
- src/sim/hil_interface.py: NEW — HIL interface: PX4 SITL ↔ real hardware, time sync, data logging
- src/sim/autonomy_validator.py: NEW — Formal verification: reachability, safety invariants, liveness
- src/sim/digital_twin_bridge.py: NEW — Twin ↔ hardware bridge, state sync, latency compensation
- tests/test_digital_twin.py: NEW — 4 tests (state sync, HIL loop, reachability, safety invariants)

## Architecture
```
Real Drone (PX4) ←→ HIL Interface ←→ Digital Twin (sim)
    ↕                    ↕                    ↕
  Sensors             Actuators            Physics
    ↕                    ↕                    ↕
  State Est.        Actuator Cmds         Physics Engine
    ↕                    ↕                    ↕
  State Sync  ←→  State Sync  ←→  Physics Step
```

## Digital Twin Components
| Component | Purpose | Latency Target |
|-----------|---------|----------------|
| State Estimator | EKF/UKF fusion (IMU + GPS + VO) | < 5ms |
| Physics Engine | Rigid body + aerodynamics | < 10ms |
| Environment | Wind, obstacles, thermal | < 5ms |
| Sensor Sim | Camera, LiDAR, IMU, GPS | < 10ms |
| Actuator Model | Motor + prop dynamics | < 2ms |

## HIL Interface
- PX4 SITL ↔ Real hardware (MAVLink)
- Time sync: NTP/PTP < 1ms
- Data rate: 100Hz state, 50Hz control
- Latency budget: < 20ms round-trip

## Autonomy Validator
| Check | Method | Target |
|-------|--------|--------|
| Reachability | Hamilton-Jacobi / SOS | 100% coverage |
| Safety invariants | Barrier functions | 100% |
| Liveness | Temporal logic (LTL) | 100% |
| Collision avoidance | CBF verification | 100% |

## Acceptance
- State sync error: < 5cm position, < 0.1 m/s velocity
- HIL loop latency: < 20ms round-trip
- Reachability: 100% coverage on test scenarios
- Safety invariants: 0 violations in 1000 episodes

## DONE
DONE-22 | files: src/sim/digital_twin.py,src/sim/hil_interface.py,src/sim/autonomy_validator.py,src/sim/digital_twin_bridge.py,tests/test_digital_twin.py | tests: 4/4 green | metrics: {sync_error_cm: <5, hil_latency_ms: <20, reachability: 1.0, safety_violations: 0}