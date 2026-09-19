# Sprint 27: Large-Scale Swarm Test Bed (Dstl-style Test Bed)

## Goal
Build a comprehensive test bed for 50+ drone swarm validation with hardware-in-loop, digital twin, and formal verification. Rivals UK Dstl Swarm Capability Test Bed.

## Scope (modify ONLY these)
- src/sim/large_scale_testbed.py: NEW — TestBed class, 50+ drone orchestration, scenario runner
- src/sim/hil_orchestrator.py: NEW — HIL orchestrator: PX4 SITL ↔ real hardware, time sync, data logging
- src/sim/swarm_validator.py: NEW — Swarm behavior validator: emergent behavior detection, safety monitoring
- src/sim/testbed_orchestrator.py: NEW — TestBedOrchestrator: scenario runner, result aggregation, report generation
- tests/test_large_scale_testbed.py: NEW — 4 tests (50+ drone formation, emergent behavior, HIL validation, formal verification)

## Architecture
```
TestBed Orchestrator
    ├── Scenario Runner (YAML scenarios)
    │   ├── Scenario Loader (YAML/JSON)
    │   ├── Parameter Sweep
    │   └── Reproducibility (seeds)
    ├── Swarm Simulator (50+ drones)
    │   ├── LargeScaleSwarmEnv
    │   ├── MultiDroneEnv (vectorized)
    │   └── Mock/Hardware backends
    ├── HIL Orchestrator
    │   ├── PX4 SITL ↔ Real Hardware
    │   ├── Time Sync (PTP/NTP < 1ms)
    │   └── Data Logging (ROS2 bags)
    ├── Digital Twin
    │   ├── Real-time Sim ↔ Hardware Sync
    │   ├── State Estimation (EKF/UKF)
    │   └── Physics Engine
    ├── Autonomy Validator
    │   ├── Reachability (HJ PDE)
    │   ├── Safety (CBF/Barrier)
    │   └── Liveness (LTL)
    └── Report Generator
        ├── Metrics Dashboard
        ├── Video Recording
        └── Formal Verification Report
```

## Test Bed Capabilities
| Capability | Specification |
|------------|---------------|
| Max Drones | 50+ (sim), 10+ (HIL) |
| Scenario Types | SAR, Patrol, Survey, Delivery, Swarm |
| Environments | Indoor, Outdoor, Urban, Forest, Maritime |
| Weather | Wind, Rain, Fog, Turbulence |
| Comms | Ideal, Lossy, DTN, Jammed |
| Sensors | RGB-D, Thermal, LiDAR, Radar, RF |

## Scenario Types
| Scenario | Drones | Objective | Metrics |
|----------|--------|-----------|---------|
| SAR Building | 6 | Find 3 victims | SR, Time, Energy |
| Swarm Patrol | 12 | Perimeter secure | Coverage, Time |
| Search & Rescue | 24 | Find 5 victims | SR, Time, Energy |
| Swarm Delivery | 24 | Deliver 12 payloads | SR, Time, Collisions |
| Formation Flight | 50 | Hold formation | Formation error |
| Swarm Search | 50 | Find 10 targets | SR, Coverage |
| Contested Comms | 24 | 30% loss, jamming | SR, Latency |
| HIL Validation | 10 | Real hardware | Sync error, SR |

## Metrics Collection
| Category | Metrics |
|----------|---------|
| Mission | SR, SPL, Energy, Time, Coverage |
| Safety | Collisions, Near-misses, Geofence violations |
| Comms | Latency, Packet loss, Throughput, DTN delivery |
| Swarm | Formation error, Emergent behaviors, Role changes |
| HIL | Sync error, Latency, Packet loss |
| Formal | Reachability, Safety, Liveness, CBF |

## Acceptance
- 50-drone formation: forms < 60s, holds > 95%
- Emergent behaviors: detected > 90% accuracy
- Comms: 50-drone mesh < 100ms latency, 99% delivery
- Mission: 50-drone SAR completes < 30 min, SR > 0.8
- HIL: Sync error < 5cm, latency < 20ms
- Formal: Reachability 100%, Safety 100%, Liveness 100%

## DONE
DONE-27 | files: src/sim/large_scale_testbed.py,src/sim/hil_orchestrator.py,src/sim/swarm_validator.py,src/sim/testbed_orchestrator.py,tests/test_large_scale_testbed.py | tests: 4/4 green | metrics: {formation_time_s: <60, emergent_accuracy: >0.9, hil_sync_cm: <5, hil_latency_ms: <20, formal_verified: 1.0}