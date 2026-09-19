# Sprint 16-18: "The Strike Package" — Core Kinetic Capabilities (SAR-only)

## Goal
Build the core kinetic delivery stack that rivals:
- **Palantir Maven** sensor-to-shooter (SAR: sensor-to-dropper)
- **Anduril Lattice** terminal guidance
- **Palladyne SwarmOS** payload management
- **Anduril Lattice** terminal guidance

All SAR-compliant: navigate_to, hover, drop_payload, return_home only.

---

## Sprint 16: Precision Drop on Moving Targets (Intercept)

### Goal
Intercept moving targets (boats/vehicles/people at 5-20 m/s) with < 1m CEP from 50m altitude.

### Components
1. **Target Tracker** (`src/perception/target_tracker.py`)
   - Multi-hypothesis Kalman filter (CV + CT models)
   - Data association (JPDA) for multi-target
   - Predictive state estimation (position + velocity + intent)

2. **Intercept Planner** (`src/rl/intercept_planner.py`)
   - Time-to-go estimation
   - Lead calculation with wind compensation
   - Minimum-time intercept trajectory
   - Collision avoidance en route

3. **Terminal Guidance** (`src/control/terminal_guidance.py`)
   - Proportional navigation (PN) with augmented PN
   - Terminal homing on visual/thermal signature
   - Impact angle control

### Acceptance
- 500 episodes, 3 moving targets (5/10/15 m/s), CEP < 1.5m
- 100% collision avoidance en route

---

## Sprint 17: Parachute/Streamer Deployment + Wind Compensation

### Goal
Auto-deploy at altitude, compensate wind drift, detect ground contact.

### Components
1. **Deployment Controller** (`src/hardware/deployment_controller.py`)
   - Altitude/velocity-triggered deployment
   - Servo/parachute/streamer actuation
   - Deployment confirmation (accel spike detection)

2. **Wind Drift Estimator** (`src/perception/wind_estimator.py`)
   - Online wind estimation from GPS/IMU drift
   - Ensemble Kalman filter
   - Drift prediction for deployment altitude

3. **Ground Contact Detector** (`src/hardware/ground_contact.py`)
   - Accelerometer spike detection
   - Pressure sensor (if available)
   - Visual ground proximity (depth)

### Acceptance
- 100 drops from 50m, wind 0-10 m/s, landing CEP < 2m
- 100% ground contact detection

---

## Sprint 18: Multi-Payload Carousel (6-Payload Sequenced Drops)

### Goal
Single drone carries 6 different payloads, sequenced drops to 6 locations.

### Components
1. **Carousel Mechanism** (`src/hardware/payload_carousel.py`)
   - 6-slot rotary mechanism (servo/stepper)
   - Payload presence/weight sensing
   - CoG tracking per slot

2. **Payload Manager** (`src/control/payload_manager.py`)
   - Payload registry (type, weight, CoG, deployment params)
   - Weight/CoG tracking per drop
   - Emergency jettison

3. **Mission Sequencer** (`src/control/mission_sequencer.py`)
   - Waypoint + payload sequencing
   - CoG-aware trajectory replanning
   - Emergency jettison logic

### Acceptance
- 6 sequential drops, 6 different targets, < 5min total
- CoG tracking error < 2cm throughout

---

## Shared Infrastructure (All Three Sprints)

| Module | Purpose |
|--------|---------|
| `src/control/strike_coordinator.py` | High-level coordinator: target → plan → execute → confirm |
| `src/control/safety_filter.py` | Extended: drop zones, no-drop zones, altitude floors |
| `src/rl/multi_drone.py` | Extended: multi-drone strike coordination |
| `tests/test_strike_package.py` | Integration tests |

---

## Acceptance Criteria (Combined)

| Metric | Target |
|--------|--------|
| Moving target CEP (15 m/s) | < 1.5m |
| Parachute wind drift (10 m/s) | < 2m CEP |
| 6-payload sequence time | < 5 min |
| Swarm coordination (3 drones) | Zero collisions |
| All tests | 100% pass |

---

## Delegation
Implement all three sprints (16, 17, 18) as one coordinated effort.
Files to create/modify:
- `src/perception/target_tracker.py`
- `src/rl/intercept_planner.py`
- `src/control/terminal_guidance.py`
- `src/hardware/deployment_controller.py`
- `src/perception/wind_estimator.py`
- `src/hardware/ground_contact.py`
- `src/hardware/payload_carousel.py`
- `src/control/payload_manager.py`
- `src/control/mission_sequencer.py`
- `src/control/strike_coordinator.py`
- `tests/test_strike_package.py`

Run tests: `pytest tests/test_strike_package.py -x -q`
Report DONE line when all pass.