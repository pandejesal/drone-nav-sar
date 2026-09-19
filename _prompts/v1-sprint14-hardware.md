# Sprint 14 — Real Hardware Integration: Crazyflie + PX4 (SAR-only)

## Goal
Deploy trained policies to real Crazyflie 2.1 / Bolt via PX4 SITL → hardware. Target: >50% SR on real hardware, zero safety violations.

## Scope (modify ONLY these)
- src/hardware/crazyflie_bridge.py: NEW — Crazyflie Python lib wrapper (cflib), logging, param sync
- src/hardware/px4_interface.py: NEW — MAVLink + ROS2 bridge, offboard mode, safety checks
- src/hardware/safety_monitor.py: NEW — geofence, altitude ceiling, battery failsafe, RC override
- src/hardware/calibration.py: NEW — IMU bias, motor thrust curve, vibration analysis
- src/control/hardware_bridge.py: NEW — unified interface (sim ↔ hardware), policy hot-swap
- tests/test_hardware.py: NEW — 4 tests (calibration, safety monitor, policy deploy, failsafe)

## Hardware Stack
- Crazyflie 2.1 + Bolt deck (or Bitcraze AI deck)
- PX4 Autopilot (v1.14+) on Pixhawk / STM32
- MAVLink 2.0 over USB / radio
- Motion capture (OptiTrack) or UWB for ground truth

## Safety (non-negotiable)
- Geofence: 4x4x2.5m hardcoded in PX4 + Python monitor
- Altitude ceiling: 2.5m max
- Battery failsafe: RTL at 3.3V/cell
- RC override: always active, kills offboard on loss
- Emergency stop: physical button + software watchdog

## Acceptance
- Calibration: IMU bias < 0.01 rad/s, thrust curve R² > 0.98
- Safety monitor: 100% catch rate on simulated violations
- Policy deploy: 500-ep sim → 20-ep real, SR > 0.5, zero collisions
- Failsafe: 100% RTL on signal loss, battery low, geofence breach

## DONE
DONE-14 | files: src/hardware/*.py, tests/test_hardware.py | tests: 4/4 green | metrics: {sim2real_sr: >0.5, collisions: 0}