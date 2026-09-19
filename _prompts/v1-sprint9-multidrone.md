# Sprint 9 — Multi-Drone Decentralized Coordination (SAR-only)

## Goal
Multiple drones cooperatively search a building, share victim detections, avoid collisions, and deliver med-kits. Target: 3 drones, >70% joint mission SR with zero collisions.

## Scope (modify ONLY these)
- src/rl/multi_drone.py: NEW — MultiDroneEnv (vectorized multi-agent), collision avoidance reward, communication protocol
- src/rl/ppo_nav.py: ADD --multi-drone flag, shared policy with agent_id conditioning
- src/control/mission_planner.py: INTEGRATE multi-drone coordination (assignment, deconfliction)
- src/control/comms.py: NEW — lightweight message bus (victim spots, intent, position)
- tests/test_multi_drone.py: NEW — 4 tests (collision avoidance, victim sharing, formation, full mission)

## Architecture
```
GlobalPlanner (shared) → Subgoal assignment per drone
       ↓
LocalPolicy (shared weights, agent_id input) → Motor commands
       ↓
Comms bus (broadcast: victim_found, intent, position) → Collision avoidance
```

## Multi-Drone Details
- 3 drones, shared policy weights, agent_id one-hot concatenated to obs
- Collision avoidance: reciprocal velocity obstacles (RVO) reward penalty
- Victim sharing: broadcast detection → all drones update belief
- Task assignment: greedy auction on victim spots
- Formation: loose (5m separation) during transit, tight at victims

## Acceptance
.venv/Scripts/python -m pytest tests/test_multi_drone.py -q (4 passed)
.venv/Scripts/python -m src.rl.ppo_nav --episodes 2000 --seed 0 --multi-drone 3 --hierarchical --curriculum --dynamic-obstacles --sar-mission --save-path policy_sprint9.pt → joint SR > 0.7, zero collisions

## DONE
DONE-9 | files: src/rl/multi_drone.py,src/rl/ppo_nav.py,src/control/mission_planner.py,src/control/comms.py,tests/test_multi_drone.py | tests: 4/4 green | metrics: {joint_sr:..., collisions:0}