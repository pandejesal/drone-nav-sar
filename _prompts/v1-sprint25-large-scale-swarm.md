# Sprint 25: Large-Scale Swarm (50+ Drones, CETC/PLA Scale)

## Goal
Scale swarm to 50+ drones with emergent behaviors, hierarchical command, and resilient comms. Rivals CETC/PLA swarm research.

## Scope (modify ONLY these)
- src/sim/large_scale_swarm.py: NEW — LargeScaleSwarmEnv, 50+ drone simulation, emergent behaviors
- src/control/swarm_coordinator.py: EXTEND — Hierarchical command (platoon/company/battalion), emergent behavior detection
- src/comms/mesh_c2.py: EXTEND — Scalable routing (OLSRv2/BATMAN-adv), hierarchical addressing
- src/control/swarm_coordinator.py: EXTEND — Hierarchical command (platoon→company→battalion), emergent behavior detection
- src/rl/multi_drone.py: EXTEND — 50+ drone PPO, shared critic, centralized training decentralized execution
- tests/test_large_scale_swarm.py: NEW — 4 tests (50+ drone formation, emergent behavior, comms scalability, mission completion)

## Architecture
```
Battalion Commander (1)
    ├── Company A (16 drones)
    │   ├── Platoon 1 (4 drones) — Scout
    │   ├── Platoon 2 (4 drones) — Search
    │   ├── Platoon 3 (4 drones) — Relay
    │   └── Platoon 4 (4 drones) — Reserve
    ├── Company B (16 drones) — Search/Rescue
    ├── Company C (16 drones) — Perimeter
    └── Reserve (2 drones) — HQ
```

## Scalable Mesh C2
| Scale | Routing | Addressing | Latency |
|-------|---------|------------|---------|
| 10 drones | AODV | Flat | < 10ms |
| 50 drones | OLSRv2 | Hierarchical (platoon/company) | < 50ms |
| 100+ drones | BATMAN-adv | Hierarchical (company/battalion) | < 100ms |

## Emergent Behaviors
| Behavior | Trigger | Detection |
|----------|---------|-----------|
| Flocking | Proximity + alignment | Velocity alignment > 0.8 |
| Flanking | Enemy detected | Flanking maneuver detection |
| Encircling | Target surrounded | Encirclement metric > 0.9 |
| Swarming | High density | Density > 0.5 drones/m³ |
| Self-healing | Node loss | Topology repair < 5s |

## Hierarchical Command
| Level | Drones | Authority | Latency |
|-------|--------|-----------|---------|
| Drone | 1 | Execute | < 10ms |
| Platoon (4) | 4 | Tactical | < 50ms |
| Company (16) | 16 | Operational | < 100ms |
| Battalion (50+) | 50+ | Strategic | < 500ms |

## Emergent Behavior Detection
| Behavior | Metric | Threshold |
|----------|--------|-----------|
| Flocking | Velocity alignment | > 0.8 |
| Flanking | Lateral spread | > 2x formation width |
| Encircling | Encirclement ratio | > 0.9 |
| Swarming | Density | > 0.5 drones/m³ |

## Hierarchical Command
| Level | Drones | Authority | Latency |
|-------|--------|-----------|---------|
| Drone | 1 | Execute | < 10ms |
| Platoon (4) | 4 | Tactical | < 50ms |
| Company (16) | 16 | Operational | < 100ms |
| Battalion (50+) | 50+ | Strategic | < 500ms |

## Emergent Behavior Detection
| Behavior | Metric | Threshold |
|----------|--------|-----------|
| Flocking | Velocity alignment | > 0.8 |
| Flanking | Lateral spread | > 2x formation width |
| Encircling | Encirclement ratio | > 0.9 |
| Swarming | Density | > 0.5 drones/m³ |

## Hierarchical Command
| Level | Drones | Authority | Latency |
|-------|--------|-----------|---------|
| Drone | 1 | Execute | < 10ms |
| Platoon (4) | 4 | Tactical | < 50ms |
| Company (16) | 16 | Operational | < 100ms |
| Battalion (50+) | 50+ | Strategic | < 500ms |

## Emergent Behavior Detection
| Behavior | Metric | Threshold |
|----------|--------|-----------|
| Flocking | Velocity alignment | > 0.8 |
| Flanking | Lateral spread | > 2x formation width |
| Encircling | Encirclement ratio | > 0.9 |
| Swarming | Density | > 0.5 drones/m³ |

## Acceptance
- 50-drone formation: forms in < 60s, holds formation > 95%
- Emergent behaviors: detected > 90% accuracy
- Comms: 50-drone mesh < 100ms latency, 99% delivery
- Mission: 50-drone SAR completes in < 30 min

## DONE
DONE-25 | files: src/sim/large_scale_swarm.py,src/control/swarm_coordinator.py,src/comms/mesh_c2.py,src/rl/multi_drone.py,tests/test_large_scale_swarm.py | tests: 4/4 green | metrics: {formation_time_s: <60, emergent_accuracy: >0.9, comms_latency_ms: <100, mission_sr: >0.8}