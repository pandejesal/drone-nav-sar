# Sprint 21: Human-Swarm Teaming (SwarmOS-like)

## Goal
Natural language → swarm intent → autonomous execution. Rivals Palladyne SwarmOS human-swarm teaming.

## Scope (modify ONLY these)
- src/control/swarm_llm.py: NEW — LLM/SLM intent parser → swarm intent (task graph)
- src/control/swarm_coordinator.py: EXTEND — Intent → task graph → role assignment
- src/control/mission_planner.py: INTEGRATE — Language → swarm intent → COA
- src/control/swarm_llm.py: NEW — Few-shot prompt templates, schema validation
- tests/test_swarm_llm.py: NEW — 4 tests (parser accuracy, intent→task graph, execution, latency)

## Architecture
```
Operator ("search building 3 for victims")
       ↓
SwarmLLM (few-shot + schema) → SwarmIntent {task_graph, roles, constraints}
       ↓
SwarmCoordinator (role election, task allocation, health)
       ↓
SwarmCoordinator → MissionPlanner (COA) → LocalPolicy (per drone)
```

## Swarm Intent Schema
```json
{
  "intent": "search_and_rescue",
  "area": {"building": "building_3", "floors": [1,2,3]},
  "objectives": [
    {"type": "search", "area": "floor_1", "priority": 1},
    {"type": "search", "area": "floor_2", "priority": 2},
    {"type": "search", "area": "floor_3", "priority": 3}
  ],
  "constraints": {
    "max_drones": 6,
    "max_time_min": 30,
    "safety_margin_m": 2.0,
    "comms_range_m": 500
  },
  "roles": {
    "scout": 2,
    "searcher": 3,
    "relay": 1
  }
}
```

## Swarm LLM Pipeline
```
User text → Few-shot prompt → LLM/SLM → JSON intent → Schema validation
    → SwarmIntent {task_graph, roles, constraints, priority}
    → SwarmCoordinator.role_election() → role assignment
    → SwarmCoordinator.task_allocation() → per-drone task graphs
    → MissionPlanner → LocalPolicy per drone
```

## Acceptance
- Parser accuracy: > 90% on 100 test commands
- Intent→task graph latency: < 500ms
- Role election convergence: < 5s for 10 drones
- Task allocation optimality: > 90% vs optimal
- End-to-end latency (text → drone action): < 2s

## DONE
DONE-21 | files: src/control/swarm_llm.py,src/control/swarm_coordinator.py,tests/test_swarm_llm.py | tests: 4/4 green | metrics: {parser_accuracy: >0.9, latency_ms: <500, role_election_s: <5}