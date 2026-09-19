# Sprint 8 — Long-Horizon Hierarchical Policy (SAR-only)

## Goal
Hierarchical RL: global topological planner (high-level) + local PPO policy (low-level).
Global planner operates on room graph, outputs subgoals (room centers, doorways).
Local policy executes subgoals (navigate_to, hover, drop_payload).
Target: 10-room building, >70% mission SR with 5+ victims.

## Scope (modify ONLY these)
- src/rl/hierarchical.py: NEW — HierarchicalPPO wrapper (global + local)
- src/rl/global_planner.py: NEW — Topological planner on room graph (A* / Dijkstra)
- src/rl/local_policy.py: NEW — Local PPO policy (subgoal-conditioned)
- src/rl/ppo_nav.py: ADD --hierarchical flag, multi-task heads (task_embedding_id)
- src/control/mission_planner.py: INTEGRATE global planner for subgoal generation
- tests/test_hierarchical.py: NEW — 4 tests (global planner, local policy, hierarchical wrapper, integration)

## Architecture
```
Global Planner (room graph) → Subgoal sequence (room centers, doorways)
       ↓
Local Policy (PPO, obs + subgoal embedding) → Motor commands
       ↓
Mock Backend → Reward (subgoal progress + SAR rewards)
```

## Global Planner
- Room graph: nodes = room centers + doorways, edges = traversable corridors
- A* on graph with dynamic edge weights (door open/closed, obstacle density)
- Outputs: sequence of subgoals [(x,y,z), ...] for local policy
- Replans when: door state changes, obstacle blocks path, victim detected

## Local Policy
- Subgoal-conditioned PPO: obs (21) + subgoal (3) + task_embedding (4) → action (4)
- Task embeddings: 0=nav, 1=hover, 2=detect, 3=drop, 4=return
- Shared backbone (MLP 64-64) + task-specific heads
- Trained with curriculum: static → dynamic → SAR mission

## Acceptance
.venv/Scripts/python -m pytest tests/test_hierarchical.py -q (4 passed)
.venv/Scripts/python -m src.rl.ppo_nav --episodes 2000 --seed 0 --hierarchical --curriculum --dynamic-obstacles --sar-mission --save-path policy_sprint8.pt → mission SR > 0.7

## DONE
DONE-8 | files: src/rl/hierarchical.py,src/rl/global_planner.py,src/rl/local_policy.py,src/rl/ppo_nav.py,src/control/mission_planner.py,tests/test_hierarchical.py | tests: 4/4 green | metrics: {mission_sr:...}