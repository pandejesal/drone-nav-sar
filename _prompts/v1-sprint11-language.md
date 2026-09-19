# Sprint 11 — Human-in-the-Loop: Language Commands → Task Embeddings (SAR-only)

## Goal
Natural language interface for operators: type/speak "go to kitchen" or "deliver medkit to room 3" → LLM/SLM parses to task embedding → hierarchical policy executes. Target: < 2s command-to-execution latency, > 90% command understanding accuracy on test set.

## Scope (modify ONLY these)
- src/control/language_interface.py: NEW — LLM/SLM parser (OpenAI-compatible API, local fallback), prompt templates, few-shot examples
- src/control/language_to_task.py: NEW — maps parsed intent → task_embedding_id + params (room, victim, item)
- src/rl/local_policy.py: MODIFY — accept language-conditioned task embedding (concatenate to obs)
- src/control/mission_planner.py: MODIFY — accept language commands, update subgoals
- src/rl/ppo_nav.py: ADD --language flag, integrate language interface
- tests/test_language.py: NEW — 4 tests (parser accuracy, embedding mapping, execution, latency)

## Architecture
```
Operator ("deliver medkit to room 3")
       ↓
LanguageInterface (LLM/SLM + few-shot) → {task: "deliver", room: 3, item: "medkit"}
       ↓
LanguageToTask → task_embedding_id=3 + params={room:3, item:"medkit"}
       ↓
MissionPlanner (global planner subgoals) → LocalPolicy (obs + task_embedding)
       ↓
MultiDroneEnv (3 drones, shared policy, comms) → Med-kit delivery
```

## Language Commands Supported (SAR-only)
- Navigation: "go to room 3", "fly to kitchen", "return to base"
- Delivery: "deliver medkit to room 2", "drop supplies at victim 1"
- Inspection: "inspect hallway 3", "search room 1 for victims"
- Emergency: "return to base immediately", "abort mission"

## Acceptance
.venv/Scripts/python -m pytest tests/test_language.py -q (4 passed)
.venv/Scripts/python -m src.rl.ppo_nav --episodes 1000 --seed 0 --language --multi-drone 3 --hierarchical --curriculum --dynamic-obstacles --sar-mission --save-path policy_sprint11.pt → mission SR > 0.7, command latency < 2s
Command understanding accuracy > 90% on 100 test phrases

## DONE
DONE-11 | files: src/control/language_interface.py,src/control/language_to_task.py,src/rl/local_policy.py,src/control/mission_planner.py,src/rl/ppo_nav.py,tests/test_language.py | tests: 4/4 green | metrics: {accuracy:..., latency_ms:...}