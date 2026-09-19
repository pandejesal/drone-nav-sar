# Tutorial 6 — SAR Mission: 4-Room Building, Language Commands, Med-Kit Drop (~30 min, SAR-only)

Capstone: fly a full search-and-rescue mission in a 4-room building — language
commands in, victim found, med-kit dropped, drone home. Everything composes
Tutorials 1-5.

## Mission script

| # | Command | Skill chain |
|---|---|---|
| 1 | "search room A" | `navigate_to(room_A)` |
| 2 | "search room B" | `navigate_to(room_B)` |
| 3 | "hold position" | `hover` (victim detected via `yolo_detector.py`) |
| 4 | "drop the med-kit" | `drop_payload(medkit)` |
| 5 | "come home" | `return_home` |

## Steps

### 1. Load the mission mesh + policy

```bash
python scripts/dataset_build.py --spaces 1 --quick --output-dir data/mission_building
pytest tests/test_sar.py -q
```

### 2. Run the mission (mock backend)

```python
from src.sim import make_env
from src.control.mission_planner import MissionPlanner

env = make_env(backend="mock", mesh="data/mission_building/mesh/space.glb")
planner = MissionPlanner(policy_path="checkpoints/policy.pt")
report = planner.run_mission(env, commands=[
    "search room A",
    "search room B",
    "hold position",
    "drop the med-kit",
    "come home",
])
print(report)  # per-room coverage, victim found, drop accuracy, SPL, time
```

### 3. Check the report gates

`src/eval/metrics.py` thresholds: all 4 rooms visited, victim detection
logged, drop within 1 m of victim, geofence never violated, RC override
available throughout.

### 4. (Hardware, supervised) Replay on the vehicle

```bash
python -m src.control.mavlink_bridge --mission report.json --dry-run
```

Remove `--dry-run` only with a safety pilot on RC override and the geofence
parameters from `docs/architecture.md` loaded.

## Mission-complete criteria

- 4/4 rooms searched, victim found, med-kit within 1 m, drone home
- No geofence/altitude violations in the log
- Report JSON archived (CI artifact pattern from Tutorial 2)

You finished the series — see `docs/api_reference.md` for the full API and
`README.md` for community links (Discord, Discussions, office hours).
