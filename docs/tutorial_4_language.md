# Tutorial 4 — Language: SAR Commands to Task Embeddings (~30 min, SAR-only)

Turn constrained SAR voice/text commands into task embeddings that steer the
hierarchical policy. The language layer can only emit the four SAR skills —
anything else is rejected by the validator.

## Allowed skills (closed vocabulary)

`navigate_to / hover / drop_payload (aid kits) / return_home`

`src/control/language_validator.py` rejects everything outside this set —
weapons, targeting, kinetic, and surveillance requests are refused, never
re-interpreted.

## Steps

### 1. Parse a command

```python
from src.control.language_to_task import parse_command
task = parse_command("go to room B and hover")
# Task(skill='navigate_to', target='room_B') + Task(skill='hover')
```

### 2. Embed and condition the policy

```python
from src.control.language_interface import embed_task
emb = embed_task(task)  # fixed-size task embedding -> hierarchical policy input
```

### 3. Run the integration test

```bash
python test_language_integration.py
pytest tests/test_language.py -q
```

### 4. Try the SAR command set

| Say | Parsed skill chain |
|---|---|
| "search room B" | `navigate_to(room_B)` |
| "hold position" | `hover` |
| "drop the med-kit" | `drop_payload(medkit)` |
| "come home" | `return_home` |
| "track that person" | REJECTED (surveillance) |
| "drop the <weapon>" | REJECTED (kinetic) |

## Pipeline

```text
command -> language_to_task -> language_validator -> embed_task
        -> slm_adapter -> safety_filter -> mavlink/ros2 bridge
```

The safety filter re-checks the skill against geofence/altitude/payload rules
at execution time, so a compromised upstream component cannot smuggle a
forbidden skill through.

## Next

Tutorial 5 — export the conditioned policy to edge hardware.
