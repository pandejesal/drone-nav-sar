# Tutorial 3 — RL Training: PPO + Curriculum + Hierarchical + Multi-Drone (~30 min, SAR-only)

Train a SAR navigation policy: PPO on vectorized mock envs, curriculum over
spaces, hierarchical global/local split, multi-drone option.

## Steps

### 1. Smoke training (minutes, CI-safe)

```bash
python -m src.rl.ppo_nav --quick --spaces 2 --eps 5
```

### 2. Full training (2000 eps/space, curriculum)

```bash
python -m src.rl.ppo_nav --curriculum --spaces 100 --eps-per-space 2000
```

Curriculum (`policy_sprint4_curriculum*.pt`): fixed → shaped → long-horizon
spaces. Vectorized envs (`src/rl/vec_env.py`, N=8 seeded) keep it reproducible.

### 3. Hierarchical policy

```python
from src.rl.hierarchical import HierarchicalPolicy
policy = HierarchicalPolicy.load("checkpoints/hierarchical.pt")
# global_planner.py: room-level waypoints; local_policy.py: motor-level control
```

### 4. Multi-drone

```python
from src.rl.multi_drone import MultiDroneTeam
team = MultiDroneTeam(n_drones=2, policy_path="checkpoints/policy.pt")
```

Drones share one mesh with deconflicted goals; collision separation is a
safety-filter invariant, not learned.

### 5. Evaluate (SPL, success, energy, time)

```bash
python scripts/eval_all.py --policy checkpoints/policy.pt --spaces 10
```

Gates (`src/eval/metrics.py`): success rate, SPL, energy proxy, episode time.
Regression = CI fail.

## Checkpoints

| File | What |
|---|---|
| `policy.pt` | Latest full SAR policy |
| `policy_sprint4_curriculum*.pt` | Curriculum stages |
| `policy_sprint5_dynamic*.pt` | Dynamic-obstacle stages |
| `policy.onnx` | Edge-exported policy (see Tutorial 5) |

## Next

Tutorial 4 — steer the policy with SAR language commands.
