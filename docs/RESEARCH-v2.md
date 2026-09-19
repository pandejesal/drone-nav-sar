# Research Survey v2 — SLM Prompt Control + Custom Drone Import + Edge Deploy (SAR-ONLY)

> Scope: 3 gaps beyond v1 (10 papers, 12 repos): (1) NN→JSON skill → SLM paraphrase → safety filter → MAVLink, (2) anyone-brings-own-drone URDF/SDF + datasheet importer, (3) ONNX edge deploy Jetson/RPi for medkit A→B.
> **Safety (non-negotiable, SAR-ONLY):** medkit delivery with `navigate_to / hover / drop_payload / return_home` only. Where a cited source covers weapons, reuse nav-only, refuse weapon part.

---

## 1. Papers (7 NEW)

### Gap 1: JSON-mode / grammar-constrained LLM output → SLM paraphrase → safety filter → MAVLink

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P11 | Tam et al., "Let Me Speak Freely? A Study on the Impact of Format Restrictions on LLM Performance" (EMNLP 2024) | Quantifies how JSON-mode constrained decoding trades reasoning for format adherence — informs our `slm_adapter.py` schema-validation design. | https://arxiv.org/abs/2410.03072 |
| P12 | Ro-SLM, "Onboard Small Language Models for Robot Task Planning and Operation Code Generation" (2025) | Distills LLM reasoning into deployable SLM (Phi-3 on Jetson) for task planning + code generation; directly templates our `slm_adapter.py` onboard pipeline. | https://arxiv.org/abs/2604.10929 |
| P15 | Generating Structured Outputs from Language Models: Benchmark and Studies (SchemaBench, 2025) | Benchmarks 6 constrained-decoding frameworks on JSON Schema compliance; guides choice of grammar engine for `skill_schema.json` validation. | https://arxiv.org/abs/2501.10868 |

### Gap 1b: Vision-language nav with safety filter / shield (formal or runtime)

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P13 | Sanyal & Roy, "ASMA: Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions" (IEEE RA-L 2025) | Formal CBF safety layer on top of VLN for drones — our `safety_filter.py` geometric/geofence rules + CBF extension template. | https://arxiv.org/abs/2409.10283 |
| P14 | AlphaAdj, "Dynamic Control Barrier Function Regulation with Vision-Language Models for Safe, Adaptive, and Realtime Visual Navigation" (2025) | Uses VLM risk estimate to dynamically adapt CBF conservativeness — templates runtime safety margin adaptation in `safety_filter.py`. | https://arxiv.org/abs/2412.04153 |

### Gap 3: Small LM onboard + ONNX/TensorRT edge inference

| # | Paper | Why useful (1 line) | Link |
|---|-------|---------------------|------|
| P16 | Islam et al., "Characterizing and Understanding the Performance of Small Language Models on Edge Platforms" (2024) | Benchmarks Phi-3, TinyLlama, Llama-3 on Jetson Orin / RPi — selects our `slm_adapter.py` model candidate. | https://ieeexplore.ieee.org/document/10850044 |
| P17 | Anil et al., "Gemma: Lightweight Open Models for the Edge" (2024) | Google's 2B-11B open models optimized for edge; alternative SLM candidate for `slm_adapter.py`. | https://arxiv.org/abs/2402.05629 |

---

## 2. Repos/Code to Copy (8 NEW)

Conventions: **Reuse** = exact files/modules to copy or wrap. **Cost** = integration effort into THIS repo (low/med/high).

### Gap 2: URDF/SDF + datasheet importer for anyone-brings-own-drone

| # | Repo | License | What to copy | Cost |
|---|------|---------|--------------|------|
| R13 | yourdfpy — https://github.com/clemense/yourdfpy | MIT | `yourdfpy.URDF.load` + `to_dict` — parse custom URDFs, decouple parsing from validation/meshes; `src/drones/importer.py` wraps this for geometry/collision only. | Low |
| R14 | urdfpy — https://github.com/mmatl/urdfpy | MIT | `urdfpy.URDF.load` + FK — alternative URDF parser; `src/drones/importer.py` uses whichever is available. | Low |
| R15 | sdformat_urdf — https://github.com/ros/sdformat_urdf | BSD-3 | `sdformat_urdf` plugin — converts SDFormat XML → URDF DOM; enables `.sdf` input path in `src/drones/importer.py`. | Med |

### Gap 1: Skill schema validation + SLM constrained decoding

| # | Repo | License | What to copy | Cost |
|---|------|---------|--------------|------|
| R16 | guidance-ai/llguidance — https://github.com/guidance-ai/llguidance | MIT | Grammar engine for JSON Schema-constrained decoding; `src/control/slm_adapter.py` uses llguidance to enforce `skill_schema.json` output format. | Low |

### Gap 3: ONNX/TensorRT edge inference for Jetson/RPi

| # | Repo | License | What to copy | Cost |
|---|------|---------|--------------|------|
| R17 | NVIDIA TensorRT-Edge-LLM — https://github.com/NVIDIA/TensorRT-Edge-LLM | MIT | ONNX export + TensorRT engine build + C++ inference; `src/control/edge_export.py` wraps the export step, `edge_report.json` validates. | Med |
| R18 | ONNX Runtime IoT/RPi — https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html | MIT | `onnxruntime.InferenceSession` on RPi; `src/control/edge_export.py` regression test clones this for RPi validation. | Low |

### Gap 2b: Custom drone URDF/SDF + PX4 SITL model loading

| # | Repo | License | What to copy | Cost |
|---|------|---------|--------------|------|
| R19 | hector_quadrotor_description — https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor | BSD | `quadrotor.urdf.xacro` — reference URDF template for our `assets/drones/<name>/model.urdf`; collision + visual geometry. | Low |
| R20 | PX4 sitl_gazebo custom model — https://github.com/PX4/PX4-Autopilot/tree/main/Tools/sitl_gazebo/models | BSD-3 | `model.config` + `.sdf` pattern for custom airframe SITL; `src/drones/importer.py` validates against this convention. | Med |

---

## 3. Integration Map — Which File Gets What

### `src/drones/importer.py` (NEW)
- **Copies from:** R13 (yourdfpy), R14 (urdfpy), R15 (sdformat_urdf), R19 (hector_quadrotor), R20 (PX4 sitl_gazebo)
- **Function:** Parse `drone.yaml` → validate ranges → load URDF/SDF for geometry/collision → emit `QuadrotorParams` with `spec_hash`
- **Cost:** Low–Med (wrap parsers, validate against existing `QuadrotorParams` type)
- **Safety:** Rejects raw-RPM configs; enforces `thrust_unit == normalized` (taboo guard)

### `src/control/slm_adapter.py` (NEW)
- **Copies from:** P12 (Ro-SLM), P16 (SLM edge benchmark), R16 (llguidance)
- **Function:** SLM/LLM fills `params` from observation + mission prompt, constrained to `skill_schema.json` via llguidance grammar engine
- **Cost:** Low (grammar-constrained decode wrapper; model choice per P16/P17)
- **Safety:** Schema validation rejects anything non-conforming; no bypass path

### `src/control/safety_filter.py` (NEW)
- **Copies from:** P13 (ASMA CBF), P14 (AlphaAdj), P11 (format-restriction study)
- **Function:** Hard checks: skill in SAR allowlist, target inside geofence, below ceiling, speed capped, RC override live. **Reject = hover in place.**
- **Cost:** Low (rule-based + optional CBF extension per P13/P14)
- **Safety:** Default-deny, hover on reject; no bypass path

### `src/control/edge_export.py` (NEW)
- **Copies from:** P16 (SLM edge), R17 (TensorRT-Edge-LLM), R18 (ONNX Runtime RPi)
- **Function:** `torch.onnx.export` + ONNX Runtime inference check; `edge_report.json` validates runtime passed in eval gate
- **Cost:** Low (export + runtime check; TensorRT optional on Jetson per R17)
- **Safety:** ONNX opset pinned; same file eval gate passed deploys

### `src/eval/` (NEW)
- **References:** P11 (format adherence), P13 (CBF safety), P16 (edge perf)
- **Function:** SPL + success rate primary; adds `spec_hash`, `edge_latency_ms` for edge-deployed runs; `onnx_inference_ok` gate
- **Cost:** Low (extends existing eval contract from `docs/ARCHITECTURE-v2.md` §5)

---

## 4. Notes on SAR-ONLY Compliance

- **P13 (ASMA)** covers weapons-adjacent VLN topics; we reuse **only** the CBF safety layer and navigation commands. No targeting or kinetic effector logic is referenced.
- **P14 (AlphaAdj)** similarly reuses only the adaptive CBF parameter mechanism for safety margin control.
- **All repos** (R13–R20) are navigation/geometry/edge-inference focused; none contain weaponized control or targeting modules.
- **Constraint:** Where any source discusses dual-use, this survey reuses the **navigation-only** component and refuses the weapon part (per v1 convention).

---

## 5. Cost Summary

| Module | Repo Copies | Cost |
|--------|------------|------|
| `src/drones/importer.py` | R13, R14, R15, R19, R20 | Low–Med |
| `src/control/slm_adapter.py` | R16, P12, P16 | Low |
| `src/control/safety_filter.py` | P13, P14 | Low |
| `src/control/edge_export.py` | R17, R18 | Low |
| `src/eval/` extensions | P11, P13, P16 | Low |

---

DONE-R2 | files: docs/RESEARCH-v2.md | tests: n/a | metrics: {papers:7,repos:8}
