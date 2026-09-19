# Architecture — DroneNav-SAR (SAR-only)

## System diagram

```text
photos -> [reconstruction] -> mesh.glb + hulls
                                     |
mesh.glb -> [sim: mock|isaac|airsim|gazebo] -> BaseDroneEnv
                                     |
                     [rl: PPO + curriculum + hierarchical + multi-drone] -> policy.pt
                                     |
command -> [language_to_task -> validator -> embed_task -> slm_adapter]
                                     |
policy + task_emb -> [safety_filter] -> [mavlink_bridge | ros2_bridge] -> vehicle
                                     |
                              [eval: SPL/success/energy/time, sim2real, edge_bench]
```

## Data flows

1. **Reconstruction flow**: `photos/ -> colmap_sparse/ -> mesh/space.glb ->
   domain_rand.json -> manifest.json` (per space bundle).
2. **Training flow**: space bundles -> vec_env (N=8, seeded) -> PPO ->
   `checkpoints/policy.pt` (+ curriculum stage files).
3. **Mission flow**: language command -> skills -> safety filter -> bridge ->
   `report.json` (coverage, detection, drop accuracy, SPL, violations).
4. **Edge flow**: `policy.pt -> policy.onnx -> policy.plan (TRT FP16, Jetson) /
   policy_int8.onnx (ORT, RPi CM4)`.
5. **Release flow**: `v*` tag -> wheels + ONNX + TRT manifest -> HF Hub.

## Safety invariants (non-negotiable, SAR-only)

- **Closed skill set**: `navigate_to / hover / drop_payload (aid kits only) /
  return_home`. The language validator rejects weapons, targeting, kinetic
  payloads, and human surveillance — refused, never re-interpreted.
- **Geofence + altitude ceiling** hardcoded in the control layer, not learned.
- **MAVLink heartbeat + RC override** always active on hardware.
- **Safety filter re-checks** every skill at execution time (defense in depth
  past the language layer).
- **Multi-drone separation** enforced by the filter, not the policy.
- **Payload interlock**: `drop_payload` fires only for registered aid-kit
  payload types within the mission geofence.

See `docs/ARCHITECTURE-v2.md` for the Sprint 4+ module proposal this
implements, and `docs/api_reference.md` for the module API.
