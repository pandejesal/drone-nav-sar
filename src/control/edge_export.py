#!/usr/bin/env python3
"""ONNX export + TensorRT optimization + runtime validation (Sprint 6).

Exports the deterministic actor of an ActorCritic policy
(obs (B,21) -> thrusts (B,4) in [0,1]) to a valid ONNX graph and
validates numeric equivalence against PyTorch.

Two export backends:
- "torch": torch.onnx.export (used when the `onnx` package is installed).
- "vendored": dependency-free minimal ONNX writer in this file (Gemm +
  Tanh graph, opset 11). The output is a structurally valid ONNX
  ModelProto: parseable by onnx.load / onnx.checker once `onnx` is
  installed, and verifiable here with the built-in decoder.

The (tanh(u)+1)/2 squash is expressed as a final Gemm with a scaled
identity (0.5*I) plus a 0.5 bias, so the graph needs only Gemm + Tanh
(no broadcasting Add/Div, which older opsets restrict).
"""

import argparse
import importlib.util
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rl.policies import ActorCritic, OBS_DIM, ACT_DIM, DEVICE
from src.rl.local_policy import HierarchicalPolicy, LocalPolicy, SUBGOAL_DIM, TASK_EMBED_DIM

VENDORED_OPSET = 11
VENDORED_IR_VERSION = 6  # historical pair with opset 11 (ONNX 1.6)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

class _DeterministicActor(torch.nn.Module):
    """Deterministic thrust head: obs -> (tanh(trunk(obs))+1)/2.

    Works with ActorCritic (obs 21) and HierarchicalPolicy (obs 24+subgoal+task).
    """

    def __init__(self, policy):
        super().__init__()
        # ActorCritic has .actor_trunk, HierarchicalPolicy has .local_policy.backbone
        if hasattr(policy, "actor_trunk"):
            self.trunk = policy.actor_trunk
        elif hasattr(policy, "local_policy"):
            self.trunk = policy.local_policy.backbone
        else:
            raise TypeError(f"edge_export: unsupported policy type {type(policy)}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        u = self.trunk(obs)
        return (torch.tanh(u) + 1.0) / 2.0


def load_policy(policy_path: str):
    """Load an ActorCritic or HierarchicalPolicy from a state-dict .pt."""
    sd = torch.load(policy_path, map_location=DEVICE, weights_only=True)
    # HierarchicalPolicy state dict has local_policy.actor_trunk.0.weight etc.
    if isinstance(sd, dict) and "local_policy.actor_trunk.0.weight" in sd:
        # HierarchicalPolicy state dict
        sd_local = {k.replace("local_policy.", ""): v for k, v in sd.items() if k.startswith("local_policy.")}
        w0 = sd_local["actor_trunk.0.weight"]
        w2 = sd_local["actor_trunk.2.weight"]
        w4 = sd_local["actor_trunk.4.weight"]
        local_policy = LocalPolicy(
            obs_dim=int(w0.shape[1]),
            act_dim=int(w4.shape[0]),
            hidden_dims=(int(w0.shape[0]), int(w2.shape[0])),
        )
        local_policy.load_state_dict(sd_local)
        policy = HierarchicalPolicy(local_policy=local_policy)
    elif isinstance(sd, dict) and "actor_trunk.0.weight" in sd:
        w0 = sd["actor_trunk.0.weight"]
        w2 = sd["actor_trunk.2.weight"]
        w4 = sd["actor_trunk.4.weight"]
        policy = ActorCritic(
            obs_dim=int(w0.shape[1]),
            act_dim=int(w4.shape[0]),
            hidden_dims=(int(w0.shape[0]), int(w2.shape[0])),
        )
        policy.load_state_dict(sd)
    elif hasattr(sd, "get_action"):
        policy = sd
    else:
        raise ValueError(f"edge_export: unrecognized policy file {policy_path!r}")
    policy.eval()
    policy.to(DEVICE)
    return policy


def deterministic_action(policy, obs: np.ndarray) -> np.ndarray:
    """Deterministic thrusts (B,4) in [0,1].
    
    ActorCritic expects obs (B,21). HierarchicalPolicy expects obs (B, 21+3+4=28) + subgoal + task_id.
    """
    arr = np.asarray(obs, dtype=np.float32)
    if hasattr(policy, "local_policy"):
        # HierarchicalPolicy: needs subgoal and task_id
        # For edge export, we use the raw LocalPolicy directly
        local = policy.local_policy
        if arr.shape[1] == 21:
            # Pad with zeros for subgoal + task_embedding
            pad = np.zeros((arr.shape[0], 7), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        task_id = np.zeros(arr.shape[0], dtype=np.int64)
        action, _, _ = local.get_action(arr, deterministic=True)
    else:
        action, _, _ = policy.get_action(arr, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(arr.shape[0], -1)


# ---------------------------------------------------------------------------
# Minimal ONNX protobuf writer (vendored, dependency-free)
# ---------------------------------------------------------------------------

def _uvarint(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _uvarint((field << 3) | wire)


def _vfield(field: int, value: int) -> bytes:
    return _tag(field, 0) + _uvarint(int(value))


def _bfield(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _uvarint(len(payload)) + payload


def _sfield(field: int, text: str) -> bytes:
    return _bfield(field, text.encode("utf-8"))


def _ffield(field: int, value: float) -> bytes:
    return _tag(field, 5) + struct.pack("<f", float(value))


def _float_attr(name: str, value: float) -> bytes:
    return _sfield(1, name) + _ffield(2, value) + _vfield(20, 1)  # FLOAT=1


def _tensor_init(name: str, arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    out = b""
    out += _bfield(1, b"".join(_uvarint(d) for d in a.shape))  # dims (packed)
    out += _vfield(2, 1)  # data_type FLOAT=1
    out += _bfield(4, struct.pack(f"<{a.size}f", *a.ravel().tolist()))  # float_data
    out += _sfield(8, name)
    return out


def _node(op: str, inputs: List[str], outputs: List[str],
          attrs: Optional[List[Tuple[str, str, Any]]] = None,
          name: str = "") -> bytes:
    out = b"".join(_sfield(1, i) for i in inputs)
    out += b"".join(_sfield(2, o) for o in outputs)
    if name:
        out += _sfield(3, name)
    out += _sfield(4, op)
    for attr_name, kind, value in attrs or []:
        if kind == "f":
            out += _bfield(6, _float_attr(attr_name, value))
        elif kind == "i":
            out += _bfield(6, _sfield(1, attr_name) + _vfield(4, int(value)) + _vfield(20, 2))
        else:
            raise ValueError(f"unsupported attr kind {kind!r}")
    return out


def _dim(value: Any) -> bytes:
    if isinstance(value, str):
        return _sfield(1, value)  # dim_param
    return _vfield(2, int(value))  # dim_value


def _value_info(name: str, dims: List[Any]) -> bytes:
    shape = b"".join(_bfield(1, _dim(d)) for d in dims)
    tensor_type = _vfield(1, 1) + _bfield(2, shape)  # elem FLOAT=1
    return _sfield(1, name) + _bfield(2, _bfield(1, tensor_type))


def build_actor_onnx_bytes(state_dict: Dict[str, Any],
                           opset: int = VENDORED_OPSET,
                           ir_version: int = VENDORED_IR_VERSION) -> bytes:
    """Serialize the deterministic actor as ONNX ModelProto bytes."""
    def arr(key: str) -> np.ndarray:
        v = state_dict[key]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return np.asarray(v, dtype=np.float32)

    W1, b1 = arr("actor_trunk.0.weight"), arr("actor_trunk.0.bias")
    W2, b2 = arr("actor_trunk.2.weight"), arr("actor_trunk.2.bias")
    W3, b3 = arr("actor_trunk.4.weight"), arr("actor_trunk.4.bias")
    act_dim = int(W3.shape[0])
    Ws = (0.5 * np.eye(act_dim, dtype=np.float32))  # squash scale
    bs = (0.5 * np.ones((act_dim,), dtype=np.float32))  # squash shift

    gemm = [("alpha", "f", 1.0), ("beta", "f", 1.0)]
    nodes = [
        _node("Gemm", ["obs", "W1", "b1"], ["h1"], gemm, "gemm1"),
        _node("Tanh", ["h1"], ["t1"], None, "tanh1"),
        _node("Gemm", ["t1", "W2", "b2"], ["h2"], gemm, "gemm2"),
        _node("Tanh", ["h2"], ["t2"], None, "tanh2"),
        _node("Gemm", ["t2", "W3", "b3"], ["u"], gemm, "gemm3"),
        _node("Tanh", ["u"], ["tu"], None, "tanh3"),
        _node("Gemm", ["tu", "Ws", "bs"], ["action"], gemm, "squash"),
    ]
    graph = b"".join(_bfield(1, n) for n in nodes)
    graph += _sfield(2, "dronenav_sar_actor")
    for name, a in (("W1", W1.T), ("b1", b1), ("W2", W2.T), ("b2", b2),
                    ("W3", W3.T), ("b3", b3), ("Ws", Ws), ("bs", bs)):
        graph += _bfield(5, _tensor_init(name, a))
    graph += _bfield(11, _value_info("obs", ["batch", int(W1.shape[1])]))
    graph += _bfield(12, _value_info("action", ["batch", act_dim]))

    opset_id = _sfield(1, "") + _vfield(2, opset)
    model = _vfield(1, ir_version)
    model += _sfield(2, "DroneNav-SAR edge_export")
    model += _bfield(7, graph)
    model += _bfield(8, opset_id)
    return model


# ---------------------------------------------------------------------------
# Minimal ONNX decoder (for validation without the onnx package)
# ---------------------------------------------------------------------------

def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _parse_fields(buf: bytes) -> List[Tuple[int, int, Any]]:
    fields = []
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x07
        if wire == 0:
            val, pos = _read_varint(buf, pos)
            fields.append((field, wire, val))
        elif wire == 2:
            n, pos = _read_varint(buf, pos)
            fields.append((field, wire, buf[pos:pos + n]))
            pos += n
        elif wire == 5:
            fields.append((field, wire, buf[pos:pos + 4]))
            pos += 4
        elif wire == 1:
            fields.append((field, wire, buf[pos:pos + 8]))
            pos += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
    return fields


def _parse_tensor(payload: bytes) -> Tuple[str, np.ndarray]:
    name, dims, dtype, floats, raw = "", [], 1, None, None
    for field, wire, val in _parse_fields(payload):
        if field == 1:  # dims (packed varints)
            p = 0
            while p < len(val):
                d, p = _read_varint(val, p)
                dims.append(d)
        elif field == 2:
            dtype = int(val)
        elif field == 4:  # float_data packed
            n = len(val) // 4
            floats = np.array(struct.unpack(f"<{n}f", val), dtype=np.float32)
        elif field == 8:
            name = val.decode("utf-8")
        elif field == 9:
            raw = val
    if floats is None and raw is not None:
        floats = np.frombuffer(raw, dtype=np.float32).copy()
    if floats is None:
        raise ValueError(f"tensor {name!r} has no data")
    shape = tuple(dims) if dims else ()
    return name, np.asarray(floats, dtype=np.float32).reshape(shape)


def decode_actor_onnx(payload: bytes) -> Dict[str, Any]:
    """Decode a vendored actor ModelProto into nodes/initializers."""
    graph_payload = None
    ir_version, producer = None, ""
    for field, wire, val in _parse_fields(payload):
        if field == 1:
            ir_version = int(val)
        elif field == 2:
            producer = val.decode("utf-8")
        elif field == 7:
            graph_payload = val
    if graph_payload is None:
        raise ValueError("no graph in ModelProto")
    nodes, inits, inputs, outputs = [], {}, [], []
    graph_name = ""
    for field, wire, val in _parse_fields(graph_payload):
        if field == 1:  # node
            op, ins, outs, attrs, nname = "", [], [], {}, ""
            for f, w, v in _parse_fields(val):
                if f == 1:
                    ins.append(v.decode("utf-8"))
                elif f == 2:
                    outs.append(v.decode("utf-8"))
                elif f == 3:
                    nname = v.decode("utf-8")
                elif f == 4:
                    op = v.decode("utf-8")
                elif f == 6:
                    aname, aval = "", None
                    for af, aw, av in _parse_fields(v):
                        if af == 1:
                            aname = av.decode("utf-8")
                        elif af == 2:
                            aval = struct.unpack("<f", av)[0]
                        elif af == 4:
                            aval = int(av)
                    attrs[aname] = aval
            nodes.append({"op": op, "inputs": ins, "outputs": outs,
                          "attrs": attrs, "name": nname})
        elif field == 2:
            graph_name = val.decode("utf-8")
        elif field == 5:
            n, a = _parse_tensor(val)
            inits[n] = a
        elif field == 11:
            for f, w, v in _parse_fields(val):
                if f == 1:
                    inputs.append(v.decode("utf-8"))
        elif field == 12:
            for f, w, v in _parse_fields(val):
                if f == 1:
                    outputs.append(v.decode("utf-8"))
    return {"ir_version": ir_version, "producer": producer,
            "graph_name": graph_name, "nodes": nodes,
            "initializers": inits, "inputs": inputs, "outputs": outputs}


def numpy_forward(decoded: Dict[str, Any], obs: np.ndarray) -> np.ndarray:
    """Run the decoded Gemm/Tanh graph in numpy."""
    env: Dict[str, np.ndarray] = {"obs": np.asarray(obs, dtype=np.float32)}
    inits = decoded["initializers"]
    for node in decoded["nodes"]:
        op = node["op"]
        if op == "Gemm":
            a = env[node["inputs"][0]]
            b = inits[node["inputs"][1]]
            alpha = float(node["attrs"].get("alpha", 1.0))
            beta = float(node["attrs"].get("beta", 1.0))
            trans = int(node["attrs"].get("transB", 0))
            y = alpha * (a @ (b.T if trans else b))
            if len(node["inputs"]) > 2:
                c = inits[node["inputs"][2]]
                y = y + beta * c
            env[node["outputs"][0]] = y.astype(np.float32)
        elif op == "Tanh":
            env[node["outputs"][0]] = np.tanh(env[node["inputs"][0]]).astype(np.float32)
        else:
            raise ValueError(f"numpy_forward: unsupported op {op!r}")
    return env[decoded["outputs"][0]]


# ---------------------------------------------------------------------------
# Export + validation + TensorRT
# ---------------------------------------------------------------------------

def _has_pkg(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def export_onnx(policy_path: str, output_path: str, opset: int = 17,
                validate: bool = True) -> Dict[str, Any]:
    """Export policy to ONNX. Returns a validation report dict."""
    t0 = time.time()
    policy = load_policy(policy_path)
    sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}

    # Detect input dimension from policy
    if hasattr(policy, "local_policy"):
        # HierarchicalPolicy or LocalPolicy wrapped
        local = policy.local_policy
        obs_dim = local.obs_dim if hasattr(local, "obs_dim") else OBS_DIM + SUBGOAL_DIM + TASK_EMBED_DIM
    else:
        # ActorCritic: infer from first layer
        w0 = policy.actor_trunk[0].weight
        obs_dim = int(w0.shape[1])

    backend = "vendored"
    if _has_pkg("onnx"):
        try:
            wrapper = _DeterministicActor(policy)
            dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=DEVICE)
            torch.onnx.export(
                wrapper, dummy, output_path, export_params=True,
                opset_version=int(opset), do_constant_folding=True,
                input_names=["obs"], output_names=["action"],
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            )
            backend = "torch"
        except Exception as exc:  # fall back to vendored writer
            print(f"edge_export: torch.onnx failed ({exc}); using vendored writer")

    if backend == "vendored":
        payload = build_actor_onnx_bytes(sd)
        Path(output_path).write_bytes(payload)

    report: Dict[str, Any] = {
        "onnx_path": str(output_path),
        "opset": int(opset) if backend == "torch" else VENDORED_OPSET,
        "backend": backend,
        "bytes": Path(output_path).stat().st_size,
        "export_time_s": round(time.time() - t0, 3),
    }
    if validate:
        report.update(validate_onnx(output_path, policy_path=policy_path))
    else:
        report["valid"] = None
    report["all_match"] = bool(report.get("valid"))
    return report


def validate_onnx(onnx_path: str, policy_path: Optional[str] = None,
                  tol: float = 1e-4, n_checks: int = 5) -> Dict[str, Any]:
    """Validate an exported ONNX file. Returns {valid, max_abs_err, ...}."""
    payload = Path(onnx_path).read_bytes()
    structural_ok, struct_reason, checker = False, "", "vendored"
    if _has_pkg("onnx"):
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            structural_ok, struct_reason, checker = True, "onnx.checker passed", "onnx"
        except Exception as exc:
            structural_ok, struct_reason = False, f"onnx.checker failed: {exc}"
    try:
        decoded = decode_actor_onnx(payload)
        expected_ops = ["Gemm", "Tanh", "Gemm", "Tanh", "Gemm", "Tanh", "Gemm"]
        ops = [n["op"] for n in decoded["nodes"]]
        if checker == "vendored":
            structural_ok = (
                ops == expected_ops
                and decoded["inputs"] == ["obs"]
                and decoded["outputs"] == ["action"]
            )
            struct_reason = f"vendored decode: ops={ops}" if structural_ok else \
                f"vendored decode mismatch: ops={ops}"
        struct_ok_for_numeric = True
    except Exception as exc:
        return {"valid": False, "max_abs_err": None, "structural_ok": False,
                "reason": f"decode failed: {exc}", "checker": checker}

    max_err: Optional[float] = None
    numeric_reason = "no policy reference; structural check only"
    if policy_path is not None:
        try:
            policy = load_policy(policy_path)
            decoded = decode_actor_onnx(payload)
            rng = np.random.default_rng(0)
            worst = 0.0
            # Detect input dimension from policy
            if hasattr(policy, "local_policy"):
                local = policy.local_policy
                input_dim = local.obs_dim if hasattr(local, "obs_dim") else OBS_DIM + SUBGOAL_DIM + TASK_EMBED_DIM
            else:
                w0 = policy.actor_trunk[0].weight
                input_dim = int(w0.shape[1])
            probes = [np.zeros((1, input_dim), np.float32),
                      np.ones((1, input_dim), np.float32)]
            for _ in range(max(0, n_checks - len(probes))):
                probes.append(rng.uniform(-2, 2, size=(2, input_dim)).astype(np.float32))
            for obs in probes:
                ref = deterministic_action(policy, obs)
                got = numpy_forward(decoded, obs)
                worst = max(worst, float(np.max(np.abs(ref - got))))
            max_err = worst
            numeric_reason = f"max_abs_err={worst:.2e} over {len(probes)} probes (tol={tol:.0e})"
        except Exception as exc:
            return {"valid": False, "max_abs_err": None,
                    "structural_ok": bool(structural_ok),
                    "reason": f"numeric check failed: {exc}", "checker": checker}

    valid = bool(structural_ok) and (max_err is None or max_err <= tol)
    reason = f"{struct_reason}; {numeric_reason}"
    out: Dict[str, Any] = {"valid": valid, "max_abs_err": max_err,
                           "structural_ok": bool(structural_ok),
                           "reason": reason, "checker": checker}
    # Optional cross-check with onnxruntime when installed.
    if _has_pkg("onnxruntime"):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            # Get input dimension from ONNX model
            input_shape = sess.get_inputs()[0].shape
            input_dim = input_shape[1] if len(input_shape) > 1 else 1
            probe = np.zeros((1, input_dim), dtype=np.float32)
            ort_out = np.asarray(sess.run(None, {"obs": probe})[0])
            out["ort_available"] = True
            out["ort_shape"] = list(ort_out.shape)
        except Exception as exc:
            out["ort_available"] = False
            out["ort_error"] = str(exc)
    else:
        out["ort_available"] = False
    return out


def optimize_tensorrt(onnx_path: str, output_path: str, fp16: bool = True,
                      precision: str = "fp16", device: str = "jetson") -> Dict[str, Any]:
    """Optimize ONNX with TensorRT for Jetson (FP16/INT8).

    SAR-only edge target: Jetson Orin (FP16 default, INT8 optional).
    On CPU CI without TensorRT/CUDA this writes a JSON build-manifest to
    `output_path` describing the would-be engine (marked simulated=True)
    and returns optimized=False so pipelines degrade gracefully instead
    of crashing. Returns a dict with keys:
    optimized, engine_path, precision, device, simulated, reason.
    """
    precision = str(precision).lower()
    if fp16 and precision == "fp32":
        precision = "fp16"
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"optimize_tensorrt: unknown precision {precision!r}")
    base: Dict[str, Any] = {
        "optimized": False,
        "engine_path": str(output_path),
        "precision": precision,
        "device": str(device),
        "simulated": True,
        "reason": "",
    }

    def _write_manifest(reason: str) -> Dict[str, Any]:
        import json
        manifest = {
            "source_onnx": str(onnx_path),
            "precision": precision,
            "device": str(device),
            "simulated": True,
            "reason": reason,
            "target_latency_ms": 5.0,
        }
        try:
            Path(output_path).write_text(json.dumps(manifest, indent=2))
            base["reason"] = reason + f"; manifest written to {output_path}"
        except Exception as exc:
            base["reason"] = f"{reason}; manifest write failed: {exc}"
        return base

    if _has_pkg("torch_tensorrt"):
        try:
            import torch_tensorrt  # noqa: F401
            return _write_manifest(
                "torch_tensorrt present but compile needs CUDA GPU; skipped on CPU CI")
        except Exception as exc:
            return _write_manifest(f"torch_tensorrt import failed: {exc}")
    if _has_pkg("tensorrt"):
        try:
            import tensorrt as trt  # noqa: F401
            return _write_manifest(
                "tensorrt present but engine build needs GPU builder; skipped on CPU CI")
        except Exception as exc:
            return _write_manifest(f"tensorrt import failed: {exc}")
    return _write_manifest("tensorrt not installed (Jetson-only); onnx artifact kept as-is")


def quantize_dynamic(onnx_path: str, output_path: str) -> bool:
    """Dynamic quantization for edge deployment (RPi/CPU)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
        return True
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False


def quantize_for_device(onnx_path: str, output_path: str,
                        precision: str = "int8",
                        device: str = "rpi") -> Dict[str, Any]:
    """Quantize ONNX for an edge device (RPi CM4 dynamic INT8).

    Tries real onnxruntime dynamic quantization; when onnxruntime is
    absent (CPU CI) it copies the artifact and records simulated=True so
    downstream accuracy checks still run. Returns a dict with keys:
    quantized, output_path, precision, device, simulated, reason.
    """
    import shutil
    precision = str(precision).lower()
    report: Dict[str, Any] = {
        "quantized": False,
        "output_path": str(output_path),
        "precision": precision,
        "device": str(device),
        "simulated": True,
        "reason": "",
    }
    if precision in ("fp32", "fp16"):
        # No quantization needed at fp precision: copy through.
        try:
            shutil.copyfile(onnx_path, output_path)
            report.update(quantized=True,
                          reason=f"passthrough copy at {precision} (no quant)")
            return report
        except Exception as exc:
            report["reason"] = f"copy failed: {exc}"
            return report
    if _has_pkg("onnxruntime"):
        ok = quantize_dynamic(onnx_path, output_path)
        report.update(quantized=bool(ok), simulated=False,
                      reason="onnxruntime dynamic INT8" if ok else "quantize_dynamic failed")
        return report
    try:
        shutil.copyfile(onnx_path, output_path)
        report.update(quantized=True,
                      reason="onnxruntime absent; artifact copied, marked simulated INT8")
    except Exception as exc:
        report["reason"] = f"copy failed: {exc}"
    return report


def validate_with_onnxruntime(onnx_path: str, n_warmup: int = 3,
                              n_iters: int = 20) -> Dict[str, Any]:
    """Validate an ONNX file with ONNX Runtime (CPU provider).

    Returns {available, latency_ms_mean, ...}; available=False with a
    reason when onnxruntime is not installed.
    """
    if not _has_pkg("onnxruntime"):
        return {"available": False,
                "reason": "onnxruntime not installed; vendored numpy check applies"}
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        probe = np.zeros((1, OBS_DIM), dtype=np.float32)
        for _ in range(int(n_warmup)):
            sess.run(None, {"obs": probe})
        t0 = time.time()
        for _ in range(int(n_iters)):
            sess.run(None, {"obs": probe})
        dt_ms = (time.time() - t0) / max(1, int(n_iters)) * 1000.0
        out = np.asarray(sess.run(None, {"obs": probe})[0])
        return {"available": True, "latency_ms_mean": round(dt_ms, 3),
                "output_shape": list(out.shape)}
    except Exception as exc:
        return {"available": False, "reason": f"onnxruntime run failed: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export policy to ONNX + validate (Sprint 10 edge)")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy .pt")
    parser.add_argument("--output", type=str, default="policy.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (torch backend)")
    parser.add_argument("--no-validate", action="store_true", help="Skip runtime validation")
    parser.add_argument("--tensorrt", action="store_true", help="Optimize with TensorRT")
    parser.add_argument("--quantize", action="store_true", help="Dynamic quantize for edge")
    parser.add_argument("--fp16", action="store_true", help="FP16 for TensorRT")
    parser.add_argument("--int8", action="store_true", help="INT8 for TensorRT")
    parser.add_argument("--device", type=str, default="jetson",
                        choices=["jetson", "rpi", "cpu"],
                        help="Edge target device (jetson=TensorRT, rpi=dynamic INT8)")
    parser.add_argument("--precision", type=str, default=None,
                        choices=["fp32", "fp16", "int8"],
                        help="Precision (default: fp16 on jetson, int8 on rpi, fp32 on cpu)")
    args = parser.parse_args()

    # Resolve precision defaults per device (Sprint 10 pipeline).
    precision = args.precision
    if precision is None:
        precision = {"jetson": "fp16", "rpi": "int8", "cpu": "fp32"}[args.device]
    if args.fp16:
        precision = "fp16"
    if args.int8:
        precision = "int8"
    want_trt = bool(args.tensorrt or args.device == "jetson")
    want_quant = bool(args.quantize or args.device == "rpi")

    print(f"Exporting {args.policy} -> {args.output} (device={args.device} precision={precision})")
    report = export_onnx(args.policy, args.output, opset=args.opset,
                         validate=not args.no_validate)
    print(f"backend={report['backend']} bytes={report['bytes']} "
          f"valid={report.get('valid')} err={report.get('max_abs_err')}")
    if report.get("valid") is False:
        print(f"VALIDATION FAILED: {report.get('reason')}")
        sys.exit(1)

    ort = validate_with_onnxruntime(args.output)
    print(f"ONNX Runtime: {ort}")

    if want_trt:
        trt = optimize_tensorrt(args.output, args.output.replace(".onnx", "_trt.engine"),
                                fp16=(precision == "fp16"),
                                precision=precision, device=args.device)
        print(f"TensorRT: {trt}")

    if want_quant:
        q_path = args.output.replace(".onnx", "_quant.onnx")
        print(f"Quantizing ({precision}) -> {q_path}")
        qrep = quantize_for_device(args.output, q_path,
                                   precision=precision, device=args.device)
        print(f"Quantization: {qrep}")

    print("Export complete")


if __name__ == "__main__":
    main()
