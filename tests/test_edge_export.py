#!/usr/bin/env python3
"""Sprint 10 edge-export tests (3 tests, SAR-only).

1. ONNX validity: export ActorCritic -> ONNX validates with tiny max diff.
2. TensorRT build: optimize_tensorrt degrades gracefully on CPU CI and
   writes a JSON build-manifest describing the Jetson FP16 engine.
3. Quantization accuracy: INT8 artifact (simulated copy on CPU CI) stays
   numerically equivalent to the FP32 ONNX.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestEdgeExport(unittest.TestCase):
    def _policy_file(self, tmp: Path) -> str:
        from src.rl.policies import ActorCritic
        torch.manual_seed(0)
        policy = ActorCritic()
        policy.eval()
        pt = str(tmp / "policy_sprint10_test.pt")
        torch.save(policy.state_dict(), pt)
        return pt

    def test_onnx_export_valid(self):
        from src.control.edge_export import export_onnx
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt = self._policy_file(tmp)
            out = str(tmp / "policy.onnx")
            report = export_onnx(pt, out, validate=True)
            self.assertTrue(Path(out).exists())
            self.assertTrue(report.get("valid"), report.get("reason"))
            self.assertLessEqual(float(report["max_abs_err"]), 1e-4)

    def test_tensorrt_build_graceful(self):
        import json as _json
        from src.control.edge_export import export_onnx, optimize_tensorrt
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt = self._policy_file(tmp)
            onnx_path = str(tmp / "policy.onnx")
            export_onnx(pt, onnx_path, validate=False)
            engine = str(tmp / "policy_trt.engine")
            rep = optimize_tensorrt(onnx_path, engine, fp16=True,
                                    precision="fp16", device="jetson")
            self.assertEqual(rep["precision"], "fp16")
            self.assertEqual(rep["device"], "jetson")
            self.assertTrue(Path(engine).exists())
            manifest = _json.loads(Path(engine).read_text())
            self.assertEqual(manifest["precision"], "fp16")

    def test_quantization_accuracy(self):
        from src.control.edge_export import (
            decode_actor_onnx,
            export_onnx,
            numpy_forward,
            quantize_for_device,
        )
        from src.rl.policies import OBS_DIM
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt = self._policy_file(tmp)
            onnx_path = str(tmp / "policy.onnx")
            export_onnx(pt, onnx_path, validate=False)
            q_path = str(tmp / "policy_quant.onnx")
            qrep = quantize_for_device(onnx_path, q_path,
                                       precision="int8", device="rpi")
            self.assertTrue(qrep["quantized"], qrep.get("reason"))
            self.assertTrue(Path(q_path).exists())
            ref = numpy_forward(decode_actor_onnx(Path(onnx_path).read_bytes()),
                                np.zeros((2, OBS_DIM), np.float32))
            got = numpy_forward(decode_actor_onnx(Path(q_path).read_bytes()),
                                np.zeros((2, OBS_DIM), np.float32))
            self.assertLessEqual(float(np.max(np.abs(ref - got))), 1e-3)


if __name__ == "__main__":
    unittest.main()
