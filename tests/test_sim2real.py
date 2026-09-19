#!/usr/bin/env python3
"""Sim2Real tests for DroneNav-SAR (Sprint 6).

Tests:
1. ONNX export produces valid model with matching outputs (skipped if deps missing)
2. MAVLink bridge connects (mock mode)
3. System ID data format validation
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check optional dependencies
try:
    import onnx
    import onnxruntime as ort
    import onnxscript
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestONNXExport(unittest.TestCase):
    """Test ONNX export and validation."""

    @unittest.skipUnless(ONNX_AVAILABLE and TORCH_AVAILABLE, "ONNX dependencies not installed")
    def test_onnx_export_valid(self):
        """Test that policy exports to valid ONNX with matching outputs."""
        import torch
        from src.rl.policies import ActorCritic, OBS_DIM, ACT_DIM
        from src.control.edge_export import export_onnx

        # Create a temporary policy
        policy = ActorCritic()
        policy_path = "/tmp/test_policy.pt"
        torch.save(policy.state_dict(), policy_path)

        # Export to ONNX
        onnx_path = "/tmp/test_policy.onnx"
        report = export_onnx(policy_path, onnx_path)

        # Validate report
        self.assertTrue(report["all_match"], f"ONNX outputs don't match: {report}")
        self.assertEqual(report["input_shape"], [1, OBS_DIM])
        self.assertEqual(len(report["output_shapes"]), 3)  # action, value, logprob
        self.assertEqual(report["output_shapes"][0], (1, ACT_DIM))  # action
        self.assertEqual(report["output_shapes"][1], (1,))  # value
        self.assertEqual(report["output_shapes"][2], (1,))  # logprob

    @unittest.skipUnless(ONNX_AVAILABLE and TORCH_AVAILABLE, "ONNX dependencies not installed")
    def test_onnx_runtime_inference(self):
        """Test ONNX Runtime can load and run the model."""
        import torch
        import onnxruntime as ort
        from src.rl.policies import ActorCritic, OBS_DIM, ACT_DIM
        from src.control.edge_export import export_onnx

        policy = ActorCritic()
        policy_path = "/tmp/test_policy2.pt"
        torch.save(policy.state_dict(), policy_path)

        onnx_path = "/tmp/test_policy2.onnx"
        export_onnx(policy_path, onnx_path)

        # Load with ONNX Runtime
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_input = np.random.randn(1, OBS_DIM).astype(np.float32)
        outputs = sess.run(None, {"obs": dummy_input})

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, (1, ACT_DIM))
        self.assertEqual(outputs[1].shape, (1,))
        self.assertEqual(outputs[2].shape, (1,))


class TestMAVLinkBridge(unittest.TestCase):
    """Test MAVLink bridge (mock mode)."""

    def test_mock_transport(self):
        """Test mock transport records setpoints."""
        from src.control.mavlink_bridge import MavlinkBridge, MockTransport

        transport = MockTransport()
        bridge = MavlinkBridge(transport=transport)

        cmd = {
            "skill": "navigate_to",
            "params": {"x": 1.0, "y": 2.0, "z": 1.5, "yaw_deg": 90, "speed_ms": 0.8},
            "constraints": {"timeout_s": 20},
        }
        result = bridge.execute(cmd)

        self.assertEqual(result["skill"], "navigate_to")
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(transport.sent[0]["skill"], "navigate_to")
        self.assertTrue(bridge.heartbeat())

    def test_real_available_flag(self):
        """Test real_available property reflects MAVSDK availability."""
        from src.control.mavlink_bridge import MavlinkBridge, _REAL_AVAILABLE

        bridge = MavlinkBridge()
        self.assertEqual(bridge.real_available, _REAL_AVAILABLE)


class TestSystemIDFormat(unittest.TestCase):
    """Test system identification data format."""

    def test_sysid_data_structure(self):
        """Test that system ID data has expected fields."""
        # Expected fields for system ID
        required_fields = [
            "mass_kg",
            "arm_length_m",
            "max_thrust_N",
            "motor_tau_s",
            "inertia_xx",
            "inertia_yy",
            "inertia_zz",
            "thrust_coeff",
            "drag_coeff",
        ]

        # Mock system ID data (what would be collected from real flights)
        sysid_data = {
            "mass_kg": 0.027,
            "arm_length_m": 0.046,
            "max_thrust_N": 0.62,
            "motor_tau_s": 0.02,
            "inertia_xx": 1.4e-5,
            "inertia_yy": 1.4e-5,
            "inertia_zz": 2.2e-5,
            "thrust_coeff": 1.0e-5,
            "drag_coeff": 1.0e-6,
        }

        for field in required_fields:
            self.assertIn(field, sysid_data, f"Missing required field: {field}")
            self.assertIsInstance(sysid_data[field], (int, float))
            self.assertGreater(sysid_data[field], 0)

    def test_sysid_to_quadrotor_params(self):
        """Test conversion from sysid data to QuadrotorParams."""
        from src.sim.drone_dynamics import QuadrotorParams

        sysid = {
            "mass_kg": 0.027,
            "arm_length_m": 0.046,
            "max_thrust_N": 0.62,
            "motor_tau_s": 0.02,
            "inertia_xx": 1.4e-5,
            "inertia_yy": 1.4e-5,
            "inertia_zz": 2.2e-5,
        }

        params = QuadrotorParams(
            mass=sysid["mass_kg"],
            arm_length=sysid["arm_length_m"],
            max_thrust=sysid["max_thrust_N"],
            motor_time_constant=sysid["motor_tau_s"],
            inertia=np.diag([sysid["inertia_xx"], sysid["inertia_yy"], sysid["inertia_zz"]]).astype(np.float32),
        )

        self.assertAlmostEqual(params.mass, 0.027)
        self.assertAlmostEqual(params.arm_length, 0.046)
        self.assertAlmostEqual(params.max_thrust, 0.62)
        self.assertAlmostEqual(params.motor_time_constant, 0.02)


if __name__ == "__main__":
    unittest.main()