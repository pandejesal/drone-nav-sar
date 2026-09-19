"""DroneNav-SAR perception package (Sprint 15).

SAR-only: victim detection, obstacle avoidance, SLAM.
Edge-optimized for Jetson Orin (ONNX -> TensorRT FP16).
"""

from src.perception.yolo_detector import YOLODetector, Detection, create_detector
from src.perception.depth_estimator import DepthEstimator, create_depth_estimator
from src.perception.slam import VIOSLAM, Pose, create_slam
from src.perception.fusion import MultiSensorFusion, FusionOutput, create_fusion

__all__ = [
    "YOLODetector", "Detection", "create_detector",
    "DepthEstimator", "create_depth_estimator",
    "VIOSLAM", "Pose", "create_slam",
    "MultiSensorFusion", "FusionOutput", "create_fusion",
]
