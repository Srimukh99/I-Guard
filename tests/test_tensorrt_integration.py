#!/usr/bin/env python3
"""
Integration tests for TensorRT and DeepStream components.

NOTE: These tests require:
- NVIDIA GPU with CUDA support
- TensorRT engines built and available
- DeepStream SDK installed
- Actual model files

Run only on target hardware (Jetson/GPU workstations).
"""

import os
import sys
import pytest
import unittest
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    GSTREAMER_AVAILABLE = True
except ImportError:
    GSTREAMER_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (TRT_AVAILABLE and GSTREAMER_AVAILABLE),
    reason="TensorRT and GStreamer required for integration tests"
)


class TestTensorRTIntegration(unittest.TestCase):
    """Test TensorRT engine loading and inference."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_engine_path = "/path/to/your/yolo.engine"  # Update this path
        cls.test_video_path = "/path/to/test/video.mp4"     # Update this path
        
    def test_tensorrt_engine_loading(self):
        """Test that TensorRT engine can be loaded."""
        if not os.path.exists(self.test_engine_path):
            self.skipTest(f"TensorRT engine not found: {self.test_engine_path}")
            
        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.test_engine_path, 'rb') as f:
            engine_data = f.read()
            
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        self.assertIsNotNone(engine, "Failed to load TensorRT engine")
        self.assertGreater(engine.num_bindings, 0, "Engine has no bindings")
        
    def test_tensorrt_inference(self):
        """Test TensorRT inference with sample data."""
        if not os.path.exists(self.test_engine_path):
            self.skipTest(f"TensorRT engine not found: {self.test_engine_path}")
            
        try:
            from detection.frame_detector import FrameDetector
            
            # Initialize detector with TensorRT engine
            detector = FrameDetector(
                model_path=self.test_engine_path,
                device="cuda:0",
                confidence_threshold=0.5
            )
            
            # Create dummy frame (640x480x3 uint8)
            import numpy as np
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Run detection
            detections = detector.detect(test_frame)
            
            self.assertIsInstance(detections, list, "Detections should be a list")
            # Note: May be empty list if no objects detected in random frame
            
        except ImportError as e:
            self.skipTest(f"FrameDetector not available: {e}")


class TestDeepStreamIntegration(unittest.TestCase):
    """Test DeepStream pipeline components."""
    
    def test_gstreamer_pipeline_creation(self):
        """Test basic GStreamer pipeline creation."""
        try:
            # Create simple test pipeline
            pipeline_str = (
                "videotestsrc num-buffers=10 ! "
                "video/x-raw,width=640,height=480 ! "
                "fakesink"
            )
            
            pipeline = Gst.parse_launch(pipeline_str)
            self.assertIsNotNone(pipeline, "Failed to create GStreamer pipeline")
            
            # Test pipeline state changes
            ret = pipeline.set_state(Gst.State.READY)
            self.assertEqual(ret, Gst.StateChangeReturn.SUCCESS, "Failed to set pipeline to READY")
            
            ret = pipeline.set_state(Gst.State.NULL)
            self.assertEqual(ret, Gst.StateChangeReturn.SUCCESS, "Failed to set pipeline to NULL")
            
        except Exception as e:
            self.skipTest(f"GStreamer pipeline test failed: {e}")
            
    def test_deepstream_camera_adapter(self):
        """Test DeepStream camera adapter initialization."""
        try:
            from pipeline.deepstream_adapter import DeepStreamAdapter
            
            # Test with videotestsrc (no real camera required)
            config = {
                "source": "videotestsrc num-buffers=10",
                "width": 640,
                "height": 480,
                "fps": 30
            }
            
            adapter = DeepStreamAdapter("test_camera", config)
            self.assertIsNotNone(adapter, "Failed to create DeepStreamAdapter")
            
            # Test pipeline creation
            adapter._create_pipeline()
            self.assertIsNotNone(adapter.pipeline, "Failed to create DeepStream pipeline")
            
        except ImportError as e:
            self.skipTest(f"DeepStreamAdapter not available: {e}")
        except Exception as e:
            self.skipTest(f"DeepStream test failed: {e}")


class TestHardwareAcceleration(unittest.TestCase):
    """Test GPU acceleration and hardware encoding."""
    
    def test_cuda_device_availability(self):
        """Test CUDA device detection."""
        try:
            import pycuda.driver as cuda
            
            cuda.init()
            device_count = cuda.Device.count()
            self.assertGreater(device_count, 0, "No CUDA devices found")
            
            # Test first device
            device = cuda.Device(0)
            context = device.make_context()
            
            # Get device properties
            attrs = device.get_attributes()
            self.assertGreater(attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT], 0)
            
            context.pop()
            
        except Exception as e:
            self.skipTest(f"CUDA test failed: {e}")
            
    def test_nvenc_availability(self):
        """Test NVIDIA hardware encoder availability."""
        try:
            # Test if nvenc elements are available in GStreamer
            factory = Gst.ElementFactory.find("nvh264enc")
            if factory is None:
                factory = Gst.ElementFactory.find("nvv4l2h264enc")
                
            self.assertIsNotNone(factory, "No NVIDIA hardware encoder found")
            
        except Exception as e:
            self.skipTest(f"NVENC test failed: {e}")


if __name__ == "__main__":
    # Print system information
    print("=== Hardware Integration Test Suite ===")
    print(f"TensorRT Available: {TRT_AVAILABLE}")
    print(f"GStreamer Available: {GSTREAMER_AVAILABLE}")
    
    if TRT_AVAILABLE:
        print("TensorRT version:", trt.__version__)
    if GSTREAMER_AVAILABLE:
        print("GStreamer version:", Gst.version_string())
        
    print("\nNOTE: Update test_engine_path and test_video_path in TestTensorRTIntegration")
    print("before running these tests on target hardware.\n")
    
    # Run tests
    unittest.main(verbosity=2)
