"""
Comprehensive test suite for hybrid backend system.

Tests automatic backend selection, fallback behavior, performance comparison,
and integration with the inference pipeline.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.backends import (
    BaseBackend, 
    BackendCapabilities, 
    Detection, 
    PythonBackend, 
    BackendFactory
)


class MockBackend(BaseBackend):
    """Mock backend for testing purposes."""
    
    def __init__(self, config, fail_init=False, fail_processing=False):
        self.fail_init = fail_init
        self.fail_processing = fail_processing
        super().__init__(config)
    
    def _get_capabilities(self):
        return BackendCapabilities(
            max_streams=4,
            gpu_required=False,
            platform_requirements=[],
            performance_tier='low',
            memory_usage='low',
            cpu_intensive=True
        )
    
    def initialize(self):
        if self.fail_init:
            return False
        self._is_initialized = True
        return True
    
    def process_frame(self, frame, frame_id, timestamp):
        if self.fail_processing:
            raise RuntimeError("Mock processing failure")
        
        # Return mock detection
        return [Detection(
            class_id=0,
            confidence=0.8,
            bbox=(0.1, 0.1, 0.9, 0.9),
            class_name="person",
            timestamp=timestamp,
            frame_id=frame_id
        )]
    
    def process_batch(self, frames, frame_ids, timestamps):
        results = []
        for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps):
            results.append(self.process_frame(frame, frame_id, timestamp))
        return results
    
    def cleanup(self):
        self._is_initialized = False


class TestBaseBackend(unittest.TestCase):
    """Test BaseBackend abstract interface."""
    
    def test_abstract_methods(self):
        """Test that BaseBackend cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseBackend({})
    
    def test_detection_dataclass(self):
        """Test Detection dataclass functionality."""
        detection = Detection(
            class_id=1,
            confidence=0.95,
            bbox=(0.2, 0.3, 0.8, 0.7),
            class_name="gun",
            timestamp=time.time(),
            frame_id=123
        )
        
        # Test to_dict conversion
        det_dict = detection.to_dict()
        expected_keys = ['class_id', 'confidence', 'bbox', 'class_name', 'timestamp', 'frame_id']
        self.assertEqual(set(det_dict.keys()), set(expected_keys))
        self.assertEqual(det_dict['class_id'], 1)
        self.assertEqual(det_dict['confidence'], 0.95)
    
    def test_backend_capabilities(self):
        """Test BackendCapabilities dataclass."""
        caps = BackendCapabilities(
            max_streams=8,
            gpu_required=True,
            platform_requirements=['linux'],
            performance_tier='high',
            memory_usage='high',
            cpu_intensive=False
        )
        
        self.assertEqual(caps.max_streams, 8)
        self.assertTrue(caps.gpu_required)
        self.assertIn('linux', caps.platform_requirements)
    
    def test_mock_backend(self):
        """Test our mock backend implementation."""
        backend = MockBackend({})
        
        # Test initialization
        self.assertTrue(backend.initialize())
        self.assertTrue(backend.is_initialized)
        
        # Test frame validation
        valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertTrue(backend.validate_frame(valid_frame))
        
        invalid_frame = np.zeros((480, 640), dtype=np.uint8)  # Missing channel dimension
        self.assertFalse(backend.validate_frame(invalid_frame))
        
        # Test processing
        detections = backend.process_frame(valid_frame, 1, time.time())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "person")
        
        # Test cleanup
        backend.cleanup()
        self.assertFalse(backend.is_initialized)
    
    def test_context_manager(self):
        """Test backend context manager functionality."""
        with MockBackend({}) as backend:
            self.assertTrue(backend.is_initialized)
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = backend.process_frame(frame, 1, time.time())
            self.assertEqual(len(detections), 1)
        
        # Should be cleaned up automatically
        self.assertFalse(backend.is_initialized)


class TestPythonBackend(unittest.TestCase):
    """Test Python backend implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'path': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.4
            },
            'clip_verification': {
                'enabled': False  # Disable CLIP for testing
            }
        }
    
    @patch('pipeline.backends.python_backend.FrameDetector')
    def test_python_backend_initialization(self, mock_detector_class):
        """Test Python backend initialization."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        backend = PythonBackend(self.config)
        
        # Test capabilities
        caps = backend.capabilities
        self.assertEqual(caps.max_streams, 8)
        self.assertFalse(caps.gpu_required)
        self.assertEqual(caps.performance_tier, 'medium')
        
        # Test initialization
        self.assertTrue(backend.initialize())
        self.assertTrue(backend.is_initialized)
        
        # Verify detector was created with correct parameters
        mock_detector_class.assert_called_once()
    
    @patch('pipeline.backends.python_backend.FrameDetector')
    def test_python_backend_processing(self, mock_detector_class):
        """Test Python backend frame processing."""
        # Import Detection class for proper mock
        from detection.frame_detector import Detection
        
        # Set up mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            Detection(
                label='person',
                confidence=0.85,
                bbox=(100, 200, 80, 80)  # x, y, w, h format
            )
        ]
        mock_detector_class.return_value = mock_detector
        
        backend = PythonBackend(self.config)
        backend.initialize()
        
        # Test frame processing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = backend.process_frame(frame, 1, time.time())
        
        self.assertEqual(len(detections), 1)
        # The backend generates class_id from label hash and normalizes bbox
        self.assertEqual(detections[0].class_name, 'person')
        self.assertEqual(detections[0].confidence, 0.85)
        # Check that bbox is normalized (values between 0 and 1)
        self.assertTrue(all(0 <= coord <= 1 for coord in detections[0].bbox))
        
        # Verify detector was called
        mock_detector.detect.assert_called_once_with(frame)
    
    @patch('pipeline.backends.python_backend.FrameDetector')
    def test_python_backend_batch_processing(self, mock_detector_class):
        """Test Python backend batch processing."""
        mock_detector = Mock()
        mock_detector.detect.return_value = []
        mock_detector_class.return_value = mock_detector
        
        backend = PythonBackend(self.config)
        backend.initialize()
        
        # Test batch processing
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        frame_ids = [1, 2, 3]
        timestamps = [time.time() + i for i in range(3)]
        
        results = backend.process_batch(frames, frame_ids, timestamps)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_detector.detect.call_count, 3)


class TestBackendFactory(unittest.TestCase):
    """Test backend factory and automatic selection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = BackendFactory()
        self.config = {
            'backend': {'type': 'auto'},
            'streams': {'max_concurrent': 4},
            'performance': {'priority': 'balanced'}
        }
    
    def test_backend_discovery(self):
        """Test backend discovery."""
        available = self.factory.get_available_backends()
        
        # Python backend should always be available
        self.assertIn('python', available)
        
        # DeepStream availability depends on system
        # We can't guarantee it in test environment
    
    def test_python_backend_creation(self):
        """Test creating Python backend."""
        config = self.config.copy()
        config['backend']['type'] = 'python'
        
        with patch('pipeline.backends.python_backend.FrameDetector'):
            backend = self.factory.create_backend(config)
            self.assertIsInstance(backend, PythonBackend)
    
    def test_automatic_backend_selection(self):
        """Test automatic backend selection logic."""
        # Test high stream count preference
        config = self.config.copy()
        config['streams']['max_concurrent'] = 12
        
        with patch.object(self.factory, '_available_backends', {'python': PythonBackend}):
            backend_name = self.factory._select_best_backend(config)
            self.assertEqual(backend_name, 'python')  # Only available option

    def test_selection_logic_from_config(self):
        """Test the backend selection logic with various configs."""

        # --- Test Cases when DeepStream IS available ---
        # Mock that the factory has discovered the deepstream backend
        self.factory._available_backends['deepstream'] = MagicMock()

        test_cases_ds_available = [
            # High Priority -> DeepStream
            {'priority': 'high', 'streams': 2, 'expected': 'deepstream', 'msg': "High priority should select DeepStream"},
            # Compatibility Priority -> Python
            {'priority': 'compatibility', 'streams': 10, 'expected': 'python', 'msg': "Compatibility priority should select Python"},
            # Balanced, High stream count -> DeepStream
            {'priority': 'balanced', 'streams': 10, 'expected': 'deepstream', 'msg': "Balanced with high stream count should select DeepStream"},
            # Balanced, Low stream count -> Python
            {'priority': 'balanced', 'streams': 2, 'expected': 'python', 'msg': "Balanced with low stream count should select Python"},
        ]

        for case in test_cases_ds_available:
            with self.subTest(msg=case['msg']):
                config = {
                    'backend': {
                        'type': 'auto',
                        'auto_selection': {
                            'performance_priority': case['priority'],
                            'stream_threshold': 8
                        }
                    },
                    'cameras': [{} for _ in range(case['streams'])]
                }
                selected_backend = self.factory._select_best_backend(config)
                self.assertEqual(selected_backend, case['expected'])

        # --- Test Cases when DeepStream is NOT available ---
        self.factory._available_backends.pop('deepstream')
        config = {
            'backend': {
                'type': 'auto',
                'auto_selection': {'performance_priority': 'high'}
            },
            'cameras': []
        }
        selected_backend = self.factory._select_best_backend(config)
        self.assertEqual(selected_backend, 'python', "Should fall back to Python if DeepStream is not available")
    
    def test_backend_validation(self):
        """Test backend configuration validation."""
        # Test valid configuration
        self.assertTrue(self.factory.validate_backend_config('python', self.config))
        
        # Test invalid backend
        self.assertFalse(self.factory.validate_backend_config('nonexistent', self.config))
    
    def test_fallback_behavior(self):
        """Test fallback when preferred backend fails."""
        config = self.config.copy()
        config['backend']['type'] = 'nonexistent'
        
        with patch.object(self.factory, '_available_backends', {'python': PythonBackend}):
            with patch('pipeline.backends.python_backend.FrameDetector'):
                backend, backend_name = self.factory.create_backend_with_fallback(config)
                self.assertEqual(backend_name, 'python')
                self.assertIsInstance(backend, PythonBackend)
    
    def test_backend_comparison(self):
        """Test backend capability comparison."""
        comparison = self.factory.get_backend_comparison()
        
        # Should have at least Python backend
        self.assertIn('python', comparison)
        
        python_info = comparison['python']
        self.assertIn('max_streams', python_info)
        self.assertIn('performance_tier', python_info)


class TestHybridIntegration(unittest.TestCase):
    """Test integration between hybrid backends and inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'backend': {'type': 'python'},
            'model': {
                'path': 'yolov8n.pt',
                'confidence_threshold': 0.5
            },
            'clip_verification': {'enabled': False},
            'cameras': [{'id': 'cam01', 'source': 0}],
            'step1': {'pre_buffer_sec': 5, 'post_buffer_sec': 5},
            'storage': {'clips_dir': tempfile.mkdtemp()}
        }
    
    @patch('pipeline.inference_pipeline.EventQueue')
    @patch('pipeline.inference_pipeline.FrameDetector')  # Patch legacy detector
    @patch('pipeline.backends.python_backend.FrameDetector')
    @patch('detection.clip_verifier.ClipVerifier')
    def test_pipeline_backend_integration(self, mock_clip_class, mock_backend_detector_class, mock_legacy_detector_class, mock_queue_class):
        """Test that inference pipeline properly uses hybrid backends."""
        from pipeline.inference_pipeline import InferencePipeline
        
        # Set up mocks for legacy detector
        mock_legacy_detector = Mock()
        mock_legacy_detector.detect.return_value = []
        mock_legacy_detector_class.return_value = mock_legacy_detector
        
        # Set up mocks for backend detector
        mock_backend_detector = Mock()
        mock_backend_detector.detect.return_value = []
        mock_backend_detector_class.return_value = mock_backend_detector
        
        mock_clip = Mock()
        mock_clip_class.return_value = mock_clip
        
        mock_queue = Mock()
        mock_queue_class.return_value = mock_queue
        
        # Create pipeline
        pipeline = InferencePipeline(self.config, mock_queue)
        
        # Verify backend factory was created
        self.assertIsNotNone(pipeline.backend_factory)
        
        # Verify available backends
        available = pipeline.backend_factory.get_available_backends()
        self.assertIn('python', available)


class TestPerformanceComparison(unittest.TestCase):
    """Test performance comparison between backends."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        self.frame_ids = list(range(10))
        self.timestamps = [time.time() + i * 0.1 for i in range(10)]
    
    def test_mock_backend_performance(self):
        """Test performance measurement with mock backend."""
        backend = MockBackend({})
        
        with backend:
            start_time = time.time()
            
            for frame, frame_id, timestamp in zip(self.test_frames, self.frame_ids, self.timestamps):
                detections = backend.process_frame(frame, frame_id, timestamp)
                self.assertIsInstance(detections, list)
            
            elapsed = time.time() - start_time
            fps = len(self.test_frames) / elapsed
            
            # Mock backend should be very fast
            self.assertGreater(fps, 100)  # Should process >100 FPS
    
    def test_batch_vs_individual_processing(self):
        """Compare batch vs individual frame processing."""
        backend = MockBackend({})
        
        with backend:
            # Test individual processing
            start_time = time.time()
            individual_results = []
            for frame, frame_id, timestamp in zip(self.test_frames, self.frame_ids, self.timestamps):
                result = backend.process_frame(frame, frame_id, timestamp)
                individual_results.append(result)
            individual_time = time.time() - start_time
            
            # Test batch processing
            start_time = time.time()
            batch_results = backend.process_batch(self.test_frames, self.frame_ids, self.timestamps)
            batch_time = time.time() - start_time
            
            # Results should be equivalent
            self.assertEqual(len(individual_results), len(batch_results))
            
            # Batch processing might be faster (though not necessarily with mock)
            print(f"Individual: {individual_time:.4f}s, Batch: {batch_time:.4f}s")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_backend_initialization_failure(self):
        """Test handling of backend initialization failure."""
        backend = MockBackend({}, fail_init=True)
        
        self.assertFalse(backend.initialize())
        self.assertFalse(backend.is_initialized)
    
    def test_backend_processing_failure(self):
        """Test handling of backend processing failure."""
        backend = MockBackend({}, fail_processing=True)
        backend.initialize()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with self.assertRaises(RuntimeError):
            backend.process_frame(frame, 1, time.time())
    
    def test_invalid_frame_handling(self):
        """Test handling of invalid frames."""
        backend = MockBackend({})
        backend.initialize()
        
        # Test None frame
        self.assertFalse(backend.validate_frame(None))
        
        # Test wrong type
        self.assertFalse(backend.validate_frame("not_a_frame"))
        
        # Test wrong dimensions
        wrong_dims = np.zeros((480, 640), dtype=np.uint8)
        self.assertFalse(backend.validate_frame(wrong_dims))
        
        # Test wrong channels
        wrong_channels = np.zeros((480, 640, 4), dtype=np.uint8)
        self.assertFalse(backend.validate_frame(wrong_channels))
    
    def test_context_manager_with_failure(self):
        """Test context manager behavior when initialization fails."""
        with self.assertRaises(RuntimeError):
            with MockBackend({}, fail_init=True) as backend:
                pass  # Should not reach here


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test suite
    unittest.main(verbosity=2)
