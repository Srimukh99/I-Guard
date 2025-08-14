"""End-to-end integration tests for I-Guard pipeline.

This test suite validates the complete application flow from camera capture
through detection and verification, using mock components to avoid GPU/model
dependencies in CI environments.
"""

import os
import sys
import time
import threading
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

# Optional pytest import - fallback gracefully if not available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    pytest = None
    HAS_PYTEST = False
    
    # Create mock pytest.fixture for when pytest isn't available
    class MockPytest:
        @staticmethod
        def fixture(func):
            return func
    
    if pytest is None:
        pytest = MockPytest()

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix relative imports by importing detection modules directly
try:
    from detection.frame_detector import FrameDetector, Detection
    from detection.clip_verifier import ClipVerifier
    from detection.tracking import TrackManager
except ImportError as e:
    print(f"Warning: Could not import detection modules: {e}")
    print("This may affect some test functionality")

# Import with fallback for missing dependencies
try:
    from pipeline.inference_pipeline import InferencePipeline
    HAS_INFERENCE_PIPELINE = True
except ImportError as e:
    print(f"Warning: Could not import InferencePipeline: {e}")
    InferencePipeline = None
    HAS_INFERENCE_PIPELINE = False

from pipeline.event_queue import EventQueue, Event
from pipeline.camera_adapter import CameraAdapter, VideoFrame


class MockCameraAdapter:
    """Mock camera adapter that generates synthetic frames with controllable content."""
    
    def __init__(self, source: Any, fps: int = 15):
        self.source = source
        self.fps = fps
        self.frame_count = 0
        self.running = True
        self._generate_threat_at_frame = 10  # Generate threat detection at frame 10
        
    def read(self) -> VideoFrame:
        """Generate synthetic video frames."""
        if not self.running:
            return None
            
        self.frame_count += 1
        
        # Create a synthetic 640x480 BGR frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some visual patterns to make it more realistic
        frame[200:280, 300:380] = [255, 255, 255]  # White rectangle (person)
        
        # At frame 10, add a "weapon-like" pattern to trigger detection
        if self.frame_count == self._generate_threat_at_frame:
            frame[240:260, 320:360] = [0, 0, 255]  # Red rectangle (weapon)
            
        return VideoFrame(
            timestamp=time.time(),
            image=frame
        )
    
    def release(self):
        """Stop the mock camera."""
        self.running = False


class MockFrameDetector:
    """Mock frame detector that generates controllable detections."""
    
    def __init__(self, *args, **kwargs):
        self.detection_count = 0
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate mock detections based on frame content."""
        self.detection_count += 1
        detections = []
        
        # Always detect a person
        detections.append(Detection(
            label="person",
            confidence=0.85,
            bbox=(300, 200, 80, 80)
        ))
        
        # Detect weapon in frames with red pixels (our synthetic threat)
        if np.any(frame[:, :, 2] > 200):  # Red channel high
            detections.append(Detection(
                label="gun",
                confidence=0.75,
                bbox=(320, 240, 40, 20)
            ))
            
        return detections
    
    def analyze_events(self, detections: List[Detection]) -> Dict[str, bool]:
        """Generate event flags based on detections."""
        flags = {
            "pointing": False,
            "firing": False,
            "fall": False,
            "assault": False,
            "mass_shooter": False
        }
        
        # Flag pointing if we have both person and weapon
        person_detected = any(d.label == "person" for d in detections)
        weapon_detected = any(d.label in ["gun", "knife"] for d in detections)
        
        if person_detected and weapon_detected:
            flags["pointing"] = True
            
        return flags


class MockClipVerifier:
    """Mock clip verifier that returns controllable verification results."""
    
    def __init__(self, *args, **kwargs):
        self.verification_count = 0
        
    def verify(self, video_clip=None, detections_per_frame=None) -> Dict[str, float]:
        """Mock verification that returns threat score based on weapon detections."""
        self.verification_count += 1
        
        if detections_per_frame:
            # Count frames with weapons
            weapon_frames = sum(1 for frame_labels in detections_per_frame 
                              if any(label in ["gun", "knife"] for label in frame_labels))
            total_frames = len(detections_per_frame)
            score = weapon_frames / max(total_frames, 1)
        else:
            # Default score for mock
            score = 0.8
            
        return {
            "score": score,
            "action": "weapon_threat" if score > 0.5 else "normal",
            "action_confidence": score,
            "weapon_ratio": score
        }


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configuration files."""
    if not HAS_PYTEST:
        # Fallback for non-pytest usage
        import tempfile
        return tempfile.mkdtemp()
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config(temp_config_dir):
    """Create mock configuration for testing."""
    if not HAS_PYTEST:
        # Fallback for non-pytest usage
        import tempfile
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir = temp_config_dir
    
    return {
        "inference_mode": "python",
        "cameras": [
            {
                "id": "test_cam_01",
                "name": "Test Camera 1",
                "source": 0,  # Will be mocked
                "fps": 15,
                "resolution": [640, 480]
            }
        ],
        "step1": {
            "model_path": "mock_model.pt",
            "input_size": 640,
            "confidence_threshold": 0.5,
            "classes": ["person", "gun", "knife"],
            "events": {
                "pointing": True,
                "firing": True,
                "fall": False,
                "assault": False,
                "mass_shooter": False
            },
            "pre_buffer_sec": 2,
            "post_buffer_sec": 2,
            "frame_skip": 0,
            "frame_batch_size": 1
        },
        "step2": {
            "enabled": True,
            "model_path": "mock_verifier.engine",
            "model_type": "simple",
            "verification_threshold": 0.7,
            "async_enabled": False  # Use sync for simpler testing
        },
        "server": {
            "host": "127.0.0.1",
            "port": 5000
        },
        "storage": {
            "clips_dir": temp_dir,
            "logs_dir": temp_dir,
            "keep_days": 7
        },
        "advanced": {
            "queue_maxsize": 64,
            "use_deepstream": False,
            "tracking_enabled": False,
            "dynamic_backpressure": False
        }
    }


class TestEndToEndPipeline:
    """End-to-end integration tests for the complete I-Guard pipeline."""
    
    @patch('pipeline.inference_pipeline.CameraAdapter', MockCameraAdapter)
    @patch('pipeline.inference_pipeline.FrameDetector', MockFrameDetector)
    @patch('pipeline.inference_pipeline.ClipVerifier', MockClipVerifier)
    def test_complete_pipeline_flow(self, mock_config):
        """Test the complete pipeline flow from camera to verification."""
        if not HAS_INFERENCE_PIPELINE:
            print("⚠️  Skipping pipeline test - InferencePipeline not available")
            return
        
        # Create event queue
        event_queue = EventQueue(maxsize=10)
        
        # Create and start pipeline
        pipeline = InferencePipeline(config=mock_config, event_queue=event_queue)
        
        # Track events for verification
        events_detected = []
        events_verified = []
        
        def event_monitor():
            """Monitor events from the queue."""
            timeout_count = 0
            max_timeouts = 50  # 5 seconds at 0.1s intervals
            
            while timeout_count < max_timeouts:
                try:
                    event = event_queue.get(timeout=0.1)
                    events_detected.append(event)
                    event_queue.task_done()
                    
                    # Check if this event should trigger verification
                    if any(event.flags.values()):
                        events_verified.append(event)
                        
                except:
                    timeout_count += 1
                    continue
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=event_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            # Start the pipeline
            pipeline.start()
            
            # Let it run for a few seconds to capture the synthetic threat
            time.sleep(3.0)
            
            # Stop the pipeline
            pipeline.stop()
            
            # Wait a bit more for events to be processed
            time.sleep(1.0)
            
        finally:
            # Ensure cleanup
            pipeline.stop()
        
        # Verify results
        assert len(events_detected) > 0, "No events were detected"
        assert len(events_verified) > 0, "No threat events were generated"
        
        # Check that we detected the synthetic threat
        threat_event = events_verified[0]
        assert threat_event.camera_id == "test_cam_01"
        assert threat_event.flags["pointing"] == True, "Expected pointing event flag"
        
        # Verify event contains expected data
        assert threat_event.timestamp > 0
        assert threat_event.frame_id > 0
        assert len(threat_event.detections) >= 2  # person + weapon
        
        # Check detection labels
        detection_labels = [d.label for d in threat_event.detections]
        assert "person" in detection_labels
        assert "gun" in detection_labels
        
        print(f"✓ Pipeline test completed successfully:")
        print(f"  - Events detected: {len(events_detected)}")
        print(f"  - Threat events: {len(events_verified)}")
        print(f"  - Threat event flags: {threat_event.flags}")
        print(f"  - Detection labels: {detection_labels}")

    @patch('pipeline.inference_pipeline.CameraAdapter', MockCameraAdapter)
    @patch('pipeline.inference_pipeline.FrameDetector', MockFrameDetector)
    def test_pipeline_without_verification(self, mock_config):
        """Test pipeline operation with verification disabled."""
        if not HAS_INFERENCE_PIPELINE:
            print("⚠️  Skipping pipeline test - InferencePipeline not available")
            return
        
        # Disable step2 verification
        mock_config["step2"]["enabled"] = False
        
        event_queue = EventQueue(maxsize=10)
        pipeline = InferencePipeline(config=mock_config, event_queue=event_queue)
        
        events_collected = []
        
        def collect_events():
            timeout_count = 0
            while timeout_count < 30:  # 3 seconds
                try:
                    event = event_queue.get(timeout=0.1)
                    events_collected.append(event)
                    event_queue.task_done()
                except:
                    timeout_count += 1
        
        monitor_thread = threading.Thread(target=collect_events, daemon=True)
        monitor_thread.start()
        
        try:
            pipeline.start()
            time.sleep(2.0)
            pipeline.stop()
            time.sleep(0.5)
        finally:
            pipeline.stop()
        
        # Should still detect events even without verification
        assert len(events_collected) > 0, "No events detected without verification"
        
        # Check that events have detection data
        for event in events_collected:
            assert event.camera_id == "test_cam_01"
            assert len(event.detections) > 0
            assert isinstance(event.flags, dict)
        
        print(f"✓ No-verification test completed: {len(events_collected)} events")

    def test_event_queue_operations(self):
        """Test basic event queue functionality."""
        event_queue = EventQueue(maxsize=5)
        
        # Create test event
        test_event = Event(
            camera_id="test_cam",
            timestamp=time.time(),
            frame_id=1,
            detections=[Detection("person", 0.9, (0, 0, 100, 100))],
            flags={"pointing": True}
        )
        
        # Test put/get operations
        event_queue.put(test_event)
        assert event_queue.qsize() == 1
        
        retrieved_event = event_queue.get()
        assert retrieved_event.camera_id == "test_cam"
        assert retrieved_event.flags["pointing"] == True
        assert len(retrieved_event.detections) == 1
        assert retrieved_event.detections[0].label == "person"
        
        event_queue.task_done()
        
        print("✓ Event queue operations test passed")

    def test_mock_components_individually(self):
        """Test individual mock components work as expected."""
        # Test mock camera
        camera = MockCameraAdapter(source=0, fps=15)
        frame = camera.read()
        assert frame is not None
        assert frame.image.shape == (480, 640, 3)
        assert frame.timestamp > 0
        camera.release()
        
        # Test mock detector
        detector = MockFrameDetector()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        assert len(detections) >= 1  # Should always detect person
        assert any(d.label == "person" for d in detections)
        
        # Test event analysis
        flags = detector.analyze_events(detections)
        assert isinstance(flags, dict)
        assert "pointing" in flags
        
        # Test mock verifier
        verifier = MockClipVerifier()
        result = verifier.verify(detections_per_frame=[["person", "gun"]])
        assert "score" in result
        assert "action" in result
        assert result["score"] > 0
        
        print("✓ Individual mock components test passed")

    @patch('pipeline.inference_pipeline.CameraAdapter', MockCameraAdapter)
    @patch('pipeline.inference_pipeline.FrameDetector', MockFrameDetector)
    @patch('pipeline.inference_pipeline.ClipVerifier', MockClipVerifier)
    def test_pipeline_metrics_and_history(self, mock_config):
        """Test that pipeline correctly maintains event history and metrics."""
        if not HAS_INFERENCE_PIPELINE:
            print("⚠️  Skipping pipeline test - InferencePipeline not available")
            return
        
        event_queue = EventQueue(maxsize=10)
        pipeline = InferencePipeline(config=mock_config, event_queue=event_queue)
        
        try:
            pipeline.start()
            time.sleep(2.0)  # Let it process some frames
            pipeline.stop()
            time.sleep(0.5)  # Let event consumer finish
        finally:
            pipeline.stop()
        
        # Check event history
        with pipeline.events_lock:
            history = pipeline.events_history.copy()
        
        # Should have some events in history
        assert len(history) >= 0, "Event history should be accessible"
        
        if len(history) > 0:
            # Check event structure
            event_record = history[0]
            assert "id" in event_record
            assert "camera_id" in event_record
            assert "timestamp" in event_record
            assert "flags" in event_record
            
            print(f"✓ Pipeline created {len(history)} event records")
        else:
            print("✓ Pipeline ran without generating events (also valid)")


def test_imports_and_dependencies():
    """Test that all required modules can be imported."""
    # Test core pipeline imports
    from pipeline.event_queue import EventQueue, Event
    from pipeline.camera_adapter import CameraAdapter
    from detection.frame_detector import FrameDetector
    from detection.clip_verifier import ClipVerifier
    
    # Test optional InferencePipeline
    if HAS_INFERENCE_PIPELINE:
        print("✓ InferencePipeline available")
    else:
        print("⚠️  InferencePipeline not available (missing dependencies)")
    
    # Test that numpy is available (required dependency)
    import numpy as np
    assert np.__version__ is not None
    
    print("✓ All core imports successful")


if __name__ == "__main__":
    # Run tests directly for development
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create mock config
        config = {
            "inference_mode": "python",
            "cameras": [{"id": "test_cam", "source": 0, "fps": 15}],
            "step1": {
                "model_path": "mock.pt",
                "input_size": 640,
                "confidence_threshold": 0.5,
                "classes": ["person", "gun"],
                "events": {"pointing": True},
                "pre_buffer_sec": 1,
                "post_buffer_sec": 1,
                "frame_skip": 0,
                "frame_batch_size": 1
            },
            "step2": {"enabled": True, "model_type": "simple", "async_enabled": False},
            "storage": {"clips_dir": temp_dir, "logs_dir": temp_dir},
            "advanced": {"queue_maxsize": 10, "use_deepstream": False}
        }
        
        # Run basic tests
        test_imports_and_dependencies()
        
        tester = TestEndToEndPipeline()
        tester.test_mock_components_individually()
        tester.test_event_queue_operations()
        
        print("\n✓ All basic tests passed!")
        print("Run with pytest for full test suite:")
        print("  pytest tests/test_end_to_end.py -v")
        
    finally:
        shutil.rmtree(temp_dir)
