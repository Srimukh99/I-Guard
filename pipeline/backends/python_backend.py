"""
Python backend implementation using YOLO and existing detection pipeline.

Wraps the current Python-based detection system to conform to the
unified backend interface while maintaining all existing functionality.
"""

import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

# Add the project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .base_backend import BaseBackend, BackendCapabilities, Detection
from detection.frame_detector import FrameDetector
from detection.clip_verifier import ClipVerifier


class PythonBackend(BaseBackend):
    """Python-based backend using YOLO and CLIP verification."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Python backend with existing detection pipeline."""
        super().__init__(config)
        
        # Detection components
        self.frame_detector = None
        self.clip_verifier = None
        
        # Performance tracking
        self._processing_times = []
        self._frame_count = 0
    
    def _get_capabilities(self) -> BackendCapabilities:
        """Return Python backend capabilities."""
        return BackendCapabilities(
            max_streams=8,  # Conservative estimate for Python backend
            gpu_required=False,  # Can run on CPU, but benefits from GPU
            platform_requirements=[],  # Cross-platform
            performance_tier='medium',
            memory_usage='medium',
            cpu_intensive=True
        )
    
    def initialize(self) -> bool:
        """Initialize the Python detection pipeline."""
        try:
            # Initialize frame detector
            model_config = self.config.get('model', {})
            self.frame_detector = FrameDetector(
                model_path=model_config.get('path', 'yolov8n.pt'),
                input_size=model_config.get('input_size', 640),
                classes=model_config.get('classes', ["person", "gun", "knife"]),
                confidence_threshold=model_config.get('confidence_threshold', 0.5),
                events_config=model_config.get('events', {}),
                inference_mode="python"
            )
            
            # Initialize CLIP verifier if enabled
            clip_config = self.config.get('clip_verification', {})
            if clip_config.get('enabled', True):
                self.clip_verifier = ClipVerifier(
                    model_path=clip_config.get('model_path', 'mock_clip.pt'),
                    model_type=clip_config.get('model_type', 'simple'),
                    threshold=clip_config.get('threshold', 0.8)
                )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Python backend: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """Process a single frame using Python detection pipeline."""
        if not self.is_initialized:
            raise RuntimeError("Backend not initialized")
        
        if not self.validate_frame(frame):
            return []
        
        start_time = time.time()
        
        try:
            # Run YOLO detection
            detections = self.frame_detector.detect(frame)
            
            # Convert to standardized format
            results = []
            for det in detections:
                # For now, skip CLIP verification in backend (done in pipeline level)
                # Map class label to ID (simple mapping for now)
                class_id = hash(det.label) % 1000
                
                # Convert bbox to normalized format if needed
                x, y, w, h = det.bbox
                frame_h, frame_w = frame.shape[:2]
                norm_bbox = (x/frame_w, y/frame_h, (x+w)/frame_w, (y+h)/frame_h)
                
                detection = Detection(
                    class_id=class_id,
                    confidence=det.confidence,
                    bbox=norm_bbox,
                    class_name=det.label,
                    timestamp=timestamp,
                    frame_id=frame_id
                )
                results.append(detection)
            
            # Track performance
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            self._frame_count += 1
            
            # Keep only recent processing times
            if len(self._processing_times) > 100:
                self._processing_times = self._processing_times[-100:]
            
            return results
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            return []
    
    def process_batch(self, frames: List[np.ndarray], frame_ids: List[int], 
                     timestamps: List[float]) -> List[List[Detection]]:
        """Process a batch of frames (sequential for Python backend)."""
        if not self.is_initialized:
            raise RuntimeError("Backend not initialized")
        
        results = []
        for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps):
            frame_detections = self.process_frame(frame, frame_id, timestamp)
            results.append(frame_detections)
        
        return results
    
    def cleanup(self) -> None:
        """Clean up Python backend resources."""
        if self.frame_detector:
            # YOLO cleanup (if needed)
            pass
        
        if self.clip_verifier:
            # CLIP cleanup (if needed)
            pass
        
        self._is_initialized = False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        base_metrics = super().get_performance_metrics()
        
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0
        
        python_metrics = {
            'frames_processed': self._frame_count,
            'avg_processing_time': avg_time,
            'estimated_fps': fps,
            'clip_verification_enabled': self.clip_verifier is not None,
            'model_loaded': self.frame_detector is not None
        }
        
        return {**base_metrics, **python_metrics}
