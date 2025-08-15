"""
Base backend interface for I-Guard processing pipeline.

Defines the abstract interface that all backend implementations must follow,
ensuring consistent behavior across Python and DeepStream backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Standardized detection result across all backends."""
    
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    class_name: str
    timestamp: float
    frame_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        return {
            'class_id': self.class_id,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'class_name': self.class_name,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id
        }


@dataclass
class BackendCapabilities:
    """Backend capability information for automatic selection."""
    
    max_streams: int
    gpu_required: bool
    platform_requirements: List[str]  # e.g., ['linux', 'nvidia-gpu']
    performance_tier: str  # 'high', 'medium', 'low'
    memory_usage: str  # 'low', 'medium', 'high'
    cpu_intensive: bool
    
    def is_available(self) -> bool:
        """Check if backend is available on current system."""
        # Implementation will be provided by concrete backends
        return True


class BaseBackend(ABC):
    """Abstract base class for all processing backends."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration."""
        self.config = config
        self._is_initialized = False
        self._capabilities = self._get_capabilities()
    
    @abstractmethod
    def _get_capabilities(self) -> BackendCapabilities:
        """Return backend capabilities for selection logic."""
        pass
    
    @property
    def capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        return self._capabilities
    
    @property
    def is_initialized(self) -> bool:
        """Check if backend is properly initialized."""
        return self._is_initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the backend.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """
        Process a single frame and return detections.
        
        Args:
            frame: Input frame as numpy array
            frame_id: Unique frame identifier
            timestamp: Frame timestamp
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def process_batch(self, frames: List[np.ndarray], frame_ids: List[int], 
                     timestamps: List[float]) -> List[List[Detection]]:
        """
        Process a batch of frames for improved performance.
        
        Args:
            frames: List of input frames
            frame_ids: List of frame identifiers
            timestamps: List of frame timestamps
            
        Returns:
            List of detection lists (one per frame)
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from backend.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            'backend_type': self.__class__.__name__,
            'max_streams': self.capabilities.max_streams,
            'performance_tier': self.capabilities.performance_tier,
            'memory_usage': self.capabilities.memory_usage
        }
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate input frame format.
        
        Args:
            frame: Input frame to validate
            
        Returns:
            True if frame is valid
        """
        if frame is None:
            return False
        
        if not isinstance(frame, np.ndarray):
            return False
        
        if len(frame.shape) != 3:
            return False
        
        if frame.shape[2] != 3:  # Expecting RGB/BGR
            return False
        
        return True
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize {self.__class__.__name__}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
